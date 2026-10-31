/** curans - streaming rANS compressor */
#include "frame.h"
#include "gpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const size_t kMaxFrame = 256u << 20;

static void usage(const char* argv0) {
    fprintf(stderr,
            "usage:\n"
            "  %s -c [in] [out]     compress  (default: stdin -> stdout)\n"
            "  %s -d [in] [out]     decompress\n"
            "  %s --self-test       check GPU output against the CPU reference\n"
            "options:\n"
            "  -f, --frame-mib N    plaintext frame size in MiB (default 16)\n"
            "      --cpu            use the CPU path only\n",
            argv0, argv0, argv0);
}

static double now_s() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static bool read_exact(void* p, size_t n, FILE* f) {
    return fread(p, 1, n, f) == n;
}

static void* host_buf(size_t n, bool want_pinned, bool* pinned) {
    void* p = nullptr;
    *pinned = false;
    if (want_pinned && cudaHostAlloc(&p, n, cudaHostAllocDefault) == cudaSuccess) {
        *pinned = true;
        return p;
    }
    return malloc(n);
}

static void host_buf_free(void* p, bool pinned) {
    if (!p) return;
    if (pinned) cudaFreeHost(p);
    else free(p);
}

static int do_compress(FILE* in, FILE* out, size_t frame_bytes, bool want_gpu) {
    StreamHdr sh;
    memset(&sh, 0, sizeof(sh));
    sh.magic = kStreamMagic;
    sh.version = kStreamVersion;
    sh.chunk_size = kChunk;
    sh.lanes = kLanes;
    sh.frame_size = (uint32_t)frame_bytes;
    if (fwrite(&sh, sizeof(sh), 1, out) != 1) { fprintf(stderr, "write failed\n"); return 1; }

    GpuCtx gpu;
    bool on_gpu = want_gpu && gpu_init(&gpu, frame_bytes);
    if (want_gpu && !on_gpu) fprintf(stderr, "note: no usable GPU, using CPU path\n");
    if (on_gpu) fprintf(stderr, "device: %s\n", gpu_name());

    const size_t nlanes = (frame_bytes / kChunk + 1) * kLanes;
    const size_t paycap = 2 * frame_bytes + 8 * nlanes + 1024;
    const size_t reccap = frame_bound(frame_bytes);

    bool pin_src = false, pin_pay = false, pin_rec = false;
    uint8_t* src = (uint8_t*)host_buf(frame_bytes, on_gpu, &pin_src);
    uint8_t* pay = (uint8_t*)host_buf(paycap, on_gpu, &pin_pay);
    uint8_t* rec = (uint8_t*)host_buf(reccap, on_gpu, &pin_rec);
    uint16_t* lens = (uint16_t*)malloc(nlanes * sizeof(uint16_t));
    uint16_t freq[256];
    if (!src || !pay || !rec || !lens) { fprintf(stderr, "out of memory\n"); return 1; }
    uint64_t tin = 0, tout = 0;
    double t0 = now_s();
    size_t n;
    while ((n = fread(src, 1, frame_bytes, in)) > 0) {
        tin += n;
        size_t m;
        if (on_gpu) {
            size_t plen = 0;
            if (!gpu_compress_frame(&gpu, src, n, pay, paycap, &plen, freq, lens)) {
                fprintf(stderr, "gpu compress failed\n");
                return 1;
            }
            m = frame_assemble(rec, reccap, src, n, freq, lens, plen, pay);
        } else {
            m = cpu_compress_frame(src, n, rec, reccap);
        }
        if (!m) { fprintf(stderr, "frame encode failed\n"); return 1; }
        if (fwrite(rec, 1, m, out) != m) { fprintf(stderr, "write failed\n"); return 1; }
        tout += m;
    }
    double dt = now_s() - t0;

    FrameHdr end;
    memset(&end, 0, sizeof(end));
    if (fwrite(&end, sizeof(end), 1, out) != 1) { fprintf(stderr, "write failed\n"); return 1; }

    if (ferror(in)) { fprintf(stderr, "read failed\n"); return 1; }
    fprintf(stderr, "in %.1f MiB  out %.1f MiB  ratio %.3fx  %.0f MB/s (%s)\n",
            tin / 1048576.0, tout / 1048576.0, tout ? (double)tin / tout : 0.0,
            dt > 0 ? tin / 1048576.0 / dt : 0.0, on_gpu ? "gpu" : "cpu");
    if (on_gpu) gpu_free(&gpu);
    host_buf_free(src, pin_src);
    host_buf_free(pay, pin_pay);
    host_buf_free(rec, pin_rec);
    free(lens);
    return 0;
}

static int do_decompress(FILE* in, FILE* out, bool want_gpu) {
    StreamHdr sh;
    if (!read_exact(&sh, sizeof(sh), in) || sh.magic != kStreamMagic ||
        sh.version != kStreamVersion || sh.chunk_size != kChunk || sh.lanes != kLanes ||
        sh.frame_size == 0 || sh.frame_size > kMaxFrame) {
        fprintf(stderr, "not a curans stream (or unsupported version)\n");
        return 1;
    }
    const size_t frame = sh.frame_size;

    GpuCtx gpu;
    bool on_gpu = want_gpu && gpu_init(&gpu, frame);
    if (want_gpu && !on_gpu) fprintf(stderr, "note: no usable GPU, using CPU path\n");

    const size_t nlanes = (frame / kChunk + 1) * kLanes;
    const size_t paycap = 2 * frame + 8 * nlanes + 1024;
    const size_t reccap = frame_bound(frame);

    bool pin_pay = false, pin_out = false;
    uint8_t* pay = (uint8_t*)host_buf(paycap, on_gpu, &pin_pay);
    uint8_t* outbuf = (uint8_t*)host_buf(frame, on_gpu, &pin_out);
    uint8_t* rec = (uint8_t*)malloc(reccap);
    uint16_t* lens = (uint16_t*)malloc(nlanes * sizeof(uint16_t));
    if (!pay || !outbuf || !rec || !lens) { fprintf(stderr, "out of memory\n"); return 1; }

    uint64_t tin = 0, tout = 0;
    double t0 = now_s();
    bool saw_end = false;
    for (;;) {
        FrameHdr fh;
        size_t got = fread(&fh, 1, sizeof(fh), in);
        if (got == 0) break;
        if (got != sizeof(fh)) { fprintf(stderr, "truncated frame header\n"); return 1; }
        if (!fh.nchunks && !fh.raw_len && !fh.flags) { saw_end = true; break; }
        tin += got;
        if (fh.raw_len > frame) { fprintf(stderr, "corrupt frame (size)\n"); return 1; }
        if (fh.flags & ~(uint32_t)kFlagStored) { fprintf(stderr, "corrupt frame (flags)\n"); return 1; }

        if (fh.flags & kFlagStored) {
            if (!read_exact(pay, fh.raw_len, in)) { fprintf(stderr, "truncated\n"); return 1; }
            if (fwrite(pay, 1, fh.raw_len, out) != fh.raw_len) return 1;
            tin += fh.raw_len;
            tout += fh.raw_len;
            continue;
        }
        const uint32_t nchunks = (uint32_t)((fh.raw_len + kChunk - 1) / kChunk);
        if (fh.nchunks != nchunks || nchunks == 0) {
            fprintf(stderr, "corrupt frame (chunks)\n");
            return 1;
        }
        uint32_t fsum = 0;
        for (int i = 0; i < 256; i++) fsum += fh.freq[i];
        if (fsum != RANS_M) { fprintf(stderr, "corrupt frame (table)\n"); return 1; }

        const size_t nl = (size_t)nchunks * kLanes;
        if (nl > nlanes) { fprintf(stderr, "corrupt frame (lanes)\n"); return 1; }
        if (!read_exact(lens, nl * sizeof(uint16_t), in)) { fprintf(stderr, "truncated\n"); return 1; }
        size_t plen = 0;
        for (size_t i = 0; i < nl; i++) plen += lens[i];
        if (plen > paycap) { fprintf(stderr, "corrupt frame (payload)\n"); return 1; }
        if (!read_exact(pay, plen, in)) { fprintf(stderr, "truncated\n"); return 1; }
        tin += nl * sizeof(uint16_t) + plen;

        if (on_gpu) {
            if (!gpu_decompress_frame(&gpu, pay, plen, fh.freq, lens, nchunks,
                                      fh.raw_len, outbuf)) {
                fprintf(stderr, "gpu decompress failed\n");
                return 1;
            }
        } else {
            memcpy(rec, &fh, sizeof(fh));
            memcpy(rec + sizeof(fh), lens, nl * sizeof(uint16_t));
            memcpy(rec + frame_payload_off(nchunks), pay, plen);
            if (cpu_decompress_frame(rec, frame_payload_off(nchunks) + plen, outbuf,
                                     frame) != frame_payload_off(nchunks) + plen) {
                fprintf(stderr, "corrupt frame (decode)\n");
                return 1;
            }
        }
        if (fwrite(outbuf, 1, fh.raw_len, out) != fh.raw_len) return 1;
        tout += fh.raw_len;
    }
    if (!saw_end) { fprintf(stderr, "truncated stream (missing end marker)\n"); return 1; }
    double dt = now_s() - t0;
    fprintf(stderr, "in %.1f MiB  out %.1f MiB  %.0f MB/s (%s)\n", tin / 1048576.0,
            tout / 1048576.0, dt > 0 ? tout / 1048576.0 / dt : 0.0,
            on_gpu ? "gpu" : "cpu");
    if (on_gpu) gpu_free(&gpu);
    host_buf_free(pay, pin_pay);
    host_buf_free(outbuf, pin_out);
    free(rec);
    free(lens);
    return 0;
}

static int self_test() {
    const size_t frame = 2u << 20;
    GpuCtx gpu;
    if (!gpu_init(&gpu, frame)) { fprintf(stderr, "no usable GPU\n"); return 1; }
    fprintf(stderr, "device: %s\n", gpu_name());

    const size_t nlanes = (frame / kChunk + 1) * kLanes;
    const size_t paycap = 2 * frame + 8 * nlanes + 1024;
    const size_t reccap = frame_bound(frame);
    uint8_t* src = (uint8_t*)malloc(frame);
    uint8_t* pay = (uint8_t*)malloc(paycap);
    uint8_t* a = (uint8_t*)malloc(reccap);
    uint8_t* b = (uint8_t*)malloc(reccap);
    uint8_t* out = (uint8_t*)malloc(frame);
    uint8_t* out2 = (uint8_t*)malloc(frame);
    uint16_t* lens = (uint16_t*)malloc(nlanes * sizeof(uint16_t));
    uint16_t freq[256];

    const size_t sizes[] = {1, 63, 65535, 65536, 65537, 300000, frame};
    uint32_t rng = 0x9e3779b9u;
    bool all = true;
    for (size_t n : sizes) {
        for (int kind = 0; kind < 3; kind++) {
            for (size_t i = 0; i < n; i++) {
                rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
                if (kind == 0) src[i] = (uint8_t)rng;
                else if (kind == 1) src[i] = (uint8_t)("etaoin shrdlu"[rng % 13]);
                else src[i] = (uint8_t)((i / 512) & 0xff);
            }
            size_t m1 = cpu_compress_frame(src, n, a, reccap);
            size_t plen = 0;
            bool ok = m1 != 0 &&
                      gpu_compress_frame(&gpu, src, n, pay, paycap, &plen, freq, lens);
            size_t m2 = ok ? frame_assemble(b, reccap, src, n, freq, lens, plen, pay) : 0;
            bool same = ok && m1 == m2 && memcmp(a, b, m1) == 0;
            bool rt1 = m1 && cpu_decompress_frame(a, m1, out, frame) == m1 &&
                       memcmp(out, src, n) == 0;
            bool rt2 = m2 && gpu_decompress_frame(&gpu, pay, plen, freq, lens,
                                                  (n + kChunk - 1) / kChunk, n, out2) &&
                       memcmp(out2, src, n) == 0;
            bool pass = same && rt1 && rt2;
            all &= pass;
            printf("n=%8zu kind=%d  cpu=%8zu gpu=%8zu  %5.2fx  bytes:%s  cpu-rt:%s  gpu-rt:%s\n",
                   n, kind, m1, m2, m2 ? (double)n / m2 : 0.0,
                   same ? "ok" : "DIFF", rt1 ? "ok" : "BAD", rt2 ? "ok" : "BAD");
        }
    }
    printf("\n%s\n", all ? "SELF-TEST PASSED" : "SELF-TEST FAILED");
    gpu_free(&gpu);
    free(src); free(pay); free(a); free(b); free(out); free(out2); free(lens);
    return all ? 0 : 1;
}

/** entry point */
int main(int argc, char** argv) {
    bool comp = false, decomp = false, cpu_only = false, selftest = false;
    size_t frame_mib = 16;
    const char* pos[2] = {nullptr, nullptr};
    int npos = 0;

    for (int i = 1; i < argc; i++) {
        const char* a = argv[i];
        if (!strcmp(a, "-c")) comp = true;
        else if (!strcmp(a, "-d")) decomp = true;
        else if (!strcmp(a, "--cpu")) cpu_only = true;
        else if (!strcmp(a, "--self-test")) selftest = true;
        else if (!strcmp(a, "-f") || !strcmp(a, "--frame-mib")) {
            if (++i >= argc) { usage(argv[0]); return 1; }
            frame_mib = strtoul(argv[i], nullptr, 10);
        } else if (!strcmp(a, "-h") || !strcmp(a, "--help")) { usage(argv[0]); return 0; }
        else if (a[0] == '-' && a[1]) { usage(argv[0]); return 1; }
        else if (npos < 2) pos[npos++] = a;
        else { usage(argv[0]); return 1; }
    }
    if (selftest) return self_test();
    if (comp == decomp) { usage(argv[0]); return 1; }
    if (frame_mib < 1 || frame_mib > 256) { fprintf(stderr, "bad frame size\n"); return 1; }

    FILE* in = stdin;
    FILE* out = stdout;
    if (npos > 0 && strcmp(pos[0], "-")) {
        in = fopen(pos[0], "rb");
        if (!in) { perror(pos[0]); return 1; }
    }
    if (npos > 1 && strcmp(pos[1], "-")) {
        out = fopen(pos[1], "wb");
        if (!out) { perror(pos[1]); return 1; }
    }
    int rc = comp ? do_compress(in, out, frame_mib << 20, !cpu_only)
                  : do_decompress(in, out, !cpu_only);
    if (in != stdin) fclose(in);
    if (out != stdout && fclose(out) != 0) { perror("close"); return 1; }
    return rc;
}
