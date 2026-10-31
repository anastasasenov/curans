/** CUDA kernels and the GPU driver for one frame. */
#include "gpu.h"
#include "frame.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

/** @note __match_any_sync() was added in CUDA 11 and only exists for sm_70+. */
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11000)
#define CURANS_HAS_MATCH_ANY_SYNC 1
#else
#define CURANS_HAS_MATCH_ANY_SYNC 0
#endif

#if CURANS_HAS_MATCH_ANY_SYNC && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
#define CURANS_WARP_MATCH 1
#else
#define CURANS_WARP_MATCH 0
#endif

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t err_ = (expr);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(err_), __FILE__, __LINE__);              \
            return false;                                                       \
        }                                                                       \
    } while (0)

/** One 256-bin histogram per block in shared memory */
__global__ void k_hist(const uint8_t* __restrict__ src, size_t n,
                       uint32_t* __restrict__ hist) {
    __shared__ uint32_t h[4 * 256];
    for (int i = threadIdx.x; i < 4 * 256; i += blockDim.x) h[i] = 0;
    __syncthreads();

    const int warp = threadIdx.x >> 5;
    uint32_t* mine = h + (warp & 3) * 256;
    const int lane = threadIdx.x & 31;
    const size_t stride = (size_t)gridDim.x * blockDim.x;

    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += stride) {
        uint32_t b = src[i];
#if CURANS_WARP_MATCH
        unsigned active = __activemask();
        unsigned peers = __match_any_sync(active, b);
        if ((__ffs(peers) - 1) == lane) atomicAdd(&mine[b], (uint32_t)__popc(peers));
#else
        (void)lane;
        atomicAdd(&mine[b], 1u);
#endif
    }
    __syncthreads();

    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        uint32_t s = h[i] + h[256 + i] + h[512 + i] + h[768 + i];
        if (s) atomicAdd(&hist[i], s);
    }
}

/** One block sweeps chunks */
__global__ void k_encode(const uint8_t* __restrict__ src, int total, int nchunks,
                         const uint16_t* __restrict__ freq,
                         const uint16_t* __restrict__ start,
                         uint8_t* __restrict__ scratch,
                         uint16_t* __restrict__ lens) {
    __shared__ uint16_t s_freq[256];
    __shared__ uint16_t s_start[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        s_freq[i] = freq[i];
        s_start[i] = start[i];
    }
    __syncthreads();

    const int lane = threadIdx.x;
    for (int chunk = blockIdx.x; chunk < nchunks; chunk += gridDim.x) {
        const int base = chunk * (int)kChunk;
        const int clen = min((int)kChunk, total - base);
        const int m = (lane < clen) ? (clen - lane + (int)kLanes - 1) / (int)kLanes : 0;

        uint8_t* end = scratch + ((size_t)chunk * kLanes + lane) * kSlot + kSlot;
        uint8_t* p = end;
        uint32_t x = RANS_L;
        for (int i = m - 1; i >= 0; i--) {
            uint8_t s = src[base + lane + (size_t)i * kLanes];
            x = rans_put(x, &p, s_start[s], s_freq[s]);
        }
        rans_flush(x, &p);
        lens[(size_t)chunk * kLanes + lane] = (uint16_t)(end - p);
    }
}

/** Copy each lane's stream out */
__global__ void k_gather(const uint8_t* __restrict__ scratch,
                         const uint16_t* __restrict__ lens,
                         const uint32_t* __restrict__ offs,
                         uint8_t* __restrict__ payload, int nlanes) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nlanes;
         i += gridDim.x * blockDim.x) {
        const int len = lens[i];
        const uint8_t* s = scratch + (size_t)i * kSlot + (kSlot - len);
        uint8_t* d = payload + offs[i];
        for (int k = 0; k < len; k++) d[k] = s[k];
    }
}

/**  decode */
__global__ void k_decode(const uint8_t* __restrict__ payload,
                         const uint32_t* __restrict__ offs,
                         const uint16_t* __restrict__ freq,
                         const uint16_t* __restrict__ start,
                         int nchunks, int total, uint8_t* __restrict__ out) {
    __shared__ uint16_t s_freq[256];
    __shared__ uint16_t s_start[256];
    __shared__ uint16_t s_end[256];
    __shared__ uint8_t s_tab[4096];

    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        s_freq[i] = freq[i];
        s_start[i] = start[i];
        s_end[i] = (uint16_t)(start[i] + freq[i]);
    }
    __syncthreads();

    for (int slot = threadIdx.x; slot < (int)RANS_M; slot += blockDim.x) {
        int lo = 0, hi = 255;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if ((int)s_end[mid] > slot) hi = mid;
            else lo = mid + 1;
        }
        s_tab[slot] = (uint8_t)lo;
    }
    __syncthreads();

    const int lane = threadIdx.x;
    for (int chunk = blockIdx.x; chunk < nchunks; chunk += gridDim.x) {
        const int base = chunk * (int)kChunk;
        const int clen = min((int)kChunk, total - base);
        const int m = (lane < clen) ? (clen - lane + (int)kLanes - 1) / (int)kLanes : 0;
        if (!m) continue;
        const uint8_t* p = payload + offs[(size_t)chunk * kLanes + lane];
        uint32_t x = rans_ld32(p);
        p += 4;
        for (int i = 0; i < m; i++)
            out[base + lane + (size_t)i * kLanes] =
                rans_get(&x, &p, s_tab, s_freq, s_start);
    }
}

/** host driver */
const char* gpu_name() {
    static char name[256] = {0};
    if (!name[0]) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess)
            snprintf(name, sizeof(name), "%s (sm_%d%d, %d SMs)", prop.name,
                     prop.major, prop.minor, prop.multiProcessorCount);
        else
            snprintf(name, sizeof(name), "unknown");
    }
    return name;
}

bool gpu_init(GpuCtx* c, size_t max_frame) {
    memset(c, 0, sizeof(*c));
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return false;
    c->sm_count = prop.multiProcessorCount;
    c->max_frame = max_frame;
    c->max_chunks = max_frame / kChunk + 1;

    const size_t nlanes = c->max_chunks * kLanes;
    const size_t payload_cap = 2 * max_frame + 8 * nlanes + 1024;
    bool ok = cudaMalloc((void**)&c->d_src, max_frame + kChunk) == cudaSuccess &&
              cudaMalloc((void**)&c->d_scratch, nlanes * kSlot) == cudaSuccess &&
              cudaMalloc((void**)&c->d_lens, nlanes * sizeof(uint16_t)) == cudaSuccess &&
              cudaMalloc((void**)&c->d_offs, nlanes * sizeof(uint32_t)) == cudaSuccess &&
              cudaMalloc((void**)&c->d_payload, payload_cap) == cudaSuccess &&
              cudaMalloc((void**)&c->d_counts, 256 * sizeof(uint32_t)) == cudaSuccess &&
              cudaMalloc((void**)&c->d_freq, 256 * sizeof(uint16_t)) == cudaSuccess &&
              cudaMalloc((void**)&c->d_start, 256 * sizeof(uint16_t)) == cudaSuccess;
    if (!ok) {
        fprintf(stderr, "CUDA allocation failed (%zu MiB frame)\n", max_frame >> 20);
        gpu_free(c);
        return false;
    }
    return true;
}

void gpu_free(GpuCtx* c) {
    cudaFree(c->d_src);
    cudaFree(c->d_scratch);
    cudaFree(c->d_lens);
    cudaFree(c->d_offs);
    cudaFree(c->d_payload);
    cudaFree(c->d_counts);
    cudaFree(c->d_freq);
    cudaFree(c->d_start);
    memset(c, 0, sizeof(*c));
}

bool gpu_compress_frame(GpuCtx* c, const uint8_t* src, size_t n,
                        uint8_t* out_payload, size_t cap, size_t* out_len,
                        uint16_t* out_freq, uint16_t* out_lens) {
    if (!n || n > c->max_frame) return false;
    const uint32_t nchunks = (uint32_t)((n + kChunk - 1) / kChunk);
    const int nlanes = (int)(nchunks * kLanes);

    CUDA_CHECK(cudaMemcpy(c->d_src, src, n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(c->d_counts, 0, 256 * sizeof(uint32_t)));

    int grid = (int)((n + 255) / 256);
    if (grid > 2048) grid = 2048;
    k_hist<<<grid, 256>>>(c->d_src, n, c->d_counts);
    CUDA_CHECK(cudaGetLastError());

    uint32_t counts[256];
    CUDA_CHECK(cudaMemcpy(counts, c->d_counts, sizeof(counts), cudaMemcpyDeviceToHost));

    uint16_t freq[256], start[256];
    rans_normalize(counts, freq, start);
    memcpy(out_freq, freq, sizeof(freq));
    CUDA_CHECK(cudaMemcpy(c->d_freq, freq, sizeof(freq), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(c->d_start, start, sizeof(start), cudaMemcpyHostToDevice));

    int egrid = c->sm_count * 8;
    if (egrid > (int)nchunks) egrid = (int)nchunks;
    k_encode<<<egrid, kLanes>>>(c->d_src, (int)n, (int)nchunks, c->d_freq,
                                c->d_start, c->d_scratch, c->d_lens);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(out_lens, c->d_lens, nlanes * sizeof(uint16_t),
                          cudaMemcpyDeviceToHost));

    uint32_t* offs = (uint32_t*)malloc(nlanes * sizeof(uint32_t));
    if (!offs) return false;
    size_t total = 0;
    for (int i = 0; i < nlanes; i++) { offs[i] = (uint32_t)total; total += out_lens[i]; }
    if (total > cap) { free(offs); return false; }
    CUDA_CHECK(cudaMemcpy(c->d_offs, offs, nlanes * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    free(offs);

    int ggrid = (nlanes + 255) / 256;
    if (ggrid > 2048) ggrid = 2048;
    k_gather<<<ggrid, 256>>>(c->d_scratch, c->d_lens, c->d_offs, c->d_payload, nlanes);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(out_payload, c->d_payload, total, cudaMemcpyDeviceToHost));
    *out_len = total;
    return true;
}

bool gpu_decompress_frame(GpuCtx* c, const uint8_t* payload, size_t payload_len,
                          const uint16_t* freq, const uint16_t* lens,
                          uint32_t nchunks, size_t raw_len, uint8_t* out) {
    if (raw_len > c->max_frame) return false;
    const int nlanes = (int)(nchunks * kLanes);

    uint16_t start[256];
    uint32_t acc = 0;
    for (int i = 0; i < 256; i++) { start[i] = (uint16_t)acc; acc += freq[i]; }

    uint32_t* offs = (uint32_t*)malloc(nlanes * sizeof(uint32_t));
    if (!offs) return false;
    size_t total = 0;
    for (int i = 0; i < nlanes; i++) { offs[i] = (uint32_t)total; total += lens[i]; }
    if (total > payload_len) { free(offs); return false; }

    CUDA_CHECK(cudaMemcpy(c->d_offs, offs, nlanes * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(c->d_freq, freq, 256 * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(c->d_start, start, 256 * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(c->d_payload, payload, total, cudaMemcpyHostToDevice));
    free(offs);

    int grid = c->sm_count * 8;
    if (grid > (int)nchunks) grid = (int)nchunks;
    k_decode<<<grid, kLanes>>>(c->d_payload, c->d_offs, c->d_freq, c->d_start,
                               (int)nchunks, (int)raw_len, c->d_src);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(out, c->d_src, raw_len, cudaMemcpyDeviceToHost));
    return true;
}
