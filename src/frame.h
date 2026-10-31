#pragma once

/** Frame format and CPU reference codec */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "rans.h"

static const uint32_t kChunk   = 65536;  /** bytes per chunk */
static const uint32_t kLanes   = 128;    /** rANS streams per chunk */
static const uint32_t kLaneMax = kChunk / kLanes;
static const uint32_t kSlot    = 2 * kLaneMax + 8;  /** worst-case */

static const uint32_t kStreamMagic = 0x534E5243u;  /** CRNS little-endian */
static const uint32_t kStreamVersion = 1u;

enum { kFlagStored = 1u };

#pragma pack(push, 1)
struct StreamHdr {
    uint32_t magic;
    uint32_t version;
    uint32_t chunk_size;
    uint32_t lanes;
    uint32_t frame_size;
};
struct FrameHdr {
    uint32_t nchunks;
    uint32_t raw_len;
    uint32_t flags;
    uint16_t freq[256];
};
#pragma pack(pop)

static inline size_t frame_lens_off(uint32_t) { return sizeof(FrameHdr); }
static inline size_t frame_payload_off(uint32_t nchunks) {
    return sizeof(FrameHdr) + (size_t)nchunks * kLanes * sizeof(uint16_t);
}
static inline size_t frame_bound(size_t n) {
    size_t nchunks = (n + kChunk - 1) / kChunk;
    return sizeof(FrameHdr) + nchunks * kLanes * 2 + 2 * n + nchunks * kLanes * 8;
}

/** Serialise a frame from its parts */
static size_t frame_assemble(uint8_t* dst, size_t cap, const uint8_t* src, size_t n,
                             const uint16_t* freq, const uint16_t* lens,
                             size_t payload_len, const uint8_t* payload) {
    const uint32_t nchunks = (uint32_t)((n + kChunk - 1) / kChunk);
    const size_t poff = frame_payload_off(nchunks);
    FrameHdr h;
    memset(&h, 0, sizeof(h));
    h.raw_len = (uint32_t)n;
    if (payload_len >= n || poff + payload_len > cap) {
        h.flags = kFlagStored;
        if (sizeof(h) + n > cap) return 0;
        memcpy(dst, &h, sizeof(h));
        memcpy(dst + sizeof(h), src, n);
        return sizeof(h) + n;
    }
    h.nchunks = nchunks;
    memcpy(h.freq, freq, sizeof(h.freq));
    memcpy(dst, &h, sizeof(h));
    memcpy(dst + sizeof(h), lens, (size_t)nchunks * kLanes * sizeof(uint16_t));
    memcpy(dst + poff, payload, payload_len);
    return poff + payload_len;
}

/** CPU codec */

static size_t cpu_compress_frame(const uint8_t* src, size_t n,
                                 uint8_t* dst, size_t cap) {
    if (!n) return 0;
    const uint32_t nchunks = (uint32_t)((n + kChunk - 1) / kChunk);
    const size_t payload_off = frame_payload_off(nchunks);
    if (payload_off + n > cap) return 0;

    uint32_t counts[256] = {0};
    for (size_t i = 0; i < n; i++) counts[src[i]]++;
    uint16_t freq[256], start[256];
    rans_normalize(counts, freq, start);

    std::vector<uint8_t> scratch((size_t)nchunks * kLanes * kSlot + 16);
    uint16_t* lens = (uint16_t*)(dst + frame_lens_off(nchunks));
    uint32_t* offs = (uint32_t*)malloc((size_t)nchunks * kLanes * sizeof(uint32_t));
    if (!offs) return 0;

    for (uint32_t c = 0; c < nchunks; c++) {
        const size_t base = (size_t)c * kChunk;
        const int clen = (int)((n - base < kChunk) ? n - base : kChunk);
        for (uint32_t lane = 0; lane < kLanes; lane++) {
            const int m = (lane < (uint32_t)clen)
                              ? (clen - (int)lane + (int)kLanes - 1) / (int)kLanes
                              : 0;
            uint8_t* end = scratch.data() +
                           ((size_t)c * kLanes + lane) * kSlot + kSlot;
            uint8_t* p = end;
            uint32_t x = RANS_L;
            for (int i = m - 1; i >= 0; i--) {
                uint8_t s = src[base + lane + (size_t)i * kLanes];
                x = rans_put(x, &p, start[s], freq[s]);
            }
            rans_flush(x, &p);
            lens[(size_t)c * kLanes + lane] = (uint16_t)(end - p);
        }
    }

    size_t off = 0;
    for (size_t i = 0; i < (size_t)nchunks * kLanes; i++) {
        offs[i] = (uint32_t)off;
        off += lens[i];
    }
    const size_t payload_len = off;

    if (payload_off + payload_len >= payload_off + n) {
        free(offs);
        FrameHdr h;
        memset(&h, 0, sizeof(h));
        h.nchunks = 0;
        h.raw_len = (uint32_t)n;
        h.flags = kFlagStored;
        memcpy(dst, &h, sizeof(h));
        memcpy(dst + sizeof(FrameHdr), src, n);
        return sizeof(FrameHdr) + n;
    }

    FrameHdr h;
    memset(&h, 0, sizeof(h));
    h.nchunks = nchunks;
    h.raw_len = (uint32_t)n;
    h.flags = 0;
    memcpy(h.freq, freq, sizeof(freq));
    memcpy(dst, &h, sizeof(h));
    for (size_t i = 0; i < (size_t)nchunks * kLanes; i++) {
        const uint8_t* s = scratch.data() + i * kSlot + (kSlot - lens[i]);
        memcpy(dst + payload_off + offs[i], s, lens[i]);
    }
    free(offs);
    return payload_off + payload_len;
}

/** Returns bytes consumed */
static size_t cpu_decompress_frame(const uint8_t* rec, size_t reclen,
                                   uint8_t* out, size_t outcap) {
    if (reclen < sizeof(FrameHdr)) return 0;
    FrameHdr h;
    memcpy(&h, rec, sizeof(h));
    if (h.raw_len > outcap) return 0;

    if (h.flags & kFlagStored) {
        if (reclen < sizeof(FrameHdr) + h.raw_len) return 0;
        memcpy(out, rec + sizeof(FrameHdr), h.raw_len);
        return sizeof(FrameHdr) + h.raw_len;
    }
    if (h.nchunks == 0 || h.nchunks != (h.raw_len + kChunk - 1) / kChunk) return 0;

    uint32_t sum = 0;
    for (int i = 0; i < 256; i++) sum += h.freq[i];
    if (sum != RANS_M) return 0;

    uint16_t start[256];
    uint32_t acc = 0;
    for (int i = 0; i < 256; i++) { start[i] = (uint16_t)acc; acc += h.freq[i]; }

    const size_t payload_off = frame_payload_off(h.nchunks);
    if (reclen < payload_off) return 0;
    const uint16_t* lens = (const uint16_t*)(rec + frame_lens_off(h.nchunks));
    const uint8_t* payload = rec + payload_off;

    size_t off = 0;
    for (size_t i = 0; i < (size_t)h.nchunks * kLanes; i++) off += lens[i];
    if (payload_off + off > reclen) return 0;

    uint8_t tab[4096];
    rans_build_table(h.freq, start, tab);

    size_t poff = 0;
    for (uint32_t c = 0; c < h.nchunks; c++) {
        const size_t base = (size_t)c * kChunk;
        const int clen = (int)((h.raw_len - base < kChunk) ? h.raw_len - base : kChunk);
        for (uint32_t lane = 0; lane < kLanes; lane++) {
            const int m = (lane < (uint32_t)clen)
                              ? (clen - (int)lane + (int)kLanes - 1) / (int)kLanes
                              : 0;
            uint16_t len = lens[(size_t)c * kLanes + lane];
            if (poff + len > off) return 0;
            const uint8_t* p = payload + poff;
            poff += len;
            if (!m) continue;
            uint32_t x = rans_ld32(p);
            p += 4;
            for (int i = 0; i < m; i++) {
                out[base + lane + (size_t)i * kLanes] =
                    rans_get(&x, &p, tab, h.freq, start);
            }
            if (p != payload + poff) return 0;
        }
    }
    return payload_off + off;
}


