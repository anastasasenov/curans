#pragma once

/** rANS (range Asymmetric Numeral Systems) */

#include <stdint.h>
#include <string.h>

#define RANS_SCALE_BITS 12u
#define RANS_M          (1u << RANS_SCALE_BITS)
#define RANS_MASK       (RANS_M - 1u)
#define RANS_L          (1u << 23)

#if defined(__CUDACC__)
#define RANS_HD __host__ __device__ __forceinline__
#else
#define RANS_HD static inline
#endif

RANS_HD void rans_st32(uint8_t* p, uint32_t v) {
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
}

RANS_HD uint32_t rans_ld32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

RANS_HD uint32_t rans_put(uint32_t x, uint8_t** pp, uint32_t start, uint32_t freq) {
    uint32_t x_max = ((RANS_L >> RANS_SCALE_BITS) << 8) * freq;
    uint8_t* p = *pp;
    while (x >= x_max) {
        *--p = (uint8_t)(x & 0xffu);
        x >>= 8;
    }
    *pp = p;
    return ((x / freq) << RANS_SCALE_BITS) + (x % freq) + start;
}

RANS_HD void rans_flush(uint32_t x, uint8_t** pp) {
    uint8_t* p = *pp - 4;
    rans_st32(p, x);
    *pp = p;
}

RANS_HD uint8_t rans_get(uint32_t* xp, const uint8_t** pp,
                         const uint8_t* tab, const uint16_t* freq,
                         const uint16_t* start) {
    uint32_t x = *xp;
    uint32_t slot = x & RANS_MASK;
    uint8_t s = tab[slot];
    x = (uint32_t)freq[s] * (x >> RANS_SCALE_BITS) + slot - start[s];
    const uint8_t* p = *pp;
    while (x < RANS_L) {
        x = (x << 8) | *p++;
    }
    *pp = p;
    *xp = x;
    return s;
}

static void rans_normalize(const uint32_t* counts, uint16_t* freq, uint16_t* start) {
    uint64_t total = 0;
    int nz = 0;
    for (int i = 0; i < 256; i++) {
        total += counts[i];
        nz += counts[i] != 0;
    }
    if (total == 0 || nz == 0) {
        memset(freq, 0, 256 * sizeof(uint16_t));
        memset(start, 0, 256 * sizeof(uint16_t));
        return;
    }
    int sum = 0;
    for (int i = 0; i < 256; i++) {
        if (!counts[i]) { freq[i] = 0; continue; }
        double v = (double)counts[i] * (double)(RANS_M - nz) / (double)total;
        uint32_t f = 1u + (uint32_t)v;
        freq[i] = (uint16_t)f;
        sum += (int)f;
    }
    while (sum < (int)RANS_M) {
        int best = -1;
        uint32_t bc = 0;
        for (int i = 0; i < 256; i++)
            if (counts[i] && freq[i] < RANS_M && counts[i] > bc) { bc = counts[i]; best = i; }
        if (best < 0) break;
        freq[best]++;
        sum++;
    }
    while (sum > (int)RANS_M) {
        int best = -1;
        uint16_t bf = 0;
        for (int i = 0; i < 256; i++)
            if (freq[i] > 1 && freq[i] > bf) { bf = freq[i]; best = i; }
        if (best < 0) break;
        freq[best]--;
        sum--;
    }
    uint32_t acc = 0;
    for (int i = 0; i < 256; i++) { start[i] = (uint16_t)acc; acc += freq[i]; }
}

static void rans_build_table(const uint16_t* freq, const uint16_t* start, uint8_t* tab) {
    for (int s = 0; s < 256; s++)
        for (uint32_t k = 0; k < freq[s]; k++)
            tab[start[s] + k] = (uint8_t)s;
}


