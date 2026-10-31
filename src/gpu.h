#pragma once

/** gpu api */

#include <stddef.h>
#include <stdint.h>

#include <cuda_runtime.h>

struct GpuCtx {
    int sm_count;
    size_t max_frame;
    size_t max_chunks;
    uint8_t* d_src;
    uint8_t* d_scratch;
    uint16_t* d_lens;
    uint32_t* d_offs;
    uint8_t* d_payload;
    uint32_t* d_counts;
    uint16_t* d_freq;
    uint16_t* d_start;
};

bool gpu_init(GpuCtx* ctx, size_t max_frame);
void gpu_free(GpuCtx* ctx);
const char* gpu_name();

bool gpu_compress_frame(GpuCtx* ctx, const uint8_t* src, size_t n,
                        uint8_t* out_payload, size_t cap, size_t* out_len,
                        uint16_t* out_freq, uint16_t* out_lens);


bool gpu_decompress_frame(GpuCtx* ctx, const uint8_t* payload, size_t payload_len,
                          const uint16_t* freq, const uint16_t* lens,
                          uint32_t nchunks, size_t raw_len, uint8_t* out);


