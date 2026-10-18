#pragma once
#include "freetoken-cache.cuh"

void freetoken_quantize_draft(const void * src, uint8_t * packed, uint8_t * scales,
                             uint16_t * globals, int rows, int cols, int * invalid, cudaStream_t stream);

// W4A16 GPU decode over canonical FT banks. Scratch holds [tokens*top_k, intermediate]
// BF16 SiLU(gate)*up results, and [tokens*top_k, hidden] BF16 weighted down results.
// out is F32 [tokens, hidden]. CPU addends are combined by the dispatch bridge.
// All pointers are device-accessible. Caller owns scratch and serializes cache access.
void freetoken_gpu_moe(const freetoken_cache_view & cache, int hidden, int intermediate,
                      int top_k, int tokens, const float * x, const float * weights,
                      uint16_t * gate_up, uint16_t * down, float * out, cudaStream_t stream);
// Leave weighted BF16 route outputs in down for the CPU/GPU completion kernel.
void freetoken_gpu_moe_deferred(const freetoken_cache_view & cache, int hidden, int intermediate,
                               int top_k, int tokens, const float * x, const float * weights,
                               uint16_t * gate_up, uint16_t * down, cudaStream_t stream);
