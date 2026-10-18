// SPDX-License-Identifier: Apache-2.0
// Native adaptation of FreeToken kernel/triton/nvfp4_fused_moe.py; see licenses/LICENSE-freetoken.
// Source SHA256: f35616fed3f8ad0626024095c697e3579cf94606489b0dd51892aecf38596759
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include "freetoken-moe.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#if CUDART_VERSION >= 12080
#include <cuda_fp4.h>
#endif
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace {
static __device__ __forceinline__ float fp4(unsigned code) {
    const unsigned mag = code & 7;
    // Normal E2M1 codes map directly to FP32 exponent and mantissa bits.
    const unsigned bits = mag < 2 ? mag * 0x3f000000u : (mag + 252u) << 22;
    return __uint_as_float(bits | ((code & 8u) << 28));
}
static __device__ __forceinline__ float2 fp4_pair(unsigned codes) {
#if CUDART_VERSION >= 12080
    return __half22float2(static_cast<__half2>(__nv_cvt_fp4x2_to_halfraw2(codes & 255, __NV_E2M1)));
#else
    return make_float2(fp4(codes & 15), fp4((codes >> 4) & 15));
#endif
}
static __device__ __forceinline__ float fp8(uint8_t code) {
    const unsigned exp = (code >> 3) & 15, man = code & 7;
    // Normal E4M3 scales need an exponent bias change from 7 to 127.
    const unsigned bits = exp == 0 ? __float_as_uint(float(man) * (1.0f / 512.0f)) :
        ((unsigned(code & 127) + 960u) << 20);
    return __uint_as_float(bits | (unsigned(code & 128) << 24));
}
static __device__ __forceinline__ float bf16(float x) { return __bfloat162float(__float2bfloat16(x)); }

// Eight adjacent codes per load; defer warp reduction until the full dot is accumulated.
template<bool DOWN, bool HARDWARE = false, bool VECTOR = false>
__global__ void gemv(freetoken_cache_view cache, int h, int intermediate, int top_k,
                     const float * x, const float * weights, const uint16_t * gate_up, uint16_t * dst) {
    const int route = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 4 + threadIdx.x / 32;
    const int n = DOWN ? h : intermediate;
    const int bank_rows = DOWN ? h : 2 * intermediate;
    const int k = DOWN ? intermediate : h;
    if (row >= n) { return; }
    const int slot = cache.gpu_ids[route];
    if (slot < 0) {
        if (lane == 0) { dst[size_t(route) * n + row] = 0; }
        return;
    }
    const int wb = DOWN ? 1 : 0, sb = DOWN ? 3 : 2, gb = DOWN ? 5 : 4;
    const size_t wr = size_t(slot) * bank_rows + row;
    const uint32_t * packed = reinterpret_cast<const uint32_t *>(cache.banks[wb]) + wr * (k / 8);
    const uint8_t * scales = cache.banks[sb] + wr * (k / 16);
    float sum = 0.0f, up_sum = 0.0f;
    for (int word = lane; word < k / 8; word += 32) {
        const uint32_t codes = packed[word];
        const uint32_t up_codes = DOWN ? 0 : packed[size_t(intermediate) * (k / 8) + word];
        float activation[8];
        if constexpr (VECTOR) {
            if constexpr (DOWN) {
                const uint4 values = *reinterpret_cast<const uint4 *>(gate_up + size_t(route)*intermediate + word*8);
                const uint32_t pairs[] = {values.x,values.y,values.z,values.w};
#pragma unroll
                for (int j = 0; j < 8; ++j) { activation[j] = __bfloat162float(__ushort_as_bfloat16(uint16_t(pairs[j/2] >> (16*(j%2))))); }
            } else {
                const float4 * input = reinterpret_cast<const float4 *>(x + size_t(route/top_k)*h + word*8);
                const float4 a = input[0], b = input[1];
                const float values[] = {a.x,a.y,a.z,a.w,b.x,b.y,b.z,b.w};
#pragma unroll
                for (int j = 0; j < 8; ++j) { activation[j] = bf16(values[j]); }
            }
        }
        float dot = 0.0f, up_dot = 0.0f;
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const unsigned c = codes >> (8*pair), u = up_codes >> (8*pair);
            const float2 value = HARDWARE ? fp4_pair(c) : make_float2(fp4(c & 15), fp4((c >> 4) & 15));
            const float2 up_value = DOWN ? make_float2(0,0) :
                (HARDWARE ? fp4_pair(u) : make_float2(fp4(u & 15), fp4((u >> 4) & 15)));
#pragma unroll
            for (int part = 0; part < 2; ++part) {
                const int j = 2*pair + part;
                const int col = word * 8 + j;
                float a;
                if constexpr (VECTOR) { a = activation[j]; }
                else if (DOWN) {
                    const size_t base = size_t(route) * intermediate + col;
                    a = __bfloat162float(__ushort_as_bfloat16(gate_up[base]));
                } else { a = bf16(x[size_t(route / top_k) * h + col]); }
                dot += a * (part ? value.y : value.x);
                if (!DOWN) { up_dot += a * (part ? up_value.y : up_value.x); }
            }
        }
        sum += dot * fp8(scales[word / 2]);
        if (!DOWN) { up_sum += up_dot * fp8(scales[size_t(intermediate) * (k / 16) + word / 2]); }
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (!DOWN) { up_sum += __shfl_down_sync(0xffffffff, up_sum, offset); }
    }
    if (lane == 0) {
        sum *= __half2float(reinterpret_cast<const __half *>(cache.banks[gb])[wr]);
        if (DOWN) { sum *= weights[route]; }
        else {
            const float gate = bf16(sum);
            const float up = bf16(up_sum * __half2float(reinterpret_cast<const __half *>(cache.banks[gb])[wr + intermediate]));
            sum = (gate / (1.0f + expf(-gate))) * up;
        }
        dst[size_t(route) * n + row] = __bfloat16_as_ushort(__float2bfloat16(sum));
    }
}
__global__ void reduce(const uint16_t * down, float * out, int hidden, int top_k, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) { return; }
    const int token = i / hidden, col = i % hidden;
    float sum = 0;
    for (int k = 0; k < top_k; ++k) {
        sum += __bfloat162float(__ushort_as_bfloat16(down[(size_t(token) * top_k + k) * hidden + col]));
    }
    out[i] = sum;
}
static void check_launch() {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) { throw std::runtime_error(std::string("FreeToken GEMV: ") + cudaGetErrorString(error)); }
}
} // namespace

static void gpu_moe_impl(const freetoken_cache_view & cache, int h, int i, int k, int tokens,
                      const float * x, const float * weights, uint16_t * gate_up, uint16_t * down,
                      float * out, cudaStream_t stream, bool deferred) {
    if (h < 64 || h > 65536 || h % 64 || i < 64 || i > 65536 || i % 64 ||
        k < 1 || k > 64 || tokens*k > cache.route_capacity || tokens < 1 || tokens > 4 || !x || !weights || !gate_up || !down || (!out && !deferred) || !cache.gpu_ids) {
        throw std::invalid_argument("Invalid FreeToken GPU MoE dimensions");
    }
    const size_t expected[6] = {size_t(i)*h, size_t(h)*i/2, size_t(i)*h/8,
                               size_t(h)*i/16, size_t(i)*4, size_t(h)*2};
    for (int b = 0; b < 6; ++b) {
        if (!cache.banks[b] || cache.bytes_per_expert[b] != expected[b]) {
            throw std::invalid_argument("FreeToken GPU MoE bank dimensions do not match");
        }
    }
    static const bool hardware = [] {
        const char * value = std::getenv("GGML_FT_FP4_HW");
        return value && std::strcmp(value, "1") == 0;
    }();
    static const bool vector = [] {
        const char * value = std::getenv("GGML_FT_MOE_VECTOR");
        return value && std::strcmp(value,"1") == 0;
    }();
    const bool aligned = (reinterpret_cast<uintptr_t>(x) & 15) == 0 && (reinterpret_cast<uintptr_t>(gate_up) & 15) == 0;
    if (vector && aligned && hardware) {
        gemv<false,true,true><<<dim3((i+3)/4, tokens*k), 128, 0, stream>>>(cache,h,i,k,x,weights,gate_up,gate_up);
        check_launch();
        gemv<true,true,true><<<dim3((h+3)/4, tokens*k), 128, 0, stream>>>(cache,h,i,k,x,weights,gate_up,down);
    } else if (vector && aligned) {
        gemv<false,false,true><<<dim3((i+3)/4, tokens*k), 128, 0, stream>>>(cache,h,i,k,x,weights,gate_up,gate_up);
        check_launch();
        gemv<true,false,true><<<dim3((h+3)/4, tokens*k), 128, 0, stream>>>(cache,h,i,k,x,weights,gate_up,down);
    } else if (hardware) {
        gemv<false,true><<<dim3((i+3)/4, tokens*k), 128, 0, stream>>>(cache,h,i,k,x,weights,gate_up,gate_up);
        check_launch();
        gemv<true,true><<<dim3((h+3)/4, tokens*k), 128, 0, stream>>>(cache,h,i,k,x,weights,gate_up,down);
    } else {
        gemv<false><<<dim3((i+3)/4, tokens*k), 128, 0, stream>>>(cache,h,i,k,x,weights,gate_up,gate_up);
        check_launch();
        gemv<true><<<dim3((h+3)/4, tokens*k), 128, 0, stream>>>(cache,h,i,k,x,weights,gate_up,down);
    }
    check_launch();
    if (deferred) { return; }
    reduce<<<(tokens*h+255)/256, 256, 0, stream>>>(down,out,h,k,tokens*h);
    check_launch();
}

void freetoken_gpu_moe(const freetoken_cache_view & cache, int h, int i, int k, int tokens,
        const float * x, const float * weights, uint16_t * gate_up, uint16_t * down, float * out, cudaStream_t stream) {
    gpu_moe_impl(cache,h,i,k,tokens,x,weights,gate_up,down,out,stream,false);
}
void freetoken_gpu_moe_deferred(const freetoken_cache_view & cache, int h, int i, int k, int tokens,
        const float * x, const float * weights, uint16_t * gate_up, uint16_t * down, cudaStream_t stream) {
    gpu_moe_impl(cache,h,i,k,tokens,x,weights,gate_up,down,nullptr,stream,true);
}

#endif // CUDA-only FreeToken path
