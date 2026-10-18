#include "freetoken-dense.cuh"
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include "freetoken-moe.cuh"
#include "ggml-backend.h"
#include <cuda_fp8.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
static void check(cudaError_t err) {
    if (err != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(err)); }
}
__global__ void quantize(const __nv_bfloat16 * src, uint8_t * dst, float * scales, int k, int * invalid) {
    const int row = blockIdx.x, tid = threadIdx.x;
    float amax = 0;
    for (int i = tid; i < k; i += blockDim.x) {
        float x = __bfloat162float(src[size_t(row)*k+i]);
        if (!isfinite(x)) { atomicExch(invalid,1); }
        amax = fmaxf(amax,fabsf(x));
    }
    __shared__ float maxima[256];
    maxima[tid] = amax;
    __syncthreads();
    for (int d = 128; d; d >>= 1) {
        if (tid < d) { maxima[tid] = fmaxf(maxima[tid],maxima[tid+d]); }
        __syncthreads();
    }
    const float scale = fmaxf(maxima[0]/448.0f,1e-12f);
    if (!tid) { scales[row] = scale; }
    for (int i = tid; i < k; i += blockDim.x) {
        dst[size_t(row)*k+i] = __nv_fp8_e4m3(fminf(448.0f,fmaxf(-448.0f,__bfloat162float(src[size_t(row)*k+i])/scale))).__x;
    }
}

template<int M, bool FIXUP, int ROWS = 4> __global__ void gemv(const uint8_t * w, const float * scales, const float * x,
                                    float * partial, unsigned int * counters, float * out, int k, int n, int splits) {
    const int row = blockIdx.x*ROWS + threadIdx.x/32;
    const bool active = row < n;
    if (!FIXUP && !active) { return; }
    const int lane = threadIdx.x%32;
    float acc[M] = {};
    const int begin = (k/4)*blockIdx.y/splits, end = (k/4)*(blockIdx.y+1)/splits;
    for (int j = begin+lane; (!FIXUP || active) && j < end; j += 32) {
        __nv_fp8x4_e4m3 packed;
        packed.__x = reinterpret_cast<const uint32_t *>(w)[size_t(row)*(k/4)+j];
        const float4 v = static_cast<float4>(packed);
        #pragma unroll
        for (int t = 0; t < M; ++t) {
            const float4 a = reinterpret_cast<const float4 *>(x + size_t(t)*k)[j];
            acc[t] = fmaf(v.x,a.x,acc[t]); acc[t] = fmaf(v.y,a.y,acc[t]);
            acc[t] = fmaf(v.z,a.z,acc[t]); acc[t] = fmaf(v.w,a.w,acc[t]);
        }
    }
    #pragma unroll
    for (int t = 0; t < M; ++t) {
        for (int d = 16; d; d >>= 1) { acc[t] += __shfl_down_sync(0xffffffff,acc[t],d); }
        if ((!FIXUP || active) && !lane) {
            if (splits == 1) { out[size_t(t)*n+row] = __bfloat162float(__float2bfloat16(acc[t]*scales[row])); }
            else {
                if constexpr (FIXUP) { static_cast<volatile float *>(partial)[(size_t(t)*splits+blockIdx.y)*n+row] = acc[t]; }
                else { partial[(size_t(t)*splits+blockIdx.y)*n+row] = acc[t]; }
            }
        }
    }
    if constexpr (FIXUP) {
        // Publish all warp stores before the block increments its tile counter.
        __threadfence();
        __syncthreads();
        __shared__ bool last;
        // The last increment resets the counter; launches sharing scratch must be serialized.
        if (!threadIdx.x) { last = atomicInc(counters+blockIdx.x,splits-1) == unsigned(splits-1); }
        __syncthreads();
        if (last && active && !lane) {
            #pragma unroll
            for (int t = 0; t < M; ++t) {
                float sum = 0;
                for (int s = 0; s < splits; ++s) { sum += static_cast<volatile float *>(partial)[(size_t(t)*splits+s)*n+row]; }
                out[size_t(t)*n+row] = __bfloat162float(__float2bfloat16(sum*scales[row]));
            }
        }
    }
}

// Each warp handles one token; all four warps reuse the same weight row.
__global__ void gemv_tokens4(const uint8_t * w, const float * scales, const float * x,
                             float * partial, float * out, int k, int n, int splits) {
    const int row = blockIdx.x, token = 4*blockIdx.z + threadIdx.x/32, lane = threadIdx.x%32;
    const int begin = (k/4)*blockIdx.y/splits, end = (k/4)*(blockIdx.y+1)/splits;
    float acc = 0;
    for (int j = begin+lane; j < end; j += 32) {
        __nv_fp8x4_e4m3 packed;
        packed.__x = reinterpret_cast<const uint32_t *>(w)[size_t(row)*(k/4)+j];
        const float4 v = static_cast<float4>(packed);
        const float4 a = reinterpret_cast<const float4 *>(x + size_t(token)*k)[j];
        acc = fmaf(v.x,a.x,acc); acc = fmaf(v.y,a.y,acc);
        acc = fmaf(v.z,a.z,acc); acc = fmaf(v.w,a.w,acc);
    }
    for (int d = 16; d; d >>= 1) { acc += __shfl_down_sync(0xffffffff,acc,d); }
    if (!lane) {
        if (splits == 1) { out[size_t(token)*n+row] = __bfloat162float(__float2bfloat16(acc*scales[row])); }
        else { partial[(size_t(token)*splits+blockIdx.y)*n+row] = acc; }
    }
}

__global__ void finish(const float * partial, const float * scales, float * out, int n, int m, int splits) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= n*m) { return; }
    int row = i%n, t = i/n;
    float sum = 0;
    for (int s = 0; s < splits; ++s) { sum += partial[(size_t(t)*splits+s)*n+row]; }
    out[i] = __bfloat162float(__float2bfloat16(sum*scales[row]));
}
struct bank {
    int device;
    uint8_t * weights = nullptr;
    float * scales = nullptr;
    uint8_t * block_scales = nullptr;
    uint16_t * globals = nullptr;
    ~bank() { cudaSetDevice(device); cudaFree(weights); cudaFree(scales); cudaFree(block_scales); cudaFree(globals); }
};

template<int M> __global__ void gemv_draft_head(const uint8_t * packed, const uint8_t * scales,
        const uint16_t * globals, const float * x, float * out, int k, int n) {
    const int row = blockIdx.x*4 + threadIdx.x/32, lane = threadIdx.x%32;
    if (row >= n) { return; }
    float sum[M] = {};
    for (int word = lane; word < k/8; word += 32) {
        const uint32_t codes = reinterpret_cast<const uint32_t *>(packed)[size_t(row)*(k/8)+word];
        __nv_fp8_e4m3 scale; scale.__x = scales[size_t(row)*(k/16)+word/2];
        float dot[M] = {};
        float4 first[M], second[M];
#pragma unroll
        for (int t = 0; t < M; ++t) {
            const auto * input = reinterpret_cast<const float4 *>(x+size_t(t)*k);
            first[t] = input[word*2]; second[t] = input[word*2+1];
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const unsigned code = (codes >> (4*j)) & 15, mag = code & 7;
            const unsigned bits = mag < 2 ? mag*0x3f000000u : (mag+252u)<<22;
            const float w = __uint_as_float(bits | ((code & 8u)<<28));
#pragma unroll
            for (int t = 0; t < M; ++t) {
                const float value = j == 0 ? first[t].x : j == 1 ? first[t].y : j == 2 ? first[t].z : j == 3 ? first[t].w :
                                    j == 4 ? second[t].x : j == 5 ? second[t].y : j == 6 ? second[t].z : second[t].w;
                const float a = __bfloat162float(__float2bfloat16(value));
                dot[t] += a*w;
            }
        }
#pragma unroll
        for (int t = 0; t < M; ++t) { sum[t] += dot[t]*static_cast<float>(scale); }
    }
#pragma unroll
    for (int t = 0; t < M; ++t) {
        for (int d = 16; d; d >>= 1) { sum[t] += __shfl_down_sync(0xffffffff,sum[t],d); }
        if (!lane) { out[size_t(t)*n+row] = __bfloat162float(__float2bfloat16(sum[t]*__half2float(__ushort_as_half(globals[row])))); }
    }
}

static std::shared_ptr<bank> pack_draft_head(int device, const ggml_tensor * w) {
    const int k = w->ne[0], n = w->ne[1];
    const size_t bytes = size_t(n)*(k/2+k/16+sizeof(uint16_t));
    size_t free_bytes, all_bytes; check(cudaMemGetInfo(&free_bytes,&all_bytes));
    if (bytes+(size_t(2)<<30)+8*1024*1024 > free_bytes) { throw std::runtime_error("FreeToken NVFP4 draft head requires 2 GiB free VRAM"); }
    auto result = std::make_shared<bank>(); result->device = device;
    check(cudaMalloc(&result->weights,size_t(n)*(k/2)));
    check(cudaMalloc(&result->block_scales,size_t(n)*(k/16)));
    check(cudaMalloc(&result->globals,size_t(n)*sizeof(uint16_t)));
    void * staging = nullptr; int * invalid = nullptr;
    try {
        const int chunk = std::min(256,(4*1024*1024)/k);
        std::vector<uint16_t> host(size_t(chunk)*k);
        check(cudaMalloc(&staging,host.size()*sizeof(uint16_t)));
        check(cudaMalloc(&invalid,sizeof(int))); check(cudaMemset(invalid,0,sizeof(int)));
        for (int row = 0; row < n; row += chunk) {
            const int rows = std::min(chunk,n-row);
            const size_t len = size_t(rows)*k*sizeof(uint16_t);
            ggml_backend_tensor_get(w,host.data(),size_t(row)*k*sizeof(uint16_t),len);
            check(cudaMemcpy(staging,host.data(),len,cudaMemcpyHostToDevice));
            freetoken_quantize_draft(staging,result->weights+size_t(row)*(k/2),
                result->block_scales+size_t(row)*(k/16),result->globals+row,rows,k,invalid,nullptr);
        }
        int bad; check(cudaMemcpy(&bad,invalid,sizeof(int),cudaMemcpyDeviceToHost));
        if (bad) { throw std::runtime_error("Non-finite BF16 weights in FreeToken NVFP4 draft head"); }
        check(cudaFree(staging)); staging = nullptr;
        check(cudaFree(invalid)); invalid = nullptr;
    } catch (...) { cudaFree(staging); cudaFree(invalid); throw; }
    fprintf(stderr,"FreeToken: NVFP4 draft head %d x %d, %zu MiB private weights; target FP8 retained\n",n,k,bytes/(1024*1024));
    return result;
}
struct dense_timing { unsigned long long start, calls[4], ns[4]; };
__global__ void dense_stamp(dense_timing * timing, int m, bool finish) {
    unsigned long long now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    if (finish) { ++timing->calls[m-1]; timing->ns[m-1] += now-timing->start; }
    else { timing->start = now; }
}
struct matrix {
    std::shared_ptr<bank> storage;
    int device, k, n, splits;
    int m4_rows = 4;
    uint8_t * weights = nullptr;
    float * scales = nullptr;
    float * partial = nullptr;
    unsigned int * counters = nullptr;
    dense_timing * timing = nullptr;
    std::string name;
    bool cloned = false;
    void init_timing() {
        const char * value = std::getenv("GGML_FT_DENSE_TIMING");
        if (value && std::strcmp(value,"1") == 0) {
            check(cudaMalloc(&timing,sizeof(dense_timing)));
            check(cudaMemset(timing,0,sizeof(dense_timing)));
        }
    }
    ~matrix() {
        cudaSetDevice(device);
        if (timing) {
            dense_timing data{};
            const auto error = cudaMemcpy(&data,timing,sizeof(data),cudaMemcpyDeviceToHost);
            if (error == cudaSuccess) {
                for (int m = 0; m < 4; ++m) if (data.calls[m]) {
                    fprintf(stderr,"FreeToken dense span: clone=%d name=%s k=%d n=%d m=%d calls=%llu total_ms=%.3f mean_us=%.3f\n",
                        int(cloned),name.c_str(),k,n,m+1,data.calls[m],data.ns[m]/1e6,double(data.ns[m])/data.calls[m]/1e3);
                }
            } else { fprintf(stderr,"FreeToken dense span report failed: %s\n",cudaGetErrorString(error)); }
            cudaFree(timing);
        }
        cudaFree(partial); cudaFree(counters);
    }
};
struct owner {
    int device;
    std::unordered_map<const ggml_tensor *,std::unique_ptr<matrix>> matrices;
    explicit owner(int dev) : device(dev) {}
    ~owner() { cudaSetDevice(device); cudaDeviceSynchronize(); }
};
[[noreturn]] static void marker(ggml_tensor *, int, int, void *) { GGML_ABORT("FreeToken FP8 dense operation requires CUDA"); }
static matrix * get_op(const ggml_tensor * op) {
    if (op->op != GGML_OP_CUSTOM) { return nullptr; }
    ggml_custom_op_params params; memcpy(&params,op->op_params,sizeof(params));
    return params.fun == marker ? static_cast<matrix *>(params.userdata) : nullptr;
}
static int m4_layout(int k, int n) {
    // Calibrated SM120 Qwen4Exp shapes; other matrices retain the original layout.
    if ((k == 10240 && n == 320) || (k == 2560 && n == 640)) { return 16; }
    if (k == 2560 && (n == 512 || n == 12288)) { return 8; }
    if ((n == 2560 && (k == 2560 || k == 5120 || k == 6144)) || (k == 2560 && n == 48)) { return 1; }
    return 4;
}
static void * create(int device, ggml_tensor * const * tensors, int count) {
    check(cudaSetDevice(device));
    auto p = std::unique_ptr<owner>(new owner(device));
    cudaDeviceProp properties; check(cudaGetDeviceProperties(&properties,device));
    const char * layout = std::getenv("GGML_FT_DENSE_M4_LAYOUT");
    const bool tune_m4 = properties.major == 12 && properties.minor == 0 && (!layout || std::strcmp(layout,"0") != 0);
    void * staging = nullptr; int * invalid = nullptr;
    size_t total = 0;
    try {
        check(cudaMalloc(&staging,8*1024*1024));
        check(cudaMalloc(&invalid,sizeof(int))); check(cudaMemset(invalid,0,sizeof(int)));
        std::vector<uint16_t> host(4*1024*1024);
        for (int i = 0; i < count; ++i) {
            const auto * w = tensors[i];
            if (!w || w->type != GGML_TYPE_BF16 || !ggml_is_contiguous(w) || w->ne[2] != 1 || w->ne[3] != 1 ||
                w->ne[0] < 256 || w->ne[0] > 65536 || w->ne[0]%4 || w->ne[1] < 16 || w->ne[1] > 1048576 ||
                p->matrices.count(w)) { continue; }
            auto b = std::unique_ptr<matrix>(new matrix{});
            b->device = device; b->k = w->ne[0]; b->n = w->ne[1];
            b->name = w->name; b->init_timing();
            if (tune_m4) { b->m4_rows = m4_layout(b->k,b->n); }
            b->splits = std::min(16,std::max(1,1024/((b->n+3)/4)));
            b->splits = std::min(b->splits,std::max(1,b->k/256));
            size_t bytes = size_t(b->k)*b->n + size_t(b->n)*sizeof(float)*(1+4*b->splits) + size_t((b->n+3)/4)*sizeof(unsigned int);
            size_t free_bytes, all_bytes; check(cudaMemGetInfo(&free_bytes,&all_bytes));
            if (bytes + (size_t(2)<<30) > free_bytes) { throw std::runtime_error("FreeToken FP8 requires 2 GiB free VRAM after packing"); }
            b->storage = std::make_shared<bank>(); b->storage->device = device;
            check(cudaMalloc(&b->storage->weights,size_t(b->k)*b->n));
            check(cudaMalloc(&b->storage->scales,size_t(b->n)*sizeof(float)));
            b->weights = b->storage->weights; b->scales = b->storage->scales;
            check(cudaMalloc(&b->partial,size_t(b->n)*4*b->splits*sizeof(float)));
            check(cudaMalloc(&b->counters,size_t((b->n+3)/4)*sizeof(unsigned int)));
            check(cudaMemset(b->counters,0,size_t((b->n+3)/4)*sizeof(unsigned int)));
            int chunk = std::min(256,int(host.size()/b->k));
            for (int row = 0; row < b->n; row += chunk) {
                int rows = std::min(chunk,b->n-row);
                size_t len = size_t(rows)*b->k*sizeof(uint16_t);
                ggml_backend_tensor_get(w,host.data(),size_t(row)*b->k*sizeof(uint16_t),len);
                check(cudaMemcpy(staging,host.data(),len,cudaMemcpyHostToDevice));
                quantize<<<rows,256>>>(static_cast<const __nv_bfloat16 *>(staging),b->weights+size_t(row)*b->k,b->scales+row,b->k,invalid);
                check(cudaGetLastError());
            }
            total += bytes;
            p->matrices.emplace(w,std::move(b));
        }
        int bad; check(cudaMemcpy(&bad,invalid,sizeof(int),cudaMemcpyDeviceToHost));
        if (bad) { throw std::runtime_error("Non-finite BF16 weights in FreeToken FP8 conversion"); }
        check(cudaFree(staging)); staging = nullptr;
        check(cudaFree(invalid)); invalid = nullptr;
    } catch (...) { cudaFree(staging); cudaFree(invalid); throw; }
    fprintf(stderr,"FreeToken: FP8 dense experimental: %zu matrices, %zu MiB additional VRAM; BF16 prefill retained\n",p->matrices.size(),total/(1024*1024));
    return p.release();
}
static void * clone(void * ptr) {
    const auto & source = *static_cast<owner *>(ptr);
    check(cudaSetDevice(source.device));
    auto p = std::make_unique<owner>(source.device);
    size_t scratch = 0;
    bool has_nvfp4_head = false;
    for (const auto & entry : source.matrices) {
        const auto & original = *entry.second;
        auto b = std::make_unique<matrix>();
        b->device = original.device; b->k = original.k; b->n = original.n; b->splits = original.splits;
        b->m4_rows = original.m4_rows;
        b->name = original.name; b->cloned = true; b->init_timing();
        b->storage = original.storage; b->weights = original.weights; b->scales = original.scales;
        const char * draft_head = std::getenv("GGML_FT_MTP_HEAD_NVFP4");
        if (draft_head && std::strcmp(draft_head,"1") == 0 && original.name == "output.weight" && original.k%16 == 0 && !b->storage->block_scales) {
            b->storage = pack_draft_head(source.device,entry.first);
            b->weights = b->storage->weights; b->scales = nullptr;
        }
        has_nvfp4_head = has_nvfp4_head || b->storage->block_scales != nullptr;
        // Contexts share immutable banks, but never reduction scratch or counters.
        check(cudaMalloc(&b->partial,size_t(b->n)*4*b->splits*sizeof(float)));
        check(cudaMalloc(&b->counters,size_t((b->n+3)/4)*sizeof(unsigned int)));
        check(cudaMemset(b->counters,0,size_t((b->n+3)/4)*sizeof(unsigned int)));
        scratch += size_t(b->n)*4*b->splits*sizeof(float) + size_t((b->n+3)/4)*sizeof(unsigned int);
        p->matrices.emplace(entry.first,std::move(b));
    }
    // Backend streams can be nonblocking; finish counter initialization before publication.
    check(cudaStreamSynchronize(nullptr));
    if (has_nvfp4_head) {
        size_t free_bytes, all_bytes; check(cudaMemGetInfo(&free_bytes,&all_bytes));
        if (free_bytes < (size_t(2)<<30)) { throw std::runtime_error("FreeToken NVFP4 draft head requires 2 GiB free VRAM after scratch allocation"); }
    }
    fprintf(stderr,"FreeToken: shared dense: %zu matrices, %zu MiB private scratch; optional draft head reported separately\n",p->matrices.size(),scratch/(1024*1024));
    return p.release();
}
[[noreturn]] static void hc_marker(ggml_tensor *, int, int, void *) { GGML_ABORT("FreeToken HC reduction requires CUDA"); }
static owner * get_hc(const ggml_tensor * op) {
    if (op->op != GGML_OP_CUSTOM) { return nullptr; }
    ggml_custom_op_params params; memcpy(&params, op->op_params, sizeof(params));
    return params.fun == hc_marker ? static_cast<owner *>(params.userdata) : nullptr;
}
static bool hc_shape(const ggml_tensor * x, const ggml_tensor * logits, int width) {
    return x && logits && width > 0 && width <= 65536 && x->type == GGML_TYPE_F32 && logits->type == GGML_TYPE_F32 &&
           ggml_are_same_shape(x, logits) && ggml_is_contiguous(x) && ggml_is_contiguous(logits) &&
           x->ne[0] == 4*width && x->ne[1] >= 1 && x->ne[1] <= 4 && x->ne[2] == 1 && x->ne[3] == 1;
}
static ggml_tensor * build_hc(void * ptr, ggml_context * ctx, ggml_tensor * x, ggml_tensor * logits, int width) {
    if (!ptr || !hc_shape(x, logits, width)) { return nullptr; }
    ggml_tensor * args[] = {x, logits};
    return ggml_custom_4d(ctx, GGML_TYPE_F32, width, x->ne[1], 1, 1, args, 2, hc_marker, 1, ptr);
}
__global__ void hc_reduce(const float * x, const float * logits, float * out, int width, int tokens) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= width*tokens) { return; }
    const int base = (i/width)*4*width + i%width;
    float sum = 0.0f;
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        const int j = base + c*width;
        const float gate = 1.0f / (1.0f + expf(-logits[j]));
        const float value = __fmul_rn(x[j], gate);
        sum = c == 0 ? value : __fadd_rn(sum, value);
    }
    out[i] = __fmul_rn(sum, 0.25f);
}
[[noreturn]] static void hc_combine_marker(ggml_tensor *, int, int, void *) { GGML_ABORT("FreeToken HC combine requires CUDA"); }
static owner * get_hc_combine(const ggml_tensor * op) {
    if (op->op != GGML_OP_CUSTOM) { return nullptr; }
    ggml_custom_op_params params; memcpy(&params, op->op_params, sizeof(params));
    return params.fun == hc_combine_marker ? static_cast<owner *>(params.userdata) : nullptr;
}
static bool hc_combine_shape(const ggml_tensor * r, const ggml_tensor * b, const ggml_tensor * w) {
    return r && b && w && r->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 && w->type == GGML_TYPE_F32 &&
        ggml_is_contiguous(r) && ggml_is_contiguous(b) && ggml_is_contiguous(w) &&
        r->ne[0] > 0 && r->ne[0] <= 65536 && r->ne[1] == 4 && r->ne[2] >= 1 && r->ne[2] <= 4 && r->ne[3] == 1 &&
        b->ne[0] == r->ne[0] && b->ne[1] == r->ne[2] && b->ne[2] == 1 && b->ne[3] == 1 &&
        w->ne[0] == 4 && w->ne[1] == r->ne[2] && w->ne[2] == 1 && w->ne[3] == 1;
}
static ggml_tensor * build_hc_combine(void * ptr, ggml_context * ctx, ggml_tensor * r, ggml_tensor * b, ggml_tensor * w) {
    if (!ptr || !hc_combine_shape(r, b, w)) { return nullptr; }
    ggml_tensor * args[] = {r, b, w};
    return ggml_custom_4d(ctx, GGML_TYPE_F32, r->ne[0], 4, r->ne[2], 1, args, 3, hc_combine_marker, 1, ptr);
}
__global__ void hc_combine(const float * residual, const float * block, const float * inject, float * out, int width, int tokens) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= width*4*tokens) { return; }
    const int stream_row = i/width;
    const float scaled = __fmul_rn(inject[stream_row], 0.25f);
    const float weight = __fmul_rn(1.0f / (1.0f + expf(-scaled)), 2.0f);
    const float value = __fmul_rn(block[(stream_row/4)*width + i%width], weight);
    out[i] = __fadd_rn(residual[i], value);
}
static void destroy(void * ptr) { delete static_cast<owner *>(ptr); }
static ggml_tensor * dense_input(ggml_context * ctx, ggml_tensor * x) {
    static const bool skip_copy = [] {
        const char * value = std::getenv("GGML_FT_DENSE_SKIP_COPY");
        return value && std::strcmp(value, "1") == 0;
    }();
    // Native GEMV loads float4 values; contiguous views can still be misaligned.
    const bool aligned = x->view_offs % 16 == 0 && (!x->data || reinterpret_cast<uintptr_t>(x->data) % 16 == 0);
    return skip_copy && aligned && ggml_is_contiguous(x) ? x : ggml_cont(ctx, x);
}
static ggml_tensor * build(void * ptr, ggml_context * ctx, ggml_tensor * w, ggml_tensor * x) {
    auto & p = *static_cast<owner *>(ptr);
    auto it = p.matrices.find(w);
    static const bool mtp_projection = [] {
        const char * value = std::getenv("GGML_FT_MTP_PROJECTION_FP8");
        return value && std::strcmp(value, "1") == 0;
    }();
    if (mtp_projection && it != p.matrices.end() && std::strstr(w->name, ".nextn.eh_proj.weight") &&
        x->type == GGML_TYPE_F32 && x->ne[0] == w->ne[0] && x->ne[1] == 4 &&
        x->ne[2] >= 2 && x->ne[2] <= 4 && x->ne[3] == 1) {
        ggml_tensor * args[] = {dense_input(ctx,x)};
        return ggml_custom_4d(ctx, GGML_TYPE_F32, w->ne[1], 4, x->ne[2], 1, args, 1, marker, 1, it->second.get());
    }
    if (it == p.matrices.end() || x->type != GGML_TYPE_F32 || x->ne[0] != w->ne[0] ||
        x->ne[1] < 1 || x->ne[1] > 4 || x->ne[2] != 1 || x->ne[3] != 1) { return nullptr; }
    ggml_tensor * args[] = {dense_input(ctx,x)};
    return ggml_custom_4d(ctx,GGML_TYPE_F32,w->ne[1],x->ne[1],1,1,args,1,marker,1,it->second.get());
}
} // namespace
const ggml_cuda_ft_dense_api * ggml_cuda_ft_dense_get_api() {
    static const ggml_cuda_ft_dense_api api{create,destroy,build,clone,build_hc,build_hc_combine}; return &api;
}
bool ggml_cuda_ft_dense_supports(const ggml_tensor * op, int device) {
    if (const auto * hc = get_hc_combine(op)) {
        return hc->device == device && op->type == GGML_TYPE_F32 && ggml_is_contiguous(op) &&
            hc_combine_shape(op->src[0], op->src[1], op->src[2]) && ggml_are_same_shape(op, op->src[0]);
    }
    if (const auto * hc = get_hc(op)) {
        return hc->device == device && op->type == GGML_TYPE_F32 && ggml_is_contiguous(op) &&
               hc_shape(op->src[0], op->src[1], op->ne[0]) && op->ne[1] == op->src[0]->ne[1] && op->ne[2] == 1 && op->ne[3] == 1;
    }
    const auto * p = get_op(op);
    return p && p->device == device && op->type == GGML_TYPE_F32 && ggml_is_contiguous(op) &&
           op->src[0] && op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0]);
}
void ggml_cuda_ft_dense_compute(ggml_backend_cuda_context & ctx, ggml_tensor * op) {
    if (get_hc_combine(op)) {
        GGML_ASSERT(ggml_cuda_ft_dense_supports(op, ctx.device));
        hc_combine<<<(ggml_nelements(op)+255)/256,256,0,ctx.stream()>>>(
            (const float *) op->src[0]->data, (const float *) op->src[1]->data, (const float *) op->src[2]->data,
            (float *) op->data, op->ne[0], op->ne[2]);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (get_hc(op)) {
        GGML_ASSERT(ggml_cuda_ft_dense_supports(op, ctx.device));
        hc_reduce<<<(ggml_nelements(op)+255)/256,256,0,ctx.stream()>>>(
            (const float *) op->src[0]->data, (const float *) op->src[1]->data, (float *) op->data, op->ne[0], op->ne[1]);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    auto * p = get_op(op); GGML_ASSERT(p && ggml_cuda_ft_dense_supports(op,ctx.device));
    const int m = op->ne[1];
    if (p->timing) { dense_stamp<<<1,1,0,ctx.stream()>>>(p->timing,m,false); CUDA_CHECK(cudaGetLastError()); }
    if (p->storage->block_scales) {
        GGML_ASSERT(op->ne[2] == 1);
        const auto * x = static_cast<const float *>(op->src[0]->data);
        auto * y = static_cast<float *>(op->data);
        const dim3 grid((p->n+3)/4);
        switch (m) {
            case 1: gemv_draft_head<1><<<grid,128,0,ctx.stream()>>>(p->weights,p->storage->block_scales,p->storage->globals,x,y,p->k,p->n); break;
            case 2: gemv_draft_head<2><<<grid,128,0,ctx.stream()>>>(p->weights,p->storage->block_scales,p->storage->globals,x,y,p->k,p->n); break;
            case 3: gemv_draft_head<3><<<grid,128,0,ctx.stream()>>>(p->weights,p->storage->block_scales,p->storage->globals,x,y,p->k,p->n); break;
            case 4: gemv_draft_head<4><<<grid,128,0,ctx.stream()>>>(p->weights,p->storage->block_scales,p->storage->globals,x,y,p->k,p->n); break;
            default: GGML_ABORT("Invalid FreeToken draft head batch");
        }
        CUDA_CHECK(cudaGetLastError());
        if (p->timing) { dense_stamp<<<1,1,0,ctx.stream()>>>(p->timing,m,true); CUDA_CHECK(cudaGetLastError()); }
        return;
    }
    static const bool batch_projection = [] {
        const char * value = std::getenv("GGML_FT_MTP_PROJECTION_BATCH");
        return value && std::strcmp(value, "1") == 0;
    }();
    if (batch_projection && m == 4 && op->ne[2] > 1 && p->m4_rows == 1 && p->splits == 1) {
        // No reduction scratch is needed when each block covers the full input row.
        gemv_tokens4<<<dim3(p->n,1,op->ne[2]),128,0,ctx.stream()>>>(p->weights,p->scales,
            static_cast<const float *>(op->src[0]->data),p->partial,static_cast<float *>(op->data),p->k,p->n,1);
        CUDA_CHECK(cudaGetLastError());
        if (p->timing) { dense_stamp<<<1,1,0,ctx.stream()>>>(p->timing,m,true); CUDA_CHECK(cudaGetLastError()); }
        return;
    }
    // Each four-stream slice reuses scratch on the same CUDA stream.
    for (int64_t t = 0; t < op->ne[2]; ++t) {
        const float * x = static_cast<const float *>(op->src[0]->data) + t*m*p->k;
        float * y = static_cast<float *>(op->data) + t*m*p->n;
        const dim3 grid((p->n+3)/4,p->splits);
        switch (m) {
            case 1:
                if (p->splits > 1) { gemv<1,true><<<grid,128,0,ctx.stream()>>>(p->weights,p->scales,x,p->partial,p->counters,y,p->k,p->n,p->splits); }
                else { gemv<1,false><<<grid,128,0,ctx.stream()>>>(p->weights,p->scales,x,p->partial,p->counters,y,p->k,p->n,p->splits); }
                break;
            case 2: gemv<2,false><<<grid,128,0,ctx.stream()>>>(p->weights,p->scales,x,p->partial,p->counters,y,p->k,p->n,p->splits); break;
            case 3: gemv<3,false><<<grid,128,0,ctx.stream()>>>(p->weights,p->scales,x,p->partial,p->counters,y,p->k,p->n,p->splits); break;
            case 4:
                switch (p->m4_rows) {
                    case 1: gemv_tokens4<<<dim3(p->n,p->splits),128,0,ctx.stream()>>>(p->weights,p->scales,x,p->partial,y,p->k,p->n,p->splits); break;
                    case 8: gemv<4,false,8><<<dim3((p->n+7)/8,p->splits),256,0,ctx.stream()>>>(p->weights,p->scales,x,p->partial,p->counters,y,p->k,p->n,p->splits); break;
                    case 16: gemv<4,false,16><<<dim3((p->n+15)/16,p->splits),512,0,ctx.stream()>>>(p->weights,p->scales,x,p->partial,p->counters,y,p->k,p->n,p->splits); break;
                    default: gemv<4,false><<<grid,128,0,ctx.stream()>>>(p->weights,p->scales,x,p->partial,p->counters,y,p->k,p->n,p->splits); break;
                }
                break;
            default: GGML_ABORT("Invalid FreeToken dense batch");
        }
        if (m > 1 && p->splits > 1) { finish<<<(p->n*m+255)/256,256,0,ctx.stream()>>>(p->partial,p->scales,y,p->n,m,p->splits); }
        CUDA_CHECK(cudaGetLastError());
    }
    if (p->timing) { dense_stamp<<<1,1,0,ctx.stream()>>>(p->timing,m,true); CUDA_CHECK(cudaGetLastError()); }
}
#endif
