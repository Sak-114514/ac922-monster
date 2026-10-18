#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include "freetoken-ggml.cuh"
#include "freetoken-cache.cuh"
#include "freetoken-dispatch.cuh"
#include "freetoken-moe.cuh"
#include "ggml-impl.h"
#include <cuda_fp8.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <fstream>
#include <set>
#if defined(__linux__)
#include <sched.h>
#endif

namespace {
static std::vector<int> worker_cores(int threads) {
    std::vector<int> result;
#if defined(__linux__)
    cpu_set_t allowed;
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) { return result; }
    std::set<std::pair<int, int>> seen;
    for (int cpu = 0; cpu < CPU_SETSIZE && int(result.size()) < threads; ++cpu) {
        if (!CPU_ISSET(cpu, &allowed)) { continue; }
        const std::string root = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
        int core, package;
        if (!(std::ifstream(root + "core_id") >> core) || !(std::ifstream(root + "physical_package_id") >> package)) { continue; }
        if (seen.emplace(package, core).second) { result.push_back(cpu); }
    }
    if (int(result.size()) != threads) { result.clear(); }
#else
    (void) threads;
#endif
    return result;
}
// FreeToken's draft quantization: FP16 row global, E4M3 scale per 16 values, E2M1 codes.
__global__ void quantize_draft(const __nv_bfloat16 * src, uint8_t * packed, uint8_t * scales,
                               uint16_t * globals, int cols, int * invalid) {
    const int row = blockIdx.x, tid = threadIdx.x;
    __shared__ float maxima[256];
    float amax = 0;
    for (int j = tid; j < cols; j += 256) {
        const float x = __bfloat162float(src[size_t(row)*cols+j]);
        if (!isfinite(x)) { atomicExch(invalid,1); }
        amax = fmaxf(amax,fabsf(x));
    }
    maxima[tid] = amax;
    __syncthreads();
    for (int stride = 128; stride; stride >>= 1) {
        if (tid < stride) { maxima[tid] = fmaxf(maxima[tid],maxima[tid+stride]); }
        __syncthreads();
    }
    const __half global = __float2half_rn(fmaxf(maxima[0]/2688.0f,0x1p-24f));
    const float g = __half2float(global);
    if (!isfinite(g)) { atomicExch(invalid,1); }
    if (!tid) { globals[row] = __half_as_ushort(global); }
    for (int block = tid; block < cols/16; block += 256) {
        float values[16], maximum = 0;
        for (int j = 0; j < 16; ++j) {
            values[j] = __bfloat162float(src[size_t(row)*cols+block*16+j]);
            maximum = fmaxf(maximum,fabsf(values[j]));
        }
        const __nv_fp8_e4m3 scale(fminf(448.0f,fmaxf(0x1p-6f,__fdiv_rn(maximum,6.0f*g))));
        scales[size_t(row)*(cols/16)+block] = scale.__x;
        const float divisor = static_cast<float>(scale)*g;
        unsigned codes[16];
        for (int j = 0; j < 16; ++j) {
            const float q = __fdiv_rn(values[j],divisor), a = fabsf(q);
            // bucketize uses the lower code at exact midpoints.
            codes[j] = (a > .25f)+(a > .75f)+(a > 1.25f)+(a > 1.75f)+(a > 2.5f)+(a > 3.5f)+(a > 5.f);
            if (q < 0) { codes[j] += 8; }
        }
        for (int j = 0; j < 8; ++j) {
            packed[size_t(row)*(cols/2)+block*8+j] = codes[2*j] | (codes[2*j+1]<<4);
        }
    }
}
__global__ void draft_routes(const int32_t * ids, int32_t * slots, int n, int experts) {
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < n) { slots[i] = ids[i] >= 0 && ids[i] < experts ? ids[i] : -1; }
}

struct owner;
struct layer_op { owner * parent; int layer; };
static void marker(ggml_tensor *, int, int, void *) { GGML_ABORT("FreeToken MoE requires the CUDA backend"); }
static void marker_begin(ggml_tensor *, int, int, void *) { GGML_ABORT("FreeToken submission requires CUDA"); }
static void marker_finish(ggml_tensor *, int, int, void *) { GGML_ABORT("FreeToken completion requires CUDA"); }
struct owner {
    int device, h = 0, intermediate = 0, experts = 0, top_k;
    std::vector<void *> pinned;
    std::vector<layer_op> ops;
    std::vector<freetoken_cpu_bank> cpu_banks;
    std::unique_ptr<freetoken_gpu_cache> cache;
    std::unique_ptr<freetoken_cpu_dispatch> dispatch;
    uint16_t * activation = nullptr, * down = nullptr;
    float * gpu = nullptr;
    int fetch_cap = 1;
    bool resident_mode = false;
    freetoken_cache_view resident{};

    static void check(cudaError_t error) {
        if (error != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(error)); }
    }
    void allocate(void * & host, const void * & mapped, size_t bytes) {
        check(cudaHostAlloc(&host, bytes, cudaHostAllocMapped));
        pinned.push_back(host);
        void * ptr;
        check(cudaHostGetDevicePointer(&ptr, host, 0)); mapped = ptr;
    }
    void release() noexcept {
        cudaSetDevice(device);
        cudaDeviceSynchronize();
        dispatch.reset(); cache.reset();
        for (auto * bank : resident.banks) { cudaFree(bank); }
        cudaFree(resident.gpu_ids);
        cudaFree(activation); cudaFree(down); cudaFree(gpu);
        for (void * p : pinned) { cudaFreeHost(p); }
    }
    static bool eligible(const ggml_cuda_ft_layer & l) {
        return l.gate && l.up && l.down && l.gate->type == GGML_TYPE_NVFP4 &&
               l.up->type == GGML_TYPE_NVFP4 && l.down->type == GGML_TYPE_NVFP4;
    }
    void pack(ggml_tensor * tensor, ggml_tensor * scale, int rows, int cols,
              uint8_t * packed, uint8_t * block_scales, uint16_t * globals,
              int output_rows, int row_offset) {
        if (!ggml_is_contiguous(tensor) || tensor->ne[0] != cols || tensor->ne[1] != rows ||
            tensor->ne[2] != experts || tensor->ne[3] != 1 ||
            (scale && (scale->type != GGML_TYPE_F32 || ggml_nelements(scale) != experts))) {
            throw std::runtime_error("Unsupported FreeToken expert tensor layout");
        }
        std::vector<float> factors(experts, 1.0f);
        if (scale) { ggml_backend_tensor_get(scale, factors.data(), 0, factors.size()*sizeof(float)); }
        const size_t bytes = ggml_row_size(GGML_TYPE_NVFP4, cols) * rows;
        std::vector<uint8_t> staging(bytes);
        for (int e = 0; e < experts; ++e) {
            ggml_backend_tensor_get(tensor, staging.data(), size_t(e)*bytes, bytes);
            const size_t row = size_t(e)*output_rows + row_offset;
            freetoken_unpack_nvfp4(staging.data(), rows, cols, packed + row*(cols/2), block_scales + row*(cols/16));
            std::fill(globals + row, globals + row + rows, ggml_fp32_to_fp16(factors[e]));
        }
    }
    owner(int dev, int slots, int threads, int k, const ggml_cuda_ft_layer * layers, int count) : device(dev), top_k(k) {
        if (!layers || count < 1 || count > 128 || slots < 1 || threads < 1 || threads > 256 || k < 1 || k > 64) {
            throw std::invalid_argument("Invalid FreeToken owner dimensions");
        }
        if (const char * value = std::getenv("GGML_FT_FETCH_CAP")) {
            char * end = nullptr;
            const long cap = std::strtol(value,&end,10);
            if (end == value || *end || cap < 1 || cap > 4*k) {
                throw std::invalid_argument("GGML_FT_FETCH_CAP must be an integer between 1 and 4*top_k");
            }
            fetch_cap = int(cap);
        }
        check(cudaSetDevice(device));
        for (int l = 0; l < count; ++l) if (eligible(layers[l])) {
            h = layers[l].gate->ne[0]; intermediate = layers[l].gate->ne[1]; experts = layers[l].gate->ne[2]; break;
        }
        if (h <= 0 || intermediate <= 0 || h % 64 || intermediate % 64 || h > 65536 || intermediate > 65536 || experts < k) {
            throw std::runtime_error("No supported NVFP4 expert layers for FreeToken");
        }
        const size_t bytes[6] = {size_t(intermediate)*h, size_t(h)*intermediate/2, size_t(intermediate)*h/8,
                                size_t(h)*intermediate/16, size_t(intermediate)*4, size_t(h)*2};
        pinned.reserve(size_t(count)*6); ops.resize(count, {this, -1});
        std::vector<freetoken_cache_layer> mapped;
        try {
            for (int l = 0; l < count; ++l) {
                if (!eligible(layers[l])) { continue; }
                void * host[6]; freetoken_cache_layer m{};
                for (int b = 0; b < 6; ++b) { allocate(host[b], m.banks[b], size_t(experts)*bytes[b]); }
                freetoken_cpu_bank bank{(uint8_t *)host[0], (uint8_t *)host[1], (uint8_t *)host[2],
                                       (uint8_t *)host[3], (uint16_t *)host[4], (uint16_t *)host[5]};
                pack(layers[l].gate, layers[l].gate_scale, intermediate,h,(uint8_t *)host[0],(uint8_t *)host[2],(uint16_t *)host[4],2*intermediate,0);
                pack(layers[l].up, layers[l].up_scale, intermediate,h,(uint8_t *)host[0],(uint8_t *)host[2],(uint16_t *)host[4],2*intermediate,intermediate);
                pack(layers[l].down, layers[l].down_scale,h,intermediate,(uint8_t *)host[1],(uint8_t *)host[3],(uint16_t *)host[5],h,0);
                ops[l].layer = mapped.size(); mapped.push_back(m); cpu_banks.push_back(bank);
                fprintf(stderr, "FreeToken: packed layer %d (%zu MiB canonical host banks)\n", l,
                        mapped.size()*size_t(experts)*(bytes[0]+bytes[1]+bytes[2]+bytes[3]+bytes[4]+bytes[5])/(1024*1024));
            }
            fprintf(stderr,"FreeToken: demand fetch cap %d experts per layer call\n",fetch_cap);
            cache.reset(new freetoken_gpu_cache(experts,slots,4*k,bytes,mapped));
            dispatch.reset(new freetoken_cpu_dispatch(h,intermediate,experts,k,4,threads,cpu_banks,worker_cores(threads)));
            check(cudaMalloc(&activation,size_t(4*k)*intermediate*sizeof(uint16_t)));
            check(cudaMalloc(&down,size_t(4*k)*h*sizeof(uint16_t)));
            check(cudaMalloc(&gpu,size_t(4)*h*sizeof(float)));
        } catch (...) { release(); throw; }
    }
    void pack_draft(ggml_tensor * tensor, int rows, int cols, int output_rows, int offset,
                    int wb, int sb, int gb) {
        // Bound staging independently of the number of experts and the full checkpoint.
        const int chunk = 128;
        std::vector<ggml_bf16_t> host(size_t(chunk)*cols);
        void * stage = nullptr; int * invalid = nullptr;
        try {
            check(cudaMalloc(&stage,host.size()*sizeof(ggml_bf16_t)));
            check(cudaMalloc(&invalid,sizeof(int))); check(cudaMemset(invalid,0,sizeof(int)));
            for (int e = 0; e < experts; ++e) for (int r = 0; r < rows; r += chunk) {
                const int nr = std::min(chunk,rows-r);
                const size_t bytes = size_t(nr)*cols*sizeof(ggml_bf16_t);
                ggml_backend_tensor_get(tensor,host.data(),(size_t(e)*rows+r)*cols*sizeof(ggml_bf16_t),bytes);
                check(cudaMemcpy(stage,host.data(),bytes,cudaMemcpyHostToDevice));
                const size_t dest = size_t(e)*output_rows+offset+r;
                quantize_draft<<<nr,256>>>(static_cast<const __nv_bfloat16 *>(stage),resident.banks[wb]+dest*(cols/2),
                        resident.banks[sb]+dest*(cols/16),reinterpret_cast<uint16_t *>(resident.banks[gb])+dest,cols,invalid);
                check(cudaGetLastError());
            }
            int bad; check(cudaMemcpy(&bad,invalid,sizeof(int),cudaMemcpyDeviceToHost));
            if (bad) { throw std::runtime_error("Non-finite or unrepresentable FreeToken draft weights"); }
        } catch (...) { cudaFree(stage); cudaFree(invalid); throw; }
        cudaFree(stage); cudaFree(invalid);
    }
    owner(int dev, int k, int layer, const ggml_cuda_ft_layer & l) : device(dev), top_k(k), resident_mode(true) {
        if (layer < 0 || layer >= 128 || k < 1 || k > 64 || !l.gate || !l.up || !l.down ||
            l.gate_scale || l.up_scale || l.down_scale) { throw std::invalid_argument("Invalid FreeToken draft layer"); }
        h = l.gate->ne[0]; intermediate = l.gate->ne[1]; experts = l.gate->ne[2];
        if (h < 64 || intermediate < 64 || h > 65536 || intermediate > 65536 || h%64 || intermediate%64 || experts < k || experts > 65536) {
            throw std::invalid_argument("Invalid FreeToken draft dimensions");
        }
        for (const auto * t : {l.gate,l.up,l.down}) {
            const bool down_proj = t == l.down;
            if (t->type != GGML_TYPE_BF16 || !ggml_is_contiguous(t) || t->ne[0] != (down_proj ? intermediate : h) ||
                t->ne[1] != (down_proj ? h : intermediate) || t->ne[2] != experts || t->ne[3] != 1) {
                throw std::runtime_error("FreeToken draft requires contiguous BF16 experts without scales");
            }
        }
        check(cudaSetDevice(device));
        const size_t bytes[6] = {size_t(intermediate)*h,size_t(h)*intermediate/2,size_t(intermediate)*h/8,
                                size_t(h)*intermediate/16,size_t(intermediate)*4,size_t(h)*2};
        size_t total = 0;
        for (size_t b : bytes) { total += size_t(experts)*b; }
        const size_t scratch = size_t(4*k)*(h+intermediate)*sizeof(uint16_t)+size_t(4)*h*sizeof(float)+size_t(4*k)*sizeof(int32_t);
        size_t free_bytes, all_bytes; check(cudaMemGetInfo(&free_bytes,&all_bytes));
        if (total+scratch+size_t(128)*std::max(h,intermediate)*2+(size_t(2)<<30) > free_bytes) {
            throw std::runtime_error("FreeToken resident MTP NVFP4 needs its expert banks plus 2 GiB free VRAM");
        }
        ops.resize(layer+1,{this,-1}); ops[layer].layer = 0;
        try {
            for (int b = 0; b < 6; ++b) {
                resident.bytes_per_expert[b] = bytes[b];
                check(cudaMalloc(&resident.banks[b],size_t(experts)*bytes[b]));
            }
            resident.route_capacity = 4*k;
            check(cudaMalloc(&resident.gpu_ids,size_t(4*k)*sizeof(int32_t)));
            check(cudaMalloc(&activation,size_t(4*k)*intermediate*sizeof(uint16_t)));
            check(cudaMalloc(&down,size_t(4*k)*h*sizeof(uint16_t)));
            pack_draft(l.gate,intermediate,h,2*intermediate,0,0,2,4);
            pack_draft(l.up,intermediate,h,2*intermediate,intermediate,0,2,4);
            pack_draft(l.down,h,intermediate,h,0,1,3,5);
            check(cudaStreamSynchronize(nullptr));
            fprintf(stderr,"FreeToken: resident MTP NVFP4 layer %d, %d experts, %zu MiB VRAM; no CPU draft dispatch\n",layer,experts,(total+scratch)/(1024*1024));
        } catch (...) { release(); throw; }
    }
    ~owner() { release(); }
};
static void * create(int dev, int slots, int threads, int k, const ggml_cuda_ft_layer * layers, int count) {
    return new owner(dev,slots,threads,k,layers,count);
}
static void * create_draft(int dev, int k, int layer, const ggml_cuda_ft_layer * weights) {
    if (!weights) { throw std::invalid_argument("Missing FreeToken draft weights"); }
    return new owner(dev,k,layer,*weights);
}
static void destroy(void * p) { delete static_cast<owner *>(p); }
static ggml_tensor * build(void * ptr, ggml_context * ctx, int layer, ggml_tensor * x, ggml_tensor * ids, ggml_tensor * weights) {
    auto & p = *static_cast<owner *>(ptr);
    if (layer < 0 || layer >= int(p.ops.size()) || p.ops[layer].layer < 0 || x->ne[1] > 4 ||
        x->ne[0] != p.h || ggml_nelements(ids) != x->ne[1]*p.top_k) { return nullptr; }
    ggml_tensor * args[] = {ggml_cont(ctx,x), ggml_cont(ctx,ids), ggml_cont(ctx,weights)};
    return ggml_custom_4d(ctx,GGML_TYPE_F32,p.h,x->ne[1],1,1,args,3,marker,1,&p.ops[layer]);
}
static layer_op * get_op(const ggml_tensor * op) {
    if (op->op != GGML_OP_CUSTOM) { return nullptr; }
    ggml_custom_op_params params; memcpy(&params,op->op_params,sizeof(params));
    return params.fun == marker || params.fun == marker_begin || params.fun == marker_finish ? static_cast<layer_op *>(params.userdata) : nullptr;
}
static bool is_marker(const ggml_tensor * op, ggml_custom_op_t fun) {
    if (op->op != GGML_OP_CUSTOM) { return false; }
    ggml_custom_op_params params; memcpy(&params,op->op_params,sizeof(params));
    return params.fun == fun;
}
static ggml_tensor * begin(ggml_context * ctx, ggml_tensor * moe) {
    auto * job = get_op(moe);
    if (!job || job->parent->resident_mode || !is_marker(moe,marker)) { return nullptr; }
    ggml_tensor * args[] = {moe->src[0],moe->src[1],moe->src[2]};
    return ggml_custom_4d(ctx,GGML_TYPE_F32,moe->ne[0],moe->ne[1],1,1,args,3,marker_begin,1,job);
}
static ggml_tensor * finish(ggml_context * ctx, ggml_tensor * start, ggml_tensor * dependency) {
    auto * job = get_op(start);
    if (!job || !is_marker(start,marker_begin) || !dependency) { throw std::invalid_argument("Invalid FreeToken deferred completion"); }
    ggml_tensor * args[] = {start,dependency};
    return ggml_custom_4d(ctx,GGML_TYPE_F32,start->ne[0],start->ne[1],1,1,args,2,marker_finish,1,job);
}
__global__ void round_output(float * x, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) { x[i] = __bfloat162float(__float2bfloat16(x[i])); }
}
} // namespace
void freetoken_quantize_draft(const void * src, uint8_t * packed, uint8_t * scales,
                             uint16_t * globals, int rows, int cols, int * invalid, cudaStream_t stream) {
    quantize_draft<<<rows,256,0,stream>>>(static_cast<const __nv_bfloat16 *>(src),packed,scales,globals,cols,invalid);
    CUDA_CHECK(cudaGetLastError());
}
const ggml_cuda_ft_api * ggml_cuda_ft_get_api() {
    static const ggml_cuda_ft_api api{create,destroy,build,create_draft,begin,finish}; return &api;
}
bool ggml_cuda_ft_supports(const ggml_tensor * op, int device) {
    const auto * job = get_op(op);
    if (!job || job->parent->device != device || op->type != GGML_TYPE_F32 || !ggml_is_contiguous(op)) { return false; }
    if (is_marker(op,marker_finish)) {
        return op->src[0] && is_marker(op->src[0],marker_begin) && get_op(op->src[0]) == job && op->src[1];
    }
    for (int i = 0; i < 3; ++i) {
        if (!op->src[i] || !ggml_is_contiguous(op->src[i]) || op->src[i]->type != (i==1?GGML_TYPE_I32:GGML_TYPE_F32)) { return false; }
    }
    return true;
}
void ggml_cuda_ft_compute(ggml_backend_cuda_context & ctx, ggml_tensor * op) {
    auto * job = get_op(op); GGML_ASSERT(job && ggml_cuda_ft_supports(op,ctx.device));
    auto & p = *job->parent; const int tokens = op->ne[1], routes = tokens*p.top_k;
    const auto stream = ctx.stream();
    if (is_marker(op,marker_finish)) {
        p.dispatch->finish_routes(tokens,(float *)op->data,p.down,stream);
        return;
    }
    if (p.resident_mode) {
        draft_routes<<<(routes+127)/128,128,0,stream>>>((const int32_t *)op->src[1]->data,p.resident.gpu_ids,routes,p.experts);
        freetoken_gpu_moe(p.resident,p.h,p.intermediate,p.top_k,tokens,(const float *)op->src[0]->data,
                         (const float *)op->src[2]->data,p.activation,p.down,(float *)op->data,stream);
        round_output<<<(tokens*p.h+255)/256,256,0,stream>>>((float *)op->data,tokens*p.h);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    p.cache->route(job->layer,routes,(const int32_t *)op->src[1]->data,p.fetch_cap,stream);
    p.dispatch->submit(job->layer,tokens,(const float *)op->src[0]->data,p.cache->view().cpu_ids,(const float *)op->src[2]->data,stream);
    p.cache->fetch(job->layer,routes,p.fetch_cap,stream);
    static const bool fuse_finalize = [] {
        const char * value = std::getenv("GGML_FT_MOE_FINALIZE_FUSION");
        return value && std::strcmp(value,"1") == 0;
    }();
    if (fuse_finalize || is_marker(op,marker_begin)) {
        freetoken_gpu_moe_deferred(p.cache->view(),p.h,p.intermediate,p.top_k,tokens,
            (const float *)op->src[0]->data,(const float *)op->src[2]->data,p.activation,p.down,stream);
        if (!is_marker(op,marker_begin)) { p.dispatch->finish_routes(tokens,(float *)op->data,p.down,stream); }
        return;
    }
    freetoken_gpu_moe(p.cache->view(),p.h,p.intermediate,p.top_k,tokens,(const float *)op->src[0]->data,
                     (const float *)op->src[2]->data,p.activation,p.down,p.gpu,stream);
    p.dispatch->finish(tokens,(float *)op->data,p.gpu,stream);
    round_output<<<(tokens*p.h+255)/256,256,0,stream>>>((float *)op->data,tokens*p.h);
    CUDA_CHECK(cudaGetLastError());
}
#endif
