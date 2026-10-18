// Experimental SM70 weight-streaming MoE. One cooperative launch, no Hopper ISA.
// Router, top-k, input quantization, gated expert projections, intermediate
// quantization, down projection and expert reduction share a persistent grid.
#include "monomoe-sm70.h"
#include "common.cuh"
#include "vecdotq.cuh"
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdint>
namespace cg=cooperative_groups;
namespace {
constexpr int THREADS=256, WARPS=8, MAX_TOPK=10;
struct scratch {
    float * logits;
    int * ids;
    float * weights;
    block_q8_1 * input_q;
    float * input_d;
    float * intermediate;
    block_q8_1 * intermediate_q;
    float * intermediate_d;
    float * partial;
};
size_t align128(size_t n) { return (n+127)&~size_t(127); }
scratch layout(void * ptr,const monomoe_sm70_params & p,size_t & total) {
    total=0; char * base=(char*)ptr; scratch s{};
#define TAKE(field,type,count) s.field=(type*)(base?base+total:nullptr);total+=align128(sizeof(type)*size_t(count))
    TAKE(logits,float,p.experts*p.tokens);
    TAKE(ids,int,p.selected*p.tokens);
    TAKE(weights,float,p.selected*p.tokens);
    TAKE(input_q,block_q8_1,(p.hidden/32)*p.tokens);
    TAKE(input_d,float,(p.hidden/32)*p.tokens);
    TAKE(intermediate,float,p.ffn*p.selected*p.tokens);
    TAKE(intermediate_q,block_q8_1,(p.ffn/32)*p.selected*p.tokens);
    TAKE(intermediate_d,float,(p.ffn/32)*p.selected*p.tokens);
    TAKE(partial,float,p.hidden*p.selected*p.tokens);
#undef TAKE
    return s;
}
__device__ __forceinline__ float mono_scale(const block_q8_1 * base,const block_q8_1 * block,const float * full) {
    return full?full[block-base]:__low2float(block->ds);
}
#include "monomoe-quant-dots.cuh"
template<ggml_type Q> __device__ __forceinline__ float dot_chunk(const void * w,const block_q8_1 * y,int b,int i,const float * d) {
    if constexpr(Q==GGML_TYPE_Q2_0) return mono_dot_q2_0(w,y,b,i,d);
    if constexpr(Q==GGML_TYPE_IQ2_XXS) return mono_dot_iq2_xxs(w,y,b,i,d);
    if constexpr(Q==GGML_TYPE_IQ2_XS) return mono_dot_iq2_xs(w,y,b,i,d);
    if constexpr(Q==GGML_TYPE_IQ2_S) return mono_dot_iq2_s(w,y,b,i,d);
    if constexpr(Q==GGML_TYPE_IQ3_XXS) return mono_dot_iq3_xxs(w,y,b,i,d);
    if constexpr(Q==GGML_TYPE_IQ3_S) return mono_dot_iq3_s(w,y,b,i,d);
    if constexpr(Q==GGML_TYPE_IQ4_NL) return mono_dot_iq4_nl(w,y,b,i,d);
    return 0;
}
template<ggml_type Q> __host__ __device__ constexpr int vdr() {
    if constexpr(Q==GGML_TYPE_Q2_0) return VDR_Q2_0_Q8_1_MMVQ;
    if constexpr(Q==GGML_TYPE_IQ4_NL) return VDR_IQ4_NL_Q8_1_MMVQ;
    return 2;
}
__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for(int d=16;d;d>>=1) x+=__shfl_xor_sync(0xffffffff,x,d);
    return x;
}
__device__ __forceinline__ float warp_max(float x) {
#pragma unroll
    for(int d=16;d;d>>=1) x=fmaxf(x,__shfl_xor_sync(0xffffffff,x,d));
    return x;
}
template<ggml_type Q> __device__ __forceinline__ float row_dot(
        const void * row,const block_q8_1 * y,const float * d,int n,int width=32) {
    constexpr int QK=ggml_cuda_type_traits<Q>::qk;
    constexpr int QI=ggml_cuda_type_traits<Q>::qi;
    constexpr int V=vdr<Q>();
    const int lane=threadIdx.x&(width-1);
    float sum=0;
    for(int b=lane/(QI/V);b<n/QK;b+=V*width/QI) {
        const int by=b*(QK/32), i=V*(lane%(QI/V));
        sum+=dot_chunk<Q>(row,y+by,b,i,d?d+by:nullptr);
    }
    const unsigned mask=__activemask();
    for(int shift=width/2;shift;shift>>=1)sum+=__shfl_xor_sync(mask,sum,shift,width);
    return sum;
}
// Match the source path's inverse rounding as well as half/full d storage.
__device__ void quantize_warp(const float * x,block_q8_1 * y,float * ds,bool mmq) {
    const int lane=threadIdx.x&31;const float v=x[lane];
    const float a=warp_max(fabsf(v));
    const float sum=warp_sum(v);
    const float d=mmq?(a>0?1.0f/(127.0f/a):0.0f):a/127.0f;
    const float inv=a>0?(mmq?127.0f/a:1.0f/d):0.0f;
    y->qs[lane]=(int8_t)roundf(v*inv);
    if(lane==0) {y->ds=__floats2half2_rn(d,sum);*ds=d;}
}
int vector_limit(ggml_type q) {
    switch(q) {
        case GGML_TYPE_IQ2_XXS:case GGML_TYPE_IQ2_XS:return 5;
        case GGML_TYPE_IQ2_S:case GGML_TYPE_IQ3_XXS:case GGML_TYPE_IQ3_S:return 4;
        case GGML_TYPE_IQ4_NL:return 6;
        case GGML_TYPE_Q2_0:return 8;
        default:return 0;
    }
}
template<ggml_type G,ggml_type D>
__global__ __launch_bounds__(THREADS,2) void mono_kernel(monomoe_sm70_params p,scratch s,
        bool gate_mmq,bool down_mmq,bool down_full) {
    cg::grid_group grid=cg::this_grid();
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    const int global_warp=blockIdx.x*WARPS+warp, total_warps=gridDim.x*WARPS;
    __shared__ float reduction[WARPS];
    __shared__ float winners[WARPS];
    __shared__ int winner_ids[WARPS];
    __shared__ int selected_id;
    // Each warp evaluates one router row; tensor-parallel routers are replicated.
    for(int job=global_warp;job<p.tokens*p.experts;job+=total_warps) {
        const int t=job/p.experts,e=job%p.experts;
        const char * wr=(const char*)p.router+e*p.router_row;
        float sum=0;
        for(int c=lane;c<p.hidden;c+=32) {
            float w=p.router_type==GGML_TYPE_BF16?__bfloat162float(((const __nv_bfloat16*)wr)[c]):((const float*)wr)[c];
            sum=fmaf(w,p.input[t*p.hidden+c],sum);
        }
        sum=warp_sum(sum);if(lane==0)s.logits[job]=sum;
    }
    grid.sync();
    // One CTA per token: full softmax, deterministic top-k and selected renormalization.
    for(int t=blockIdx.x;t<p.tokens;t+=gridDim.x) {
        float mx=-INFINITY;
        for(int e=threadIdx.x;e<p.experts;e+=THREADS)mx=fmaxf(mx,s.logits[t*p.experts+e]);
        mx=warp_max(mx);if(lane==0)reduction[warp]=mx;__syncthreads();
        if(threadIdx.x==0)for(int j=1;j<WARPS;j++)reduction[0]=fmaxf(reduction[0],reduction[j]);
        __syncthreads();mx=reduction[0];
        float sum=0;
        for(int e=threadIdx.x;e<p.experts;e+=THREADS)sum+=expf(s.logits[t*p.experts+e]-mx);
        sum=warp_sum(sum);if(lane==0)reduction[warp]=sum;__syncthreads();
        if(threadIdx.x==0)for(int j=1;j<WARPS;j++)reduction[0]+=reduction[j];
        __syncthreads();const float inv=1.0f/reduction[0];
        float v0=threadIdx.x<p.experts?s.logits[t*p.experts+threadIdx.x]:-INFINITY;
        float v1=threadIdx.x+THREADS<p.experts?s.logits[t*p.experts+threadIdx.x+THREADS]:-INFINITY;
        for(int k=0;k<p.selected;k++) {
            float best=v0;int id=threadIdx.x;
            if(v1>best){best=v1;id=threadIdx.x+THREADS;}
            for(int shift=16;shift;shift>>=1) {
                const float other=__shfl_xor_sync(0xffffffff,best,shift);
                const int oid=__shfl_xor_sync(0xffffffff,id,shift);
                if(other>best||(other==best&&oid<id)){best=other;id=oid;}
            }
            if(lane==0){winners[warp]=best;winner_ids[warp]=id;}
            __syncthreads();
            if(warp==0) {
                best=lane<WARPS?winners[lane]:-INFINITY;
                id=lane<WARPS?winner_ids[lane]:INT_MAX;
                for(int shift=16;shift;shift>>=1) {
                    const float other=__shfl_xor_sync(0xffffffff,best,shift);
                    const int oid=__shfl_xor_sync(0xffffffff,id,shift);
                    if(other>best||(other==best&&oid<id)){best=other;id=oid;}
                }
                if(lane==0) {
                    selected_id=id;s.ids[t*p.selected+k]=id;
                    s.weights[t*p.selected+k]=expf(best-mx)*inv;
                }
            }
            __syncthreads();
            if(selected_id==threadIdx.x)v0=-INFINITY;
            if(selected_id==threadIdx.x+THREADS)v1=-INFINITY;
            __syncthreads();
        }
        if(threadIdx.x==0) {
            float sum_selected=0;
            for(int k=0;k<p.selected;k++)sum_selected+=s.weights[t*p.selected+k];
            const float norm=p.normalize_selected?1.0f/sum_selected:1.0f;
            for(int k=0;k<p.selected;k++)s.weights[t*p.selected+k]*=norm*p.routing_scale;
        }
        __syncthreads();
    }
    for(int job=global_warp;job<p.tokens*(p.hidden/32);job+=total_warps) {
        quantize_warp(p.input+job*32,s.input_q+job,s.input_d+job,gate_mmq);
    }
    grid.sync();
    // Four rows share the activation stream, exposing independent gate/up dots
    // to the Volta scheduler while keeping each row's accumulation order.
    constexpr int ROWS=4;
    constexpr int GQ=ggml_cuda_type_traits<G>::qk,GI=ggml_cuda_type_traits<G>::qi,GV=vdr<G>();
    const int tiles=p.ffn/ROWS;
    for(int job=global_warp;job<p.tokens*p.selected*tiles;job+=total_warps) {
        const int row0=(job%tiles)*ROWS,pair=job/tiles,t=pair/p.selected;
        const int e=s.ids[pair];
        const char * wg=(const char*)p.gate+e*p.gate_expert+row0*p.gate_row;
        const char * wu=(const char*)p.up+e*p.up_expert+row0*p.up_row;
        const auto * y=s.input_q+t*(p.hidden/32);
        const float * dy=gate_mmq?s.input_d+t*(p.hidden/32):nullptr;
        float gate[ROWS]={},up[ROWS]={};
        for(int qb=lane/(GI/GV);qb<p.hidden/GQ;qb+=GV*32/GI) {
            const int by=qb*(GQ/32),iq=GV*(lane%(GI/GV));
#pragma unroll
            for(int r=0;r<ROWS;r++) {
                gate[r]+=dot_chunk<G>(wg+r*p.gate_row,y+by,qb,iq,dy?dy+by:nullptr);
                up[r]+=dot_chunk<G>(wu+r*p.up_row,y+by,qb,iq,dy?dy+by:nullptr);
            }
        }
#pragma unroll
        for(int r=0;r<ROWS;r++) {
            float g=warp_sum(gate[r]),u=warp_sum(up[r]);
            if(lane==0)s.intermediate[pair*p.ffn+row0+r]=(g/(1.0f+expf(-g)))*u;
        }
    }
    grid.sync();
    for(int job=global_warp;job<p.tokens*p.selected*(p.ffn/32);job+=total_warps) {
        quantize_warp(s.intermediate+job*32,s.intermediate_q+job,s.intermediate_d+job,down_mmq);
    }
    grid.sync();
    // A narrow down projection needs only 4/8/16 lanes per row. Pack several
    // independent rows into one warp instead of leaving most lanes idle.
    constexpr int DQ=ggml_cuda_type_traits<D>::qk,DI=ggml_cuda_type_traits<D>::qi;
    int width=1;
    while(width<32 && width<(p.ffn/DQ)*(DI/vdr<D>()))width*=2;
    const int group=(blockIdx.x*THREADS+threadIdx.x)/width;
    const int groups=gridDim.x*THREADS/width;
    constexpr int DOWN_ROWS=4;
    const int down_tiles=p.hidden/DOWN_ROWS;
    const int glane=threadIdx.x&(width-1);
    for(int job=group;job<p.tokens*p.selected*down_tiles;job+=groups) {
        const int row0=(job%down_tiles)*DOWN_ROWS,pair=job/down_tiles,e=s.ids[pair];
        const char * wd=(const char*)p.down+e*p.down_expert+row0*p.down_row;
        const auto * y=s.intermediate_q+pair*(p.ffn/32);
        const float * dy=down_full?s.intermediate_d+pair*(p.ffn/32):nullptr;
        float acc[DOWN_ROWS]={};
        constexpr int V=vdr<D>();
        for(int qb=glane/(DI/V);qb<p.ffn/DQ;qb+=V*width/DI) {
            const int by=qb*(DQ/32),iq=V*(glane%(DI/V));
#pragma unroll
            for(int r=0;r<DOWN_ROWS;r++)acc[r]+=dot_chunk<D>(wd+r*p.down_row,y+by,qb,iq,dy?dy+by:nullptr);
        }
        const unsigned mask=__activemask();
#pragma unroll
        for(int r=0;r<DOWN_ROWS;r++) {
            float v=acc[r];
            for(int shift=width/2;shift;shift>>=1)v+=__shfl_xor_sync(mask,v,shift,width);
            if(glane==0)s.partial[pair*p.hidden+row0+r]=v*s.weights[pair];
        }
    }
    grid.sync();
    for(int i=blockIdx.x*THREADS+threadIdx.x;i<p.tokens*p.hidden;i+=gridDim.x*THREADS) {
        const int t=i/p.hidden,o=i%p.hidden;
        float sum=s.partial[t*p.selected*p.hidden+o];
        for(int k=1;k<p.selected;k++)sum+=s.partial[(t*p.selected+k)*p.hidden+o];
        p.output[i]=sum;
    }
}
template<ggml_type G,ggml_type D>
cudaError_t launch_typed(const monomoe_sm70_params & p,scratch s,cudaStream_t stream) {
    int device=0,coop=0,sm=0,active=0;
    cudaError_t err=cudaGetDevice(&device);if(err!=cudaSuccess)return err;
    cudaDeviceGetAttribute(&coop,cudaDevAttrCooperativeLaunch,device);
    cudaDeviceGetAttribute(&sm,cudaDevAttrMultiProcessorCount,device);
    if(!coop)return cudaErrorNotSupported;
    err=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,mono_kernel<G,D>,THREADS,0);if(err!=cudaSuccess)return err;
    if(active<1)return cudaErrorInvalidConfiguration;
    bool gm=p.tokens>vector_limit(G),dm=p.tokens>vector_limit(D),df=dm&&D==GGML_TYPE_Q2_0;
    monomoe_sm70_params copy=p;void * args[]={&copy,&s,&gm,&dm,&df};
    return cudaLaunchCooperativeKernel((void*)mono_kernel<G,D>,dim3(sm*std::min(active,2)),dim3(THREADS),args,0,stream);
}
template<ggml_type G> cudaError_t launch_down(const monomoe_sm70_params & p,scratch s,cudaStream_t stream) {
    switch(p.down_type) {
        case GGML_TYPE_Q2_0:return launch_typed<G,GGML_TYPE_Q2_0>(p,s,stream);
        case GGML_TYPE_IQ4_NL:return launch_typed<G,GGML_TYPE_IQ4_NL>(p,s,stream);
        default:return cudaErrorNotSupported;
    }
}
} // namespace
size_t monomoe_sm70_workspace_size(const monomoe_sm70_params & p) {size_t n;layout(nullptr,p,n);return n;}
cudaError_t monomoe_sm70_launch(const monomoe_sm70_params & p,void * workspace,size_t bytes,cudaStream_t stream) {
    if(p.tokens<1||p.tokens>32||p.experts<1||p.experts>512||p.selected<1||p.selected>MAX_TOPK||p.selected>p.experts||p.hidden<=0||p.ffn<=0||p.hidden>16384||p.ffn>16384||p.hidden%256||p.ffn%32||(p.down_type==GGML_TYPE_Q2_0&&p.ffn%64)||
       (p.router_type!=GGML_TYPE_BF16&&p.router_type!=GGML_TYPE_F32)||!p.input||!p.router||!p.gate||!p.up||!p.down||!p.output)return cudaErrorInvalidValue;
    size_t need;const scratch s=layout(workspace,p,need);if(bytes<need||!workspace)return cudaErrorInvalidValue;
    switch(p.gate_type) {
        case GGML_TYPE_IQ2_XXS:return launch_down<GGML_TYPE_IQ2_XXS>(p,s,stream);
        case GGML_TYPE_IQ2_XS:return launch_down<GGML_TYPE_IQ2_XS>(p,s,stream);
        case GGML_TYPE_IQ2_S:return launch_down<GGML_TYPE_IQ2_S>(p,s,stream);
        case GGML_TYPE_IQ3_XXS:return launch_down<GGML_TYPE_IQ3_XXS>(p,s,stream);
        case GGML_TYPE_IQ3_S:return launch_down<GGML_TYPE_IQ3_S>(p,s,stream);
        default:return cudaErrorNotSupported;
    }
}
