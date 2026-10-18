#include "monomoe-ggml.cuh"
#include "monomoe-sm70.h"

static monomoe_sm70_params parameters(const ggml_tensor * t) {
    const auto * x=t->src[0];const auto * r=t->src[1];const auto * g=t->src[2];
    const auto * u=t->src[3];const auto * d=t->src[4];
    monomoe_sm70_params p{};
    p.hidden=x->ne[0];p.tokens=x->ne[1];p.ffn=d->ne[0];p.experts=r->ne[1];
    p.input=(const float*)x->data;p.router=r->data;p.gate=g->data;
    p.up=u?u->data:(g->data?(const char*)g->data+p.ffn*g->nb[1]:nullptr);
    p.down=d->data;p.output=(float*)t->data;
    p.router_type=r->type;p.gate_type=g->type;p.down_type=d->type;
    p.router_row=r->nb[1];p.gate_row=g->nb[1];p.gate_expert=g->nb[2];
    p.up_row=u?u->nb[1]:g->nb[1];p.up_expert=u?u->nb[2]:g->nb[2];
    p.down_row=d->nb[1];p.down_expert=d->nb[2];
    p.selected=ggml_get_op_params_i32(t,0);p.routing_scale=ggml_get_op_params_f32(t,1);
    p.normalize_selected=true;
    return p;
}
bool ggml_cuda_monomoe_supported(int device,const ggml_tensor * t) {
    if(ggml_cuda_info().devices[device].cc!=GGML_CUDA_CC_VOLTA)return false;
    for(int i:{0,1,2,4})if(!t->src[i]||!ggml_is_contiguous(t->src[i]))return false;
    if(t->src[3]&&!ggml_is_contiguous(t->src[3]))return false;
    const auto p=parameters(t);
    const bool g=p.gate_type==GGML_TYPE_IQ2_XXS||p.gate_type==GGML_TYPE_IQ2_XS||p.gate_type==GGML_TYPE_IQ2_S||p.gate_type==GGML_TYPE_IQ3_XXS||p.gate_type==GGML_TYPE_IQ3_S;
    const bool d=p.down_type==GGML_TYPE_Q2_0||p.down_type==GGML_TYPE_IQ4_NL;
    return g&&d&&t->src[0]->type==GGML_TYPE_F32&&t->type==GGML_TYPE_F32&&
        (p.router_type==GGML_TYPE_BF16||p.router_type==GGML_TYPE_F32)&&
        p.hidden>0&&p.hidden%256==0&&p.ffn>0&&p.ffn%32==0&&(p.down_type!=GGML_TYPE_Q2_0||p.ffn%64==0)&&p.tokens>0&&p.tokens<=32&&p.experts<=512&&p.selected>0&&p.selected<=10;
}
size_t ggml_cuda_monomoe_alloc_size(const ggml_tensor * t) {
    const size_t output=(ggml_nbytes(t)+127)&~size_t(127);
    const auto p=parameters(t);
    // Meta can allocate an empty FFN shard; its computation is disabled there.
    return p.ffn==0?output:output+monomoe_sm70_workspace_size(p);
}
void ggml_cuda_op_monomoe(ggml_backend_cuda_context & ctx,ggml_tensor * t) {
    const auto p=parameters(t);
    GGML_ASSERT(ggml_cuda_monomoe_supported(ctx.device,t));
    const size_t offset=(ggml_nbytes(t)+127)&~size_t(127);
    static bool logged[GGML_TYPE_COUNT][GGML_TYPE_COUNT]={};
    if(!logged[p.gate_type][p.down_type]) {
        GGML_LOG_INFO("%s: using SM70 MonoMoE gate=%s down=%s T=%d hidden=%d ffn=%d experts=%d selected=%d\n",
            __func__,ggml_type_name(p.gate_type),ggml_type_name(p.down_type),p.tokens,p.hidden,p.ffn,p.experts,p.selected);
        logged[p.gate_type][p.down_type]=true;
    }
    CUDA_CHECK(monomoe_sm70_launch(p,(char*)t->data+offset,monomoe_sm70_workspace_size(p),ctx.stream()));
}
