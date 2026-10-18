#include "strided-sum.cuh"
#include "convert.cuh"

#define SS_MAX_TERMS 8

struct strided_sum_args {
    const char    * src;     // q8_0: the GET_ROWS source rows; relu: the RELU input
    const int32_t * ids;     // q8_0: GET_ROWS row ids
    int64_t         src_nb1; // q8_0: bytes per source row
    int64_t         g_ne0;   // q8_0: elements per gathered row
    int             n_terms;
    int64_t         off[SS_MAX_TERMS];    // window start, elements into the GET_ROWS / RELU output
    int64_t         stride[SS_MAX_TERMS]; // window row stride, elements
    int64_t         ne0;
    int64_t         n;
    int             has_scale;
    float           scale;
    float           scale_bias;
    const float   * bias;
    float         * dst;
};

// value k of output element (e, j), rounded exactly as GET_ROWS (dequantize_q8_0) or RELU would store it
template <bool q8>
static __device__ __forceinline__ float strided_sum_val(const strided_sum_args & a, const int k, const int64_t e, const int64_t j) {
    const int64_t f = a.off[k] + j*a.stride[k] + e;
    if constexpr (q8) {
        const int64_t r = f / a.g_ne0;
        const int64_t c = f - r*a.g_ne0;
        const block_q8_0 * b = (const block_q8_0 *) (a.src + (int64_t) a.ids[r]*a.src_nb1) + c/QK8_0;
        return __fmul_rn((float) b->qs[c % QK8_0], (float) b->d);
    } else {
        return fmaxf(((const float *) a.src)[f], 0.0f);
    }
}

template <bool q8>
static __global__ void strided_sum_kernel(const strided_sum_args a) {
    const int64_t o = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (o >= a.n) {
        return;
    }
    const int64_t j = o / a.ne0;
    const int64_t e = o - j*a.ne0;

    // same association as the ADD chain: ((t0 + t1) + t2) + ...; no contraction with the dequantize multiply
    float s = __fadd_rn(strided_sum_val<q8>(a, 0, e, j), strided_sum_val<q8>(a, 1, e, j));
    for (int k = 2; k < a.n_terms; ++k) {
        s = __fadd_rn(s, strided_sum_val<q8>(a, k, e, j));
    }
    if (a.has_scale) {
        s = __fmaf_rn(a.scale, s, a.scale_bias); // scale_f32: scale * x + bias, contracted
    }
    if (a.bias) {
        s = __fadd_rn(s, a.bias[o]);
    }
    a.dst[o] = s;
}

static bool ss_is_view_or_noop(const ggml_tensor * t) {
    return t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE || t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE ||
           t->op == GGML_OP_NONE;
}

int ggml_cuda_strided_sum_try(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i) {
    ggml_tensor * S = cgraph->nodes[i];

    bool q8;
    if (S->op == GGML_OP_GET_ROWS) {
        const ggml_tensor * s0 = S->src[0];
        const ggml_tensor * s1 = S->src[1];
        if (s0->type != GGML_TYPE_Q8_0 || s1->type != GGML_TYPE_I32 || S->type != GGML_TYPE_F32 || !ggml_is_contiguous(S) ||
                s0->ne[2] != 1 || s0->ne[3] != 1 || s1->ne[1] != 1 || s1->ne[2] != 1 || s1->ne[3] != 1 ||
                s0->nb[0] != sizeof(block_q8_0) || s0->ne[0] % QK8_0 != 0 || s1->nb[0] != sizeof(int32_t) ||
                S->ne[0] != s0->ne[0] || S->ne[1] != s1->ne[0] || S->ne[2] != 1 || S->ne[3] != 1) {
            return 0;
        }
        q8 = true;
    } else if (S->op == GGML_OP_UNARY && ggml_get_unary_op(S) == GGML_UNARY_OP_RELU) {
        if (S->type != GGML_TYPE_F32 || S->src[0]->type != GGML_TYPE_F32 || !ggml_is_contiguous(S) || !ggml_is_contiguous(S->src[0])) {
            return 0;
        }
        q8 = false;
    } else {
        return 0;
    }

    ggml_tensor * conts[SS_MAX_TERMS];
    int  cont_idx[SS_MAX_TERMS];
    bool cont_used[SS_MAX_TERMS] = {};
    int  n_conts = 0;

    const ggml_tensor * terms[SS_MAX_TERMS];
    int n_terms = 0;

    ggml_tensor * cur   = nullptr;
    ggml_tensor * scale = nullptr;
    ggml_tensor * bias  = nullptr;
    int end = -1;

    // the 2d window a term reads, or nullptr if it is not a window of S
    const auto term_of = [&](ggml_tensor * t, const ggml_tensor * like) -> const ggml_tensor * {
        if (t->op == GGML_OP_CONT) {
            int c = 0;
            for (; c < n_conts && conts[c] != t; ++c) {}
            if (c == n_conts || cont_used[c]) {
                return nullptr;
            }
            cont_used[c] = true;
            t = t->src[0];
        }
        if (t->view_src != S || t->type != GGML_TYPE_F32 || t->nb[0] != sizeof(float) || t->nb[1] % sizeof(float) != 0 ||
                t->view_offs % sizeof(float) != 0 || t->ne[2] != 1 || t->ne[3] != 1 ||
                t->ne[0] != like->ne[0] || t->ne[1] != like->ne[1]) {
            return nullptr;
        }
        return t;
    };

    for (int j = i + 1; j < cgraph->n_nodes && j <= i + 48; ++j) {
        ggml_tensor * nj = cgraph->nodes[j];
        if (ss_is_view_or_noop(nj)) {
            continue;
        }
        if ((nj->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            return 0;
        }
        if (nj->op == GGML_OP_CONT && nj->src[0]->view_src == S && nj->type == GGML_TYPE_F32 && n_conts < SS_MAX_TERMS) {
            cont_idx[n_conts] = j;
            conts[n_conts++]  = nj;
            continue;
        }
        if (nj->op == GGML_OP_ADD && nj->type == GGML_TYPE_F32 && ggml_is_contiguous(nj) && nj->ne[2] == 1 && nj->ne[3] == 1) {
            ggml_tensor * a = nj->src[0];
            ggml_tensor * b = nj->src[1];
            if (cur == nullptr) {
                const ggml_tensor * ta = term_of(a, nj);
                const ggml_tensor * tb = ta ? term_of(b, nj) : nullptr;
                if (!ta || !tb) {
                    return 0;
                }
                terms[n_terms++] = ta;
                terms[n_terms++] = tb;
                cur = nj;
                end = j;
                continue;
            }
            ggml_tensor * other = a == cur ? b : b == cur ? a : nullptr;
            if (other == nullptr || other == cur) {
                break;
            }
            if (n_terms < SS_MAX_TERMS && (other->view_src == S || other->op == GGML_OP_CONT)) {
                const ggml_tensor * t = term_of(other, nj);
                if (!t) {
                    return 0;
                }
                terms[n_terms++] = t;
                cur = nj;
                end = j;
                continue;
            }
            if (other != S && other->view_src != S && other->type == GGML_TYPE_F32 && ggml_is_contiguous(other) &&
                    ggml_are_same_shape(other, nj)) {
                bool mine = false;
                for (int c = 0; c < n_conts; ++c) {
                    mine |= other == conts[c] || other->view_src == conts[c];
                }
                if (!mine) {
                    bias = other;
                    cur  = nj;
                    end  = j;
                }
            }
            break;
        }
        if (nj->op == GGML_OP_SCALE && cur != nullptr && nj->src[0] == cur && nj->type == GGML_TYPE_F32 && ggml_is_contiguous(nj)) {
            scale = nj;
            cur   = nj;
            end   = j;
        }
        break;
    }
    if (n_terms < 2 || end < 0) {
        return 0;
    }
    for (int c = 0; c < n_conts; ++c) {
        if (cont_idx[c] <= end && !cont_used[c]) {
            return 0; // a copy inside the group that no add reads
        }
    }

    // everything the group computes except its result, and the views of it, must have no reader outside the group
    const auto internal = [&](const ggml_tensor * t) {
        const ggml_tensor * root = t->view_src ? t->view_src : t;
        if (t == cur || root == cur) {
            return false;
        }
        if (root == S) {
            return true;
        }
        for (int c = 0; c < n_conts; ++c) {
            if (root == conts[c]) {
                return true;
            }
        }
        for (int j = i + 1; j <= end; ++j) {
            if (cgraph->nodes[j] == root && root->op == GGML_OP_ADD) {
                return true;
            }
        }
        return false;
    };
    for (int x = i; x <= end; ++x) {
        const ggml_tensor * X = cgraph->nodes[x];
        if (!internal(X)) {
            continue;
        }
        if (X->flags & GGML_TENSOR_FLAG_OUTPUT) {
            return 0;
        }
        int refs = 0;
        for (int j = i + 1; j <= end; ++j) {
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                refs += cgraph->nodes[j]->src[s] == X;
            }
        }
        if (ggml_node_get_use_count(cgraph, x) != refs) {
            return 0;
        }
    }

    ggml_tensor * dst = cur;
    if (dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) || dst->ne[0] != terms[0]->ne[0] || dst->ne[1] != terms[0]->ne[1] ||
            dst->ne[2] != 1 || dst->ne[3] != 1) {
        return 0;
    }

    // the kernel writes dst while it still reads strided windows of the source (and the bias); the unfused graph may be
    // done with the source by then, so the allocator may have put dst on top of it: read a copy then
    ggml_cuda_pool_alloc<char> t_src(ctx.pool()), t_ids(ctx.pool()), t_bias(ctx.pool());

    strided_sum_args a = {};
    a.src = (const char *) ggml_cuda_stage_if_overlap(ctx, dst, S->src[0], t_src);
    if (q8) {
        a.ids     = (const int32_t *) ggml_cuda_stage_if_overlap(ctx, dst, S->src[1], t_ids);
        a.src_nb1 = S->src[0]->nb[1];
        a.g_ne0   = S->ne[0];
    }
    a.n_terms = n_terms;
    for (int k = 0; k < n_terms; ++k) {
        a.off[k]    = terms[k]->view_offs / sizeof(float);
        a.stride[k] = terms[k]->nb[1] / sizeof(float);
    }
    a.ne0 = dst->ne[0];
    a.n   = ggml_nelements(dst);
    if (scale) {
        a.has_scale = 1;
        memcpy(&a.scale,      (const float *) scale->op_params + 0, sizeof(float));
        memcpy(&a.scale_bias, (const float *) scale->op_params + 1, sizeof(float));
    }
    a.bias = (const float *) ggml_cuda_stage_if_overlap(ctx, dst, bias, t_bias);
    a.dst  = (float *) dst->data;

    const int block = 256;
    const int grid  = (int) ((a.n + block - 1) / block);
    if (q8) {
        strided_sum_kernel<true><<<grid, block, 0, ctx.stream()>>>(a);
    } else {
        strided_sum_kernel<false><<<grid, block, 0, ctx.stream()>>>(a);
    }
    CUDA_CHECK(cudaGetLastError());
    return end - i;
}

// true if no node outside [i, end] reads any of roots (nodes i..end) or a view of one, and none is a graph output
bool ggml_cuda_group_private(const ggml_cgraph * cgraph, int i, int end, const ggml_tensor * const * roots, int n_roots) {
    for (int x = i; x <= end; ++x) {
        const ggml_tensor * X = cgraph->nodes[x];
        const ggml_tensor * root = X->view_src ? X->view_src : X;
        bool in = false;
        for (int r = 0; r < n_roots; ++r) {
            in |= root == roots[r];
        }
        if (!in) {
            continue;
        }
        if (X->flags & GGML_TENSOR_FLAG_OUTPUT) {
            return false;
        }
        int refs = 0;
        for (int j = i + 1; j <= end; ++j) {
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                refs += cgraph->nodes[j]->src[s] == X;
            }
        }
        if (ggml_node_get_use_count(cgraph, x) != refs) {
            return false;
        }
    }
    return true;
}

struct expand_rows_args {
    const float   * score;   // [n_blk, n_tps, n_stream], contiguous
    const int32_t * ids;     // [n_kv, n_stream]
    int64_t         ids_nb1; // elements
    const char    * mask;    // any strides, f16 or f32, same element count as dst
    int64_t         m_ne[4];
    int64_t         m_nb[4];
    int64_t         n_blk, n_kv, n_tps, n;
    float         * dst;     // [n_kv, n_tps, n_stream]
};

template <typename mask_t>
static __global__ void expand_rows_kernel(const expand_rows_args a) {
    const int64_t o = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (o >= a.n) {
        return;
    }
    const int64_t c = o % a.n_kv;
    const int64_t t = (o / a.n_kv) % a.n_tps;
    const int64_t s = o / (a.n_kv*a.n_tps);
    const float v = a.score[(s*a.n_tps + t)*a.n_blk + a.ids[s*a.ids_nb1 + c]];

    int64_t r = o;
    const int64_t m0 = r % a.m_ne[0]; r /= a.m_ne[0];
    const int64_t m1 = r % a.m_ne[1]; r /= a.m_ne[1];
    const int64_t m2 = r % a.m_ne[2]; r /= a.m_ne[2];
    const mask_t m = *(const mask_t *) (a.mask + m0*a.m_nb[0] + m1*a.m_nb[1] + m2*a.m_nb[2] + r*a.m_nb[3]);

    a.dst[o] = __fadd_rn(v, ggml_cuda_cast<float, mask_t>(m));
}

static bool ss_is_swap01(const ggml_tensor * p, const ggml_tensor * x) {
    return p->op == GGML_OP_PERMUTE && p->view_src == x && p->view_offs == 0 &&
           p->ne[0] == x->ne[1] && p->ne[1] == x->ne[0] && p->ne[2] == x->ne[2] && p->ne[3] == x->ne[3] &&
           p->nb[0] == x->nb[1] && p->nb[1] == x->nb[0] && p->nb[2] == x->nb[2];
}

int ggml_cuda_expand_rows_try(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i) {
    ggml_tensor * c1 = cgraph->nodes[i];
    if (c1->op != GGML_OP_CONT || c1->type != GGML_TYPE_F32 || c1->src[0]->view_src == nullptr) {
        return 0;
    }
    ggml_tensor * X = c1->src[0]->view_src; // the block scores
    if (X->type != GGML_TYPE_F32 || !ggml_is_contiguous(X) || X->ne[3] != 1 || !ss_is_swap01(c1->src[0], X)) {
        return 0;
    }

    ggml_tensor * g    = nullptr;
    ggml_tensor * c2   = nullptr;
    ggml_tensor * cast = nullptr;
    ggml_tensor * add  = nullptr;
    const ggml_tensor * mask = nullptr;
    int end = -1;
    for (int j = i + 1; j < cgraph->n_nodes && j <= i + 16 && add == nullptr; ++j) {
        ggml_tensor * nj = cgraph->nodes[j];
        if (ss_is_view_or_noop(nj)) {
            continue;
        }
        if ((nj->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            return 0;
        }
        if (!g) {
            if (nj->op != GGML_OP_GET_ROWS || nj->src[0] != c1 || nj->src[1]->type != GGML_TYPE_I32 || nj->type != GGML_TYPE_F32 ||
                    nj->src[1]->ne[1] != c1->ne[2] || nj->src[1]->ne[2] != 1 || nj->src[1]->nb[0] != sizeof(int32_t) ||
                    !ggml_is_contiguous(nj)) {
                return 0;
            }
            g = nj;
        } else if (!c2) {
            if (nj->op != GGML_OP_CONT || nj->type != GGML_TYPE_F32 || !ss_is_swap01(nj->src[0], g)) {
                return 0;
            }
            c2 = nj;
        } else if (nj->op == GGML_OP_CPY && !cast && nj->src[0]->type == GGML_TYPE_F16 && nj->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(nj) && nj->view_src == nullptr) {
            cast = nj;
        } else if (nj->op == GGML_OP_ADD && nj->type == GGML_TYPE_F32 && ggml_is_contiguous(nj) && ggml_are_same_shape(nj, c2)) {
            ggml_tensor * other = nj->src[0] == c2 ? nj->src[1] : nj->src[1] == c2 ? nj->src[0] : nullptr;
            if (!other) {
                return 0;
            }
            const ggml_tensor * root = other->view_src ? other->view_src : other;
            if (cast && root == cast) {
                if (other != cast && other->op != GGML_OP_RESHAPE) {
                    return 0;
                }
                mask = cast->src[0];
            } else if (root != X && root != c1 && root != g && root != c2 && other->type == GGML_TYPE_F32 &&
                    ggml_is_contiguous(other)) {
                mask = other;
            } else {
                return 0;
            }
            add = nj;
            end = j;
        } else {
            return 0;
        }
    }
    if (!add || (cast && mask != cast->src[0]) || ggml_nelements(mask) != ggml_nelements(add)) {
        return 0;
    }
    const ggml_tensor * roots[4] = { c1, g, c2, cast };
    if (!ggml_cuda_group_private(cgraph, i, end, roots, cast ? 4 : 3)) {
        return 0;
    }

    // the kernel writes add while it still reads the block scores, the row ids and the mask at scattered offsets. The
    // unfused graph is done with the scores after c1 (and may be with the others), so the allocator may have put add on
    // top of them: read copies then
    ggml_cuda_pool_alloc<char> t_score(ctx.pool()), t_ids(ctx.pool()), t_mask(ctx.pool());

    expand_rows_args a = {};
    a.score   = (const float *) ggml_cuda_stage_if_overlap(ctx, add, X, t_score);
    a.ids     = (const int32_t *) ggml_cuda_stage_if_overlap(ctx, add, g->src[1], t_ids);
    a.ids_nb1 = g->src[1]->nb[1] / sizeof(int32_t);
    a.mask    = (const char *) ggml_cuda_stage_if_overlap(ctx, add, mask, t_mask);
    for (int d = 0; d < 4; ++d) {
        a.m_ne[d] = mask->ne[d];
        a.m_nb[d] = mask->nb[d];
    }
    a.n_blk = X->ne[0];
    a.n_kv  = add->ne[0];
    a.n_tps = add->ne[1];
    a.n     = ggml_nelements(add);
    a.dst   = (float *) add->data;
    if (a.n_tps != X->ne[1] || add->ne[2] != X->ne[2] || g->src[1]->ne[0] != a.n_kv || g->src[1]->nb[1] % sizeof(int32_t) != 0) {
        return 0;
    }

    const int block = 256;
    const int grid  = (int) ((a.n + block - 1) / block);
    if (mask->type == GGML_TYPE_F16) {
        expand_rows_kernel<half><<<grid, block, 0, ctx.stream()>>>(a);
    } else if (mask->type == GGML_TYPE_F32) {
        expand_rows_kernel<float><<<grid, block, 0, ctx.stream()>>>(a);
    } else {
        return 0;
    }
    CUDA_CHECK(cudaGetLastError());
    return end - i;
}
