#include "llama-moe-cache.h"

#include "llama-impl.h"

#include <algorithm>
#include <cinttypes>
#include <cstring>

llama_moe_cache::llama_moe_cache(const llama_moe_cache_params & params, ggml_backend_buffer_type_t buft)
    : params(params), buft(buft) {
}

llama_moe_cache::~llama_moe_cache() {
    if (buf) {
        ggml_backend_buffer_free(buf);
    }
    if (ctx) {
        ggml_free(ctx);
    }
}

llama_moe_cache_tensor * llama_moe_cache::add_tensor(ggml_tensor * src, int64_t n_expert, int64_t n_expert_used) {
    if (!enabled() || src == nullptr) {
        return nullptr;
    }

    // ne[2] indexes experts; anything else is not an expert tensor we know how to slot.
    if (src->ne[2] != n_expert) {
        LLAMA_LOG_DEBUG("%s: skipping %s, ne[2]=%" PRId64 " != n_expert=%" PRId64 "\n",
                __func__, ggml_get_name(src), src->ne[2], n_expert);
        return nullptr;
    }

    // A tensor already living on the accelerator needs no cache; caching it would just spend
    // VRAM to duplicate what is already there.
    if (src->buffer && ggml_backend_buffer_get_type(src->buffer) == buft) {
        return nullptr;
    }

    if (auto * existing = find(src)) {
        return existing;
    }

    const int32_t n_slots = std::min<int64_t>(params.n_slots, n_expert);
    if (n_slots <= 0) {
        return nullptr;
    }

    llama_moe_cache_tensor ct;
    ct.src      = src;
    ct.n_expert = n_expert;
    ct.n_slots  = n_slots;
    // one private dummy slot per routed expert, so ids stay distinct within a token
    ct.n_dummy  = (int32_t) std::max<int64_t>(n_expert_used, 1);

    ct.map_host.assign(n_expert, -1.0f);
    ct.map_src_host.resize(n_expert);
    for (int64_t e = 0; e < n_expert; ++e) {
        ct.map_src_host[e] = (int32_t) e;   // nothing resident yet: the host chain owns every route
    }
    ct.slot_expert.assign(n_slots, -1);
    ct.slot_used.assign(n_slots, 0);
    ct.touch_count.assign(n_expert, 0);
    ct.freq.assign(n_expert, 0);

    // Everything starts uncached (-1); the graph turns that into this route's position-indexed
    // dummy slot.
    ct.map_dirty = true;

    const size_t row_bytes = ggml_nbytes(src) / n_expert;
    total_bytes += row_bytes * (size_t) (n_slots + ct.n_dummy);

    by_src[src] = tensors.size();
    tensors.push_back(std::move(ct));
    pending.emplace_back();

    return &tensors.back();
}

llama_moe_cache_tensor * llama_moe_cache::find(const ggml_tensor * src) const {
    auto it = by_src.find(src);
    if (it == by_src.end()) {
        return nullptr;
    }
    return const_cast<llama_moe_cache_tensor *>(&tensors[it->second]);
}

bool llama_moe_cache::alloc() {
    if (tensors.empty()) {
        return true;
    }

    // two tensors per cached expert tensor (slots + map), plus ggml's per-tensor overhead
    ggml_init_params ip = {
        /*.mem_size   =*/ ggml_tensor_overhead() * (tensors.size() * 3 + 1) + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    ctx = ggml_init(ip);
    if (!ctx) {
        LLAMA_LOG_ERROR("%s: failed to create ggml context\n", __func__);
        return false;
    }

    for (auto & ct : tensors) {
        const int64_t n_total = ct.n_slots + ct.n_dummy;

        ct.slots = ggml_new_tensor_3d(ctx, ct.src->type, ct.src->ne[0], ct.src->ne[1], n_total);
        ggml_format_name(ct.slots, "%s#slots", ggml_get_name(ct.src));

        // [1, n_expert] so ggml_get_rows can gather it with a flattened id vector: get_rows
        // requires the table's ne[2] to match the index tensor's ne[1], which a 1-D table
        // would not satisfy for multi-token batches.
        ct.map = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, ct.n_expert);
        ggml_format_name(ct.map, "%s#map", ggml_get_name(ct.src));

        ct.map_src = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, ct.n_expert);
        ggml_format_name(ct.map_src, "%s#map_src", ggml_get_name(ct.src));
    }

    {
        // shared across tensors: n_slots + position, added to routes that have no slot
        const int64_t n_used = tensors.empty() ? 1 : tensors.front().n_dummy;
        dummy_pos = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, n_used, 1);
        ggml_set_name(dummy_pos, "moe_cache#dummy_pos");
    }

    buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buf) {
        LLAMA_LOG_ERROR("%s: failed to allocate %.2f MiB for the MoE expert cache\n",
                __func__, total_bytes / 1024.0 / 1024.0);
        ggml_free(ctx);
        ctx = nullptr;
        return false;
    }

    // Dummy slots must read as zero: an uncached route lands there and has to contribute
    // nothing to the sum. Zero the whole buffer so the real slots start defined too.
    ggml_backend_buffer_clear(buf, 0);

    {
        std::vector<float> pos(dummy_pos->ne[1]);
        for (size_t p = 0; p < pos.size(); ++p) {
            pos[p] = (float) (params.n_slots + (int32_t) p);
        }
        ggml_backend_tensor_set(dummy_pos, pos.data(), 0, ggml_nbytes(dummy_pos));
    }

    for (auto & ct : tensors) {
        ggml_backend_tensor_set(ct.map,     ct.map_host.data(),     0, ggml_nbytes(ct.map));
        ggml_backend_tensor_set(ct.map_src, ct.map_src_host.data(), 0, ggml_nbytes(ct.map_src));
        ct.map_dirty = false;
    }

    LLAMA_LOG_INFO("%s: MoE expert cache: %zu tensors, %d slots each, %.2f MiB VRAM\n",
            __func__, tensors.size(), params.n_slots, total_bytes / 1024.0 / 1024.0);

    return true;
}

void llama_moe_cache::observe(llama_moe_cache_tensor * ct, const int32_t * ids, size_t n_ids) {
    if (!ct || !ids) {
        return;
    }

    const size_t idx = by_src.at(ct->src);

    for (size_t i = 0; i < n_ids; ++i) {
        const int32_t e = ids[i];
        if (e < 0 || e >= (int32_t) ct->n_expert) {
            continue;
        }

        st.n_routes++;
        ct->freq[e]++;

        const float slot = ct->map_host[e];
        if (slot >= 0.0f) {
            st.n_hits++;
            // Refresh recency so a hot expert is not evicted by a passing one.
            ct->slot_used[(size_t) slot] = step;
            continue;
        }

        if (++ct->touch_count[e] >= (uint32_t) params.admit_after) {
            auto & q = pending[idx];
            if (std::find(q.begin(), q.end(), e) == q.end()) {
                q.push_back(e);
            }
        }
    }
}

bool llama_moe_cache::admit(llama_moe_cache_tensor & ct, int32_t expert) {
    if (ct.map_host[expert] >= 0.0f) {
        return true; // already resident
    }

    // Prefer a free slot, else the LEAST FREQUENTLY routed resident expert -- and only take
    // it if the candidate is genuinely hotter. Pure recency thrashes here: a decode step
    // touches ~n_expert_used of the slots, so almost every slot looks equally stale and the
    // hot set gets evicted by whatever happened to be routed once.
    int32_t  victim      = -1;
    uint32_t victim_freq = UINT32_MAX;

    for (int32_t s = 0; s < ct.n_slots; ++s) {
        if (ct.slot_expert[s] < 0) {
            victim = s;
            victim_freq = 0;
            break;
        }
        if (ct.slot_used[s] == step) {
            continue; // in use this step
        }
        const uint32_t f = ct.freq[ct.slot_expert[s]];
        if (f < victim_freq) {
            victim_freq = f;
            victim = s;
        }
    }

    if (victim < 0) {
        return false;
    }

    if (ct.slot_expert[victim] >= 0 && ct.freq[expert] <= victim_freq) {
        return false; // the resident expert is at least as hot; leave it alone
    }

    const size_t row_bytes = ggml_nbytes(ct.src) / ct.n_expert;

    // Copy the expert's rows from host storage into the slot. ggml_backend_tensor_set on the
    // slot buffer is a plain H2D of a contiguous range, since experts are the outermost axis.
    std::vector<uint8_t> tmp(row_bytes);
    ggml_backend_tensor_get(ct.src, tmp.data(), (size_t) expert * row_bytes, row_bytes);
    ggml_backend_tensor_set(ct.slots, tmp.data(), (size_t) victim * row_bytes, row_bytes);

    if (ct.slot_expert[victim] >= 0) {
        // the evicted expert falls back to a dummy slot
        const int32_t prev = ct.slot_expert[victim];
        ct.map_host[prev] = -1.0f;
        ct.map_src_host[prev] = prev;   // the host chain owns it again
        ct.touch_count[prev] = 0;
        st.n_evicts++;
    }

    ct.slot_expert[victim]  = expert;
    ct.slot_used[victim]    = step;
    ct.map_host[expert]     = (float) victim;
    ct.map_src_host[expert] = (int32_t) ct.n_expert;  // resident: host chain skips it
    ct.touch_count[expert] = 0;
    ct.map_dirty = true;

    st.n_admits++;

    return true;
}

void llama_moe_cache::harvest() {
    if (!enabled() || tensors.empty()) {
        return;
    }

    if (++harvests % 64 == 0 && st.n_routes > w_routes) {
        // windowed, not cumulative: a cumulative rate is dominated forever by the cold start
        const uint64_t dr = st.n_routes - w_routes;
        const uint64_t dh = st.n_hits   - w_hits;
        const uint64_t da = st.n_admits - w_admits;
        const uint64_t de = st.n_evicts - w_evicts;
        LLAMA_LOG_INFO("%s: moe cache: hit %.1f%% (window) / %.1f%% (all) | +%llu admits +%llu evicts\n",
                __func__, 100.0 * (double) dh / (double) dr,
                100.0 * (double) st.n_hits / (double) st.n_routes,
                (unsigned long long) da, (unsigned long long) de);
        w_routes = st.n_routes; w_hits = st.n_hits; w_admits = st.n_admits; w_evicts = st.n_evicts;
    }

    // Several weights of one layer share a routing tensor, so read each distinct one once.
    std::unordered_map<const ggml_tensor *, std::vector<int32_t>> seen;

    for (auto & ct : tensors) {
        ggml_tensor * ids = ct.ids_node;
        ct.ids_node = nullptr;
        if (ids == nullptr || ids->buffer == nullptr) {
            continue;
        }

        if (ids->type != GGML_TYPE_I32 || !ggml_is_contiguous(ids) ||
            ggml_nbytes(ids) != ggml_nelements(ids) * sizeof(int32_t)) {
            continue;
        }

        auto it = seen.find(ids);
        if (it == seen.end()) {
            std::vector<int32_t> buf(ggml_nelements(ids));
            ggml_backend_tensor_get(ids, buf.data(), 0, ggml_nbytes(ids));
            it = seen.emplace(ids, std::move(buf)).first;
        }

        observe(&ct, it->second.data(), it->second.size());
    }

    update();
}

void llama_moe_cache::update() {
    if (!enabled() || tensors.empty()) {
        return;
    }

    step++;

    for (auto & ct : tensors) {
        uint32_t peak = 0;
        for (uint32_t f : ct.freq) {
            peak = std::max(peak, f);
        }
        if (peak >= params.decay_at) {
            for (uint32_t & f : ct.freq) {
                f /= 2;
            }
        }
    }

    for (size_t i = 0; i < tensors.size(); ++i) {
        auto & ct = tensors[i];
        auto & q  = pending[i];

        // Admit up to the per-step budget and KEEP the rest queued: dropping them would throw
        // away everything observed beyond the budget, which is most of a prefill's routing.
        int32_t n = 0;
        while (n < params.n_inserts && !q.empty()) {
            if (!admit(ct, q.front())) {
                break;
            }
            q.erase(q.begin());
            n++;
        }
        if (q.size() > 4096) {
            q.resize(4096);
        }

        if (ct.map_dirty && ct.map) {
            ggml_backend_tensor_set(ct.map,     ct.map_host.data(),     0, ggml_nbytes(ct.map));
            ggml_backend_tensor_set(ct.map_src, ct.map_src_host.data(), 0, ggml_nbytes(ct.map_src));
            ct.map_dirty = false;
        }
    }
}
