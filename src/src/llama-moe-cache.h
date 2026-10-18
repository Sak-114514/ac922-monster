#pragma once

// GPU-resident cache for host-offloaded MoE expert weights.
//
// A large MoE model does not fit in VRAM, so its expert tensors live in host RAM and every
// decode step recomputes the routed experts on the CPU (or restreams them over PCIe). Routing
// is heavily skewed, though: a modest slice of the experts serves most tokens. This cache keeps
// `n_slots` experts per tensor resident in VRAM and splits each expert matmul in two:
//
//   hits   -> ggml_mul_mat_id over the VRAM slot buffer, remapped device-side
//   misses -> the original host tensor, computed where it already lives
//
// and sums the two. Every route is covered exactly once, so the result is arithmetically the
// same as the uncached graph.
//
// Two properties are load-bearing and easy to lose:
//
//  * The expert -> slot map is a *device* tensor, gathered with ggml_get_rows inside the graph.
//    Nothing about routing crosses to the host mid-step, so CUDA graphs stay capturable.
//  * The map is only mutated *between* steps, from routing observed on the host chain. The
//    tensor keeps its address, so a captured graph replays against updated contents safely.
//
// Uncached routes are steered to a dummy slot chosen by their POSITION in the token's route
// list, not by expert id. The batched mul_mat_id kernels (CPU and CUDA alike) assume the ids
// within a token are distinct: CUDA's mm_ids_helper keeps only the last position matching an
// expert and miscounts `nex_prev`, which corrupts the expert bounds. Position-indexed dummies
// are distinct by construction, and a dummy index can never collide with a real slot because
// the dummies live above them. This is what lets batches wider than one token (speculative
// verify) use the cache at all.

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

struct llama_moe_cache_params {
    // resident experts per expert tensor; 0 disables the cache entirely
    int32_t n_slots = 0;

    // upper bound on experts promoted into VRAM per tensor per step. Uploads share PCIe with
    // the demand traffic of the same step, so admitting the whole miss set at once can cost
    // more than it saves; a small trickle converges within a few hundred steps.
    int32_t n_inserts = 8;

    // an expert must be MISSED this many times before it is queued for a slot
    int32_t admit_after = 1;

    // halve every expert's route count once any of them reaches this, so the residency can
    // follow a change of topic instead of being pinned by whatever was hot first
    uint32_t decay_at = 4096;
};

// One cached expert tensor (e.g. blk.7.ffn_gate_up_exps.weight).
struct llama_moe_cache_tensor {
    ggml_tensor * src = nullptr;   // host-resident weights, unchanged
    ggml_tensor * slots = nullptr; // [ne0, ne1, n_slots + n_dummy] in VRAM
    ggml_tensor * map = nullptr;     // [1, n_expert] f32, device: expert -> slot, or -1 when absent
    ggml_tensor * map_src = nullptr; // [1, n_expert] i32, device: expert -> id for the HOST chain,
                                     // which is the expert itself while it is not resident and a
                                     // sentinel (== n_expert) once it is, so the host matmul skips
                                     // routes the VRAM chain already serves

    int64_t n_expert = 0;
    int32_t n_slots  = 0; // real slots; dummy slots follow at [n_slots, n_slots + n_dummy)
    int32_t n_dummy  = 0;

    // host mirror of `map`, plus the reverse mapping and the LRU/admission bookkeeping
    std::vector<float>    map_host;     // n_expert; slot index as float, or -1 when absent
    std::vector<int32_t>  map_src_host; // n_expert; expert id, or the skip sentinel when resident
    std::vector<int32_t>  slot_expert;  // n_slots; which expert occupies the slot, -1 if free
    std::vector<uint64_t> slot_used;    // n_slots; last step the slot was routed to
    std::vector<uint32_t> touch_count;  // n_expert; misses observed since last queued
    std::vector<uint32_t> freq;         // n_expert; routes observed (hit or miss), decayed

    bool map_dirty = false; // host map changed and must be re-uploaded before the next step

    // the routing tensor that fed this weight in the graph just built; valid until the next
    // graph build, which is why it is harvested immediately after the graph runs
    ggml_tensor * ids_node = nullptr;
};

// Owns the VRAM slot buffers and the residency policy for one model.
class llama_moe_cache {
public:
    llama_moe_cache(const llama_moe_cache_params & params, ggml_backend_buffer_type_t buft);
    ~llama_moe_cache();

    llama_moe_cache(const llama_moe_cache &) = delete;
    llama_moe_cache & operator=(const llama_moe_cache &) = delete;

    bool enabled() const { return params.n_slots > 0; }

    // Register an expert tensor. Returns nullptr when the tensor cannot be cached (wrong shape,
    // already resident on the accelerator, or out of VRAM), in which case the caller must build
    // the ordinary uncached graph.
    llama_moe_cache_tensor * add_tensor(ggml_tensor * src, int64_t n_expert, int64_t n_expert_used);

    // Look up a previously registered tensor.
    llama_moe_cache_tensor * find(const ggml_tensor * src) const;

    // Allocate the VRAM buffers for everything registered so far. Call once, after all
    // add_tensor() calls and before the first graph is built.
    bool alloc();

    // Record the experts routed by one forward pass. Drives admission; does not touch VRAM.
    void observe(llama_moe_cache_tensor * ct, const int32_t * ids, size_t n_ids);

    // Note the routing tensor a cached weight was built against, so the routing can be read
    // back once the graph has run.
    void note_ids(llama_moe_cache_tensor * ct, ggml_tensor * ids) { if (ct) ct->ids_node = ids; }

    // Read back this step's routing and apply the residency policy. Call after the graph has
    // completed and before the next one is built.
    void harvest();

    // Apply pending admissions and re-upload dirty maps. Must run between steps, never during
    // one: it copies expert rows into slot buffers that an in-flight graph may be reading.
    void update();

    // [1, n_expert_used, 1] f32 holding n_slots + position, added to routes with no slot.
    ggml_tensor * dummy_pos = nullptr;

    // Bytes of VRAM the configured slots require for the tensors registered so far.
    size_t size_bytes() const { return total_bytes; }

    struct stats {
        uint64_t n_routes = 0; // expert routes observed
        uint64_t n_hits   = 0; // ... that were resident
        uint64_t n_admits = 0; // experts promoted into VRAM
        uint64_t n_evicts = 0;
    };

    const stats & get_stats() const { return st; }

private:
    // Promote `expert` into the least-recently-used slot. Returns false if nothing could be
    // evicted (every slot was routed this step).
    bool admit(llama_moe_cache_tensor & ct, int32_t expert);

    llama_moe_cache_params params;
    ggml_backend_buffer_type_t buft = nullptr;

    ggml_context      * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    std::vector<llama_moe_cache_tensor> tensors;
    std::unordered_map<const ggml_tensor *, size_t> by_src;

    // pending admissions per tensor, applied by update()
    std::vector<std::vector<int32_t>> pending;

    uint64_t step = 0;
    uint64_t harvests = 0;
    uint64_t w_routes = 0, w_hits = 0, w_admits = 0, w_evicts = 0; // last window snapshot
    size_t total_bytes = 0;
    stats st;
};
