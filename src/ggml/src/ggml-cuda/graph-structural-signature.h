#pragma once

#include "ggml.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// A deliberately conservative, collision-free description of a CUDA graph.
// Tensor descriptor addresses are represented only by ordered graph indices
// or by explicit external bindings. Device-visible addresses and CUDA extras
// remain exact identity fields.
namespace ggml_cuda_graph_structural {

enum class ref_kind : uint8_t { none, graph_node, external_binding };
enum class build_failure : uint8_t {
    none,
    empty_graph,
    duplicate_or_invalid_node,
    missing_external_binding,
    unsupported_external_binding,
    cyclic_external_binding,
    too_many_external_bindings,
    unsupported_node_extra,
    count,
};

struct tensor_ref {
    ref_kind kind = ref_kind::none;
    uint32_t  index = 0;

    bool operator==(const tensor_ref & other) const {
        return kind == other.kind && index == other.index;
    }
};

struct tensor_descriptor {
    int32_t type = 0;
    uintptr_t buffer = 0;
    std::array<int64_t, GGML_MAX_DIMS> ne = {};
    std::array<size_t, GGML_MAX_DIMS> nb = {};
    int32_t op = 0;
    std::array<int32_t, GGML_MAX_OP_PARAMS / sizeof(int32_t)> op_params = {};
    int32_t flags = 0;
    size_t view_offs = 0;
    uintptr_t data = 0;
    std::array<char, GGML_MAX_NAME> name = {};
    uintptr_t extra = 0;
    std::array<char, 8> padding = {};
    std::vector<uintptr_t> extra_state;

    bool operator==(const tensor_descriptor & other) const {
        return type == other.type && buffer == other.buffer && ne == other.ne && nb == other.nb &&
               op == other.op && op_params == other.op_params && flags == other.flags &&
               view_offs == other.view_offs && data == other.data && name == other.name &&
               extra == other.extra && padding == other.padding && extra_state == other.extra_state;
    }
};

// host_identity is used only while canonicalizing one current graph. It never
// appears in a completed signature.
struct tensor_input {
    uintptr_t host_identity = 0;
    tensor_descriptor descriptor;
    std::array<uintptr_t, GGML_MAX_SRC> sources = {};
    uintptr_t view_source = 0;
    bool extra_state_valid = true;
};

struct node_signature {
    tensor_descriptor descriptor;
    std::array<tensor_ref, GGML_MAX_SRC> sources = {};
    tensor_ref view_source;

    bool operator==(const node_signature & other) const {
        return descriptor == other.descriptor && sources == other.sources && view_source == other.view_source;
    }
};

struct graph_signature {
    std::vector<node_signature> nodes;
    std::vector<node_signature> external_bindings;

    bool operator==(const graph_signature & other) const {
        return nodes == other.nodes && external_bindings == other.external_bindings;
    }
};

// This predicate is NOT replay equivalence. It may only choose an executable
// handle for a FULL recapture followed by cudaGraphExecUpdate.
inline bool same_update_layout(const tensor_descriptor & a, const tensor_descriptor & b) {
    return a.type == b.type && a.ne == b.ne && a.nb == b.nb && a.op == b.op &&
           a.op_params == b.op_params && a.flags == b.flags && a.view_offs == b.view_offs &&
           a.name == b.name && a.padding == b.padding && a.extra_state.size() == b.extra_state.size();
}
inline bool same_update_layout(const graph_signature & a, const graph_signature & b) {
    if (a.nodes.size() != b.nodes.size() || a.external_bindings.size() != b.external_bindings.size()) return false;
    const auto match = [](const std::vector<node_signature> & left, const std::vector<node_signature> & right) {
        for (size_t i = 0; i < left.size(); ++i) {
            if (!same_update_layout(left[i].descriptor, right[i].descriptor) ||
                left[i].sources != right[i].sources || !(left[i].view_source == right[i].view_source)) return false;
        }
        return true;
    };
    return match(a.nodes, b.nodes) && match(a.external_bindings, b.external_bindings);
}

inline bool is_external_bound(const tensor_input & input) {
    const auto & ne = input.descriptor.ne;
    const bool valid_shape = std::all_of(ne.begin(), ne.end(), [](int64_t value) { return value >= 0; });
    const bool empty = std::any_of(ne.begin(), ne.end(), [](int64_t value) { return value == 0; });
    // An unused zero-sized KV view legitimately has no data allocation. Its
    // complete descriptor and alias edges still participate in exact equality.
    return input.host_identity != 0 && input.descriptor.buffer != 0 && valid_shape &&
           (input.descriptor.data != 0 || empty) && input.extra_state_valid;
}

inline const char * failure_name(build_failure failure) {
    switch (failure) {
        case build_failure::none: return "none";
        case build_failure::empty_graph: return "empty_graph";
        case build_failure::duplicate_or_invalid_node: return "invalid_node";
        case build_failure::missing_external_binding: return "missing_external";
        case build_failure::unsupported_external_binding: return "unsupported_external";
        case build_failure::cyclic_external_binding: return "cycle";
        case build_failure::too_many_external_bindings: return "external_cap";
        case build_failure::unsupported_node_extra: return "opaque_node_extra";
        case build_failure::count: break;
    }
    return "unknown";
}

// External non-alias producers are already materialized inputs to this
// capture. Their off-graph computation history is not re-executed. Keep alias
// ownership and references into this graph, but stop unrelated producer DAGs.
inline tensor_input project_external_binding(tensor_input input,
                                            const std::unordered_set<uintptr_t> & graph_identities) {
    const auto op = ggml_op(input.descriptor.op);
    const bool alias = input.view_source != 0 || op == GGML_OP_VIEW || op == GGML_OP_RESHAPE ||
                       op == GGML_OP_TRANSPOSE || op == GGML_OP_PERMUTE;
    if (!alias) {
        for (auto & source : input.sources) {
            if (source != 0 && graph_identities.count(source) == 0) source = 0;
        }
    }
    return input;
}

struct cycle_edge {
    uintptr_t from = 0;
    uintptr_t to = 0;
};

inline constexpr size_t max_external_bindings = 4096;

inline bool build_signature(const std::vector<tensor_input> & ordered_nodes,
                            const std::vector<tensor_input> & external_inputs,
                            graph_signature * result,
                            build_failure * failure = nullptr, cycle_edge * cycle = nullptr) {
    if (failure != nullptr) {
        *failure = build_failure::none;
    }
    if (cycle) *cycle = {};
    if (result == nullptr || ordered_nodes.empty()) {
        if (failure != nullptr) {
            *failure = build_failure::empty_graph;
        }
        return false;
    }
    if (ordered_nodes.size() > std::numeric_limits<uint32_t>::max()) {
        if (failure != nullptr) {
            *failure = build_failure::duplicate_or_invalid_node;
        }
        return false;
    }

    std::unordered_map<uintptr_t, uint32_t> node_indices;
    node_indices.reserve(ordered_nodes.size());
    for (size_t i = 0; i < ordered_nodes.size(); ++i) {
        const uintptr_t identity = ordered_nodes[i].host_identity;
        if (!ordered_nodes[i].extra_state_valid) {
            if (failure) *failure = build_failure::unsupported_node_extra;
            return false;
        }
        if (identity == 0 || !node_indices.emplace(identity, uint32_t(i)).second) {
            if (failure != nullptr) {
                *failure = build_failure::duplicate_or_invalid_node;
            }
            return false;
        }
    }

    std::unordered_map<uintptr_t, const tensor_input *> external_by_identity;
    external_by_identity.reserve(external_inputs.size());
    for (const tensor_input & input : external_inputs) {
        if (input.host_identity == 0 || node_indices.count(input.host_identity) != 0 ||
            !external_by_identity.emplace(input.host_identity, &input).second) {
            if (failure != nullptr) {
                *failure = build_failure::duplicate_or_invalid_node;
            }
            return false;
        }
        if (!is_external_bound(input)) {
            if (failure != nullptr) {
                *failure = build_failure::unsupported_external_binding;
            }
            return false;
        }
    }

    graph_signature candidate;
    candidate.nodes.reserve(ordered_nodes.size());
    candidate.external_bindings.reserve(std::min(external_inputs.size(), max_external_bindings));
    std::unordered_map<uintptr_t, uint32_t> external_indices;
    std::vector<uintptr_t> external_identity;
    external_identity.reserve(std::min(external_inputs.size(), max_external_bindings));
    const auto resolve = [&](uintptr_t identity, tensor_ref * ref) -> bool {
        if (identity == 0) {
            *ref = {};
            return true;
        }
        const auto node = node_indices.find(identity);
        if (node != node_indices.end()) {
            *ref = {ref_kind::graph_node, node->second};
            return true;
        }
        const auto ext = external_by_identity.find(identity);
        if (ext == external_by_identity.end()) {
            if (failure != nullptr) {
                *failure = build_failure::missing_external_binding;
            }
            return false;
        }
        const auto existing = external_indices.find(identity);
        if (existing != external_indices.end()) {
            *ref = {ref_kind::external_binding, existing->second};
            return true;
        }
        if (candidate.external_bindings.size() >= max_external_bindings) {
            if (failure != nullptr) {
                *failure = build_failure::too_many_external_bindings;
            }
            return false;
        }
        auto [it, inserted] = external_indices.emplace(identity, uint32_t(candidate.external_bindings.size()));
        if (inserted) {
            node_signature binding;
            binding.descriptor = ext->second->descriptor;
            candidate.external_bindings.push_back(std::move(binding));
            external_identity.push_back(identity);
        }
        *ref = {ref_kind::external_binding, it->second};
        return true;
    };

    for (const tensor_input & input : ordered_nodes) {
        node_signature node;
        node.descriptor = input.descriptor;
        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            if (!resolve(input.sources[i], &node.sources[i])) {
                return false;
            }
        }
        if (!resolve(input.view_source, &node.view_source)) {
            return false;
        }
        candidate.nodes.push_back(std::move(node));
    }

    // First-use order follows graph nodes and source ports, then a bounded queue
    // of external bindings. It is independent of external input allocation order.
    for (size_t i = 0; i < candidate.external_bindings.size(); ++i) {
        const tensor_input & input = *external_by_identity.at(external_identity[i]);
        node_signature binding;
        binding.descriptor = input.descriptor;
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (!resolve(input.sources[s], &binding.sources[s])) return false;
        }
        if (!resolve(input.view_source, &binding.view_source)) return false;
        candidate.external_bindings[i] = std::move(binding);
    }

    // Validate cycles across both graph and external nodes without recursive
    // stack growth. Source and view edges both participate in alias topology.
    const size_t n_graph = candidate.nodes.size();
    const size_t total = n_graph + candidate.external_bindings.size();
    std::vector<uint8_t> state(total, 0);
    struct frame { size_t index; int edge; };
    std::vector<frame> stack;
    const auto at = [&](size_t index) -> const node_signature & {
        return index < n_graph ? candidate.nodes[index] : candidate.external_bindings[index - n_graph];
    };
    for (size_t root = 0; root < total; ++root) {
        if (state[root]) continue;
        state[root] = 1;
        stack.push_back({root, 0});
        while (!stack.empty()) {
            auto & top = stack.back();
            if (top.edge > GGML_MAX_SRC) {
                state[top.index] = 2;
                stack.pop_back();
                continue;
            }
            const auto & node = at(top.index);
            const int edge = top.edge++;
            const tensor_ref ref = edge < GGML_MAX_SRC ? node.sources[edge] : node.view_source;
            if (ref.kind == ref_kind::none) continue;
            const size_t child = ref.kind == ref_kind::graph_node ? ref.index : n_graph + ref.index;
            // CPY's src[1] is its write destination, not a value dependency.
            // Scheduler/Meta copies can refer to themselves as destination and
            // alias owner. Keep these storage edges in the signature but do
            // not mistake them for a cyclic read/computation dependency.
            if (node.descriptor.op == GGML_OP_CPY &&
                (edge == 1 || (edge == GGML_MAX_SRC && child == top.index))) continue;
            if (state[child] == 1) {
                if (failure) *failure = build_failure::cyclic_external_binding;
                if (cycle) {
                    const auto identity = [&](size_t index) {
                        return index < n_graph ? ordered_nodes[index].host_identity : external_identity[index - n_graph];
                    };
                    *cycle = {identity(top.index), identity(child)};
                }
                return false;
            }
            if (state[child] == 0) {
                state[child] = 1;
                stack.push_back({child, 0});
            }
        }
    }

    *result = std::move(candidate);
    return true;
}

} // namespace ggml_cuda_graph_structural
