#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

struct ggml_backend_meta_device;
struct ggml_backend_meta_buffer_type;
struct ggml_backend_meta_buffer;
struct ggml_backend_meta;

static bool ggml_meta_fifo_enabled() {
    const char * value = getenv("GGML_META_FIFO_PIPELINE");
    return value && atoi(value) == 1;
}

struct ggml_backend_meta_event_context {
    std::vector<ggml_backend_event_t> events;
};

static ggml_backend_event_t ggml_backend_meta_event_new(ggml_backend_dev_t dev);
static void ggml_backend_meta_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event);
static void ggml_backend_meta_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event);

const char * ggml_backend_meta_split_axis_name(enum ggml_backend_meta_split_axis split_axis) {
    switch (split_axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
            return "0";
        case GGML_BACKEND_SPLIT_AXIS_1:
            return "1";
        case GGML_BACKEND_SPLIT_AXIS_2:
            return "2";
        case GGML_BACKEND_SPLIT_AXIS_3:
            return "3";
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            return "MIRRORED";
        case GGML_BACKEND_SPLIT_AXIS_PARTIAL:
            return "PARTIAL";
        case GGML_BACKEND_SPLIT_AXIS_NONE:
            return "NONE";
        case GGML_BACKEND_SPLIT_AXIS_UNKNOWN:
            return "UNKNOWN";
        default:
            GGML_ABORT("fatal error");
    }
}

//
// meta backend device
//

struct ggml_backend_meta_device_context {
    std::vector<ggml_backend_dev_t>     simple_devs;
    ggml_backend_meta_get_split_state_t get_split_state;
    void *                              get_split_state_ud;

    std::string name;
    std::string description;

    ggml_backend_meta_device_context(
            std::vector<ggml_backend_dev_t> simple_devs, ggml_backend_meta_get_split_state_t get_split_state, void * get_split_state_ud) :
            simple_devs(std::move(simple_devs)), get_split_state(get_split_state), get_split_state_ud(get_split_state_ud) {
        name        = std::string("Meta(");
        description = std::string("Meta(");
        for (size_t i = 0; i < simple_devs.size(); i++) {
            if (i > 0) {
                name        += ",";
                description += ",";
            }
            name        += ggml_backend_dev_name       (simple_devs[i]);
            description += ggml_backend_dev_description(simple_devs[i]);
        }
        name        += ")";
        description += ")";
    }

    bool operator<(const ggml_backend_meta_device_context & other) const {
        return std::tie(simple_devs, get_split_state, get_split_state_ud)
            < std::tie(other.simple_devs, other.get_split_state, other.get_split_state_ud);
    }
};

static bool ggml_backend_dev_is_meta(ggml_backend_dev_t dev);

static const char * ggml_backend_meta_device_get_name(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return meta_dev_ctx->name.c_str();
}

static const char * ggml_backend_meta_device_get_description(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return meta_dev_ctx->description.c_str();
}

static void ggml_backend_meta_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    *free  = 0;
    *total = 0;
    for (ggml_backend_dev_t dev : meta_dev_ctx->simple_devs) {
        size_t tmp_free, tmp_total;
        ggml_backend_dev_memory(dev, &tmp_free, &tmp_total);
        *free  += tmp_free;
        *total += tmp_total;
    }
}

static enum ggml_backend_dev_type ggml_backend_meta_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_META;

    GGML_UNUSED(dev);
}

static void ggml_backend_meta_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;

    // TODO replace placeholders
    props->name        = ggml_backend_meta_device_get_name(dev);
    props->description = ggml_backend_meta_device_get_description(dev);
    props->type        = ggml_backend_meta_device_get_type(dev);
    props->device_id   = 0;

    ggml_backend_meta_device_get_memory(dev, &props->memory_free, &props->memory_total);

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ false, // Not implemented.
        /* .buffer_from_host_ptr  = */ false, // Not implemented.
        /* .events                = */ ggml_meta_fifo_enabled(),
        /* .mmap_support          = */ true,
    };
    for (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_dev_props tmp_props;
        ggml_backend_dev_get_props(simple_dev, &tmp_props);
        props->caps.async                = props->caps.async                && tmp_props.caps.async;
        props->caps.host_buffer          = props->caps.host_buffer          && tmp_props.caps.host_buffer;
        props->caps.buffer_from_host_ptr = props->caps.buffer_from_host_ptr && tmp_props.caps.buffer_from_host_ptr;
        props->caps.events               = props->caps.events               && tmp_props.caps.events;
        props->caps.mmap_support         = props->caps.mmap_support         && tmp_props.caps.mmap_support;
    }
}

static ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, const char * params);

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_buffer_type(ggml_backend_dev_t dev);

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_host_buffer_type(ggml_backend_dev_t dev);

static bool ggml_backend_meta_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return std::all_of(meta_dev_ctx->simple_devs.begin(), meta_dev_ctx->simple_devs.end(),
        [op](ggml_backend_dev_t simple_dev) { return ggml_backend_dev_supports_op(simple_dev, op); });
}

static bool ggml_backend_meta_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ggml_backend_dev_t dev_buft = ggml_backend_buft_get_device(buft);
    if (!ggml_backend_dev_is_meta(dev_buft)) {
        return false;
    }
    const ggml_backend_meta_device_context * meta_dev_ctx      = (const ggml_backend_meta_device_context *) dev->context;
    const ggml_backend_meta_device_context * meta_buft_dev_ctx = (const ggml_backend_meta_device_context *) dev_buft->context;
    if (meta_dev_ctx->simple_devs.size() != meta_buft_dev_ctx->simple_devs.size()) {
        return false;
    }
    for (size_t i = 0; i < meta_dev_ctx->simple_devs.size(); i++) {
        if (meta_dev_ctx->simple_devs[i] != meta_buft_dev_ctx->simple_devs[i]) {
            return false;
        }
    }
    return true;
}

static const ggml_backend_device_i ggml_backend_meta_device_iface = {
    /* .get_name             = */ ggml_backend_meta_device_get_name,
    /* .get_description      = */ ggml_backend_meta_device_get_description,
    /* .get_memory           = */ ggml_backend_meta_device_get_memory,
    /* .get_type             = */ ggml_backend_meta_device_get_type,
    /* .get_props            = */ ggml_backend_meta_device_get_props,
    /* .init_backend         = */ ggml_backend_meta_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_meta_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_meta_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_meta_device_supports_op,
    /* .supports_buft        = */ ggml_backend_meta_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ ggml_backend_meta_event_new,
    /* .event_free           = */ ggml_backend_meta_event_free,
    /* .event_synchronize    = */ ggml_backend_meta_event_synchronize,
};

static bool ggml_backend_dev_is_meta(ggml_backend_dev_t dev) {
    return dev != nullptr && dev->iface.get_name == ggml_backend_meta_device_iface.get_name;
}

static size_t ggml_backend_meta_dev_n_devs(ggml_backend_dev_t meta_dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) meta_dev->context;
    return meta_dev_ctx->simple_devs.size();
}

// number of simple devices wrapped by a meta device; lets callers check that a shared
// split-state userdata matches the meta device it is used with (island/tensor-parallel setups)
size_t ggml_backend_meta_device_n_devices(ggml_backend_dev_t meta_dev) {
    return ggml_backend_meta_dev_n_devs(meta_dev);
}

// v10 diagnostics: member i of a meta device, so callers can verify the island composition and
// per-member memory reports when an aggregate (free-memory sum) looks wrong
static ggml_backend_dev_t ggml_backend_meta_dev_simple_dev(ggml_backend_dev_t meta_dev, size_t index);
ggml_backend_dev_t ggml_backend_meta_device_member(ggml_backend_dev_t meta_dev, size_t index) {
    return ggml_backend_meta_dev_simple_dev(meta_dev, index);
}

static ggml_backend_dev_t ggml_backend_meta_dev_simple_dev(ggml_backend_dev_t meta_dev, size_t index) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) meta_dev->context;
    GGML_ASSERT(index < meta_dev_ctx->simple_devs.size());
    return meta_dev_ctx->simple_devs[index];
}

static ggml_backend_event_t ggml_backend_meta_event_new(ggml_backend_dev_t dev) {
    if (!ggml_meta_fifo_enabled()) {
        return nullptr;
    }
    auto * ctx = new ggml_backend_meta_event_context;
    for (size_t i = 0; i < ggml_backend_meta_dev_n_devs(dev); ++i) {
        auto event = ggml_backend_event_new(ggml_backend_meta_dev_simple_dev(dev, i));
        if (!event) {
            for (auto old : ctx->events) ggml_backend_event_free(old);
            delete ctx;
            return nullptr;
        }
        ctx->events.push_back(event);
    }
    return new ggml_backend_event{dev, ctx};
}

static void ggml_backend_meta_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    auto * ctx = (ggml_backend_meta_event_context *) event->context;
    for (auto child : ctx->events) ggml_backend_event_free(child);
    delete ctx;
    delete event;
}

static void ggml_backend_meta_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    auto * ctx = (ggml_backend_meta_event_context *) event->context;
    for (auto child : ctx->events) ggml_backend_event_synchronize(child);
}

ggml_backend_dev_t ggml_backend_meta_device(
        ggml_backend_dev_t * devs, size_t n_devs, ggml_backend_meta_get_split_state_t get_split_state, void * get_split_state_ud) {
    GGML_ASSERT(n_devs <= GGML_BACKEND_META_MAX_DEVICES);
    // TODO: this is not thread-safe - needs to be fixed
    static std::vector<std::unique_ptr<ggml_backend_meta_device_context>>         ctxs;
    static std::map<ggml_backend_meta_device_context, struct ggml_backend_device> meta_devs;

    std::vector<ggml_backend_dev_t> simple_devs;
    simple_devs.reserve(n_devs);
    for (size_t i = 0; i < n_devs; i++) {
        simple_devs.push_back(devs[i]);
    }
    ggml_backend_meta_device_context ctx(simple_devs, get_split_state, get_split_state_ud);

    {
        auto it = meta_devs.find(ctx);
        if (it != meta_devs.end()) {
            return &it->second;
        }
    }
    ctxs.push_back(std::make_unique<ggml_backend_meta_device_context>(ctx));

    struct ggml_backend_device meta_dev = {
        /*iface  =*/ ggml_backend_meta_device_iface,
        /*reg    =*/ nullptr,
        /*ctx    =*/ ctxs.back().get(),
    };

    auto result = meta_devs.emplace(*ctxs.back(), meta_dev);
    return &result.first->second;
}

//
// meta backend buffer type
//

struct ggml_backend_meta_buffer_type_context {
    std::vector<ggml_backend_buffer_type_t> simple_bufts;

    std::string name;

    ggml_backend_meta_buffer_type_context(std::vector<ggml_backend_buffer_type_t> simple_bufts) : simple_bufts(std::move(simple_bufts)) {
        name = "Meta(";
        for (size_t i = 0; i < simple_bufts.size(); i++) {
            if (i > 0) {
                name += ",";
            }
            name += ggml_backend_buft_name(simple_bufts[i]);
        }
        name += ")";
    }

    bool operator<(const ggml_backend_meta_buffer_type_context & other) const {
        return simple_bufts < other.simple_bufts;
    }
};

static size_t ggml_backend_meta_buft_n_bufts(ggml_backend_buffer_type_t meta_buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) meta_buft->context;
    return meta_buft_ctx->simple_bufts.size();
}

static const char * ggml_backend_meta_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) buft->context;
    return meta_buft_ctx->name.c_str();
}

static ggml_backend_buffer_type_t ggml_backend_meta_buft_simple_buft(ggml_backend_buffer_type_t meta_buft, size_t index) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) meta_buft->context;
    GGML_ASSERT(index < meta_buft_ctx->simple_bufts.size());
    return meta_buft_ctx->simple_bufts[index];
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size);

static size_t ggml_backend_meta_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alignment = 1;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        const size_t alignment = ggml_backend_buft_get_alignment(ggml_backend_meta_buft_simple_buft(buft, i));
        max_alignment = std::max(max_alignment, alignment);
        GGML_ASSERT(max_alignment % alignment == 0);
    }
    return max_alignment;
}

static size_t ggml_backend_meta_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_size = SIZE_MAX;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        max_size = std::min(max_size, ggml_backend_buft_get_max_size(ggml_backend_meta_buft_simple_buft(buft, i)));
    }
    return max_size;
}

static size_t ggml_backend_meta_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alloc_size = 0;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        const size_t alloc_size = ggml_backend_buft_get_alloc_size(ggml_backend_meta_buft_simple_buft(buft, i), tensor);
        max_alloc_size = std::max(max_alloc_size, alloc_size);
    }
    return max_alloc_size;
}

static bool ggml_backend_meta_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    for (size_t i = 0; i < n_simple_bufts; i++) {
        if (!ggml_backend_buft_is_host(ggml_backend_meta_buft_simple_buft(buft, i))) {
            return false;
        }
    }
    return true;
}

static const struct ggml_backend_buffer_type_i ggml_backend_meta_buffer_type_iface = {
    /* .get_name         = */ ggml_backend_meta_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_meta_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_meta_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_meta_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_meta_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_meta_buffer_type_is_host,
};

bool ggml_backend_buft_is_meta(ggml_backend_buffer_type_t buft) {
    return buft != nullptr && buft->iface.get_name == ggml_backend_meta_buffer_type_iface.get_name;
}

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_buffer_type(ggml_backend_dev_t dev) {
    static std::map<ggml_backend_dev_t, struct ggml_backend_buffer_type> meta_bufts;
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    {
        auto it = meta_bufts.find(dev);
        if (it != meta_bufts.end()) {
            return &it->second;
        }
    }

    const size_t n_devs = ggml_backend_meta_dev_n_devs(dev);
    std::vector<ggml_backend_buffer_type_t> simple_bufts;
    simple_bufts.reserve(n_devs);
    for (size_t i = 0; i < n_devs; i++) {
        simple_bufts.push_back(ggml_backend_dev_buffer_type(ggml_backend_meta_dev_simple_dev(dev, i)));
    }
    ggml_backend_meta_buffer_type_context * buft_ctx = new ggml_backend_meta_buffer_type_context(simple_bufts);

    struct ggml_backend_buffer_type meta_buft = {
        /*iface  =*/ ggml_backend_meta_buffer_type_iface,
        /*device =*/ dev,
        /*ctx    =*/ buft_ctx,
    };
    auto result = meta_bufts.emplace(dev, meta_buft);
    return &result.first->second;
}

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;

    ggml_backend_buffer_type_t host_buft = nullptr;
    for (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_buffer_type_t simple_host_buft = ggml_backend_dev_host_buffer_type(simple_dev);
        if (simple_host_buft == nullptr) {
            return nullptr;
        }
        if (host_buft == nullptr) {
            host_buft = simple_host_buft;
        } else if (host_buft != simple_host_buft) {
            // if different simple devices have different host buffer types,
            // we cannot provide a single host buffer type for the meta device
            return nullptr;
        }
    }
    return host_buft;
}

//
// meta backend buffer
//

// Container to hold the tensor slices per simple ggml backend buffer.
struct ggml_backend_meta_simple_tensor_container {
    std::vector<ggml_context_ptr> ctxs;
    std::map<const ggml_tensor *, std::vector<ggml_tensor *>> simple_tensors;

    ggml_backend_meta_simple_tensor_container(const ggml_init_params & params, const int n_simple) {
        ctxs.reserve(n_simple);
        for (int i = 0; i < n_simple; i++) {
            ctxs.emplace_back(ggml_init(params));
        }
    }
    ggml_backend_meta_simple_tensor_container() {}
};

#include "ggml-meta-cache.h"

struct ggml_backend_meta_buffer_context {
    // External views share weight buffers across model contexts. Active graphs retain
    // their descriptor generations; only an unreferenced generation may be reset.
    ggml_backend_meta_simple_tensor_container stc_static;
    std::shared_ptr<ggml_backend_meta_simple_tensor_container> stc_compute[2];
    int stc_compute_index      = 0;
    int stc_compute_index_next = 0;
    std::vector<ggml_backend_buffer_ptr> bufs;

    // FIXME
    // The size of the split state cache is unbounded and can theoretically grow infinitely large.
    // However, it is also expensive to build and clearing it on every rebuild in ggml_backend_meta_graph_compute is too expensive.
    static constexpr size_t nbtc = GGML_TENSOR_SIZE - sizeof(ggml_tensor::padding);
    std::map<std::pair<const ggml_tensor *, bool>, std::pair<ggml_meta_cached_split, char[nbtc]>> split_state_cache;

    int debug;

    ggml_backend_meta_buffer_context(
            ggml_backend_meta_simple_tensor_container & stc_static,
            ggml_backend_meta_simple_tensor_container & stc_compute_0,
            ggml_backend_meta_simple_tensor_container & stc_compute_1,
            const std::vector<ggml_backend_buffer_t> & bufs)
            : stc_static(std::move(stc_static)), stc_compute{
                std::make_shared<ggml_backend_meta_simple_tensor_container>(std::move(stc_compute_0)),
                std::make_shared<ggml_backend_meta_simple_tensor_container>(std::move(stc_compute_1))} {
        this->bufs.reserve(bufs.size());
        for (ggml_backend_buffer_t buf : bufs) {
            this->bufs.emplace_back(buf);
        }
        const char * GGML_META_DEBUG = getenv("GGML_META_DEBUG");
        debug = GGML_META_DEBUG ? atoi(GGML_META_DEBUG) : 0;
    }

    ggml_backend_meta_simple_tensor_container & get_simple_tensor_container(const ggml_tensor * tensor) {
        if (stc_static.simple_tensors.find(tensor) != stc_static.simple_tensors.end()) {
            return stc_static;
        }
        return *stc_compute[stc_compute_index];
    }
};

static void ggml_backend_meta_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    if (getenv("GGML_META_CACHE_TRACE")) {
        size_t stored = 0;
        for (const auto & entry : buf_ctx->split_state_cache) stored += sizeof(entry.second.first) + entry.second.first.ne.capacity() * sizeof(int64_t);
        GGML_LOG_INFO("meta-cache-storage entries=%zu descriptors_original=%zu descriptors_stored=%zu\n",
                buf_ctx->split_state_cache.size(), buf_ctx->split_state_cache.size() * sizeof(ggml_backend_meta_split_state), stored);
    }
    delete buf_ctx;
}

static size_t ggml_backend_meta_buffer_n_bufs(ggml_backend_buffer_t meta_buf) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(meta_buf));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) meta_buf->context;
    return buf_ctx->bufs.size();
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_simple_buffer(ggml_backend_buffer_t meta_buf, size_t index) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(meta_buf));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) meta_buf->context;
    GGML_ASSERT(index < buf_ctx->bufs.size());
    return buf_ctx->bufs[index].get();
}

static struct ggml_tensor * ggml_backend_meta_buffer_simple_tensor(const struct ggml_tensor * tensor, size_t index) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    GGML_ASSERT(index < buf_ctx->bufs.size());

    ggml_backend_meta_simple_tensor_container & stc = buf_ctx->get_simple_tensor_container(tensor);
    auto it = stc.simple_tensors.find(tensor);
    if (it == stc.simple_tensors.end()) {
        return nullptr;
    }
    return it->second[index];
}

static struct ggml_backend_meta_split_state ggml_backend_meta_get_split_state(const struct ggml_tensor * tensor, bool assume_sync);

static struct ggml_backend_meta_split_state ggml_backend_meta_get_split_state(
        ggml_backend_meta_simple_tensor_container & stc, const struct ggml_tensor * tensor, bool assume_sync);

static struct ggml_backend_meta_split_state ggml_backend_meta_get_split_state_impl(
        ggml_backend_meta_simple_tensor_container & stc, const struct ggml_tensor * tensor, bool assume_sync) {
    // FIXME Currently this function preserves/erases the information in n_segments and nr in an inconsistent way.
    // Since the operations in question are developed specifically for llama.cpp this currently does not manifest as a bug there.
    // However, in a broader ggml context with arbitrary ggml graphs this can lead to unexpected results.
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;

    auto split_states_equal = [&](const ggml_backend_meta_split_state & a, const ggml_backend_meta_split_state & b) -> bool {
        if (a.axis != b.axis) {
            return false;
        }
        for (size_t j = 0; j < n_bufs; j++) {
            int64_t sum_a = 0;
            for (size_t s = 0; s < a.n_segments; s++) {
                sum_a += a.ne[s*n_bufs + j] * a.nr[s];
            }
            int64_t sum_b = 0;
            for (size_t s = 0; s < b.n_segments; s++) {
                sum_b += b.ne[s*n_bufs + j] * b.nr[s];
            }
            if (sum_a != sum_b) {
                return false;
            }
        }
        return true;
    };

    auto handle_generic = [&](const std::vector<ggml_backend_meta_split_state> & src_ss, bool scalar_only) -> ggml_backend_meta_split_state {
        ggml_backend_meta_split_state ret = {GGML_BACKEND_SPLIT_AXIS_NONE, {0}, {1}, 1};
        for (size_t i = 0; i < GGML_MAX_SRC; i++) {
            if (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                continue;
            }
            if (ret.axis == GGML_BACKEND_SPLIT_AXIS_NONE) {
                ret = src_ss[i];
            } else if (!split_states_equal(src_ss[i], ret)) {
                ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
                break;
            }
        }
        if (ret.axis == GGML_BACKEND_SPLIT_AXIS_NONE) {
            ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        if (scalar_only && ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
            ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        if (ret.axis == GGML_BACKEND_SPLIT_AXIS_UNKNOWN) {
            // either the node has no sources, its sources have inconsistent split states, or
            // (scalar_only) a source is genuinely split along a dimension. Dump everything needed
            // to identify the op; GGML_META_AXIS_UNKNOWN=mirrored downgrades the abort into a
            // warning + replicated placement (island bring-up escape hatch, may cost correctness).
            static const bool fallback_mirrored = [] {
                const char * s = getenv("GGML_META_AXIS_UNKNOWN");
                return s != nullptr && strcmp(s, "mirrored") == 0;
            }();
            std::string src_info;
            for (size_t i = 0; i < GGML_MAX_SRC; i++) {
                if (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                    continue;
                }
                src_info += "src" + std::to_string(i) + " " + tensor->src[i]->name + "[" + ggml_op_name(tensor->src[i]->op) +
                    ", axis=" + ggml_backend_meta_split_axis_name(src_ss[i].axis) + "]; ";
            }
            if (fallback_mirrored) {
                GGML_LOG_WARN("%s: tensor %s [%s, type %s] has no resolvable split axis (%s)- treating as MIRRORED "
                              "(GGML_META_AXIS_UNKNOWN=mirrored; results may be wrong if the tensor is row-split)\n",
                        __func__, tensor->name, ggml_op_name(tensor->op), ggml_type_name(tensor->type), src_info.c_str());
                return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
            }
            GGML_ABORT("no split axis for tensor %s [%s, type %s, ne %" PRId64 " x %" PRId64 " x %" PRId64 " x %" PRId64 "]; %s"
                       "(run with GGML_META_DEBUG=1 to trace the split-state chain; GGML_META_AXIS_UNKNOWN=mirrored downgrades this to a warning)",
                    tensor->name, ggml_op_name(tensor->op), ggml_type_name(tensor->type),
                    tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3], src_info.c_str());
        }
        GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
        return ret;
    };

    // Some ops process data on a per-row bases:
    auto handle_per_row = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0);
        return src_ss[0];
    };

    // Some ops broadcast the src1 data across src0:
    auto handle_bin_bcast = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS &&
                tensor->src[1]->ne[src_ss[0].axis] == 1 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        if (src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && (src_ss[0].axis == src_ss[1].axis ||
           (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL)))) {
            return src_ss[0]; // GGML_OP_ADD_ID
        }
        GGML_ASSERT(tensor->src[2] == nullptr || src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        return handle_generic(src_ss, /*scalar_only =*/ false);
    };

    auto handle_concat = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        const ggml_backend_meta_split_axis concat_axis = ggml_backend_meta_split_axis(ggml_get_op_params_i32(tensor, 0));
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis >= 0 && src_ss[1].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[1].axis);
            return src_ss[1];
        }
        if (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[0].axis);
            return src_ss[0];
        }
        if (src_ss[0].axis == src_ss[1].axis && src_ss[0].axis != concat_axis) {
            return src_ss[0];
        }
        // v9: concatenating along the SPLIT axis of two sources that are split along that same
        // axis (island conv state: concat(gathered cache rows, transposed projection) on axis 0).
        // Each device concatenates its own two slices, so the result keeps the axis with the
        // per-device width being the SUM of both sources' widths. Reaching this branch used to
        // abort in handle_generic (scalar_only) - the split-state callback models exactly this
        // segment pattern for the static caches (see the r_cache segments), so the semantics
        // match the loader-side layout.
        if (src_ss[0].axis == src_ss[1].axis && src_ss[0].axis == concat_axis &&
                src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS &&
                src_ss[0].n_segments == 1 && src_ss[1].n_segments == 1) {
            ggml_backend_meta_split_state ret = src_ss[0];
            for (size_t j = 0; j < n_bufs; j++) {
                ret.ne[j] = src_ss[0].ne[j] + src_ss[1].ne[j];
            }
            return ret;
        }
        return handle_generic(src_ss, /*scalar_only =*/ true);
    };

    auto handle_mul_mat = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            ggml_backend_meta_split_state ret = src_ss[0];
            ret.axis = GGML_BACKEND_SPLIT_AXIS_0;
            ret.nr[0] = 1;
            ret.n_segments = 1;
            return ret;
        }
        if (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_1 && src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[1];
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(split_states_equal(src_ss[0], src_ss[1]));
            return {assume_sync ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_PARTIAL, {0}, {1}, 1};
        }
        if (src_ss[0].axis == src_ss[1].axis && src_ss[0].axis >= GGML_BACKEND_SPLIT_AXIS_2 &&
                src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(split_states_equal(src_ss[0], src_ss[1]));
            return src_ss[0];
        }
        // batched matmul with the batches split across devices and a replicated activation
        if (src_ss[0].axis >= GGML_BACKEND_SPLIT_AXIS_2 && src_ss[0].axis < GGML_MAX_DIMS &&
                src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        GGML_ABORT("unsupported mul_mat split states: node=%s src0=%s axis=%d src1=%s axis=%d",
            tensor->name, tensor->src[0]->name, (int) src_ss[0].axis, tensor->src[1]->name, (int) src_ss[1].axis);
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    };

    auto handle_reshape = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        switch (src_ss[0].axis) {
            case GGML_BACKEND_SPLIT_AXIS_0:
            case GGML_BACKEND_SPLIT_AXIS_1:
            case GGML_BACKEND_SPLIT_AXIS_2:
            case GGML_BACKEND_SPLIT_AXIS_3: {
                int64_t base_ne_in = 1;
                for (int dim = 0; dim <= src_ss[0].axis; dim++) {
                    base_ne_in *= tensor->src[0]->ne[dim];
                }
                if (src_ss[0].n_segments == 1) {
                    base_ne_in /= src_ss[0].nr[0];
                    if (src_ss[0].axis == ggml_n_dims(tensor->src[0]) - 1 && src_ss[0].nr[0] == 1) {
                        const int64_t src_last = tensor->src[0]->ne[src_ss[0].axis];
                        const int64_t dst_last = tensor->ne[ggml_n_dims(tensor) - 1];
                        if (tensor->src[0]->op == GGML_OP_GET_ROWS && dst_last > 0 &&
                                src_last > dst_last && src_last % dst_last == 0 &&
                                tensor->ne[0] == tensor->src[0]->ne[0] * (src_last / dst_last)) {
                            return {GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1};
                        }
                        return {ggml_backend_meta_split_axis(ggml_n_dims(tensor) - 1), {0}, {1}, 1};
                    }
                    if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && tensor->ne[0] == tensor->src[0]->ne[0] &&
                            tensor->ne[1] == 1 && src_ss[0].nr[0] == 1) {
                        bool complete_rows = true;
                        for (size_t j = 0; j < n_bufs; j++) {
                            const int64_t ne = src_ss[0].ne[j];
                            complete_rows = complete_rows && (ne == 0 || ne == tensor->src[0]->ne[0]);
                        }
                        if (complete_rows) {
                            // Move a complete dim-0 split to the following singleton dimension.
                            return {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
                        }
                    }
                }
                // Reshape outputs use one segment; split-state propagation merges source segments.
                int64_t base_ne_out = 1;
                for (int dim = 0; dim < GGML_MAX_DIMS; dim++) {
                    base_ne_out *= tensor->ne[dim];
                    if (base_ne_out % base_ne_in == 0) {
                        return {ggml_backend_meta_split_axis(dim), {0}, {uint32_t(base_ne_out/base_ne_in)}, 1};
                    }
                    if (base_ne_out > base_ne_in) {
                        GGML_ASSERT(src_ss[0].n_segments == 1);
                        GGML_ASSERT(src_ss[0].nr[0]      == 1);
                        return {ggml_backend_meta_split_axis(dim), {0}, {1}, 1};
                    }
                }
                GGML_ABORT("shape mismatch for %s", ggml_op_name(tensor->op));
            }
            case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                return src_ss[0];
            }
            default: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    auto handle_cpy = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            return handle_reshape(src_ss);
        }
        // mixed-source copy: a replicated source copied into a row-split destination (e.g. the
        // recurrent conv/SSM state write-back at an island boundary layer: src0 = the mirrored
        // activation-side view, src1 = the per-head-split cache view). The data lands in the
        // destination, so the result follows the split side; treating it as MIRRORED (the
        // GGML_META_AXIS_UNKNOWN fallback) detaches the cache from its split axis and breaks
        // every downstream consumer that expects it (observed: GATED_DELTA_NET src[5] assert).
        const bool src0_synced = src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED ||
                                  src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL;
        const bool src1_split  = src_ss[1].axis >= 0 && src_ss[1].axis < GGML_MAX_DIMS;
        if (src0_synced && src1_split) {
            return src_ss[1];
        }
        return handle_generic(src_ss, /*scalar_only =*/ false);
    };

    auto handle_view = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (ggml_is_contiguous(tensor) && ggml_is_contiguous(tensor->src[0])) {
            return handle_reshape(src_ss);
        }
        const int axis = src_ss[0].axis;
        {
            bool all_strides_the_same = true;
            for (int dim = 0; dim < GGML_MAX_DIMS; dim++) {
                if (tensor->ne[dim] == 1 && tensor->src[0]->ne[dim] == 1) {
                    continue;
                }
                if (tensor->nb[dim] != tensor->src[0]->nb[dim]) {
                    all_strides_the_same = false;
                    break;
                }
            }
            if (all_strides_the_same) {
                return src_ss[0];
            }
        }
        if (!ggml_is_permuted(tensor) && !ggml_is_permuted(tensor->src[0]) && axis >= 0 && axis < GGML_MAX_DIMS-1) {
            for (int dim = 0; dim < GGML_MAX_DIMS-1; dim++) {
                if (tensor->nb[dim+1] == tensor->src[0]->nb[axis+1]) {
                    return {ggml_backend_meta_split_axis(dim), {0}, {1}, 1};
                }
            }
            GGML_ABORT("fatal error");
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED || src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
            return src_ss[0];
        }
        GGML_ABORT("view of permuted tensor not implemented");
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    };

    auto handle_permute = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        switch (src_ss[0].axis) {
            case GGML_BACKEND_SPLIT_AXIS_0:
            case GGML_BACKEND_SPLIT_AXIS_1:
            case GGML_BACKEND_SPLIT_AXIS_2:
            case GGML_BACKEND_SPLIT_AXIS_3: {
                GGML_ASSERT(src_ss[0].n_segments == 1 || src_ss[0].nr[0] == 1);
                return {ggml_backend_meta_split_axis(tensor->op_params[src_ss[0].axis]), {0}, {src_ss[0].nr[0]}, 1};
            }
            case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                return src_ss[0];
            }
            default: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    auto handle_transpose = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        switch (src_ss[0].axis) {
            case GGML_BACKEND_SPLIT_AXIS_0:
            case GGML_BACKEND_SPLIT_AXIS_1: {
                GGML_ASSERT(src_ss[0].n_segments == 1 || src_ss[0].nr[0] == 1);
                return {ggml_backend_meta_split_axis(int(src_ss[0].axis) ^ 1), {0}, {src_ss[0].nr[0]}, 1};
            }
            case GGML_BACKEND_SPLIT_AXIS_2:
            case GGML_BACKEND_SPLIT_AXIS_3:
            case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                return src_ss[0];
            }
            default: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    auto handle_get_rows = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1 &&
                src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return {assume_sync ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_PARTIAL,
                    {0}, {1}, 1};
        }
        return handle_generic(src_ss, /*scalar_only =*/ true);
    };

    // A degenerate split state ({0} sums, e.g. returned by handle_gated_delta_net) carries no
    // per-segment sizes: treat it as "unspecified" so the sum check cannot reject a layout that
    // axis-level reasoning already accepted (island mode surfaces the GDN state write-back as
    // SET_ROWS for the first time; 6-way hides it behind the gdn->cpy graph fusion).
    auto split_states_compatible = [&](const ggml_backend_meta_split_state & a, const ggml_backend_meta_split_state & b) -> bool {
        if (a.axis != b.axis) {
            return false;
        }
        bool a_empty = true, b_empty = true;
        for (size_t j = 0; j < n_bufs && (a_empty || b_empty); j++) {
            int64_t sum_a = 0, sum_b = 0;
            for (size_t s = 0; s < a.n_segments; s++) sum_a += a.ne[s*n_bufs + j] * a.nr[s];
            for (size_t s = 0; s < b.n_segments; s++) sum_b += b.ne[s*n_bufs + j] * b.nr[s];
            a_empty = a_empty && sum_a == 0;
            b_empty = b_empty && sum_b == 0;
        }
        return a_empty || b_empty || split_states_equal(a, b);
    };

    auto handle_set_rows = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        if (!split_states_compatible(src_ss[0], src_ss[2])) {
            // v11: name everything - which cache tensor the scatter targets and where the values
            // come from decides whether a mirrored dst is legitimate bookkeeping (replicated
            // scratch / rollback slots) or a mis-downgraded cache view
            auto sums_str = [&](const ggml_backend_meta_split_state & ss) {
                std::string str;
                for (size_t j = 0; j < n_bufs && j < 6; j++) {
                    int64_t sum = 0;
                    for (size_t s = 0; s < ss.n_segments; s++) {
                        sum += ss.ne[s*n_bufs + j] * ss.nr[s];
                    }
                    str += (j ? " " : "") + std::to_string(sum);
                }
                return str;
            };
            GGML_ABORT("SET_ROWS %s incompatible: dst %s [%s] axis=%s nseg=%u sums=[%s] | "
                       "values %s [%s] axis=%s nseg=%u sums=[%s] | ids axis=%s",
                    tensor->name,
                    tensor->src[2] != nullptr ? tensor->src[2]->name : "(null)",
                    tensor->src[2] != nullptr ? ggml_op_name(tensor->src[2]->op) : "-",
                    ggml_backend_meta_split_axis_name(src_ss[2].axis), src_ss[2].n_segments, sums_str(src_ss[2]).c_str(),
                    tensor->src[0] != nullptr ? tensor->src[0]->name : "(null)",
                    tensor->src[0] != nullptr ? ggml_op_name(tensor->src[0]->op) : "-",
                    ggml_backend_meta_split_axis_name(src_ss[0].axis), src_ss[0].n_segments, sums_str(src_ss[0]).c_str(),
                    ggml_backend_meta_split_axis_name(src_ss[1].axis));
        }
        return src_ss[0];
    };

    auto handle_rope = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        return src_ss[0];
    };

    auto handle_pad = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 0] == 0);
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 1] == 0);
        }
        return src_ss[0];
    };

    auto handle_flash_attn_ext = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(tensor->src[3] == nullptr || src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

        // sparse QSA attention: q and the output are [D, n_head, n_tok], and K/V are cache views
        // [D, n_head_kv, n_cells]. src[5] contains block ids or tagged physical cell ids.
        if (tensor->src[5] != nullptr) {
            GGML_ASSERT(src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
                return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
            }
            GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1);
            GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_1 && src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_1);
            return {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
        }

        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            GGML_ASSERT(tensor->src[4] == nullptr || src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }

        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_2);
        const bool kv_split = src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_2 &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_2;
        const bool kv_mirrored = src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED;
        GGML_ASSERT(kv_split || kv_mirrored);
        GGML_ASSERT(tensor->src[4] == nullptr || src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_0);
        return {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
    };

    auto handle_lightning_indexer = [&](
            const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        for (size_t i = 0; i < 4; i++) {
            GGML_ASSERT(src_ss[i].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        }
        return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
    };

    auto handle_ssm_conv = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == src_ss[1].axis) {
            if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0) {
                return {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
            }
            if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1) {
                return {GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1};
            }
        }
        return handle_generic(src_ss, /*scalar_only =*/ false);
    };

    auto handle_gated_delta_net = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_1);
        // island boundary layer: the per-head inputs are split (axis 1) but the recurrent state
        // arrived replicated (src[5] MIRRORED, v6 AC922 finding). Accepted because heads are
        // independent: each device processes its own heads only, and the state it needs for them
        // is present in its replicated copy. init_tensor_impl injects this device's head offset
        // into op param slot 1 (0 for a split state) and the CUDA kernel indexes the state with
        // global head ids, so the deferred-gather path (reads the cache slice directly) and the
        // replicated-state path both stay correct. Output mapping is identical to the split case.
        const bool state_mirrored = src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED ||
                                    src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL;
        if (!state_mirrored) {
            GGML_ASSERT(src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_2 || src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_1 || src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_0);
        }
        // v10: with a SPLIT state (single-meta 6-way) keep the historical degenerate descriptor -
        // the cache write-back is covered by the ggml-cuda gdn->cpy graph fusion there, and the
        // dormant set_rows path must not change 6-way behavior (v8's real descriptor from the
        // src2 head basis tripped split_states_compatible against the cache-side segment basis).
        // With a REPLICATED state (island mode, fusion does not apply) return a real row
        // distribution derived from the head-split inputs so the set_rows scatter has non-empty
        // per-device source slices: ne[j] = tensor->ne[0] * heads_j / H_all.
        if (!state_mirrored) {
            return {GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1};
        }
        ggml_backend_meta_split_state ret = {GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1};
        {
            // v11: pick the head basis from the first input whose per-device distribution is
            // SELF-CONSISTENT (sum of per-device shares == that source's extent along its split
            // axis). v10 read src[2] unconditionally and an upstream-corrupted state there
            // (observed sums {0,2,2} against 48 heads) propagated straight into the descriptor
            // (values side [0,256,256], member 0 empty). If no input is consistent, abort with
            // every input dumped so the corruption is attributable.
            int      basis     = -1;
            int64_t  basis_all = 0;
            auto src_sums = [&](size_t i, int64_t * sums) {
                for (size_t j = 0; j < n_bufs; j++) {
                    int64_t sum = 0;
                    for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                        sum += src_ss[i].ne[s*n_bufs + j] * src_ss[i].nr[s];
                    }
                    sums[j] = sum;
                }
            };
            for (size_t i = 0; i < 5 && basis < 0; i++) {
                if (tensor->src[i] == nullptr || src_ss[i].axis < 0 || src_ss[i].axis >= GGML_MAX_DIMS) {
                    continue;
                }
                int64_t sums[GGML_BACKEND_META_MAX_DEVICES];
                src_sums(i, sums);
                int64_t total = 0;
                for (size_t j = 0; j < n_bufs; j++) {
                    total += sums[j];
                }
                if (total == tensor->src[i]->ne[src_ss[i].axis] && total > 0) {
                    basis     = (int) i;
                    basis_all = total;
                    for (size_t j = 0; j < n_bufs && j < GGML_BACKEND_META_MAX_DEVICES; j++) {
                        ret.ne[j] = tensor->ne[0] * sums[j] / basis_all;
                    }
                }
            }
            if (basis < 0 || basis_all <= 0) {
                std::string dump;
                for (size_t i = 0; i < 5; i++) {
                    if (tensor->src[i] == nullptr) {
                        continue;
                    }
                    int64_t sums[GGML_BACKEND_META_MAX_DEVICES];
                    src_sums(i, sums);
                    std::string s_str;
                    for (size_t j = 0; j < n_bufs && j < 6; j++) {
                        s_str += (j ? " " : "") + std::to_string(sums[j]);
                    }
                    dump += " src" + std::to_string(i) + " " + tensor->src[i]->name +
                        " axis=" + ggml_backend_meta_split_axis_name(src_ss[i].axis) +
                        " ne_axis=" + std::to_string(tensor->src[i]->ne[src_ss[i].axis >= 0 && src_ss[i].axis < GGML_MAX_DIMS ? src_ss[i].axis : 0]) +
                        " sums=[" + s_str + "];";
                }
                GGML_ABORT("gated delta net %s: no self-consistent head basis among its inputs -%s "
                           "(an upstream split state is corrupted; run with GGML_META_DEBUG=1 to trace)",
                        tensor->name, dump.c_str());
            }
            int64_t ret_total = 0;
            for (size_t j = 0; j < n_bufs; j++) {
                ret_total += ret.ne[j];
            }
            if (ret_total != tensor->ne[0]) {
                GGML_ABORT("gated delta net %s: head split is not row-aligned (total %" PRId64 " != %" PRId64 ")",
                        tensor->name, ret_total, tensor->ne[0]);
            }
        }
        return ret;
    };

    auto calculate_split_state = [&]() -> ggml_backend_meta_split_state {
        if (ggml_nelements(tensor) == 0) {
            // empty tensors (e.g. the n_outputs == 0 gather) are semantically replicated noops:
            // mirroring them keeps downstream axis resolution alive instead of poisoning it with
            // UNKNOWN (v5 finding from the AC922 island run: UNKNOWN here aborts init)
            return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }
        if (ggml_backend_buffer_get_usage(tensor->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE && tensor->view_src == nullptr) {
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer));
            const ggml_backend_meta_device_context * dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
            ggml_backend_meta_split_state ret = dev_ctx->get_split_state(tensor, dev_ctx->get_split_state_ud);
            if (ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
                const int64_t granularity = ret.axis == GGML_BACKEND_SPLIT_AXIS_0 ? ggml_blck_size(tensor->type) : 1;
                int64_t ne_sum = 0;
                for (size_t s = 0; s < ret.n_segments; s++) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        GGML_ASSERT(ret.ne[s*n_bufs + j] % granularity == 0);
                        ne_sum += ret.ne[s*n_bufs + j] * ret.nr[s];
                    }
                }
                GGML_ASSERT(ne_sum == tensor->ne[ret.axis]);
            } else if (ret.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                GGML_ASSERT(ret.n_segments == 1);
                GGML_ASSERT(ret.nr[0] == 1);
            }
            return ret;
        }

        // Scheduler allocation dependencies on NONE nodes do not describe a compute layout.
        if (tensor->op == GGML_OP_NONE) {
            return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }
        std::vector<ggml_backend_meta_split_state> src_ss(GGML_MAX_SRC, {GGML_BACKEND_SPLIT_AXIS_NONE, {0}, {1}, 1});
        for (size_t i = 0; i < GGML_MAX_SRC; i++) {
            if (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                src_ss[i] = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
                continue;
            }
            src_ss[i] = ggml_backend_meta_get_split_state(stc, tensor->src[i], /*assume_sync =*/ true);
            if (src_ss[i].axis == GGML_BACKEND_SPLIT_AXIS_UNKNOWN) {
                GGML_ABORT("source %s [%s, type %s] of tensor %s [%s] has no split state (empty tensor or unresolved axis); "
                           "run with GGML_META_DEBUG=1 to trace the chain",
                        tensor->src[i]->name, ggml_op_name(tensor->src[i]->op), ggml_type_name(tensor->src[i]->type),
                        tensor->name, ggml_op_name(tensor->op));
            }
        }

        ggml_backend_meta_split_state split_state;
        switch (tensor->op) {
            case GGML_OP_NONE: {
                split_state = {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
            } break;
            case GGML_OP_DUP: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_ADD:
            case GGML_OP_ADD_ID: {
                split_state = handle_bin_bcast(src_ss);
            } break;
            case GGML_OP_ADD1:
            case GGML_OP_ACC: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SUB:
            case GGML_OP_MUL:
            case GGML_OP_DIV: {
                split_state = handle_bin_bcast(src_ss);
            } break;
            case GGML_OP_SQR:
            case GGML_OP_SQRT:
            case GGML_OP_LOG:
            case GGML_OP_SIN:
            case GGML_OP_COS: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_SUM: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SUM_ROWS:
            case GGML_OP_CUMSUM:
            case GGML_OP_MEAN:
            case GGML_OP_ARGMAX:
            case GGML_OP_COUNT_EQUAL: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_HC_FUSED: {
                // the hyper-connection streams and their weights are mirrored
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_ARGMAX_SPLIT: {
                // each device writes its own slot: rows split along dim 0 give a partial result that the AllReduce combines
                if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0) {
                    GGML_ASSERT(src_ss[0].n_segments == 1);
                    split_state = {assume_sync ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_PARTIAL, {0}, {1}, 1};
                } else {
                    GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
                    split_state = src_ss[0];
                }
            } break;
            case GGML_OP_REPEAT:
            case GGML_OP_REPEAT_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_CONCAT: {
                split_state = handle_concat(src_ss);
            } break;
            case GGML_OP_SILU_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_NORM:
            case GGML_OP_RMS_NORM:
            case GGML_OP_RMS_NORM_BACK:
            case GGML_OP_GROUP_NORM:
            case GGML_OP_L2_NORM: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_MUL_MAT:
            case GGML_OP_MUL_MAT_ID: {
                split_state = handle_mul_mat(src_ss);
            } break;
            case GGML_OP_OUT_PROD: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SCALE: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_SET: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_CPY: {
                split_state = handle_cpy(src_ss);
            } break;
            case GGML_OP_CONT:
            case GGML_OP_RESHAPE: {
                split_state = handle_reshape(src_ss);
            } break;
            case GGML_OP_VIEW: {
                split_state = handle_view(src_ss);
            } break;
            case GGML_OP_PERMUTE: {
                split_state = handle_permute(src_ss);
            } break;
            case GGML_OP_TRANSPOSE: {
                split_state = handle_transpose(src_ss);
            } break;
            case GGML_OP_GET_ROWS: {
                split_state = handle_get_rows(src_ss);
            } break;
            case GGML_OP_GET_ROWS_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SET_ROWS: {
                split_state = handle_set_rows(src_ss);
            } break;
            case GGML_OP_DIAG:
            case GGML_OP_DIAG_MASK_INF:
            case GGML_OP_DIAG_MASK_ZERO: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SOFT_MAX:
            case GGML_OP_SOFT_MAX_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_ROPE: {
                split_state = handle_rope(src_ss);
            } break;
            case GGML_OP_ROPE_BACK: {
                split_state = handle_rope(src_ss);
            } break;
            case GGML_OP_CLAMP: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_CONV_TRANSPOSE_1D:
            case GGML_OP_IM2COL:
            case GGML_OP_IM2COL_BACK:
            case GGML_OP_IM2COL_3D:
            case GGML_OP_CONV_2D:
            case GGML_OP_CONV_3D:
            case GGML_OP_CONV_2D_DW:
            case GGML_OP_CONV_TRANSPOSE_2D:
            case GGML_OP_POOL_1D:
            case GGML_OP_POOL_2D:
            case GGML_OP_POOL_2D_BACK:
            case GGML_OP_UPSCALE: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_PAD: {
                split_state = handle_pad(src_ss);
            } break;
            case GGML_OP_ARANGE: {
                // no sources: the same values on every device (the QSA indexer builds its visibility bias from it
                // when a prompt batch is scored in chunks, at long context)
                split_state = {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
            } break;
            case GGML_OP_PAD_REFLECT_1D:
            case GGML_OP_ROLL:
            case GGML_OP_TIMESTEP_EMBEDDING: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_ARGSORT:
            case GGML_OP_TOP_K: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_LEAKY_RELU: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_TRI: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_FILL: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_FLASH_ATTN_EXT: {
                split_state = handle_flash_attn_ext(src_ss);
            } break;
            case GGML_OP_FLASH_ATTN_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SSM_CONV: {
                split_state = handle_ssm_conv(src_ss);
            } break;
            case GGML_OP_SSM_SCAN:
            case GGML_OP_WIN_PART:
            case GGML_OP_WIN_UNPART:
            case GGML_OP_GET_REL_POS:
            case GGML_OP_ADD_REL_POS:
            case GGML_OP_RWKV_WKV6:
            case GGML_OP_GATED_LINEAR_ATTN:
            case GGML_OP_RWKV_WKV7:
            case GGML_OP_SOLVE_TRI: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_GATED_DELTA_NET: {
                split_state = handle_gated_delta_net(src_ss);
            } break;
            case GGML_OP_LIGHTNING_INDEXER: {
                split_state = handle_lightning_indexer(src_ss);
            } break;
            case GGML_OP_MONOMOE_SM70: {
                GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
                GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
                if (src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                    src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                    split_state = {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
                } else {
                    GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_1);
                    GGML_ASSERT(src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_0);
                    if (tensor->src[3]) GGML_ASSERT(split_states_equal(src_ss[2], src_ss[3]));
                    const int64_t packed = tensor->src[3] ? 1 : 2;
                    for (size_t j = 0; j < n_bufs; j++) {
                        int64_t g = 0, d = 0;
                        for (size_t k = 0; k < src_ss[2].n_segments; k++) g += src_ss[2].ne[k*n_bufs+j]*src_ss[2].nr[k];
                        for (size_t k = 0; k < src_ss[4].n_segments; k++) d += src_ss[4].ne[k*n_bufs+j]*src_ss[4].nr[k];
                        GGML_ASSERT(g == packed*d);
                    }
                    split_state = {assume_sync ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_PARTIAL, {0}, {1}, 1};
                }
            } break;
            case GGML_OP_PLE_CONV_SM70: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_DSV4_HC_COMB:
            case GGML_OP_DSV4_HC_PRE:
            case GGML_OP_DSV4_HC_POST: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_UNARY: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_MAP_CUSTOM1:
            case GGML_OP_MAP_CUSTOM2:
            case GGML_OP_MAP_CUSTOM3:
            case GGML_OP_CUSTOM: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_CROSS_ENTROPY_LOSS:
            case GGML_OP_CROSS_ENTROPY_LOSS_BACK: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_OPT_STEP_ADAMW:
            case GGML_OP_OPT_STEP_SGD:
            case GGML_OP_GLU: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            default: {
                GGML_ABORT("ggml op not implemented: %s", ggml_op_name(tensor->op));
                split_state = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            } break;
        }
        if (split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS) {
            bool first_src_split_by_axis = true;
            const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);

            // v8: a real-axis source whose per-device sums are all zero is an "axis-only"
            // (unspecified) descriptor - reshape/view-style handlers emit them and expect the
            // ratio takeover to fill the shares from an informative source. Treating zero-sum
            // sources as inert in BOTH the fill and the consistency check lets a degenerate
            // source coexist with an informative one (island mode feeds set_rows exactly that:
            // a degenerate GDN-side state against the real cache-side state). When ALL real-axis
            // sources are zero-sum the fill from the first one is kept, which reproduces the
            // previous behavior bit for bit.
            auto src_ss_zero_sum = [&](size_t i) {
                for (size_t j = 0; j < n_bufs; j++) {
                    int64_t sum = 0;
                    for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                        sum += src_ss[i].ne[s*n_bufs + j] * src_ss[i].nr[s];
                    }
                    if (sum != 0) {
                        return false;
                    }
                }
                return true;
            };

            for (size_t i = 0; i < GGML_MAX_SRC; i++) {
                if (tensor->src[i] == nullptr || src_ss[i].axis < 0 || src_ss[i].axis >= GGML_MAX_DIMS) {
                    continue;
                }
                if (first_src_split_by_axis && src_ss_zero_sum(i)) {
                    continue;
                }
                if (first_src_split_by_axis) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        // Take over ratio from src:
                        for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            split_state.ne[s*n_bufs + j] = 0;
                        }
                        for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            split_state.ne[j] += src_ss[i].ne[s*n_bufs + j] * src_ss[i].nr[s];
                        }
                        split_state.ne[j] *= tensor->ne[split_state.axis];
                        if (split_state.ne[j] != 0 || tensor->src[i]->ne[src_ss[i].axis] != 0) {
                            const int64_t div = tensor->src[i]->ne[src_ss[i].axis] * split_state.nr[0];
                            GGML_ASSERT(split_state.ne[j] % div == 0);
                            split_state.ne[j] /= div;
                        }
                    }
                } else {
                    // zero-sum sources carry no ratio, they impose no consistency constraint
                    if (src_ss_zero_sum(i)) {
                        continue;
                    }
                    GGML_ASSERT(split_state.n_segments == 1);
                    for (size_t j = 0; j < n_bufs; j++) {
                        // Assert that ratio is consistent:
                        int64_t sum = 0;
                        for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            sum += src_ss[i].ne[s*n_bufs + j] * src_ss[i].nr[s];
                        }
                        const int64_t lhs = split_state.ne[j]*split_state.nr[0] * tensor->src[i]->ne[src_ss[i].axis];
                        const int64_t rhs = sum * tensor->ne[split_state.axis];
                        if (lhs != rhs) {
                            auto describe = [&](const ggml_tensor * t, const ggml_backend_meta_split_state & ss) {
                                std::string shares;
                                for (size_t b = 0; b < n_bufs; b++) {
                                    int64_t share = 0;
                                    for (size_t s = 0; s < ss.n_segments; s++) {
                                        share += ss.ne[s*n_bufs + b]*ss.nr[s];
                                    }
                                    shares += (b ? " " : "") + std::to_string(share);
                                }
                                return std::string(t->name) + " [" + ggml_op_name(t->op) + ", " + ggml_type_name(t->type) +
                                    "] axis=" + ggml_backend_meta_split_axis_name(ss.axis) + " ne=[" +
                                    std::to_string(t->ne[0]) + " " + std::to_string(t->ne[1]) + " " +
                                    std::to_string(t->ne[2]) + " " + std::to_string(t->ne[3]) + "] sums=[" + shares + "]";
                            };
                            std::string sources;
                            for (size_t k = 0; k < GGML_MAX_SRC; k++) {
                                if (tensor->src[k] != nullptr) {
                                    sources += " src" + std::to_string(k) + "=" + describe(tensor->src[k], src_ss[k]);
                                }
                            }
                            GGML_ABORT("split ratio mismatch: %s member=%zu src=%zu lhs=%" PRId64 " rhs=%" PRId64
                                       " assume_sync=%d sources:%s",
                                    describe(tensor, split_state).c_str(), j, i, lhs, rhs, assume_sync, sources.c_str());
                        }
                    }
                }
                first_src_split_by_axis = false;
            }
            if (first_src_split_by_axis) {
                // every real-axis source was zero-sum: keep the historical fill-from-first so
                // configurations that used to resolve (to an all-zero distribution) still do
                for (size_t i = 0; i < GGML_MAX_SRC; i++) {
                    if (tensor->src[i] == nullptr || src_ss[i].axis < 0 || src_ss[i].axis >= GGML_MAX_DIMS) {
                        continue;
                    }
                    for (size_t j = 0; j < n_bufs; j++) {
                        for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            split_state.ne[s*n_bufs + j] = 0;
                        }
                        for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            split_state.ne[j] += src_ss[i].ne[s*n_bufs + j] * src_ss[i].nr[s];
                        }
                        split_state.ne[j] *= tensor->ne[split_state.axis];
                        if (split_state.ne[j] != 0 || tensor->src[i]->ne[src_ss[i].axis] != 0) {
                            const int64_t div = tensor->src[i]->ne[src_ss[i].axis] * split_state.nr[0];
                            GGML_ASSERT(split_state.ne[j] % div == 0);
                            split_state.ne[j] /= div;
                        }
                    }
                    first_src_split_by_axis = false;
                    break;
                }
            }
            GGML_ASSERT(!first_src_split_by_axis);
        }
        return split_state;
    };

    const std::pair key = std::make_pair(tensor, assume_sync);
    auto it = buf_ctx->split_state_cache.find(key);
    if (it != buf_ctx->split_state_cache.end() && memcmp(it->second.second, (const char *) tensor, sizeof(it->second.second)) != 0) {
        GGML_ABORT("meta layout changed during dependency resolution: %s [%s]", tensor->name, ggml_op_name(tensor->op));
    }

    if (it == buf_ctx->split_state_cache.end()) {
        buf_ctx->split_state_cache[key].first.assign(calculate_split_state(), n_bufs);
        // v12 pollution detector: fire at the PRODUCTION point. A split state where exactly one
        // member holds zero rows while every other member holds nonzero rows is the pollution
        // signature observed on AC922 ([0 256 256] / {0,2,2} heads); legitimate granularity
        // round-downs instead concentrate the tensor on one member ({0,0,512}-style, i.e. more
        // than one zero) and never leave exactly one member empty between filled ones. With the
        // assert at the producing tensor the corrupting handler is one GGML_META_DEBUG=1 chain
        // step away, instead of the failure surfacing at a far-away consumer (set_rows abort).
        {
            // v13: enforce the REAL invariant at the production point: the per-device shares must
            // sum to the tensor's extent along the split axis. This is what the debug-only
            // assert below checks, compiled out in release builds - and the AC922 island runs
            // showed exactly that failure ({0,2,2} heads against 48). Legitimately uneven
            // layouts from the callback's granularity rounding ([0 6144 6144]: member 0 holds
            // zero whole-granularity chunks) DO sum correctly and pass - the v12
            // "exactly-one-zero-member" heuristic fired on those by mistake and is removed.
            const ggml_backend_meta_split_state & produced = buf_ctx->split_state_cache[key].first;
            if (produced.axis >= 0 && produced.axis < GGML_MAX_DIMS && n_bufs > 0) {
                int64_t total = 0;
                for (size_t j = 0; j < n_bufs; j++) {
                    int64_t sum = 0;
                    for (size_t s = 0; s < produced.n_segments; s++) {
                        sum += produced.ne[s*n_bufs + j] * produced.nr[s];
                    }
                    total += sum;
                }
                if (total != tensor->ne[produced.axis]) {
                    // Keep the exact invariant and error details; avoid formatting
                    // per-device sums on the successful allocation hot path.
                    std::string sums_str;
                    for (size_t j = 0; j < n_bufs; j++) {
                        int64_t sum = 0;
                        for (size_t s = 0; s < produced.n_segments; s++) {
                            sum += produced.ne[s*n_bufs + j] * produced.nr[s];
                        }
                        sums_str += (j ? " " : "") + std::to_string(sum);
                    }
                    std::string srcs_info;
                    for (size_t i = 0; i < GGML_MAX_SRC; i++) {
                        if (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                            continue;
                        }
                        srcs_info += std::string(" src") + std::to_string(i) + "=" + tensor->src[i]->name +
                            "[" + ggml_op_name(tensor->src[i]->op) + "];";
                    }
                    GGML_ABORT("split state invariant violated at production point: tensor %s [%s] axis=%s ne=[%" PRId64 " %" PRId64 " %" PRId64 " %" PRId64 "] "
                               "per-device sums=[%s] total %" PRId64 " != ne[axis] %" PRId64 ";%s "
                               "(rerun with GGML_META_DEBUG=1 to trace the producing chain)",
                            tensor->name, ggml_op_name(tensor->op),
                            ggml_backend_meta_split_axis_name(produced.axis),
                            tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
                            sums_str.c_str(), total, tensor->ne[produced.axis], srcs_info.c_str());
                }
            }
        }
        memcpy(buf_ctx->split_state_cache[key].second, tensor, sizeof(buf_ctx->split_state_cache[key].second));
        if (buf_ctx->debug > 0) {
            std::string srcs_info;
            for (size_t i = 0; i < GGML_MAX_SRC; i++) {
                if (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                    continue;
                }
                if (!srcs_info.empty()) {
                    srcs_info += ", ";
                }
                const ggml_tensor * src = tensor->src[i];
                if (!ggml_backend_buffer_is_meta(src->buffer)) {
                    srcs_info += std::string(src->name) + "[non-meta]";
                    continue;
                }
                auto * src_ctx = (ggml_backend_meta_buffer_context *) src->buffer->context;
                auto src_it = src_ctx->split_state_cache.find({src, true});
                if (src_it == src_ctx->split_state_cache.end()) {
                    srcs_info += std::string(src->name) + "[uncached]";
                    continue;
                }
                // Logging must not resolve layouts or invalidate an in-progress cache.
                const ggml_backend_meta_split_state & split_state = src_it->second.first;
                const char * axis_name = ggml_backend_meta_split_axis_name(split_state.axis);
                std::string ne_info;
                for (size_t j = 0; j < n_bufs; j++) {
                    if (!ne_info.empty()) {
                        ne_info += ", ";
                    }
                    int64_t sum = 0;
                    for (size_t seg = 0; seg < split_state.n_segments; seg++) {
                        sum += split_state.ne[seg*n_bufs + j] * split_state.nr[seg];
                    }
                    ne_info += std::to_string(sum);
                }
                srcs_info += std::string(tensor->src[i]->name) + "[" + ggml_op_name(tensor->src[i]->op) + ", " + axis_name + ", {" + ne_info + "}]";
            }
            std::string ne_info;
            for (size_t j = 0; j < n_bufs; j++) {
                if (!ne_info.empty()) {
                    ne_info += ", ";
                }
                const ggml_backend_meta_split_state & ss = buf_ctx->split_state_cache[key].first;
                ne_info += std::to_string(ss.ne[j]) + "x" + std::to_string(ss.nr[0]);
            }
            GGML_LOG_DEBUG("SPLIT_STATE: {%s} -> %s[%s, %s, {%s}]\n", srcs_info.c_str(), tensor->name, ggml_op_name(tensor->op),
                ggml_backend_meta_split_axis_name(buf_ctx->split_state_cache[key].first.axis), ne_info.c_str());
        }
    }

    ggml_backend_meta_split_state ret = buf_ctx->split_state_cache[key].first;
    if (ret.axis == GGML_BACKEND_SPLIT_AXIS_UNKNOWN) {
        GGML_ABORT("tensor %s [%s, type %s, ne %" PRId64 " x %" PRId64 " x %" PRId64 " x %" PRId64 "] has no split state "
                   "(empty tensor or unresolved axis); run with GGML_META_DEBUG=1 to trace the chain",
                tensor->name, ggml_op_name(tensor->op), ggml_type_name(tensor->type),
                tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    }
    GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_NONE);
#ifndef NDEBUG
    if (ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
        int64_t ne_ret = 0;
        for (size_t s = 0; s < ret.n_segments; s++) {
            for (size_t j = 0; j < n_bufs; j++) {
                ne_ret += ret.ne[s*n_bufs + j] * ret.nr[s];
            }
        }
        assert(ne_ret == tensor->ne[int(ret.axis)]);
    }
#endif // NDEBUG
    return ret;
}

// Resolve dependencies iteratively: the large split-state descriptors make deep recursive
// prefill graphs overflow an ordinary worker stack, especially with several meta devices.
static struct ggml_backend_meta_split_state ggml_backend_meta_get_split_state(
        ggml_backend_meta_simple_tensor_container & stc, const struct ggml_tensor * tensor, bool assume_sync) {
    auto cached = [](const ggml_tensor * t, bool sync) {
        GGML_ASSERT(t != nullptr && ggml_backend_buffer_is_meta(t->buffer));
        auto * ctx = (ggml_backend_meta_buffer_context *) t->buffer->context;
        auto it = ctx->split_state_cache.find({t, sync});
        if (it == ctx->split_state_cache.end()) {
            return false;
        }
        if (memcmp(it->second.second, t, sizeof(it->second.second)) != 0) {
            // Invalidate before walking sources, never from inside the layout calculator.
            ctx->split_state_cache.clear();
            return false;
        }
        return true;
    };
    if (cached(tensor, assume_sync)) {
        auto * ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
        return ctx->split_state_cache.at({tensor, assume_sync}).first;
    }
    struct frame { const ggml_tensor * tensor; int next_src; };
    std::vector<frame> pending {{tensor, 0}};
    std::set<const ggml_tensor *> active {tensor};
    while (!pending.empty()) {
        frame & f = pending.back();
        const ggml_tensor * t = f.tensor;
        const bool leaf = ggml_nelements(t) == 0 || t->op == GGML_OP_NONE ||
            (ggml_backend_buffer_get_usage(t->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE && t->view_src == nullptr);
        if (!leaf && f.next_src < GGML_MAX_SRC) {
            const ggml_tensor * src = t->src[f.next_src++];
            if (src == nullptr || src == t || cached(src, true)) {
                continue;
            }
            if (!active.insert(src).second) {
                GGML_ABORT("cyclic meta layout dependency: %s [%s] -> %s [%s]",
                        t->name, ggml_op_name(t->op), src->name, ggml_op_name(src->op));
            }
            pending.push_back({src, 0});
            continue;
        }
        const bool sync = t == tensor ? assume_sync : true;
        ggml_backend_meta_get_split_state_impl(stc, t, sync);
        active.erase(t);
        pending.pop_back();
    }
    return ggml_backend_meta_get_split_state_impl(stc, tensor, assume_sync);
}

static struct ggml_backend_meta_split_state ggml_backend_meta_get_split_state(const struct ggml_tensor * tensor, bool assume_sync) {
    GGML_ASSERT(tensor != nullptr && ggml_backend_buffer_is_meta(tensor->buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    return ggml_backend_meta_get_split_state(buf_ctx->get_simple_tensor_container(tensor), tensor, assume_sync);
}

static void * ggml_backend_meta_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    return (void *) 0x1000000000000000; // FIXME
}

// Aggregate CPU initialization attribution; never changes layout or owners.
static void ggml_backend_meta_init_time_record(const void * buffer, int64_t split_us,
                                               int64_t materialize_us, size_t descriptors) {
    struct totals { size_t calls = 0, descriptors = 0; int64_t split_us = 0, materialize_us = 0; };
    static std::mutex mutex;
    static std::map<const void *, totals> values;
    std::lock_guard<std::mutex> lock(mutex);
    auto & value = values[buffer];
    ++value.calls; value.descriptors += descriptors;
    value.split_us += split_us; value.materialize_us += materialize_us;
    if (value.calls % 1024 == 0) {
        fprintf(stderr, "meta-init-timing buffer=%p callbacks=%zu descriptors=%zu split_state_us=%" PRId64
                " materialize_backend_init_us=%" PRId64 " total_us=%" PRId64 "\n",
                buffer, value.calls, value.descriptors, value.split_us, value.materialize_us,
                value.split_us + value.materialize_us);
    }
}

static enum ggml_status ggml_backend_meta_buffer_init_tensor_impl(ggml_backend_meta_simple_tensor_container & stc, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    const size_t n_simple_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);
    static const bool init_timing = [] {
        const char * value = std::getenv("GGML_META_INIT_TIMING");
        return value && std::strcmp(value, "1") == 0;
    }();
    const int64_t init_start = init_timing ? ggml_time_us() : 0;

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(stc, tensor, /*assume_sync =*/ true);
    const int64_t split_done = init_timing ? ggml_time_us() : 0;
    GGML_ASSERT(ggml_nelements(tensor) == 0 || split_state.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
    GGML_ASSERT(split_state.n_segments <= 16);

    int split_dim = split_state.axis;
    int64_t ne[GGML_MAX_DIMS];
    size_t  nb[GGML_MAX_DIMS];
    for (size_t k = 0; k < GGML_MAX_DIMS; k++) {
        ne[k] = tensor->ne[k];
        nb[k] = tensor->nb[k];
    }

    std::vector<ggml_tensor *> simple_tensors;
    simple_tensors.reserve(n_simple_bufs);
    for (size_t j = 0; j < n_simple_bufs; j++) {
        ggml_context          * simple_ctx = stc.ctxs[j].get();
        ggml_backend_buffer_t   simple_buf = buf_ctx->bufs[j].get();

        if (split_dim >= 0 && split_dim < GGML_MAX_DIMS) {
            // TODO: the following assert fails for llama-parallel even though the results are correct:
            // GGML_ASSERT(ggml_is_contiguously_allocated(tensor));
            ne[split_dim] = 0;
            for (size_t s = 0; s < split_state.n_segments; s++) {
                ne[split_dim] += split_state.ne[s*n_simple_bufs + j] * split_state.nr[s];
            }
            for (int i = 0; i < GGML_MAX_DIMS; i++) {
                if (tensor->nb[i] > tensor->nb[split_dim]) {
                    nb[i] = tensor->nb[i] * ne[split_dim]/tensor->ne[split_dim];
                }
            }
        }

        ggml_tensor * t_ij = ggml_new_tensor(simple_ctx, tensor->type, GGML_MAX_DIMS, ne);
        t_ij->op = tensor->op;
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            t_ij->nb[i] = nb[i];
        }
        t_ij->flags = tensor->flags;
        memcpy(t_ij->op_params, tensor->op_params, sizeof(tensor->op_params));
        if (tensor->op == GGML_OP_ARGMAX_SPLIT) {
            // this device's rank and the first column of its slice
            const ggml_backend_meta_split_state ss_src = ggml_backend_meta_get_split_state(stc, tensor->src[0], /*assume_sync =*/ true);
            if (ss_src.axis == GGML_BACKEND_SPLIT_AXIS_0) {
                GGML_ASSERT(ss_src.n_segments == 1);
                GGML_ASSERT(ggml_get_op_params_i32(tensor, 0) == (int32_t) n_simple_bufs);
                int64_t offset = 0;
                for (size_t jj = 0; jj < j; jj++) {
                    offset += ss_src.ne[jj] * ss_src.nr[0];
                }
                ggml_set_op_params_i32(t_ij, 1, (int32_t) j);
                ggml_set_op_params_i32(t_ij, 2, (int32_t) offset);
                ggml_set_op_params_i32(t_ij, 3, 1);
            }
        }
        if (tensor->op == GGML_OP_GET_ROWS && tensor->src[0] != nullptr) {
            const ggml_backend_meta_split_state ss_src =
                ggml_backend_meta_get_split_state(stc, tensor->src[0], /*assume_sync =*/ true);
            if (ss_src.axis == GGML_BACKEND_SPLIT_AXIS_1) {
                int64_t row_offset = 0;
                for (size_t s = 0; s < ss_src.n_segments; s++) {
                    for (size_t jj = 0; jj < j; jj++) {
                        row_offset += ss_src.ne[s*n_simple_bufs + jj] * ss_src.nr[s];
                    }
                }
                GGML_ASSERT(row_offset <= INT32_MAX);
                ggml_set_op_params_i32(t_ij, 0, (int32_t) row_offset);
                ggml_set_op_params_i32(t_ij, 1, 1);
            }
        }
        if (tensor->op == GGML_OP_GATED_DELTA_NET && tensor->src[2] != nullptr && tensor->src[5] != nullptr) {
            // island boundary layer: per-head inputs split on axis 1, recurrent state replicated.
            // Each device must index the replicated state with its GLOBAL head id; record this
            // device's head offset (cumulative heads of src_v on the earlier devices) in op param
            // slot 1 (slot 0 holds K). For a split state nothing is injected and the kernel's
            // offset stays 0, which keeps the single-meta behavior bit-identical.
            const ggml_backend_meta_split_state ss_state =
                ggml_backend_meta_get_split_state(stc, tensor->src[5], /*assume_sync =*/ true);
            const ggml_backend_meta_split_state ss_v =
                ggml_backend_meta_get_split_state(stc, tensor->src[2], /*assume_sync =*/ true);
            const bool state_replicated = ss_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED ||
                                          ss_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL;
            const bool inputs_head_split = ss_v.axis == GGML_BACKEND_SPLIT_AXIS_1;
            if (state_replicated && inputs_head_split && j > 0) {
                int64_t head_offset = 0;
                for (size_t s = 0; s < ss_v.n_segments; s++) {
                    for (size_t jj = 0; jj < j; jj++) {
                        head_offset += ss_v.ne[s*n_simple_bufs + jj] * ss_v.nr[s];
                    }
                }
                GGML_ASSERT(head_offset <= INT32_MAX);
                ggml_set_op_params_i32(t_ij, 1, (int32_t) head_offset);
            }
        }
        ggml_set_name(t_ij, tensor->name);
        t_ij->buffer = simple_buf;
        t_ij->view_src = tensor->view_src;
        t_ij->view_offs = tensor->view_offs;
        if (t_ij->view_src != nullptr && ggml_backend_buffer_is_meta(t_ij->view_src->buffer)) {
            t_ij->view_src = ggml_backend_meta_buffer_simple_tensor(tensor->view_src, j);
            if (t_ij->view_offs > 0 && split_dim >= 0 && split_dim < GGML_MAX_DIMS) {
                GGML_ASSERT(tensor->ne[split_dim] != 0);
                const int split_dim_view_src = ggml_backend_meta_get_split_state(tensor->view_src, /*assume_sync =*/ true).axis;
                GGML_ASSERT(split_dim_view_src >= 0 && split_dim_view_src < GGML_MAX_DIMS);

                // The offset can be internal to the data split, in those cases the view offset should not be scaled.
                // If however, the offset is larger than the data split then it needs to be scaled proportionally.
                bool split_internal_offset = t_ij->view_offs <= tensor->view_src->nb[split_dim_view_src];
                for (int i = 0; i < GGML_MAX_DIMS; i++) {
                    const size_t dim_size = tensor->ne[i] * tensor->nb[i];
                    if (tensor->view_offs <= dim_size && dim_size < tensor->nb[split_dim]) {
                        split_internal_offset = true;
                        break;
                    }
                }
                if (!split_internal_offset) {
                    t_ij->view_offs = t_ij->view_offs * ne[split_dim]/tensor->ne[split_dim];
                }
            }
        }
        // TODO: revisit once the graph allocator has been refactored, see https://github.com/ggml-org/llama.cpp/pull/25051#issuecomment-4842873396
        ggml_backend_buffer_t init_buf = simple_buf;
        if (t_ij->view_src != nullptr) {
            t_ij->data = (char *) t_ij->view_src->data + t_ij->view_offs;
            // views inherit the source slice's concrete sub-buffer (issue 22197)
            if (tensor->view_src != nullptr && ggml_backend_buffer_is_meta(tensor->view_src->buffer)
                    && t_ij->view_src->buffer != nullptr) {
                t_ij->buffer = t_ij->view_src->buffer;
                init_buf     = t_ij->view_src->buffer;
            }
        } else if (simple_buf != nullptr) {
            if (ggml_backend_buffer_is_multi_buffer(simple_buf)) {
                GGML_ABORT("multi buffers are not supported by the meta backend");
            }
            t_ij->data = (char *) ggml_backend_buffer_get_base(simple_buf)
                + size_t(tensor->data) - size_t(ggml_backend_buffer_get_base(tensor->buffer));
        }

        if (init_buf) {
            // the backend that owns the buffer will set .extra
            ggml_backend_buffer_init_tensor(init_buf, t_ij);
        } else {
            t_ij->extra = tensor->extra;
        }

        for (int i = 0; i < GGML_MAX_SRC; i++) {
            t_ij->src[i] = tensor->src[i];
            if (tensor->src[i] == tensor) {
                t_ij->src[i] = t_ij;
            } else if (t_ij->src[i] != nullptr && ggml_backend_buffer_is_meta(t_ij->src[i]->buffer)) {
                t_ij->src[i] = ggml_backend_meta_buffer_simple_tensor(tensor->src[i], j);
            }
        }

        simple_tensors.push_back(t_ij);
    }

    // If one of the sources has a zero-sized slice, disable the computation:
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (tensor->src[i] == nullptr || !ggml_backend_buffer_is_meta(tensor->src[i]->buffer)) {
            continue;
        }

        const ggml_backend_meta_split_state split_state_src = ggml_backend_meta_get_split_state(tensor->src[i], /*assume_sync =*/ true);
        if (split_state_src.axis < 0 || split_state_src.axis >= GGML_MAX_DIMS) {
            continue;
        }
        for (size_t j = 0; j < n_simple_bufs; j++) {
            int64_t ne_sum = 0;
            for (size_t s = 0; s < split_state_src.n_segments; s++) {
                ne_sum += split_state_src.ne[s*n_simple_bufs + j] * split_state_src.nr[s];
            }
            if (ne_sum == 0) {
                simple_tensors[j]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
                // v8 detector: a set_rows whose source rows are empty on a device that owns
                // destination rows would silently skip the scatter there and leave that part of
                // the cache stale (numerically wrong from the next step on). Zeroed slices are
                // legitimate for pure compute shards, so only warn for the scatter op.
                if (tensor->op == GGML_OP_SET_ROWS && ggml_nelements(simple_tensors[j]) != 0) {
                    GGML_LOG_WARN("%s: SET_ROWS %s: device %zu owns destination rows but its source slice is empty - "
                                  "the cache would go stale on this device (degenerate GDN-side split state, NOTES 0.3)\n",
                            __func__, tensor->name, j);
                }
            }
        }
    }

    stc.simple_tensors[tensor] = simple_tensors;
    if (init_timing) ggml_backend_meta_init_time_record(tensor->buffer, split_done - init_start,
                                                      ggml_time_us() - split_done, n_simple_bufs);
    return GGML_STATUS_SUCCESS;
}

static enum ggml_status ggml_backend_meta_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    buf_ctx->stc_compute_index = buf_ctx->stc_compute_index_next;
    return ggml_backend_meta_buffer_init_tensor_impl(buf_ctx->get_simple_tensor_container(tensor), tensor);
}

static void ggml_backend_meta_buffer_memset_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    const ggml_backend_meta_split_state split_state =
            ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    if (split_state.n_segments != 1 || split_state.nr[0] != 1) {
        GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(split_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        std::vector<size_t> simple_offsets(n_bufs, 0);
        if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            const size_t row_stride = tensor->nb[1];
            GGML_ASSERT(offset % row_stride == 0);
            GGML_ASSERT(size   % row_stride == 0);
            const int64_t row_start = offset / row_stride;
            const int64_t row_count = size   / row_stride;
            GGML_ASSERT(row_start + row_count <= tensor->ne[1]);

            const int64_t blck_size = ggml_blck_size(tensor->type);
            for (size_t s = 0; s < split_state.n_segments; s++) {
                for (size_t r = 0; r < split_state.nr[s]; r++) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                        GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_size == 0);
                        const size_t nbytes = split_state.ne[s*n_bufs + j]/blck_size * tensor->nb[0];
                        for (int64_t row = 0; row < row_count; row++) {
                            ggml_backend_tensor_memset(simple_tensor, value,
                                    simple_offsets[j] + (row_start + row)*simple_tensor->nb[1], nbytes);
                        }
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            return;
        }

        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);

        const size_t row_stride = tensor->nb[2];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);
        const int64_t row_start = offset / row_stride;
        const int64_t row_count = size   / row_stride;
        GGML_ASSERT(row_start + row_count <= tensor->ne[2]);

        for (size_t s = 0; s < split_state.n_segments; s++) {
            for (size_t r = 0; r < split_state.nr[s]; r++) {
                for (size_t j = 0; j < n_bufs; j++) {
                    ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    const size_t nbytes = split_state.ne[s*n_bufs + j] * tensor->nb[1];
                    for (int64_t row = 0; row < row_count; row++) {
                        ggml_backend_tensor_memset(simple_tensor, value,
                                simple_offsets[j] + (row_start + row)*simple_tensor->nb[2], nbytes);
                    }
                    simple_offsets[j] += nbytes;
                }
            }
        }
        return;
    }

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        / chunk_size_full;
            const int64_t i_stop  = (offset + size) / chunk_size_full;
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size = simple_tensor->nb[split_state.axis + 1];
                if (chunk_size == 0) {
                    continue;
                }
                for (int64_t i = i_start; i < i_stop; i++) {
                    ggml_backend_tensor_memset(simple_tensor, value, i*chunk_size, chunk_size);
                }
            }
        } break;
        case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
            GGML_ASSERT(value == 0);
            [[fallthrough]];
        }
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ggml_backend_tensor_memset(simple_tensor, value, offset, size);
            }
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    if (split_state.n_segments != 1 || split_state.nr[0] != 1) {
        GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(split_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        size_t offset_data = 0;
        std::vector<size_t> simple_offsets(n_bufs, 0);
        if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            const size_t row_stride = tensor->nb[1];
            GGML_ASSERT(offset % row_stride == 0);
            GGML_ASSERT(size   % row_stride == 0);
            const int64_t row_start = offset / row_stride;
            const int64_t row_count = size   / row_stride;
            GGML_ASSERT(row_start + row_count <= tensor->ne[1]);

            const int64_t blck_size = ggml_blck_size(tensor->type);
            for (size_t s = 0; s < split_state.n_segments; s++) {
                for (size_t r = 0; r < split_state.nr[s]; r++) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                        GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_size == 0);
                        const size_t nbytes = split_state.ne[s*n_bufs + j]/blck_size * tensor->nb[0];
                        ggml_backend_tensor_set_2d(simple_tensor, (const char *) data + offset_data,
                            simple_offsets[j] + row_start * simple_tensor->nb[1], nbytes,
                            row_count, simple_tensor->nb[1], tensor->nb[1]);
                        offset_data       += nbytes;
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            GGML_ASSERT(offset_data*row_count == size);
            return;
        }
        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);

        const size_t row_stride = tensor->nb[2];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);
        const int64_t row_start = offset / row_stride;
        const int64_t row_count = size   / row_stride;
        GGML_ASSERT(row_start + row_count <= tensor->ne[2]);

        for (size_t s = 0; s < split_state.n_segments; s++) {
            for (size_t r = 0; r < split_state.nr[s]; r++) {
                for (size_t j = 0; j < n_bufs; j++) {
                    ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    const size_t nbytes = split_state.ne[s*n_bufs + j] * tensor->nb[1];
                    ggml_backend_tensor_set_2d(simple_tensor, (const char *) data + offset_data,
                        simple_offsets[j] + row_start * simple_tensor->nb[2], nbytes,
                        row_count, simple_tensor->nb[2], tensor->nb[2]);
                    offset_data       += nbytes;
                    simple_offsets[j] += nbytes;
                }
            }
        }
        GGML_ASSERT(offset_data*row_count == size);
        return;
    }

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                if (chunk_size_j == 0) {
                    continue;
                }
                const size_t simple_offset = i_start * chunk_size_j;
                if (i_stop - i_start == 1) {
                    // A single contiguous shard can exceed the device's 2D pitch limit.
                    ggml_backend_tensor_set(simple_tensor, (const char *) data + offset_j, simple_offset, chunk_size_j);
                } else {
                    ggml_backend_tensor_set_2d(simple_tensor, (const char *) data + offset_j, simple_offset, chunk_size_j, i_stop - i_start, chunk_size_j, chunk_size_full);
                }
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ggml_backend_tensor_set(simple_tensor, data, offset, size);
            }
        } break;
        case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
            GGML_ASSERT(tensor->type == GGML_TYPE_F32);
            GGML_ASSERT(offset % sizeof(float) == 0);
            GGML_ASSERT(size   % sizeof(float) == 0);
            const size_t n_values = size / sizeof(float);
            size_t n_contributors = 0;
            for (size_t j = 0; j < n_bufs; j++) {
                n_contributors += split_state.ne[j] != 0;
            }
            const bool has_contributor_mask = n_contributors != 0;
            if (!has_contributor_mask) {
                n_contributors = n_bufs;
            }
            std::vector<float> tmp(n_values);
            for (size_t i = 0; i < n_values; i++) {
                tmp[i] = ((const float *) data)[i] / n_contributors;
            }
            std::vector<float> zero;
            if (has_contributor_mask) {
                zero.resize(n_values, 0.0f);
            }
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const float * partial = has_contributor_mask && split_state.ne[j] == 0 ? zero.data() : tmp.data();
                ggml_backend_tensor_set(simple_tensor, partial, offset, size);
            }
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    if (split_state.n_segments != 1 || split_state.nr[0] != 1) {
        GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(split_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        size_t offset_data = 0;
        std::vector<size_t> simple_offsets(n_bufs, 0);
        if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            const size_t row_stride = tensor->nb[1];
            GGML_ASSERT(offset % row_stride == 0);
            GGML_ASSERT(size   % row_stride == 0);
            const int64_t row_start = offset / row_stride;
            const int64_t row_count = size   / row_stride;
            GGML_ASSERT(row_start + row_count <= tensor->ne[1]);

            const int64_t blck_size = ggml_blck_size(tensor->type);
            for (size_t s = 0; s < split_state.n_segments; s++) {
                for (size_t r = 0; r < split_state.nr[s]; r++) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                        GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_size == 0);
                        const size_t nbytes = split_state.ne[s*n_bufs + j]/blck_size * tensor->nb[0];
                        ggml_backend_tensor_get_2d(simple_tensor, (char *) data + offset_data,
                            simple_offsets[j] + row_start * simple_tensor->nb[1], nbytes,
                            row_count, simple_tensor->nb[1], tensor->nb[1]);
                        offset_data       += nbytes;
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            GGML_ASSERT(offset_data*row_count == size);
            return;
        }
        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);

        const size_t row_stride = tensor->nb[2];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);
        const int64_t row_start = offset / row_stride;
        const int64_t row_count = size   / row_stride;
        GGML_ASSERT(row_start + row_count <= tensor->ne[2]);

        for (size_t s = 0; s < split_state.n_segments; s++) {
            for (size_t r = 0; r < split_state.nr[s]; r++) {
                for (size_t j = 0; j < n_bufs; j++) {
                    const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    const size_t nbytes = split_state.ne[s*n_bufs + j] * tensor->nb[1];
                    ggml_backend_tensor_get_2d(simple_tensor, (char *) data + offset_data,
                        simple_offsets[j] + row_start * simple_tensor->nb[2], nbytes,
                        row_count, simple_tensor->nb[2], tensor->nb[2]);
                    offset_data       += nbytes;
                    simple_offsets[j] += nbytes;
                }
            }
        }
        GGML_ASSERT(offset_data*row_count == size);
        return;
    }

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_bufs; j++){
                const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                if (chunk_size_j == 0) {
                    continue;
                }
                const size_t simple_offset = i_start * chunk_size_j;
                ggml_backend_tensor_get_2d(simple_tensor, (char *) data + offset_j, simple_offset, chunk_size_j, i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, 0);
            ggml_backend_tensor_get(simple_tensor, data, offset, size);
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    const size_t n_buffers = ggml_backend_meta_buffer_n_bufs(buffer);
    for (size_t i = 0; i < n_buffers; i++) {
        ggml_backend_buffer_clear(ggml_backend_meta_buffer_simple_buffer(buffer, i), value);
    }
}

static void ggml_backend_meta_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    // Gallocr reuses tensor addresses across graph allocations. An unchanged parent
    // descriptor can otherwise retain a split computed from a changed child at the
    // same address. Weight buffers outside this reset retain their immutable cache.
    buf_ctx->split_state_cache.clear();
    for (size_t i = 0; i < buf_ctx->bufs.size(); i++) {
        ggml_backend_buffer_reset(ggml_backend_meta_buffer_simple_buffer(buffer, i));
    }
}

static const ggml_backend_buffer_i ggml_backend_meta_buffer_iface = {
    /* .free_buffer     = */ ggml_backend_meta_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_meta_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_meta_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_meta_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_meta_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_meta_buffer_get_tensor,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ ggml_backend_meta_buffer_clear,
    /* .reset           = */ ggml_backend_meta_buffer_reset,
};

bool ggml_backend_buffer_is_meta(ggml_backend_buffer_t buf) {
    return buf != nullptr && buf->iface.free_buffer == ggml_backend_meta_buffer_iface.free_buffer;
}

void ggml_backend_meta_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    for (size_t i = 0; i < buf_ctx->bufs.size(); i++) {
        if (buf_ctx->bufs[i]) {
            ggml_backend_buffer_set_usage(buf_ctx->bufs[i].get(), usage);
        }
    }
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    const ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024*ggml_tensor_overhead(), // FIXME
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_backend_meta_simple_tensor_container stc_static;
    ggml_backend_meta_simple_tensor_container stc_compute_0(params, n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_1(params, n_simple_bufts);

    size_t max_size = 0;
    std::vector<ggml_backend_buffer_t> bufs;
    bufs.reserve(n_simple_bufts);
    for (size_t i = 0; i < n_simple_bufts; i++) {
        bufs.push_back(ggml_backend_buft_alloc_buffer(ggml_backend_meta_buft_simple_buft(buft, i), size));
        GGML_ASSERT(bufs.back() != nullptr);
        max_size = std::max(max_size, ggml_backend_buffer_get_size(bufs.back()));
    }
    ggml_backend_meta_buffer_context * buf_ctx = new ggml_backend_meta_buffer_context(stc_static, stc_compute_0, stc_compute_1, bufs);

    return ggml_backend_buffer_init(buft, ggml_backend_meta_buffer_iface, buf_ctx, max_size);
}

struct ggml_backend_buffer * ggml_backend_meta_alloc_ctx_tensors_from_buft(struct ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    constexpr size_t compute_headroom = 16; // Maximum number of views per statically allocated tensor that can be created between evals.
    const ggml_init_params params_static = {
        /*.mem_size   =*/ ggml_get_mem_size(ctx),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    const ggml_init_params params_compute = {
        /*.mem_size   =*/ compute_headroom*ggml_get_mem_size(ctx),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_backend_meta_simple_tensor_container stc_static   (params_static,  n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_0(params_compute, n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_1(params_compute, n_simple_bufts);

    std::vector<ggml_backend_buffer_t> bufs(n_simple_bufts, nullptr);
    ggml_backend_meta_buffer_context * meta_buf_ctx = new ggml_backend_meta_buffer_context(stc_static, stc_compute_0, stc_compute_1, bufs);

    ggml_backend_buffer_t meta_buf = ggml_backend_buffer_init(buft, ggml_backend_meta_buffer_iface, meta_buf_ctx, 0);
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        t->buffer = meta_buf;
        ggml_backend_meta_buffer_init_tensor_impl(meta_buf_ctx->stc_static, t);
        t->data = (void *) 0x2000000000000000; // FIXME
    }
    for (size_t i = 0; i < n_simple_bufts; i++) {
        ggml_context * ctx = meta_buf_ctx->stc_static.ctxs[i].get();
        ggml_backend_buffer_type_t simple_buft = ggml_backend_meta_buft_simple_buft(buft, i);

        // If a ggml_context only has zero-sized tensors, ggml_backend_alloc_ctx_tensors_from_buft returns NULL.
        // For those edge cases, allocate a dummy buffer instead.
        bool any_nonzero_slice = false;
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            if (ggml_nelements(t) != 0) {
                any_nonzero_slice = true;
                break;
            }
        }
        if (any_nonzero_slice) {
            meta_buf_ctx->bufs[i].reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx, simple_buft));
        } else {
            meta_buf_ctx->bufs[i].reset(ggml_backend_buft_alloc_buffer(simple_buft, 0));
            for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
                t->buffer = meta_buf_ctx->bufs[i].get();
            }
        }
        if (meta_buf_ctx->bufs[i] == nullptr) {
            // v10: a hole here means the per-device context allocation failed without throwing
            // (observed once alongside a wrong island free-memory aggregate after a driver
            // glitch) - name the device and index instead of dereferencing a null buffer later
            GGML_ABORT("meta buffer of %s: member %zu context allocation returned null (%s)",
                    ggml_backend_buft_name(buft), i, any_nonzero_slice ? "real buffer" : "dummy buffer");
        }
        meta_buf->size = std::max(meta_buf->size, ggml_backend_buffer_get_size(meta_buf_ctx->bufs[i].get()));
    }
    return meta_buf;
}

//
// meta backend
//

static ggml_guid_t ggml_backend_meta_guid() {
    static ggml_guid guid = {0xf1, 0x0e, 0x34, 0xcf, 0x9c, 0x6f, 0x43, 0xcb, 0x96, 0x92, 0xbe, 0x8e, 0xbb, 0x71, 0x3f, 0xda};
    return &guid;
}

// runs jobs for one simple backend on its own host thread
struct ggml_backend_meta_worker {
    std::thread             thread;
    std::mutex              mutex;
    std::condition_variable cv;
    std::function<void()>   job;
    bool                    busy = false;
    bool                    stop = false;

    ggml_backend_meta_worker() : thread([this] { run(); }) {}

    ~ggml_backend_meta_worker() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stop = true;
        }
        cv.notify_all();
        thread.join();
    }

    void run() {
        std::unique_lock<std::mutex> lock(mutex);
        while (true) {
            cv.wait(lock, [this] { return busy || stop; });
            if (stop) {
                return;
            }
            lock.unlock();
            job();
            lock.lock();
            busy = false;
            cv.notify_all();
        }
    }

    void post(std::function<void()> fn) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            job  = std::move(fn);
            busy = true;
        }
        cv.notify_all();
    }

    void wait() {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [this] { return !busy; });
    }
};

typedef void * (*ggml_backend_p2p_ar_init_t)      (ggml_backend_t * backends, size_t n_backends);
typedef void   (*ggml_backend_p2p_ar_free_t)      (void * ar);
typedef bool   (*ggml_backend_p2p_ar_reserve_t)   (void * ar, size_t nbytes);
typedef void   (*ggml_backend_p2p_ar_set_params_t)(void * ar, size_t rank, struct ggml_tensor * node);
typedef bool   (*ggml_backend_p2p_ar_allreduce_t)(void * ar, ggml_backend_t * backends, size_t n_backends, struct ggml_tensor ** nodes);
typedef bool   (*ggml_backend_graph_ready_t)      (ggml_backend_t backend, const struct ggml_cgraph * cgraph);
typedef void * (*ggml_backend_cuda_meta_copy_init_t)(ggml_backend_t * src_backends, ggml_backend_t * dst_backends, size_t n_backends);
typedef void   (*ggml_backend_cuda_meta_copy_free_t)(void * copy_ctx);
typedef bool   (*ggml_backend_cuda_meta_copy_async_t)(void * copy_ctx, ggml_backend_t src_backend,
        const struct ggml_tensor * src, ggml_backend_t * dst_backends, struct ggml_tensor ** dst, size_t n_backends);

typedef void * (*ggml_backend_cuda_meta_span_init_t)(ggml_backend_t);
typedef void * (*ggml_backend_cuda_meta_span_begin_t)(void *, size_t, size_t);
typedef void   (*ggml_backend_cuda_meta_span_end_t)(void *, void *);
typedef void   (*ggml_backend_cuda_meta_span_free_t)(void *);

struct ggml_meta_span_scope {
    ggml_backend_cuda_meta_span_end_t end;
    std::vector<void *> & traces;
    std::vector<void *> tokens;
    ggml_meta_span_scope(ggml_backend_cuda_meta_span_begin_t begin, ggml_backend_cuda_meta_span_end_t end,
                        std::vector<void *> & traces, size_t sequence, size_t nodes) : end(end), traces(traces) {
        if (begin) {
            for (auto trace : traces) tokens.push_back(begin(trace, sequence, nodes));
        }
    }
    ~ggml_meta_span_scope() {
        for (size_t i = 0; i < tokens.size(); ++i) end(traces[i], tokens[i]);
    }
};

struct ggml_backend_meta_async_copy {
    void *                         ctx  = nullptr;
    ggml_backend_cuda_meta_copy_free_t free_fn = nullptr;
    ggml_backend_cuda_meta_copy_async_t copy = nullptr;
    size_t copies = 0;
};

struct ggml_backend_meta_context {
    struct cgraph_config {
        ggml_cgraph * cgraph_main = nullptr;
        int           offset      = 0; // Node offset vs. original graph

        std::vector<ggml_cgraph *> cgraphs_aux;
    };
    struct backend_config {
        ggml_backend_t backend;

        std::vector<cgraph_config>           cgraphs;
        std::vector<ggml_tensor *>           nodes;
        std::vector<ggml_backend_buffer_ptr> bufs;

        backend_config(ggml_backend_t backend, const size_t n_reduce_steps) : backend(backend) {
            bufs.resize(n_reduce_steps);
        }
    };
    std::vector<void *> gpu_spans;
    ggml_backend_cuda_meta_span_begin_t gpu_span_begin = nullptr;
    ggml_backend_cuda_meta_span_end_t gpu_span_end = nullptr;
    ggml_backend_cuda_meta_span_free_t gpu_span_free = nullptr;
    size_t gpu_span_sequence = 0;
    std::string                 name;
    std::vector<backend_config> backend_configs;
    ggml_context_ptr            ctx;
    std::vector<ggml_cgraph *>  cgraphs_aux;
    std::vector<ggml_tensor *>  nodes_aux;
    size_t                      n_reduce_steps;
    int                         max_nnodes    = 0;
    std::vector<size_t>          graph_node_caps;
    size_t                      max_tmp_size  = 0;
    size_t                      max_subgraphs = 0;
    size_t                      n_subgraphs   = 0;
    uint64_t                    uid           = 0;

    // Cached split graphs can outlive a rebuild in another context sharing their weights.
    std::vector<std::shared_ptr<ggml_backend_meta_simple_tensor_container>> tensor_arenas;

    void *                               comm_ctx       = nullptr;
    ggml_backend_comm_allreduce_tensor_t comm_allreduce = nullptr;

    // in-graph AllReduce: the whole graph runs as one graph per simple backend, each launched from its own thread
    void *                           p2p_ar            = nullptr;
    ggml_backend_p2p_ar_free_t       p2p_ar_free       = nullptr;
    ggml_backend_p2p_ar_reserve_t    p2p_ar_reserve    = nullptr;
    ggml_backend_p2p_ar_set_params_t p2p_ar_set_params = nullptr;
    ggml_backend_p2p_ar_allreduce_t  p2p_ar_allreduce  = nullptr;
    ggml_backend_graph_ready_t       graph_ready       = nullptr; // chunks are launched without lockstep only if this says so
    ggml_context_ptr                 ctx_merged;
    std::vector<ggml_cgraph *>       cgraphs_merged; // per simple backend, empty if the graph can not be merged
    // per simple backend: the merged graph cut into chunks, launched one after the other. Launching a CUDA graph costs
    //     host time per node and the GPU starts only when the launch returns, so with chunks the GPU starts early.
    std::vector<std::vector<ggml_cgraph *>> cgraphs_chunks;
    ggml_context_ptr                        ctx_chunks;
    bool                             merge_tried = false;
    // Optional host-staged MIRRORED copies into another Meta island. The source
    // Meta context owns each bridge so it can drain copies before its CUDA streams
    // are destroyed. The CUDA helper keeps only device IDs, pinned slots, and events.
    std::map<ggml_backend_t, ggml_backend_meta_async_copy> async_copies;
    std::mutex                         async_copies_mutex;
    // shapes that already ran once: a new shape first runs unmerged, from one thread. First runs do blocking driver calls
    //     (pool growth, lazy kernel loads) that can deadlock while a peer GPU spins in the AllReduce.
    std::set<uint64_t>               seen_shapes;
    std::vector<std::unique_ptr<ggml_backend_meta_worker>> workers; // for simple backends 1..n-1

    ggml_backend_meta_context(ggml_backend_dev_t meta_dev, const char * params) {
        const size_t n_devs = ggml_backend_meta_dev_n_devs(meta_dev);
        n_reduce_steps = std::ceil(std::log2(n_devs));
        name = "Meta(";
        std::vector<ggml_backend_t> simple_backends;
        backend_configs.reserve(n_devs);
        simple_backends.reserve(n_devs);
        for (size_t i = 0; i < n_devs; i++) {
            ggml_backend_dev_t simple_dev = ggml_backend_meta_dev_simple_dev(meta_dev, i);
            if (i > 0) {
                name += ",";
            }
            name += ggml_backend_dev_name(simple_dev);
            simple_backends.push_back(ggml_backend_dev_init(simple_dev, params));
            backend_configs.emplace_back(simple_backends.back(), n_reduce_steps);
        }
        name += ")";

        const char * spans_env = getenv("GGML_META_GPU_SPANS");
        if (spans_env && atoi(spans_env) == 1) {
            auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(simple_backends[0]));
            auto init = (ggml_backend_cuda_meta_span_init_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_meta_span_init");
            gpu_span_begin = (ggml_backend_cuda_meta_span_begin_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_meta_span_begin");
            gpu_span_end = (ggml_backend_cuda_meta_span_end_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_meta_span_end");
            gpu_span_free = (ggml_backend_cuda_meta_span_free_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_meta_span_free");
            if (init && gpu_span_begin && gpu_span_end && gpu_span_free) {
                for (auto child : simple_backends) gpu_spans.push_back(init(child));
                GGML_LOG_INFO("meta GPU stream spans active: %s; includes stream gaps, not kernel busy time\n", name.c_str());
            }
        }

        if (n_devs > 1) {
            ggml_backend_comm_init_t comm_init = (ggml_backend_comm_init_t) ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_init");
            if (comm_init != nullptr) {
                comm_ctx = comm_init(simple_backends.data(), simple_backends.size());
            }
        }
        if (n_devs > 1) {
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(simple_backends[0]));
            ggml_backend_p2p_ar_init_t p2p_ar_init = (ggml_backend_p2p_ar_init_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_p2p_ar_init");
            if (p2p_ar_init != nullptr) {
                p2p_ar = p2p_ar_init(simple_backends.data(), simple_backends.size());
            }
            if (p2p_ar != nullptr) {
                p2p_ar_free       = (ggml_backend_p2p_ar_free_t)       ggml_backend_reg_get_proc_address(reg, "ggml_backend_p2p_ar_free");
                p2p_ar_reserve    = (ggml_backend_p2p_ar_reserve_t)    ggml_backend_reg_get_proc_address(reg, "ggml_backend_p2p_ar_reserve");
                p2p_ar_set_params = (ggml_backend_p2p_ar_set_params_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_p2p_ar_set_params");
                graph_ready       = (ggml_backend_graph_ready_t)       ggml_backend_reg_get_proc_address(reg, "ggml_backend_graph_ready");
                GGML_ASSERT(p2p_ar_free && p2p_ar_reserve && p2p_ar_set_params);
                const char * cold = getenv("GGML_META_P2P_COLD");
                if (cold != nullptr && atoi(cold) != 0) {
                    p2p_ar_allreduce = (ggml_backend_p2p_ar_allreduce_t)
                        ggml_backend_reg_get_proc_address(reg, "ggml_backend_p2p_ar_allreduce");
                    GGML_ASSERT(p2p_ar_allreduce != nullptr);
                    GGML_LOG_INFO("%s: cold collectives use rank-ordered P2P AllReduce\n", __func__);
                }
                for (size_t i = 1; i < n_devs; i++) {
                    workers.emplace_back(new ggml_backend_meta_worker());
                }
                GGML_LOG_INFO("%s: using in-graph P2P AllReduce\n", __func__);
            }
        }
        if (comm_ctx != nullptr) {
            comm_allreduce = (ggml_backend_comm_allreduce_tensor_t)
                ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(
                    ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_allreduce_tensor");
            GGML_ASSERT(comm_allreduce != nullptr);
        }
    }

    ~ggml_backend_meta_context() {
        for (auto & copy : async_copies) {
            if (copy.second.ctx != nullptr && copy.second.free_fn != nullptr) {
                copy.second.free_fn(copy.second.ctx);
            }
        }
        async_copies.clear();
        workers.clear();
        for (auto trace : gpu_spans) gpu_span_free(trace);
        if (p2p_ar != nullptr) {
            p2p_ar_free(p2p_ar);
        }
        if (comm_ctx != nullptr) {
            ggml_backend_comm_free_t comm_free = (ggml_backend_comm_free_t) ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_configs[0].backend)), "ggml_backend_comm_free");
            GGML_ASSERT(comm_free != nullptr);
            comm_free(comm_ctx);
        }
        for (auto & bc : backend_configs) {
            ggml_backend_free(bc.backend);
        }
    }
};

static const char * ggml_backend_meta_get_name(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) backend->context;
    return backend_ctx->name.c_str();
}

static void ggml_backend_meta_free(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;
    delete backend_ctx;
    delete backend;
}

static void ggml_backend_meta_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    if (size == 0) {
        return; // empty tensors have no split state
    }
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(split_state.n_segments == 1);
    GGML_ASSERT(split_state.nr[0]      == 1);

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_backends; j++){
                ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                if (chunk_size_j == 0) {
                    continue;
                }
                ggml_backend_tensor_set_2d_async(simple_backend, simple_tensor, (const char *) data + offset_j, offset, chunk_size_j,
                    i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            for (size_t j = 0; j < n_backends; j++) {
                ggml_backend_tensor_set_async(
                    ggml_backend_meta_simple_backend(backend, j), ggml_backend_meta_buffer_simple_tensor(tensor, j), data, offset, size);
            }
        } break;
        default: {
            GGML_ABORT("%s: unsupported split axis %d for tensor %s", __func__, (int) split_state.axis, tensor->name);
        }
    }
}

static void ggml_backend_meta_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    if (size == 0) {
        return;
    }
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(split_state.n_segments == 1);
    GGML_ASSERT(split_state.nr[0]      == 1);

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_backends; j++){
                ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                if (chunk_size_j == 0) {
                    continue;
                }
                ggml_backend_tensor_get_2d_async(simple_backend, simple_tensor, (char *) data + offset_j, offset, chunk_size_j,
                    i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, 0);
            const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, 0);
            ggml_backend_tensor_get_async(simple_backend, simple_tensor, data, offset, size);
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static bool ggml_backend_meta_async_copy_enabled() {
    static const bool enabled = [] {
        const char * value = getenv("GGML_META_ASYNC_COPY");
        return value != nullptr && atoi(value) == 1;
    }();
    return enabled;
}

// Candidate-only path for a full, replicated activation crossing between two
// three-GPU Meta devices. Unsupported copies return false and use the existing
// synchronized backend fallback in ggml_backend_tensor_copy_async().
static bool ggml_backend_meta_cpy_tensor_async(
        ggml_backend_t backend_src, ggml_backend_t backend_dst,
        const ggml_tensor * src, ggml_tensor * dst) {
    constexpr size_t n_island_members = 3;

    if (!ggml_backend_meta_async_copy_enabled() || backend_src == backend_dst ||
            !ggml_backend_is_meta(backend_src) || !ggml_backend_is_meta(backend_dst) ||
            !ggml_backend_buffer_is_meta(src->buffer) || !ggml_backend_buffer_is_meta(dst->buffer) ||
            !ggml_are_same_layout(src, dst) || !ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        return false;
    }

    const ggml_backend_meta_split_state src_state = ggml_backend_meta_get_split_state(src, /*assume_sync =*/ false);
    const ggml_backend_meta_split_state dst_state = ggml_backend_meta_get_split_state(dst, /*assume_sync =*/ false);
    if (src_state.axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED ||
            dst_state.axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED ||
            src_state.n_segments != 1 || dst_state.n_segments != 1 ||
            src_state.nr[0] != 1 || dst_state.nr[0] != 1) {
        return false;
    }

    auto * src_ctx = (ggml_backend_meta_context *) backend_src->context;
    auto * dst_ctx = (ggml_backend_meta_context *) backend_dst->context;
    static const bool shared_device_copy = [] { const char * s = getenv("GGML_META_SHARED_DEVICE_COPY"); return s && atoi(s) != 0; }();
    if (shared_device_copy && src_ctx->backend_configs.size() > dst_ctx->backend_configs.size()) {
        std::vector<size_t> ranks;
        for (size_t j = 0; j < dst_ctx->backend_configs.size(); ++j) {
            const auto dev = ggml_backend_get_device(dst_ctx->backend_configs[j].backend);
            size_t match = src_ctx->backend_configs.size();
            for (size_t i = 0; i < src_ctx->backend_configs.size(); ++i) {
                if (ggml_backend_get_device(src_ctx->backend_configs[i].backend) == dev) { match = i; break; }
            }
            if (match == src_ctx->backend_configs.size()) return false;
            auto * source = ggml_backend_meta_buffer_simple_tensor(src, match);
            auto * target = ggml_backend_meta_buffer_simple_tensor(dst, j);
            if (!ggml_are_same_layout(source, target) || !ggml_are_same_layout(src, source)) return false;
            ranks.push_back(match);
        }
        // Protect the destination's previous frame before a source-stream copy overwrites it.
        ggml_backend_synchronize(backend_dst);
        for (size_t j = 0; j < ranks.size(); ++j) {
            const size_t i = ranks[j];
            ggml_backend_tensor_copy_async(src_ctx->backend_configs[i].backend, dst_ctx->backend_configs[j].backend,
                    ggml_backend_meta_buffer_simple_tensor(src, i), ggml_backend_meta_buffer_simple_tensor(dst, j));
        }
        static std::atomic<bool> logged{false};
        if (!logged.exchange(true, std::memory_order_relaxed)) GGML_LOG_INFO("meta shared-device copy active: %zu -> %zu members\n", src_ctx->backend_configs.size(), dst_ctx->backend_configs.size());
        return true;
    }
    if (src_ctx->backend_configs.size() != n_island_members || dst_ctx->backend_configs.size() != n_island_members) {
        return false;
    }

    ggml_backend_t src_backends[n_island_members];
    ggml_backend_t dst_backends[n_island_members];
    ggml_tensor * dst_simple_tensors[n_island_members];
    const ggml_tensor * src_simple_tensor = ggml_backend_meta_buffer_simple_tensor(src, 0);
    if (src_simple_tensor == nullptr || !ggml_is_contiguous(src_simple_tensor) ||
            !ggml_are_same_layout(src, src_simple_tensor)) {
        return false;
    }

    ggml_backend_reg_t cuda_reg = nullptr;
    for (size_t i = 0; i < n_island_members; i++) {
        src_backends[i] = ggml_backend_meta_simple_backend(backend_src, i);
        dst_backends[i] = ggml_backend_meta_simple_backend(backend_dst, i);
        dst_simple_tensors[i] = ggml_backend_meta_buffer_simple_tensor(dst, i);
        if (dst_simple_tensors[i] == nullptr || !ggml_is_contiguous(dst_simple_tensors[i]) ||
                !ggml_are_same_layout(dst, dst_simple_tensors[i])) {
            return false;
        }

        ggml_backend_reg_t src_reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(src_backends[i]));
        ggml_backend_reg_t dst_reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(dst_backends[i]));
        if (src_reg == nullptr || src_reg != dst_reg || (cuda_reg != nullptr && src_reg != cuda_reg)) {
            return false;
        }
        cuda_reg = src_reg;
    }

    const auto init = (ggml_backend_cuda_meta_copy_init_t)
        ggml_backend_reg_get_proc_address(cuda_reg, "ggml_backend_cuda_meta_copy_init");
    const auto free_fn = (ggml_backend_cuda_meta_copy_free_t)
        ggml_backend_reg_get_proc_address(cuda_reg, "ggml_backend_cuda_meta_copy_free");
    const auto copy = (ggml_backend_cuda_meta_copy_async_t)
        ggml_backend_reg_get_proc_address(cuda_reg, "ggml_backend_cuda_meta_copy_async");
    if (init == nullptr || free_fn == nullptr || copy == nullptr) {
        return false;
    }

    std::lock_guard<std::mutex> lock(src_ctx->async_copies_mutex);
    auto it = src_ctx->async_copies.find(backend_dst);
    if (it == src_ctx->async_copies.end()) {
        void * copy_ctx = init(src_backends, dst_backends, n_island_members);
        if (copy_ctx == nullptr) {
            // Remember unsupported topology/resource initialization so an opted-in
            // but ineligible pair does not repeat nine peer queries on every copy.
            src_ctx->async_copies.emplace(backend_dst, ggml_backend_meta_async_copy{nullptr, free_fn, copy});
            return false;
        }
        it = src_ctx->async_copies.emplace(backend_dst, ggml_backend_meta_async_copy{copy_ctx, free_fn, copy}).first;
    }

    if (it->second.ctx == nullptr || it->second.copy == nullptr) {
        return false;
    }
    const bool copied = it->second.copy(it->second.ctx, src_backends[0], src_simple_tensor,
            dst_backends, dst_simple_tensors, n_island_members);
    if (copied && it->second.copies++ == 0) {
        GGML_LOG_INFO("%s: async MIRRORED island copy active %s -> %s, bytes=%zu\n", __func__,
                      ggml_backend_name(backend_src), ggml_backend_name(backend_dst), ggml_nbytes(src));
    }
    return copied;
}

static void ggml_backend_meta_synchronize(ggml_backend_t backend) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    for (size_t i = 0; i < n_backends; i++) {
        ggml_backend_synchronize(ggml_backend_meta_simple_backend(backend, i));
    }
}

static void ggml_backend_meta_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_ASSERT(event->device == ggml_backend_get_device(backend));
    auto * ctx = (ggml_backend_meta_event_context *) event->context;
    for (size_t i = 0; i < ctx->events.size(); ++i) {
        ggml_backend_event_record(ctx->events[i], ggml_backend_meta_simple_backend(backend, i));
    }
}

static void ggml_backend_meta_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    if (event->device != ggml_backend_get_device(backend)) {
        static const bool gpu_wait = [] {
            const char * env = getenv("GGML_META_GPU_EVENT_WAIT");
            return env && atoi(env) == 1;
        }();
        if (gpu_wait && ggml_backend_dev_type(event->device) == GGML_BACKEND_DEVICE_TYPE_META) {
            auto * source = (ggml_backend_meta_event_context *) event->context;
            const size_t n_backends = ggml_backend_meta_n_backends(backend);
            bool cuda_only = !source->events.empty();
            for (auto child : source->events) {
                cuda_only = cuda_only && strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(child->device)), "CUDA") == 0;
            }
            for (size_t i = 0; i < n_backends; ++i) {
                const auto dev = ggml_backend_get_device(ggml_backend_meta_simple_backend(backend, i));
                cuda_only = cuda_only && strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)), "CUDA") == 0;
            }
            if (cuda_only) {
                for (size_t i = 0; i < n_backends; ++i) {
                    for (auto child : source->events) {
                        ggml_backend_event_wait(ggml_backend_meta_simple_backend(backend, i), child);
                    }
                }
                static std::atomic<bool> logged{false};
                if (!logged.exchange(true, std::memory_order_relaxed)) {
                    GGML_LOG_INFO("meta cross-device event wait active: CUDA stream dependencies\n");
                }
                return;
            }
        }
        ggml_backend_event_synchronize(event);
        return;
    }
    auto * ctx = (ggml_backend_meta_event_context *) event->context;
    for (size_t i = 0; i < ctx->events.size(); ++i) {
        ggml_backend_event_wait(ggml_backend_meta_simple_backend(backend, i), ctx->events[i]);
    }
}

static enum ggml_status ggml_backend_meta_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(cgraph->grads == nullptr);
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;

    // If the previous cgraph had a defined UID it can be used to skip rebuilding the subgraphs per simple backend.
    const bool needs_rebuild = (cgraph->uid == 0) || (cgraph->uid != backend_ctx->uid);

    // GGML_META_TIMING=1: host time of rebuilds and dispatches, logged every 200 calls
    static const bool timing = getenv("GGML_META_TIMING") != nullptr;
    struct meta_timing { int64_t n = 0, n_rebuild = 0, t_rebuild = 0, t_dispatch = 0; };
    static thread_local std::map<const void *, meta_timing> timings;
    const int64_t t_start = timing ? ggml_time_us() : 0;
    int64_t t_rebuilt = t_start;

    bool max_nnodes_raised = false;
    if (cgraph->n_nodes > backend_ctx->max_nnodes) {
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            bcj.nodes.resize(cgraph->n_nodes);
            bcj.cgraphs.resize(cgraph->n_nodes);
        }
        backend_ctx->max_nnodes = cgraph->n_nodes;
        max_nnodes_raised = true;
        assert(needs_rebuild);
    }

    if (needs_rebuild) {
        std::set<ggml_backend_buffer_t> used_buffers;
        for (int i = 0; i < cgraph->n_leafs; i++) {
            if (ggml_backend_buffer_is_meta(cgraph->leafs[i]->buffer)) {
                used_buffers.emplace(cgraph->leafs[i]->buffer);
            }
        }
        for (int i = 0; i < cgraph->n_nodes; i++) {
            if (ggml_backend_buffer_is_meta(cgraph->nodes[i]->buffer)) {
                used_buffers.emplace(cgraph->nodes[i]->buffer);
            }
        }
        backend_ctx->tensor_arenas.clear();
        for (ggml_backend_buffer_t buf : used_buffers) {
            ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buf->context;
            if (!buf_ctx->stc_compute[buf_ctx->stc_compute_index]->simple_tensors.empty()) {
                backend_ctx->tensor_arenas.push_back(buf_ctx->stc_compute[buf_ctx->stc_compute_index]);
            }
            buf_ctx->stc_compute_index_next = buf_ctx->stc_compute_index ^ 1;
            auto & next = buf_ctx->stc_compute[buf_ctx->stc_compute_index_next];
            if (!next.unique()) {
                // Another cached graph still owns this generation's tensor descriptors.
                GGML_ASSERT(!next->ctxs.empty());
                const ggml_init_params params = {
                    /*.mem_size   =*/ ggml_get_mem_size(next->ctxs[0].get()),
                    /*.mem_buffer =*/ nullptr,
                    /*.no_alloc   =*/ true,
                };
                next = std::make_shared<ggml_backend_meta_simple_tensor_container>(params, (int) next->ctxs.size());
            } else {
                for (ggml_context_ptr & ctx : next->ctxs) {
                    ggml_reset(ctx.get());
                }
                next->simple_tensors.clear();
            }
        }
        size_t n_subgraphs  = 0;
        size_t max_tmp_size = 0;

        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];

            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
                if (node->view_src != nullptr && node->view_src->op == GGML_OP_NONE && ggml_backend_buffer_is_host(node->view_src->buffer)) {
                    // FIXME s_copy_main is on the CPU and its view seems to be incorrectly added to the graph nodes.
                    // For regular usage this doesn't matter since it's a noop but trying to call ggml_backend_meta_buffer_simple_tensor results in a crash.
                    bcj.nodes[i] = node;
                    continue;
                }
                bcj.nodes[i] = ggml_backend_meta_buffer_simple_tensor(node, j);
                GGML_ASSERT(bcj.nodes[i]);
            }
        }

        {
            // Host copy views can be present in a scheduler split. They have no meta split state.
            auto is_meta_tensor = [](const ggml_tensor * t) {
                return t != nullptr && ggml_backend_buffer_is_meta(t->buffer);
            };
            // For MoE models it may make sense to delay the AllReduce in order to reduce I/O:
            auto get_i_delayed_branch = [&](const int i) -> int {
                int id = i; // i_delayed
                int idr = i; // i_delayed return, last safe return value

                ggml_tensor * node = cgraph->nodes[id];
                int32_t n_used = ggml_node_get_use_count(cgraph, id);

                // Skip MIRRORED nodes that don't consume node
                auto skip_unrelated = [&]() {
                    while (id + 1 < cgraph->n_nodes) {
                        ggml_tensor * next = cgraph->nodes[id+1];
                        if (!is_meta_tensor(next) || ggml_backend_meta_get_split_state(next, false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                            break;
                        }
                        bool safe = true;
                        for (int s = 0; s < GGML_MAX_SRC; s++) {
                            if (next->src[s] == nullptr) {
                                continue;
                            }
                            if (next->src[s] == node) {
                                safe = false;
                                break;
                            }
                            if (!is_meta_tensor(next->src[s]) || ggml_backend_meta_get_split_state(next->src[s], false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                                safe = false;
                                break;
                            }
                        }
                        if (!safe) {
                            break;
                        }
                        id++;
                    }
                };

                skip_unrelated();
                if (id + 1 >= cgraph->n_nodes) {
                    return idr;
                }
                {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op == GGML_OP_ADD_ID && next->src[0] == node &&
                            is_meta_tensor(next->src[1]) && is_meta_tensor(next->src[2]) &&
                            ggml_backend_meta_get_split_state(next->src[1], false).axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL &&
                            ggml_backend_meta_get_split_state(next->src[2], false).axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        node = next;
                        id++;
                        idr = id;
                        n_used = ggml_node_get_use_count(cgraph, id);
                    }
                }
                // Chain of MULs with MIRRORED src[1]
                while (true) {
                    skip_unrelated();
                    if (id + 1 >= cgraph->n_nodes) {
                        return idr;
                    }
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op == GGML_OP_MUL && next->src[0] == node &&
                            is_meta_tensor(next->src[1]) &&
                            ggml_backend_meta_get_split_state(next->src[1], false).axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        node = next;
                        id++;
                        idr = id;
                        n_used = ggml_node_get_use_count(cgraph, id);
                    } else {
                        break;
                    }
                }

                if (n_used != node->ne[1] || id + 2*n_used-1 >= cgraph->n_nodes) {
                    return idr;
                }
                for (int32_t k = 0; k < n_used; k++) {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op != GGML_OP_VIEW || next->view_src != node || next->view_offs != k*node->nb[1] ||
                            next->ne[0] != node->ne[0] || next->ne[1] != node->ne[2] || next->nb[1] != node->nb[2] ||
                            ggml_node_get_use_count(cgraph, id+1) != 1) {
                        return idr;
                    }
                    id++;
                }
                {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op != GGML_OP_ADD || next->src[0] != cgraph->nodes[id - (n_used-1)] ||
                            next->src[1] != cgraph->nodes[id - (n_used-2)] || ggml_node_get_use_count(cgraph, id+1) != 1) {
                        return idr;
                    }
                    id++;
                }
                for (int32_t k = 0; k < n_used - 2; k++) {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op != GGML_OP_ADD || next->src[0] != cgraph->nodes[id] ||
                            next->src[1] != cgraph->nodes[id - (n_used-2)] || ggml_node_get_use_count(cgraph, id+1) != 1) {
                        return idr;
                    }
                    id++;
                }
                idr = id;
                return idr;
            };

            // AllReduce(a) + AllReduce(b) == AllReduce(a + b) for independent partial branches.
            auto get_i_delayed = [&](const int i) -> int {
                const int i_delayed = get_i_delayed_branch(i);
                ggml_tensor * node = cgraph->nodes[i_delayed];

                if (ggml_node_get_use_count(cgraph, i_delayed) != 1) {
                    return i_delayed;
                }

                for (int id = i_delayed + 1; id < cgraph->n_nodes; id++) {
                    ggml_tensor * next = cgraph->nodes[id];
                    if (next->view_src == node) {
                        return i_delayed;
                    }
                    for (int s = 0; s < GGML_MAX_SRC; s++) {
                        if (next->src[s] == node) {
                            return i_delayed;
                        }
                    }

                    if (next->view_src != nullptr && next->view_src->op == GGML_OP_NONE && ggml_backend_buffer_is_host(next->view_src->buffer)) {
                        continue;
                    }
                    if (!is_meta_tensor(next) || ggml_backend_meta_get_split_state(next, false).axis != GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                        continue;
                    }

                    const int i_other = id;
                    const int i_other_delayed = get_i_delayed_branch(i_other);
                    ggml_tensor * other = cgraph->nodes[i_other_delayed];
                    if (ggml_node_get_use_count(cgraph, i_other_delayed) != 1 || i_other_delayed + 1 >= cgraph->n_nodes) {
                        return i_delayed;
                    }

                    ggml_tensor * sum = cgraph->nodes[i_other_delayed + 1];
                    if (sum->op != GGML_OP_ADD || !is_meta_tensor(sum) ||
                            !ggml_are_same_shape(node, other) || node->type != other->type || sum->type != node->type ||
                            !((sum->src[0] == node && sum->src[1] == other) ||
                              (sum->src[0] == other && sum->src[1] == node)) ||
                            ggml_backend_meta_get_split_state(sum, false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        return i_delayed;
                    }

                    for (size_t j = 0; j < n_backends; j++) {
                        auto & bcj = backend_ctx->backend_configs[j];
                        const bool compute       = bcj.nodes[i]->flags       & GGML_TENSOR_FLAG_COMPUTE;
                        const bool compute_other = bcj.nodes[i_other]->flags & GGML_TENSOR_FLAG_COMPUTE;
                        if (compute != compute_other) {
                            return i_delayed;
                        }
                    }
                    return i_other_delayed + 1;
                }
                return i_delayed;
            };

            int i_start = 0;
            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
                if (node->view_src != nullptr && node->view_src->op == GGML_OP_NONE && ggml_backend_buffer_is_host(node->view_src->buffer)) {
                    continue;
                }
                const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(node, /*assume_sync =*/ false);
                if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                    max_tmp_size = std::max(max_tmp_size, ggml_nbytes(node));
                }
                const bool new_subgraph = i + 1 == cgraph->n_nodes || split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL;
                if (!new_subgraph) {
                    continue;
                }

                const int i_delayed = get_i_delayed(i);

                // If we can delay the AllReduce we need to consider the interaction with zero-sized tensor slices.
                // A backend with such a slice would normally have valid data after participating in the AllReduce with a node that has
                //     its compute flag disabled and thus gets its data zeroed out.
                // If the AllReduce is delayed then the nodes until that point also need to have their compute flag disabled.
                if (i_delayed > i) {
                    for (size_t j = 0; j < n_backends; j++) {
                        auto & bcj = backend_ctx->backend_configs[j];
                        if ((bcj.nodes[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                            for (int ii = i + 1; ii <= i_delayed; ii++) {
                                bcj.nodes[ii]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
                            }
                        }
                    }
                }

                i = i_delayed;

                for (size_t j = 0; j < n_backends; j++) {
                    auto & bcj = backend_ctx->backend_configs[j];
                    bcj.cgraphs[n_subgraphs].offset = i_start;
                }
                n_subgraphs++;
                i_start = i + 1;
            }
            GGML_ASSERT(i_start == cgraph->n_nodes);
        }

        backend_ctx->uid         = cgraph->uid;
        backend_ctx->n_subgraphs = n_subgraphs;

        if (max_tmp_size > backend_ctx->max_tmp_size) {
            for (size_t j = 0; j < n_backends; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                for (size_t i = 0; i < backend_ctx->n_reduce_steps; i++) {
                    bcj.bufs[i].reset(ggml_backend_alloc_buffer(bcj.backend, max_tmp_size));
                }
            }
            backend_ctx->max_tmp_size = max_tmp_size;
        }

        static const bool compact_arena = [] { const char * s = getenv("GGML_META_COMPACT_ARENA"); return s && atoi(s) != 0; }();
        bool graph_caps_grew = false;
        if (compact_arena) {
            backend_ctx->graph_node_caps.resize(std::max(backend_ctx->graph_node_caps.size(), n_subgraphs), 1);
            const auto & graphs = backend_ctx->backend_configs[0].cgraphs;
            for (size_t i = 0; i < n_subgraphs; ++i) {
                const size_t stop = i + 1 < n_subgraphs ? graphs[i + 1].offset : cgraph->n_nodes;
                const size_t need = std::max<size_t>(1, stop - graphs[i].offset);
                if (need > backend_ctx->graph_node_caps[i]) {
                    backend_ctx->graph_node_caps[i] = need;
                    graph_caps_grew = true;
                }
            }
        }
        if ((compact_arena ? graph_caps_grew : max_nnodes_raised) || n_subgraphs > backend_ctx->max_subgraphs) {
            backend_ctx->max_subgraphs = std::max(backend_ctx->max_subgraphs, n_subgraphs);
            const size_t n_nodes_per_device = 3 * backend_ctx->n_reduce_steps;
            const size_t n_cgraphs_per_device = 2 * backend_ctx->n_reduce_steps;
            size_t mem_per_device_graphs_main = 0;
            for (size_t i = 0; i < backend_ctx->max_subgraphs; ++i) {
                const size_t cap = compact_arena ? backend_ctx->graph_node_caps[i] : backend_ctx->max_nnodes;
                mem_per_device_graphs_main += ggml_graph_overhead_custom(cap, false);
            }
            const size_t mem_per_device_graphs_aux = n_cgraphs_per_device*backend_ctx->max_subgraphs*ggml_graph_overhead_custom(1, cgraph->grads);
            const size_t mem_per_device_nodes_aux = n_nodes_per_device*backend_ctx->max_subgraphs*ggml_tensor_overhead();
            const ggml_init_params params = {
                /*.mem_size   =*/ n_backends * (mem_per_device_graphs_main + mem_per_device_graphs_aux + mem_per_device_nodes_aux),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            if (compact_arena) {
                const size_t old_main = backend_ctx->max_subgraphs * ggml_graph_overhead_custom(backend_ctx->max_nnodes, false);
                GGML_LOG_INFO("meta-compact-arena backend=%p subgraphs=%zu graph_nodes=%d bytes=%zu saved=%zu\n",
                        (void *) backend, backend_ctx->max_subgraphs, backend_ctx->max_nnodes, params.mem_size,
                        n_backends * (old_main - mem_per_device_graphs_main));
            }
            backend_ctx->ctx.reset(ggml_init(params));
            for (size_t j = 0; j < n_backends; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                // sized for the largest graph seen: a later graph may have more subgraphs or nodes without a reallocation
                for (size_t i = 0; i < backend_ctx->max_subgraphs; i++) {
                    const size_t cap = compact_arena ? backend_ctx->graph_node_caps[i] : backend_ctx->max_nnodes;
                    bcj.cgraphs[i].cgraph_main = ggml_new_graph_custom(backend_ctx->ctx.get(), cap, /*grads =*/ false);
                }
            }
            backend_ctx->cgraphs_aux.resize(n_backends*n_cgraphs_per_device*backend_ctx->max_subgraphs);
            for (size_t k = 0; k < backend_ctx->cgraphs_aux.size(); k++) {
                backend_ctx->cgraphs_aux[k] = ggml_new_graph_custom(backend_ctx->ctx.get(), 1, cgraph->grads);
            }
            backend_ctx->nodes_aux.resize(n_backends*n_nodes_per_device*backend_ctx->max_subgraphs);
            for (size_t k = 0; k < backend_ctx->nodes_aux.size(); k++) {
                backend_ctx->nodes_aux[k] = ggml_new_tensor_1d(backend_ctx->ctx.get(), GGML_TYPE_F32, 1);
            }
        }

        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            for (size_t i_graph = 0; i_graph < n_subgraphs; i_graph++) {
                ggml_cgraph * cgraph_ij = bcj.cgraphs[i_graph].cgraph_main;
                const size_t i_node_start = bcj.cgraphs[i_graph].offset;
                const size_t i_node_stop = i_graph + 1 < n_subgraphs ? bcj.cgraphs[i_graph + 1].offset : cgraph->n_nodes;
                cgraph_ij->n_nodes = i_node_stop - i_node_start;
                ggml_hash_set_reset(&cgraph_ij->visited_hash_set);
                for (size_t i_node = i_node_start; i_node < i_node_stop; i_node++) {
                    ggml_tensor * node_ij = bcj.nodes[i_node];
                    cgraph_ij->nodes[i_node - i_node_start] = node_ij;
                    const size_t hash_pos_orig = ggml_hash_find(&cgraph->visited_hash_set, cgraph->nodes[i_node]);
                    const size_t hash_pos_ij = ggml_hash_insert(&cgraph_ij->visited_hash_set, node_ij);
                    cgraph_ij->use_counts[hash_pos_ij] = cgraph->use_counts[hash_pos_orig];
                }
                cgraph_ij->uid = ggml_graph_next_uid();
            }
        }
    }

    // In-graph AllReduce: append an ALLREDUCE node after each partial node and make the later nodes read it,
    //     then run the whole graph as one graph per simple backend.
    bool try_merge = false;
    if (backend_ctx->p2p_ar != nullptr) {
        bool defer = false;
        if (needs_rebuild) {
            backend_ctx->cgraphs_merged.clear();
            backend_ctx->merge_tried = false;
            uint64_t sig = 1469598103934665603ull; // FNV-1a over ops, types and shapes
            auto mix = [&](uint64_t v) { sig = (sig ^ v) * 1099511628211ull; };
            mix(cgraph->n_nodes);
            for (int i = 0; i < cgraph->n_nodes; i++) {
                const ggml_tensor * t = cgraph->nodes[i];
                mix(t->op);
                mix(t->type);
                for (int d = 0; d < GGML_MAX_DIMS; d++) {
                    mix(t->ne[d]);
                }
            }
            defer = backend_ctx->seen_shapes.insert(sig).second;
        }
        try_merge = !defer && !backend_ctx->merge_tried;
    }
    if (try_merge) {
        backend_ctx->merge_tried = true;
        const size_t n_sub = backend_ctx->n_subgraphs;
        bool ok = n_sub > 1;
        // GGML_META_CHUNK_NODES: nodes per chunk of a merged graph (0: no chunks)
        static const size_t chunk_nodes = [] {
            const char * env = getenv("GGML_META_CHUNK_NODES");
            return env ? (size_t) atoll(env) : (size_t) 512;
        }();
        size_t max_bytes = 0;
        for (size_t i = 0; ok && i + 1 < n_sub; i++) {
            for (size_t j = 0; j < n_backends; j++) {
                const ggml_cgraph * g = backend_ctx->backend_configs[j].cgraphs[i].cgraph_main;
                const ggml_tensor * p = g->nodes[g->n_nodes - 1];
                // a partial that is not computed (zero-sized slice) is still reduced: the AllReduce adds 0 for it
                if (p->type != GGML_TYPE_F32 || !ggml_is_contiguous(p) || p->data == nullptr) {
                    static const bool debug = getenv("GGML_META_DEBUG") != nullptr;
                    ggml_log_internal(debug ? GGML_LOG_LEVEL_WARN : GGML_LOG_LEVEL_DEBUG, "%s: no in-graph AllReduce: subgraph %zu partial %s op=%s type=%s contiguous=%d compute=%d data=%p\n",
                        __func__, i, p->name, ggml_op_desc(p), ggml_type_name(p->type), (int) ggml_is_contiguous(p), (int) ((p->flags & GGML_TENSOR_FLAG_COMPUTE) != 0), p->data);
                    ok = false;
                    break;
                }
                max_bytes = std::max(max_bytes, ggml_nbytes(p));
            }
        }
        ok = ok && backend_ctx->p2p_ar_reserve(backend_ctx->p2p_ar, max_bytes);
        if (ok) {
            // same memory and allocation order on every rebuild: identical graphs get identical node addresses,
            //     so the CUDA graphs captured for them stay valid
            const size_t n_merged = cgraph->n_nodes + n_sub;
            const size_t mem_size = n_backends * (ggml_graph_overhead_custom(n_merged, false) + n_sub*ggml_tensor_overhead());
            if (!backend_ctx->ctx_merged || ggml_get_mem_size(backend_ctx->ctx_merged.get()) < mem_size) {
                const ggml_init_params params = {
                    /*.mem_size   =*/ 2*mem_size,
                    /*.mem_buffer =*/ nullptr,
                    /*.no_alloc   =*/ true,
                };
                backend_ctx->ctx_merged.reset(ggml_init(params));
            }
            ggml_context * ctx_m = backend_ctx->ctx_merged.get();
            ggml_reset(ctx_m);
            backend_ctx->cgraphs_chunks.assign(n_backends, {});

            for (size_t j = 0; j < n_backends; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                ggml_cgraph * gm = ggml_new_graph_custom(ctx_m, n_merged, false);
                auto add = [&](ggml_tensor * t, int32_t use_count) {
                    gm->nodes[gm->n_nodes++] = t;
                    gm->use_counts[ggml_hash_insert(&gm->visited_hash_set, t)] = use_count;
                };
                for (size_t i = 0; i < n_sub; i++) {
                    const int i_start = bcj.cgraphs[i].offset;
                    const int i_stop  = i + 1 < n_sub ? bcj.cgraphs[i + 1].offset : cgraph->n_nodes;
                    for (int k = i_start; k < i_stop; k++) {
                        const int32_t use_count = cgraph->use_counts[ggml_hash_find(&cgraph->visited_hash_set, cgraph->nodes[k])];
                        if (i + 1 == n_sub || k + 1 < i_stop) {
                            add(bcj.nodes[k], use_count);
                            continue;
                        }
                        ggml_tensor * p  = bcj.nodes[k];
                        ggml_tensor * ar = ggml_new_tensor(ctx_m, p->type, GGML_MAX_DIMS, p->ne);
                        memcpy(ar->nb, p->nb, sizeof(p->nb));
                        ar->op        = GGML_OP_ALLREDUCE;
                        ar->src[0]    = p;
                        ar->view_src  = p->view_src ? p->view_src : p;
                        ar->view_offs = p->view_src ? p->view_offs : 0;
                        ar->data      = p->data;
                        ar->buffer    = p->buffer;
                        ar->flags    |= GGML_TENSOR_FLAG_COMPUTE;
                        ggml_format_name(ar, "allreduce_%zu", i);
                        backend_ctx->p2p_ar_set_params(backend_ctx->p2p_ar, j, ar);
                        add(p, 1);
                        add(ar, use_count);
                        for (int k2 = i_stop; k2 < cgraph->n_nodes; k2++) {
                            for (int s = 0; s < GGML_MAX_SRC; s++) {
                                if (bcj.nodes[k2]->src[s] == p) {
                                    bcj.nodes[k2]->src[s] = ar;
                                }
                            }
                        }
                    }
                }
                gm->uid = ggml_graph_next_uid();
                backend_ctx->cgraphs_merged.push_back(gm);

            }
        }
        if (ok) {
            // cut the merged graphs into chunks, each ending with an AllReduce: the CUDA backend fuses no pattern across
            //     an AllReduce, so no fusion reads past the end of a chunk
            std::vector<std::vector<std::pair<int, int>>> bounds(n_backends);
            size_t mem_chunks = 0;
            for (size_t j = 0; j < n_backends; j++) {
                const ggml_cgraph * gm = backend_ctx->cgraphs_merged[j];
                int k0 = 0;
                for (int k = 0; k < gm->n_nodes; k++) {
                    const bool last = k + 1 == gm->n_nodes;
                    if (last || (chunk_nodes > 0 && k + 1 - k0 >= (int) chunk_nodes && gm->nodes[k]->op == GGML_OP_ALLREDUCE &&
                            gm->n_nodes - (k + 1) >= (int) chunk_nodes/2)) {
                        bounds[j].emplace_back(k0, k + 1);
                        mem_chunks += ggml_graph_overhead_custom(k + 1 - k0, false);
                        k0 = k + 1;
                    }
                }
            }
            if (!backend_ctx->ctx_chunks || ggml_get_mem_size(backend_ctx->ctx_chunks.get()) < mem_chunks) {
                const ggml_init_params params = {
                    /*.mem_size   =*/ 2*mem_chunks,
                    /*.mem_buffer =*/ nullptr,
                    /*.no_alloc   =*/ true,
                };
                backend_ctx->ctx_chunks.reset(ggml_init(params));
            }
            ggml_context * ctx_c = backend_ctx->ctx_chunks.get();
            ggml_reset(ctx_c);
            for (size_t j = 0; j < n_backends; j++) {
                ggml_cgraph * gm = backend_ctx->cgraphs_merged[j];
                if (bounds[j].size() == 1) {
                    backend_ctx->cgraphs_chunks[j].push_back(gm);
                    continue;
                }
                for (const auto & [k0, k1] : bounds[j]) {
                    ggml_cgraph * gc = ggml_new_graph_custom(ctx_c, k1 - k0, false);
                    for (int k = k0; k < k1; k++) {
                        ggml_tensor * t = gm->nodes[k];
                        gc->nodes[gc->n_nodes++] = t;
                        gc->use_counts[ggml_hash_insert(&gc->visited_hash_set, t)] = gm->use_counts[ggml_hash_find(&gm->visited_hash_set, t)];
                    }
                    gc->uid = ggml_graph_next_uid();
                    backend_ctx->cgraphs_chunks[j].push_back(gc);
                }
            }
        }
        static const bool debug = getenv("GGML_META_DEBUG") != nullptr;
        if (debug) {
            GGML_LOG_WARN("%s: graph %d nodes, %zu subgraphs: %s, %zu chunks\n", __func__, cgraph->n_nodes, n_sub, ok ? "merged" : "not merged",
                backend_ctx->cgraphs_chunks.empty() ? (size_t) 0 : backend_ctx->cgraphs_chunks[0].size());
        }
    }

    if (timing) {
        t_rebuilt = ggml_time_us();
    }
    auto log_timing = [&]() {
        if (!timing) {
            return;
        }
        meta_timing & mt = timings[backend_ctx];
        mt.n++;
        mt.n_rebuild += needs_rebuild;
        mt.t_rebuild += t_rebuilt - t_start;
        mt.t_dispatch += ggml_time_us() - t_rebuilt;
        if (mt.n % 200 == 0) {
            GGML_LOG_WARN("meta-timing ctx=%p n_nodes=%d rebuilt=%" PRId64 "/200 avg_us: rebuild=%.0f (per rebuild %.0f) dispatch=%.0f\n", (void *) backend_ctx, cgraph->n_nodes,
                mt.n_rebuild, mt.t_rebuild/200.0, mt.n_rebuild ? mt.t_rebuild/(double) mt.n_rebuild : 0.0, mt.t_dispatch/200.0);
            mt = meta_timing();
        }
    };

    ggml_meta_span_scope gpu_span(backend_ctx->gpu_span_begin, backend_ctx->gpu_span_end,
            backend_ctx->gpu_spans, backend_ctx->gpu_span_sequence++, cgraph->n_nodes);

    if (!backend_ctx->cgraphs_merged.empty()) {
        // each backend enqueues from its own thread: a host sync in one graph must not block the peer's AllReduce
        std::vector<ggml_status> status(n_backends, GGML_STATUS_SUCCESS);

        // A chunk that is not a plain replay does driver work (capture, instantiation, eager launches) on the host.
        // Driver calls can block while the own GPU spins in an AllReduce of an earlier chunk, and that deadlocks.
        // So in that case run the chunks in lockstep: all GPUs finish chunk k before any thread starts chunk k+1.
        const size_t n_chunks = backend_ctx->cgraphs_chunks[0].size();
        bool lockstep = false;
        if (n_chunks > 1) {
            for (size_t j = 0; j < n_backends && !lockstep; j++) {
                for (ggml_cgraph * gc : backend_ctx->cgraphs_chunks[j]) {
                    if (backend_ctx->graph_ready == nullptr || !backend_ctx->graph_ready(backend_ctx->backend_configs[j].backend, gc)) {
                        lockstep = true;
                        break;
                    }
                }
            }
        }
        if (lockstep) {
            for (size_t k = 0; k < n_chunks; k++) {
                for (size_t j = 1; j < n_backends; j++) {
                    backend_ctx->workers[j - 1]->post([backend_ctx, &status, j, k] {
                        status[j] = ggml_backend_graph_compute_async(backend_ctx->backend_configs[j].backend, backend_ctx->cgraphs_chunks[j][k]);
                    });
                }
                status[0] = ggml_backend_graph_compute_async(backend_ctx->backend_configs[0].backend, backend_ctx->cgraphs_chunks[0][k]);
                for (size_t j = 1; j < n_backends; j++) {
                    backend_ctx->workers[j - 1]->wait();
                }
                for (size_t j = 0; j < n_backends; j++) {
                    if (status[j] != GGML_STATUS_SUCCESS) {
                        return status[j];
                    }
                    ggml_backend_synchronize(backend_ctx->backend_configs[j].backend);
                }
            }
            log_timing();
            return GGML_STATUS_SUCCESS;
        }
        for (size_t j = 1; j < n_backends; j++) {
            backend_ctx->workers[j - 1]->post([backend_ctx, &status, j] {
                for (ggml_cgraph * gc : backend_ctx->cgraphs_chunks[j]) {
                    status[j] = ggml_backend_graph_compute_async(backend_ctx->backend_configs[j].backend, gc);
                    if (status[j] != GGML_STATUS_SUCCESS) {
                        break;
                    }
                }
            });
        }
        for (ggml_cgraph * gc : backend_ctx->cgraphs_chunks[0]) {
            status[0] = ggml_backend_graph_compute_async(backend_ctx->backend_configs[0].backend, gc);
            if (status[0] != GGML_STATUS_SUCCESS) {
                break;
            }
        }
        for (size_t j = 1; j < n_backends; j++) {
            backend_ctx->workers[j - 1]->wait();
        }
        log_timing();
        for (ggml_status st : status) {
            if (st != GGML_STATUS_SUCCESS) {
                return st;
            }
        }
        return GGML_STATUS_SUCCESS;
    }

    size_t iga = 0; // i graph aux
    size_t ina = 0; // i node aux

    auto get_node_aux = [&](ggml_tensor * t) -> ggml_tensor * {
        ggml_tensor * ret = backend_ctx->nodes_aux[ina++];
        memset(ret, 0, sizeof(ggml_tensor));
        ret->op   = GGML_OP_NONE;
        ret->type = t->type;
        for (size_t k = 0; k < GGML_MAX_DIMS; k++) {
            ret->ne[k] = t->ne[k];
            ret->nb[k] = t->nb[k];
        }
        return ret;
    };
    auto set_tmp_data = [&](ggml_tensor * tensor, const size_t j, const size_t i_buf) {
        auto & bcj = backend_ctx->backend_configs[j];
        ggml_backend_buffer_ptr & buf_ptr = bcj.bufs[i_buf];
        if (!buf_ptr || ggml_backend_buffer_get_size(buf_ptr.get()) < backend_ctx->max_tmp_size) {
            buf_ptr.reset(ggml_backend_alloc_buffer(bcj.backend, backend_ctx->max_tmp_size));
        }
        tensor->buffer = buf_ptr.get();
        tensor->data   = ggml_backend_buffer_get_base(buf_ptr.get());
    };
    // FIXME usage_counts
    auto get_cgraph_aux = [&]() -> ggml_cgraph * {
        ggml_cgraph * ret = backend_ctx->cgraphs_aux[iga++];
        return ret;
    };

    // Preferentially use backend-specific allreduce_tensor_async (e.g. NCCL for CUDA), use a generic fallback if unavailable:
    auto allreduce_fallback = [&](size_t i) -> ggml_status {
        std::vector<ggml_cgraph *> step_cgraphs(n_backends, nullptr);

        // Zero out nodes that were disabled due to having a zero-sized slice:
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            ggml_tensor * node = bcj.cgraphs[i].cgraph_main->nodes[bcj.cgraphs[i].cgraph_main->n_nodes - 1];
            if (node->flags & GGML_TENSOR_FLAG_COMPUTE) {
                continue;
            }
            ggml_tensor * node_zero = get_node_aux(node);
            node_zero->op = GGML_OP_SCALE; // FIXME 0.0f * NaN == NaN
            node_zero->src[0] = node;
            ggml_set_op_params_f32(node_zero, 0, 0.0f);
            node_zero->data = node->data;
            node_zero->buffer = node->buffer;
            node_zero->flags |= GGML_TENSOR_FLAG_COMPUTE;

            step_cgraphs[j] = get_cgraph_aux();
            step_cgraphs[j]->nodes[0] = node_zero;
            step_cgraphs[j]->n_nodes = 1;
            const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, step_cgraphs[j]);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
        }
        std::fill(step_cgraphs.begin(), step_cgraphs.end(), nullptr);

        auto push_data = [&](const size_t j_src, const size_t j_dst, const size_t i_buf) {
            assert(step_cgraphs[j_dst] == nullptr);
            auto & bcj_src = backend_ctx->backend_configs[j_src];
            auto & bcj_dst = backend_ctx->backend_configs[j_dst];

            ggml_tensor * node_src = bcj_src.cgraphs[i].cgraph_main->nodes[bcj_src.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_tensor * node_dst = bcj_dst.cgraphs[i].cgraph_main->nodes[bcj_dst.cgraphs[i].cgraph_main->n_nodes - 1];
            GGML_ASSERT(ggml_is_contiguous(node_src));
            GGML_ASSERT(ggml_is_contiguous(node_dst));

            ggml_tensor * node_tmp = get_node_aux(node_dst);
            set_tmp_data(node_tmp, j_dst, i_buf);

            ggml_backend_tensor_copy_async(bcj_src.backend, bcj_dst.backend, node_src, node_tmp);

            ggml_tensor * node_red = get_node_aux(node_dst);
            node_red->view_src = node_dst->view_src == nullptr ? node_dst : node_dst->view_src;
            node_red->view_offs = node_dst->view_offs;
            node_red->op = GGML_OP_ADD;
            node_red->src[0] = node_dst;
            node_red->src[1] = node_tmp;
            node_red->flags |= GGML_TENSOR_FLAG_COMPUTE;
            ggml_backend_view_init(node_red);

            ggml_cgraph * cgraph_aux = get_cgraph_aux();
            cgraph_aux->nodes[0] = node_red;
            cgraph_aux->n_nodes = 1;
            step_cgraphs[j_dst] = cgraph_aux;
        };

        size_t offset_j = n_backends/2;
        while ((offset_j & (offset_j - 1)) != 0) {
            offset_j--;
        }
        const size_t offset_j_max = offset_j;
        size_t i_buf = 0;

        // If n_backends is not a power of 2, fold in the excess prior to butterfly reduction:
        for (size_t j_src = 2*offset_j_max; j_src < n_backends; j_src++) {
            const size_t j_dst = j_src - 2*offset_j_max;
            push_data(j_src, j_dst, i_buf);
            const ggml_status status = ggml_backend_graph_compute_async(backend_ctx->backend_configs[j_dst].backend, step_cgraphs[j_dst]);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
            i_buf = 1;
        }

        // Butterfly reduction:
        for (; offset_j >= 1; offset_j /= 2) {
            std::fill(step_cgraphs.begin(), step_cgraphs.end(), nullptr);

            for (size_t j = 0; j < 2*offset_j_max; j++) {
                const size_t j_other = j ^ offset_j;
                if (j_other >= n_backends) {
                    continue;
                }
                push_data(j, j_other, i_buf);
            }

            for (size_t j = 0; j < 2*offset_j_max; j++) {
                if (step_cgraphs[j] == nullptr) {
                    continue;
                }
                auto & bcj = backend_ctx->backend_configs[j];
                const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, step_cgraphs[j]);
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
            }
            i_buf++;
        }
        assert(i_buf == backend_ctx->n_reduce_steps);

        // If n_backends is not a power of 2, copy back the reduced tensors to the excess:
        for (size_t j = 2*offset_j_max; j < n_backends; j++) {
            auto & bcj_src = backend_ctx->backend_configs[j - 2*offset_j_max];
            auto & bcj_dst = backend_ctx->backend_configs[j];

            ggml_tensor * node_src = bcj_src.cgraphs[i].cgraph_main->nodes[bcj_src.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_tensor * node_dst = bcj_dst.cgraphs[i].cgraph_main->nodes[bcj_dst.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_backend_tensor_copy_async(bcj_src.backend, bcj_dst.backend, node_src, node_dst);
        }

        return GGML_STATUS_SUCCESS;
    };


    for (size_t i = 0; i < backend_ctx->n_subgraphs; i++) {
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, bcj.cgraphs[i].cgraph_main);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
        }

        if (n_backends > 1 && i < backend_ctx->n_subgraphs - 1) {
            bool backend_allreduce_success = false;
            if (backend_ctx->comm_ctx || backend_ctx->p2p_ar_allreduce) {
                std::vector<ggml_tensor *> nodes;
                std::vector<ggml_backend_t> backends;
                nodes.reserve(n_backends);
                backends.reserve(n_backends);
                for (size_t j = 0; j < n_backends; j++) {
                    auto & bcj = backend_ctx->backend_configs[j];
                    ggml_cgraph * cgraph_ij = bcj.cgraphs[i].cgraph_main;
                    nodes.push_back(cgraph_ij->nodes[cgraph_ij->n_nodes-1]);
                    backends.push_back(bcj.backend);
                }
                if (backend_ctx->p2p_ar_allreduce) {
                    backend_allreduce_success = backend_ctx->p2p_ar_allreduce(
                        backend_ctx->p2p_ar, backends.data(), n_backends, nodes.data());
                    if (!backend_allreduce_success) {
                        GGML_LOG_WARN("%s: cold P2P declined %s (%s); using the existing collective fallback\n",
                            __func__, nodes[0]->name, ggml_type_name(nodes[0]->type));
                    }
                }
                if (!backend_allreduce_success && backend_ctx->comm_ctx) {
                    backend_allreduce_success = backend_ctx->comm_allreduce(backend_ctx->comm_ctx, nodes.data());
                }
            }

            if (!backend_allreduce_success) {
                const ggml_status status = allreduce_fallback(i);
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
            }
        }
    }
    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_i ggml_backend_meta_i = {
    /* .get_name                = */ ggml_backend_meta_get_name,
    /* .free                    = */ ggml_backend_meta_free,
    /* .set_tensor_async        = */ ggml_backend_meta_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_meta_get_tensor_async,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ ggml_backend_meta_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_meta_synchronize,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_meta_graph_compute,
    /* .event_record            = */ ggml_backend_meta_event_record,
    /* .event_wait              = */ ggml_backend_meta_event_wait,
    /* .graph_optimize          = */ nullptr,
};

bool ggml_backend_is_meta(ggml_backend_t backend) {
    return backend != nullptr && backend->iface.get_name == ggml_backend_meta_i.get_name;
}

static ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_meta_context * backend_ctx = new ggml_backend_meta_context(dev, params);

    ggml_backend_t backend = new struct ggml_backend;
    backend->guid    = ggml_backend_meta_guid();
    backend->iface   = ggml_backend_meta_i;
    backend->device  = dev;
    backend->context = backend_ctx;
    return backend;
}

size_t ggml_backend_meta_n_backends(ggml_backend_t meta_backend) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) meta_backend->context;
    return backend_ctx->backend_configs.size();
}

ggml_backend_t ggml_backend_meta_simple_backend(ggml_backend_t meta_backend, size_t index) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) meta_backend->context;
    return backend_ctx->backend_configs[index].backend;
}

// P3 monpack loader v2: read-only accessors over the static materialization of a
// meta-buffer tensor (the per-member simple tensors and the split shares that
// built them). Pure queries — no state change, inert unless called.
size_t ggml_backend_meta_buffer_get_n_simple(ggml_backend_buffer_t meta_buf) {
    return ggml_backend_meta_buffer_n_bufs(meta_buf);
}

struct ggml_tensor * ggml_backend_meta_tensor_get_simple(const struct ggml_tensor * tensor, size_t index) {
    return ggml_backend_meta_buffer_simple_tensor(tensor, index);
}

struct ggml_backend_meta_split_state ggml_backend_meta_tensor_get_split_state(const struct ggml_tensor * tensor) {
    // assume_sync = true matches the call init_tensor_impl used to materialize the
    // simple tensors, so the cached descriptor (if present) is exactly the one that
    // produced them.
    return ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ true);
}
