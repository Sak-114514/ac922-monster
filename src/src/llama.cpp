#include "llama.h"

#include "llama-impl.h"
#include "llama-version.h"

#include "llama-chat.h"
#include "llama-context.h"
#include "llama-mmap.h"
#include "llama-vocab.h"
#include "llama-model-loader.h"
#include "llama-model-saver.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-cpp.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <stdexcept>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

//
// interface implementation
//

const char * llama_flash_attn_type_name(enum llama_flash_attn_type flash_attn_type) {
    switch (flash_attn_type) {
        case LLAMA_FLASH_ATTN_TYPE_AUTO:
            return "auto";
        case LLAMA_FLASH_ATTN_TYPE_DISABLED:
            return "disabled";
        case LLAMA_FLASH_ATTN_TYPE_ENABLED:
            return "enabled";
    }
    GGML_ABORT("fatal error");
}

const char * llama_load_mode_name(enum llama_load_mode load_mode) {
    switch (load_mode) {
        case LLAMA_LOAD_MODE_AUTO:
            return "auto";
        case LLAMA_LOAD_MODE_NONE:
            return "none";
        case LLAMA_LOAD_MODE_MMAP:
            return "mmap";
        case LLAMA_LOAD_MODE_MLOCK:
            return "mlock";
        case LLAMA_LOAD_MODE_MMAP_MLOCK:
            return "mmap+mlock";
        case LLAMA_LOAD_MODE_DIRECT_IO:
            return "dio";
    }
    GGML_ABORT("fatal error");
}

enum llama_load_mode llama_load_mode_from_str(const char * str) {
    if (std::strcmp(str, "auto")       == 0) { return LLAMA_LOAD_MODE_AUTO;       }
    if (std::strcmp(str, "none")       == 0) { return LLAMA_LOAD_MODE_NONE;       }
    if (std::strcmp(str, "mmap")       == 0) { return LLAMA_LOAD_MODE_MMAP;       }
    if (std::strcmp(str, "mlock")      == 0) { return LLAMA_LOAD_MODE_MLOCK;      }
    if (std::strcmp(str, "mmap+mlock") == 0) { return LLAMA_LOAD_MODE_MMAP_MLOCK; }
    if (std::strcmp(str, "dio")        == 0) { return LLAMA_LOAD_MODE_DIRECT_IO;  }
    throw std::invalid_argument(std::string("unknown load mode: ") + str);
}

struct llama_sampler_chain_params llama_sampler_chain_default_params() {
    struct llama_sampler_chain_params result = {
        /*.no_perf =*/ true,
    };

    return result;
}

size_t llama_max_devices(void) {
    return 16;
}

size_t llama_max_tensor_buft_overrides() {
    return 4096;
}

bool llama_supports_mmap(void) {
    return llama_mmap::SUPPORTED;
}

bool llama_supports_mlock(void) {
    return llama_mlock::SUPPORTED;
}

bool llama_supports_gpu_offload(void) {
    if (!ggml_backend_reg_count()) {
        ggml_backend_load_all();
    }
    return ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) != nullptr ||
           ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU) != nullptr ||
           llama_supports_rpc();
}

bool llama_supports_rpc(void) {
    if (!ggml_backend_reg_count()) {
        ggml_backend_load_all();
    }
    return ggml_backend_reg_by_name("RPC") != nullptr;
}

const char * llama_version(void) {
    return LLAMA_VERSION;
}

void llama_backend_init(void) {
    ggml_time_init();

    // needed to initialize f16 tables
    {
        struct ggml_init_params params = { 0, NULL, false };
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

    if (!ggml_backend_reg_count()) {
        ggml_backend_load_all();
    }
}

void llama_numa_init(enum ggml_numa_strategy numa) {
    if (numa != GGML_NUMA_STRATEGY_DISABLED) {
        auto * dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        GGML_ASSERT(dev && "CPU backend is not loaded");
        auto * reg = ggml_backend_dev_backend_reg(dev);
        auto * numa_init_fn = (decltype(ggml_numa_init) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_numa_init");
        if (numa_init_fn) {
            numa_init_fn(numa);
        }
    }
}

void llama_backend_free(void) {
    ggml_quantize_free();
}

int64_t llama_time_us(void) {
    return ggml_time_us();
}

// LLAMA_ISLAND_SPLIT groups the tensor-parallel devices into multiple "islands", one meta device
// per island, e.g. LLAMA_ISLAND_SPLIT=3,3 on six GPUs. Each island runs tensor parallelism over
// its own devices only (so every AllReduce stays inside one NVLink domain), while the layers are
// split across the meta devices the same way LLAMA_SPLIT_MODE_LAYER splits them over plain GPUs
// (one activation copy per island boundary instead of two cross-island AllReduces per layer).
// With the variable unset a single island spans all devices, which is the previous behavior.
//
// The split-state callback is shared by all islands, so the islands must have equal sizes and the
// first n_island entries of --tensor-split should be uniform. The layer boundary between the
// islands follows the --tensor-split ratios of the meta devices (all-ones gives the first island
// one extra repeating block, e.g. 25/23+output on 48+1 blocks); LLAMA_ISLAND_LAYERS in
// llama-model.cpp can pin it explicitly.
// LLAMA_DRAFT_DETP=1: run the MTP draft head on a single device without tensor parallelism
// (see patches/draft-detp-NOTES.md). The plain tensor_split entry of the draft card is scaled
// by LLAMA_DRAFT_DETP_TRUNK_W so the card also holds the full draft weights + LM-head replica.
static bool llama_draft_detp_requested() {
    static const bool requested = [] {
        const char * s = getenv("LLAMA_DRAFT_DETP");
        return s != nullptr && atoi(s) != 0;
    }();
    return requested;
}

static bool llama_island_split_sizes(size_t n_devs, std::vector<size_t> & island_sizes) {
    const char * s = getenv("LLAMA_ISLAND_SPLIT");
    if (s == nullptr || *s == '\0') {
        island_sizes.assign(1, n_devs);
        return true;
    }

    size_t total = 0;
    const char * begin = s;
    while (true) {
        size_t size = 0;
        bool any = false;
        while (*begin >= '0' && *begin <= '9') {
            size = size * 10 + size_t(*begin - '0');
            any = true;
            begin++;
        }
        if (!any || (*begin != '\0' && *begin != ',')) {
            LLAMA_LOG_ERROR("%s: invalid LLAMA_ISLAND_SPLIT near '%s'\n", __func__, begin);
            return false;
        }
        island_sizes.push_back(size);
        total += size;
        if (*begin == '\0') {
            break;
        }
        begin++; // skip ','
    }

    if (total != n_devs) {
        LLAMA_LOG_ERROR("%s: LLAMA_ISLAND_SPLIT sizes sum to %zu, but %zu devices were given\n",
                __func__, total, n_devs);
        return false;
    }

    // the shared split-state callback shards tensors identically on every island
    for (size_t k = 1; k < island_sizes.size(); k++) {
        if (island_sizes[k] != island_sizes[0]) {
            LLAMA_LOG_ERROR("%s: LLAMA_ISLAND_SPLIT requires equal island sizes\n", __func__);
            return false;
        }
    }

    return true;
}

// PLE uses its own row partition; trunk shard ratios do not apply to this device.
ggml_backend_meta_split_state llama_ple_device_split(const ggml_tensor * tensor, void * userdata) {
    auto * model = static_cast<llama_model *>(userdata);
    if (strcmp(tensor->name, "per_layer_token_embd.weight") != 0) {
        return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
    }
    const size_t members = ggml_backend_meta_device_n_devices(model->ple_device);
    GGML_ASSERT(members > 0 && members <= GGML_BACKEND_META_MAX_DEVICES);
    ggml_backend_meta_split_state state{};
    state.axis = GGML_BACKEND_SPLIT_AXIS_1;
    state.n_segments = 1;
    state.nr[0] = 1;
    int64_t low = 0;
    for (size_t j = 0; j < members; ++j) {
        const int64_t high = tensor->ne[1] * (int64_t) (j + 1) / members;
        state.ne[j] = high - low;
        low = high;
    }
    return state;
}

// create one meta device per island and add them to model->devices
static bool llama_push_island_meta_devices(llama_model * model, ggml_backend_dev_t * devs, size_t n_devs) {
    std::vector<size_t> island_sizes;
    if (!llama_island_split_sizes(n_devs, island_sizes)) {
        return false;
    }

    // LLAMA_DRAFT_DETP and LLAMA_ISLAND_SPLIT both reinterpret the tensor_split entries: the
    // islands split the entry sequence into per-island blocks (and require uniform entries),
    // while de-TP scales one physical device's entry. They are mutually exclusive - de-TP wins
    // and the meta device spans all GPUs as one island, like an unset LLAMA_ISLAND_SPLIT.
    if (llama_draft_detp_requested() && island_sizes.size() > 1) {
        LLAMA_LOG_WARN("%s: LLAMA_ISLAND_SPLIT is ignored because LLAMA_DRAFT_DETP is enabled "
                "(the trunk AllReduces span all devices, as without islands)\n", __func__);
        island_sizes.assign(1, n_devs);
    }

    model->get_split_state_ud.n_devices = island_sizes[0];
    model->get_split_state_ud.model     = model;

    // LLAMA_DRAFT_DETP: remember a plain (non-meta) device as the candidate host for the de-TP
    // MTP draft layers. Recorded here because the plain device list is only available while the
    // meta devices are being created; unused unless the draft de-TP patch activates at load time.
    {
        size_t dev_idx = 0;
        if (const char * idx_env = getenv("LLAMA_DRAFT_DETP_DEV")) {
            const long v = atol(idx_env);
            if (v >= 0 && (size_t) v < n_devs) {
                dev_idx = (size_t) v;
            } else {
                LLAMA_LOG_WARN("%s: ignoring LLAMA_DRAFT_DETP_DEV=%s (must be in [0, %zu))\n", __func__, idx_env, n_devs);
            }
        }
        model->draft_detp_dev_candidate = devs[dev_idx];

        // asymmetric tensor split: the draft card's trunk shard is scaled down so the full
        // draft weights + LM-head replica fit next to it (the load-time memory check in
        // llama-model.cpp must use the same weight, and falls back to a plain TP draft
        // without the draft set if even the asymmetric layout does not fit)
        float trunk_w = 1.0f;
        if (llama_draft_detp_requested()) {
            trunk_w = 0.5f;
            if (const char * w_env = getenv("LLAMA_DRAFT_DETP_TRUNK_W")) {
                const float v = atof(w_env);
                if (v > 0.0f && v <= 1.0f) {
                    trunk_w = v;
                } else {
                    LLAMA_LOG_WARN("%s: ignoring LLAMA_DRAFT_DETP_TRUNK_W=%s (must be in (0, 1])\n", __func__, w_env);
                }
            }
        }
        model->get_split_state_ud.detp_trunk_w = trunk_w;
        model->get_split_state_ud.detp_dev     = dev_idx;

        model->draft_detp_devs.assign(devs, devs + n_devs);

        if (trunk_w != 1.0f) {
            LLAMA_LOG_INFO("%s: LLAMA_DRAFT_DETP asymmetric tensor split: device %zu (%s) trunk weight %.3f, others 1.000\n",
                    __func__, dev_idx, ggml_backend_dev_name(devs[dev_idx]), trunk_w);
        }
    }

    if (const char * ple = getenv("LLAMA_PLE_SEPARATE")) {
        if (atoi(ple) != 0) {
            if (island_sizes.size() != 2 || n_devs != 6) {
                LLAMA_LOG_ERROR("%s: separate PLE requires two 3-GPU islands\n", __func__);
                return false;
            }
            model->ple_device = ggml_backend_meta_device(devs, n_devs, llama_ple_device_split, model);
            LLAMA_LOG_INFO("%s: separate PLE device with %zu row shards\n", __func__, n_devs);
        }
    }

    size_t begin = 0;
    for (size_t k = 0; k < island_sizes.size(); k++) {
        LLAMA_LOG_INFO("%s: creating a Meta device with %zu devices (island %zu):\n",
                __func__, island_sizes[k], k);
        for (size_t i = 0; i < island_sizes[k]; i++) {
            LLAMA_LOG_INFO("%s: - device %zu: %s\n", __func__, i, ggml_backend_dev_name(devs[begin + i]));
        }
        model->devices.push_back({
            true, ggml_backend_meta_device(
            devs + begin, island_sizes[k], llama_meta_device_get_split_state, &model->get_split_state_ud)
        });
        begin += island_sizes[k];
    }

    return true;
}

// returns true on success
static bool llama_prepare_model_devices(const llama_model_params & params, llama_model * model) {
    // create list of devices to use with this model
    if (params.devices) {
        if (params.split_mode == LLAMA_SPLIT_MODE_TENSOR) {
            size_t n_devs = 0;
            while (params.devices[n_devs]) {
                n_devs++;
            }
            if (n_devs == 0) {
                LLAMA_LOG_ERROR("%s: LLAMA_SPLIT_MODE_TENSOR needs >= 1 devices\n", __func__);
                return false;
            }
            if (!llama_push_island_meta_devices(model, params.devices, n_devs)) {
                return false;
            }
        } else {
            for (ggml_backend_dev_t * dev = params.devices; *dev; ++dev) {
                model->devices.push_back({false, *dev});
            }
        }
    } else {
        // default device selection

        // build list of available devices
        std::vector<llama_device> gpus;
        std::vector<llama_device> igpus;
        std::vector<llama_device> rpc_servers;

        if (params.split_mode == LLAMA_SPLIT_MODE_TENSOR) {
            std::vector<ggml_backend_dev_t> devs;
            devs.reserve(ggml_backend_dev_count());
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                auto * dev = ggml_backend_dev_get(i);
                if (ggml_backend_dev_buffer_type(dev) == ggml_backend_cpu_buffer_type()) {
                    LLAMA_LOG_INFO("%s: skipping %s (%s) for tensor parallelism\n", __func__, ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));
                    continue;
                }
                devs.push_back(dev);
            }
            if (devs.empty()) {
                LLAMA_LOG_ERROR("%s: LLAMA_SPLIT_MODE_TENSOR needs >= 1 devices\n", __func__);
                return false;
            }

            if (!llama_push_island_meta_devices(model, devs.data(), devs.size())) {
                return false;
            }
        } else {
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                switch (ggml_backend_dev_type(dev)) {
                    case GGML_BACKEND_DEVICE_TYPE_CPU:
                    case GGML_BACKEND_DEVICE_TYPE_ACCEL:
                        // skip CPU backends since they are handled separately
                        break;

                    case GGML_BACKEND_DEVICE_TYPE_GPU: {
                        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
                        if (ggml_backend_reg_name(reg) == std::string("RPC")) {
                            rpc_servers.push_back({false, dev});
                        } else {
                            // check if there is already a GPU with the same device id
                            ggml_backend_dev_props props;
                            ggml_backend_dev_get_props(dev, &props);
                            auto it = std::find_if(gpus.begin(), gpus.end(), [&props](const llama_device & d) {
                                ggml_backend_dev_props d_props;
                                ggml_backend_dev_get_props(d.dev, &d_props);
                                if (props.device_id && d_props.device_id) {
                                    return strcmp(props.device_id, d_props.device_id) == 0;
                                }
                                return false;
                            });

                            if (it != gpus.end()) {
                                LLAMA_LOG_INFO("%s: skipping device %s (%s) with id %s - already using device %s (%s) with the same id\n",
                                        __func__,
                                        ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
                                        props.device_id ? props.device_id : "unknown id",
                                        ggml_backend_dev_name(it->dev), ggml_backend_dev_description(it->dev));
                            } else {
                                gpus.push_back({false, dev});
                            }
                        }
                        break;
                    }

                    case GGML_BACKEND_DEVICE_TYPE_IGPU:
                        // igpus.empty() - workaround for integrated devices seen by multiple backends
                        // ref: https://github.com/ggml-org/llama.cpp/pull/23897
                        // ggml_backend_dev_backend_reg - allow devices of the same backend regardless if integrated
                        // ref: https://github.com/ggml-org/llama.cpp/pull/23897#issuecomment-5264222997
                        if (igpus.empty() || ggml_backend_dev_backend_reg(dev) == ggml_backend_dev_backend_reg(igpus.back().dev)) {
                            igpus.push_back({false, dev});
                        }
                        break;
                    case GGML_BACKEND_DEVICE_TYPE_META:
                        GGML_ABORT("fatal error");
                }
            }
        }

        // add RPC servers at the front of the list to minimize network transfers
        model->devices.insert(model->devices.begin(), rpc_servers.begin(), rpc_servers.end());

        // add GPUs
        model->devices.insert(model->devices.end(), gpus.begin(), gpus.end());

        // add integrated GPUs only if no discrete GPUs were found
        // (RPC servers do not count, otherwise the local iGPU would be dropped on iGPU+RPC setups)
        if (gpus.empty()) {
            model->devices.insert(model->devices.end(), igpus.begin(), igpus.end());
        }
    }

    // if using single GPU mode, remove all except the main GPU
    if (params.split_mode == LLAMA_SPLIT_MODE_NONE && !model->devices.empty()) {
        if (params.main_gpu < 0) {
            model->devices.clear();
        } else {
            if (params.main_gpu >= (int)model->devices.size()) {
                LLAMA_LOG_ERROR("%s: invalid value for main_gpu: %d (available devices: %zu)\n", __func__, params.main_gpu, model->devices.size());
                return false;
            }
            llama_device main_gpu = model->devices[params.main_gpu];
            model->devices.clear();
            model->devices.push_back(main_gpu);
        }
    }

    for (const auto & dev : model->devices) {
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev.dev, &props);
        LLAMA_LOG_INFO("%s: using device %s (%s) (%s) - %zu MiB free\n", __func__,
                ggml_backend_dev_name(dev.dev), ggml_backend_dev_description(dev.dev),
                props.device_id ? props.device_id : "unknown id",
                props.memory_free/1024/1024);
    }

    return true;
}

// Returns 0 on success, -1 on error, and -2 on cancellation via llama_progress_callback
static std::pair<int, llama_model *> llama_model_load(struct gguf_context * metadata, llama_model_set_tensor_data_t set_tensor_data, void * set_tensor_data_ud,
        const std::string & fname, std::vector<std::string> & splits, FILE * file, llama_model_params & params) {
    try {
        llama_model_loader ml(metadata, set_tensor_data, set_tensor_data_ud, fname, splits, file, params.load_mode,
            params.check_tensors, params.no_alloc, params.load_mtp, params.kv_overrides, params.tensor_buft_overrides);

        ml.lazy.mode = params.lazy_mode;

        ml.print_info();
        std::unique_ptr<llama_model> model_ptr(llama_model_create(ml, params));

        bool ok = llama_prepare_model_devices(params, model_ptr.get());
        if (!ok) {
            return {-1, nullptr};
        }

        auto * model = dynamic_cast<llama_model_base *>(model_ptr.get());
        if (model == nullptr) {
            GGML_ABORT("fatal error: model does not implement llama_model_base");
        }

        // loading time will be recalculated after the first eval, so
        // we take page faults deferred by mmap() into consideration
        model->t_load_us = 0;
        time_meas tm(model->t_load_us);

        model->t_start_us = tm.t_start_us;

        model->hparams.vocab_only = params.vocab_only;
        model->hparams.no_alloc   = params.no_alloc;

        try {
            model->load_hparams(ml);
        } catch(const std::exception & e) {
            throw std::runtime_error("error loading model hyperparameters: " + std::string(e.what()));
        }
        if (model->arch == LLM_ARCH_CLIP) {
            throw std::runtime_error("CLIP cannot be used as main model, use it with --mmproj instead");
        }
        try {
            model->load_vocab(ml);
        } catch(const std::exception & e) {
            throw std::runtime_error("error loading model vocabulary: " + std::string(e.what()));
        }

        model->load_stats(ml);
        model->print_info();

        if (params.vocab_only) {
            LLAMA_LOG_INFO("%s: vocab only - skipping tensors\n", __func__);
            return {0, model_ptr.release()};
        }

        if (!model->load_tensors(ml)) {
            return {-2, nullptr};
        }

        return {0, model_ptr.release()};
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading model: %s\n", __func__, err.what());
        return {-1, nullptr};
    }
}

static struct llama_model * llama_model_load_from_file_impl(
        struct gguf_context * metadata,
        llama_model_set_tensor_data_t set_tensor_data,
        void * set_tensor_data_ud,
        const std::string & path_model,
        std::vector<std::string> & splits,
        FILE * file,
        struct llama_model_params params) {
    {
        int n_sources_defined = 0;
        if (metadata != nullptr) {
            n_sources_defined++;
        }
        if (!path_model.empty()) {
            n_sources_defined++;
        }
        if (file != nullptr) {
            n_sources_defined++;
        }
        if (n_sources_defined != 1) {
            LLAMA_LOG_ERROR("%s: exactly one out metadata, path_model, and file must be defined\n", __func__);
            return nullptr;
        }
    }
    ggml_time_init();

    if (!params.vocab_only && ggml_backend_reg_count() == 0) {
        LLAMA_LOG_ERROR("%s: no backends are loaded. hint: use ggml_backend_load() or ggml_backend_load_all() to load a backend before calling this function\n", __func__);
        return nullptr;
    }

    unsigned cur_percentage = 0;
    if (params.progress_callback == NULL) {
        params.progress_callback_user_data = &cur_percentage;
        params.progress_callback = [](float progress, void * ctx) {
            unsigned * cur_percentage_p = (unsigned *) ctx;
            unsigned percentage = (unsigned) (100 * progress);
            while (percentage > *cur_percentage_p) {
                *cur_percentage_p = percentage;
                LLAMA_LOG_CONT(".");
                if (percentage >= 100) {
                    LLAMA_LOG_CONT("\n");
                }
            }
            return true;
        };
    }

    const auto [status, model] = llama_model_load(metadata, set_tensor_data, set_tensor_data_ud, path_model, splits, file, params);
    GGML_ASSERT(status <= 0);
    if (status < 0) {
        if (status == -1) {
            LLAMA_LOG_ERROR("%s: failed to load model\n", __func__);
        } else if (status == -2) {
            LLAMA_LOG_INFO("%s: cancelled model load\n", __func__);
        }

        if (model) {
            llama_model_free(model);
        }
        return nullptr;
    }

    return model;
}

struct llama_model * llama_model_init_from_user(
        struct gguf_context * metadata,
        llama_model_set_tensor_data_t set_tensor_data,
        void * set_tensor_data_ud,
        struct llama_model_params params) {
    GGML_ASSERT(metadata != nullptr);
    std::string path_model;
    std::vector<std::string> splits = {};
    params.load_mode = LLAMA_LOAD_MODE_NONE;
    params.use_extra_bufts = false;
    return llama_model_load_from_file_impl(metadata, set_tensor_data, set_tensor_data_ud, path_model, splits, /*file*/ nullptr, params);
}
// deprecated
struct llama_model * llama_load_model_from_file(
        const char * path_model,
        struct llama_model_params params) {
    return llama_model_load_from_file(path_model, params);
}

struct llama_model * llama_model_load_from_file(
        const char * path_model,
        struct llama_model_params params) {
    std::vector<std::string> splits = {};
    return llama_model_load_from_file_impl(nullptr, nullptr, nullptr, path_model, splits, /*file*/ nullptr, params);
}

struct llama_model * llama_model_load_from_splits(
        const char ** paths,
        size_t n_paths,
        struct llama_model_params params) {
    std::vector<std::string> splits;
    if (n_paths == 0) {
        LLAMA_LOG_ERROR("%s: list of splits is empty\n", __func__);
        return nullptr;
    }
    splits.reserve(n_paths);
    for (size_t i = 0; i < n_paths; ++i) {
        splits.push_back(paths[i]);
    }
    return llama_model_load_from_file_impl(nullptr, nullptr, nullptr, splits.front(), splits, /*file*/ nullptr, params);
}

struct llama_model * llama_model_load_from_file_ptr(FILE * file, struct llama_model_params params) {
    if (!file) {
        LLAMA_LOG_ERROR("%s: file is NULL\n", __func__);
        return nullptr;
    }
    std::string path_model;
    std::vector<std::string> splits = {};
    return llama_model_load_from_file_impl(nullptr, nullptr, nullptr, path_model, splits, file, params);
}

void llama_model_save_to_file(const struct llama_model * model, const char * path_model) {
    llama_model_saver ms(model);
    ms.add_kv_from_model();
    ms.add_tensors_from_model();
    ms.save(path_model);
}

//
// chat templates
//

int32_t llama_chat_apply_template(
                              const char * tmpl,
         const struct llama_chat_message * chat,
                                  size_t   n_msg,
                                    bool   add_ass,
                                    char * buf,
                                 int32_t   length) {
    const std::string curr_tmpl(tmpl == nullptr ? "chatml" : tmpl);

    // format the chat to string
    std::vector<const llama_chat_message *> chat_vec;
    chat_vec.resize(n_msg);
    for (size_t i = 0; i < n_msg; i++) {
        chat_vec[i] = &chat[i];
    }

    std::string formatted_chat;
    llm_chat_template detected_tmpl = llm_chat_detect_template(curr_tmpl);
    if (detected_tmpl == LLM_CHAT_TEMPLATE_UNKNOWN) {
        return -1;
    }
    int32_t res = llm_chat_apply_template(detected_tmpl, chat_vec, formatted_chat, add_ass);
    if (res < 0) {
        return res;
    }
    if (buf && length > 0) {
        strncpy(buf, formatted_chat.c_str(), length);
    }
    return res;
}

//
// model split
//

int32_t llama_split_path(
    char * split_path,
    size_t maxlen,
    const char * path_prefix,
    int32_t split_no,
    int32_t split_count) {

    static const char * const SPLIT_PATH_FORMAT = "%s-%05d-of-%05d.gguf";

    const int written = snprintf(
        split_path,
        maxlen,
        SPLIT_PATH_FORMAT,
        path_prefix,
        split_no + 1,
        split_count
    );

    if (written < 0 || (size_t) written >= maxlen) {
        return 0;
    }

    return (int32_t) written;
}

int32_t llama_split_prefix(
    char * split_prefix,
    size_t maxlen,
    const char * split_path,
    int32_t split_no,
    int32_t split_count) {

    const std::string str_split_path(split_path);

    char postfix[32];
    snprintf(postfix, sizeof(postfix), "-%05d-of-%05d.gguf", split_no + 1, split_count);

    const std::string str_postfix(postfix);
    if (str_split_path.size() <= str_postfix.size()) {
        return 0;
    }

    const size_t size_prefix = str_split_path.size() - str_postfix.size();

    if (str_split_path.compare(size_prefix, std::string::npos, str_postfix) == 0) {
        const size_t copy_len = std::min(size_prefix + 1, maxlen);
        snprintf(split_prefix, copy_len, "%s", split_path);

        return (int32_t) size_prefix;
    }

    return 0;
}

const char * llama_print_system_info(void) {
    static std::string s;
    s.clear(); // Clear the string, since it's static, otherwise it will accumulate data from previous calls.

    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        auto * reg = ggml_backend_reg_get(i);
        auto * get_features_fn = (ggml_backend_get_features_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features");
        if (get_features_fn) {
            ggml_backend_feature * features = get_features_fn(reg);
            s += ggml_backend_reg_name(reg);
            s += " : ";
            for (; features->name; features++) {
                s += features->name;
                s += " = ";
                s += features->value;
                s += " | ";
            }
        }
    }

    return s.c_str();
}

