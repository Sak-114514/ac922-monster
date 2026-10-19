#include "../ggml-qsa-topk-private.h"
#include "ggml-cuda.h"
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include "ggml-cuda/freetoken-ggml.cuh"
#include "ggml-cuda/freetoken-dense.cuh"
#endif
#include "ggml-impl.h"
#include "ggml-hardware-test-policy.h"
#include "ggml-backend-impl.h"
#include "../ggml-ep-private.h"
#include "moe-t16-chunks-private.h"
#include "../ggml-meta-cold-cohort-private.h"
#include "../ggml-qsa-mask-private.h"

#include "ggml-cuda/allreduce.cuh"
#include "ggml-cuda/allreduce-p2p.cuh"
#include "ggml-cuda/hc-fused.cuh"
#include "ggml-cuda/common.cuh"
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include "layer-pinned-upload-private.h"
#endif
#include "ggml-cuda/optimization-path-log.h"
#include "ggml-cuda/graph-structural-signature.h"
#include "graph-signature-diff-private.h"
#include "graph-lease-policy-private.h"
#include "graph-structural-snapshot-cache.h"
#include "graph-device-tail-client-private.h"
#include "graph-device-tail-captured-client-v2-private.h"
#include "cublas-materialization-role-audit-private.h"
#include "ggml-cuda/acc.cuh"
#include "ggml-cuda/add-id.cuh"
#include "ggml-cuda/arange.cuh"
#include "ggml-cuda/argmax.cuh"
#include "ggml-cuda/argsort.cuh"
#include "ggml-cuda/binbcast.cuh"
#include "ggml-cuda/clamp.cuh"
#include "ggml-cuda/col2im-1d.cuh"
#include "ggml-cuda/concat.cuh"
#include "ggml-cuda/strided-sum.cuh"
#include "ggml-cuda/mmvf-multi.cuh"
#include "ggml-cuda/conv-transpose-1d.cuh"
#include "ggml-cuda/conv2d.cuh"
#include "ggml-cuda/conv2d-dw.cuh"
#include "ggml-cuda/conv2d-transpose.cuh"
#include "ggml-cuda/convert.cuh"
#include "ggml-cuda/count-equal.cuh"
#include "ggml-cuda/cpy.cuh"
#include "ggml-cuda/cross-entropy-loss.cuh"
#include "ggml-cuda/cumsum.cuh"
#include "ggml-cuda/diagmask.cuh"
#include "ggml-cuda/diag.cuh"
#include "ggml-cuda/fattn.cuh"
#include "onecat-init-ready-private.cuh"
#include "ggml-cuda/fwht.cuh"
#include "ggml-cuda/getrows.cuh"
#include "ggml-cuda/im2col.cuh"
#include "ggml-cuda/mmf.cuh"
#include "ggml-cuda/kernel_sm70_bf16.cuh"
#include "bf16-panel-private.cuh"
#include "ggml-cuda/fused-projection-sm70.cuh"
#include "ggml-cuda/mmq.cuh"
#include "ggml-cuda/mmvf.cuh"
#include "ggml-cuda/mmvq.cuh"
#include "ggml-cuda/moe-weighted-reduction.cuh"
#include "ggml-cuda/norm.cuh"
#include "ggml-cuda/opt-step-adamw.cuh"
#include "ggml-cuda/opt-step-sgd.cuh"
#include "ggml-cuda/out-prod.cuh"
#include "ggml-cuda/pad.cuh"
#include "ggml-cuda/pool2d.cuh"
#include "ggml-cuda/pool1d.cuh"
#include "ggml-cuda/quantize.cuh"
#include "ggml-cuda/rope.cuh"
#include "ggml-cuda/roll.cuh"
#include "ggml-cuda/scale.cuh"
#include "ggml-cuda/snake.cuh"
#include "ggml-cuda/softcap.cuh"
#include "ggml-cuda/softmax.cuh"
#include "ggml-cuda/ssm-conv.cuh"
#include "ggml-cuda/ssm-scan.cuh"
#include "ggml-cuda/sum.cuh"
#include "ggml-cuda/sumrows.cuh"
#include "ggml-cuda/top-k.cuh"
#include "ggml-cuda/mean.cuh"
#include "ggml-cuda/tsembd.cuh"
#include "ggml-cuda/topk-moe.cuh"
#include "ggml-cuda/unary.cuh"
#include "ggml-cuda/upscale.cuh"
#include "ggml-cuda/wkv.cuh"
#include "ggml-cuda/gla.cuh"
#include "ggml-cuda/gated_delta_net.cuh"
#include "ggml-cuda/dsv4-hc.cuh"
#include "ggml-cuda/ple-conv-sm70.cuh"
#include "ggml-cuda/monomoe-ggml.cuh"
#include "ggml-cuda/set.cuh"
#include "ggml-cuda/set-rows.cuh"
#include "ggml-cuda/pad_reflect_1d.cuh"
#include "ggml-cuda/solve_tri.cuh"
#include "ggml-cuda/tri.cuh"
#include "ggml-cuda/cumsum.cuh"
#include "ggml-cuda/fill.cuh"
#include "ggml-cuda/lightning-indexer.cuh"
#include "ggml.h"

// P3 monpack: sidecar kernel contract (monpack::is_registered / try_mul_mat / try_mul_mat_id).
// MONPACK_AVAILABLE is only defined when monpack/monpack-kernels.cu is part of the build
// (see the guard in this directory's CMakeLists.txt); without it the tree builds stock.
#ifdef MONPACK_AVAILABLE
#include "monpack/monpack.h"
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cinttypes>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cfloat>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>

// These are host dispatch/capture observations, not replay kernel counts.
void ggml_cuda_optimization_path_log(int device, const char * path, bool requested, bool selected,
                                    const char * reason, int64_t tokens, int64_t width, int64_t auxiliary) {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_OPTIMIZATION_PATH_LOG");
        return value && std::strcmp(value, "1") == 0;
    }();
    if (!enabled) return;
    struct entry {
        size_t requested = 0, selected = 0;
        std::map<std::string, size_t> fallback;
    };
    struct registry {
        std::mutex mutex;
        std::map<std::pair<int, std::string>, entry> entries;
        ~registry() {
            for (const auto & item : entries) {
                fprintf(stderr, "optimization path summary device=%d path=%s scope=host_dispatch_or_capture requested=%zu selected=%zu",
                        item.first.first, item.first.second.c_str(), item.second.requested, item.second.selected);
                for (const auto & fallback : item.second.fallback)
                    fprintf(stderr, " fallback_%s=%zu", fallback.first.c_str(), fallback.second);
                fprintf(stderr, "\n");
            }
        }
    };
    static registry stats;
    std::lock_guard<std::mutex> lock(stats.mutex);
    auto & value = stats.entries[{device, path}];
    value.requested += requested;
    const size_t count = selected ? ++value.selected : ++value.fallback[reason];
    if (count <= 2 || count % 128 == 0) {
        fprintf(stderr, "optimization path device=%d path=%s scope=host_dispatch_or_capture requested=%d selected=%d reason=%s tokens=%" PRId64 " width=%" PRId64 " auxiliary=%" PRId64 " count=%zu\n",
                device, path, requested, selected, reason, tokens, width, auxiliary, count);
    }
}

static_assert(sizeof(half) == sizeof(ggml_fp16_t), "wrong fp16 size");

#define GGML_LOG_WARN_ONCE(str) \
    { static std::once_flag warn_flag; std::call_once(warn_flag, []() { GGML_LOG_WARN(str); }); }

[[noreturn]]
void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    int id = -1; // in case cudaGetDevice fails
    (void)cudaGetDevice(&id);

    GGML_LOG_ERROR(GGML_CUDA_NAME " error: %s\n", msg);
    GGML_LOG_ERROR("  current device: %d, in function %s at %s:%d\n", id, func, file, line);
    GGML_LOG_ERROR("  %s\n", stmt);
    // abort with GGML_ABORT to get a stack trace
    GGML_ABORT(GGML_CUDA_NAME " error");
}

// map a (possibly virtual) device id to the physical CUDA device that backs it
static int ggml_cuda_get_physical_device(int device) {
    const ggml_cuda_device_info & info = ggml_cuda_info();
    GGML_ASSERT(device >= 0 && device < info.device_count);
    return info.devices[device].physical_device;
}

// this is faster on Windows
// probably because the Windows CUDA libraries forget to make this check before invoking the drivers
void ggml_cuda_set_device(int device) {
    // translate the (possibly virtual) device id to the physical CUDA device that backs it
    const int physical_device = ggml_cuda_get_physical_device(device);

    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device));

    if (physical_device == current_device) {
        return;
    }

    CUDA_CHECK(cudaSetDevice(physical_device));
}

int ggml_cuda_get_device() {
    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    return id;
}

static cudaError_t ggml_cuda_device_malloc(void ** ptr, size_t size, int device) {
    ggml_cuda_set_device(device);
    cudaError_t err;
    if (getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY") != nullptr) {
        err = cudaMallocManaged(ptr, size);
#if defined(GGML_USE_HIP)
        if (err == hipSuccess) {
            // hipMemAdviseSetCoarseGrain is an optional performance hint;
            // ignore errors (e.g. hipErrorInvalidValue on some APU/iGPU configs).
            (void)cudaMemAdvise(*ptr, size, hipMemAdviseSetCoarseGrain, device);
            (void)hipGetLastError(); // clear any error
        }

        // fall back to cudaMalloc if not supported (e.g. on Windows)
        if (err == hipErrorNotSupported) {
            static bool warned_unsupported = false;
            if (!warned_unsupported) {
                GGML_LOG_WARN("hipMallocManaged unsupported, falling back to hipMalloc.\n");
                warned_unsupported = true;
            }

            err = cudaMalloc(ptr, size);
        }
#endif // defined(GGML_USE_HIP)
    } else {
        err = cudaMalloc(ptr, size);
    }
    return err;
}

#if defined(GGML_USE_HIP)
static int ggml_cuda_parse_id(char devName[]) {
    // A list of possible Target IDs can be found under the rocclr/clr repo in device.cpp
    // these values are not stable so this is susceptible to breakage
    // https://github.com/ROCm/clr/blob/amd-staging/rocclr/device/device.cpp
    int archMajor = 0x0;
    int archMinor = 0x0;
    int archNum = GGML_CUDA_CC_OFFSET_AMD;
    int archLen = strlen(devName);
    char archName[archLen + 1];

    // strip leading 'gfx' while copying into our buffer
    if (archLen > 3) {
        strcpy(archName, &devName[3]);
        archLen -= 3;
    }

    // trim trailing :xnack- or :sramecc- statuses
    archLen = strcspn(archName, ":");
    archName[archLen] = '\0';

    // tease out the version information
    if (archLen > 8) {
        // versions labeled generic use '-' as delimiter
        // strip the trailing "-generic" then iterate through what remains
        if ((strstr(archName, "-generic"))) {
            archName[archLen - 8] = '\0';
            char * pch;
            if ((pch = strtok(archName, "-"))) {
                archMajor = (int)strtoul(pch, 0, 16);
                if ((pch = strtok(NULL, "-"))) {
                    archMinor = 0x10 * (int)strtoul(pch, 0, 16);
                }
            }
        }
    } else if (archLen >= 3) {
        // last two digits should be the minor * 0x10 + stepping
        archMinor = (int)strtoul(&archName[archLen - 2], 0, 16);
        archName[archLen - 2] = '\0';

        // only the major version remains
        archMajor = (int)strtoul(archName, 0, 16);
    }
    archNum += archMajor * 0x100;
    archNum += archMinor;

    return archNum;
}
#endif // defined(GGML_USE_HIP)

static ggml_cuda_device_info ggml_cuda_init() {
    ggml_cuda_device_info info = {};

    cudaError_t err = cudaGetDeviceCount(&info.physical_device_count);
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("%s: failed to initialize " GGML_CUDA_NAME ": %s\n", __func__, cudaGetErrorString(err));
        return info;
    }

    GGML_ASSERT(info.physical_device_count <= GGML_CUDA_MAX_DEVICES);

    // by default expose exactly the physical devices; GGML_CUDA_DEVICES can request a different
    // number of (virtual) devices to emulate multi-GPU systems on a machine with fewer GPUs
    info.device_count = info.physical_device_count;

    const char * devices_env = getenv("GGML_CUDA_DEVICES");
    if (devices_env != nullptr && info.physical_device_count > 0) {
        const int requested = atoi(devices_env);
        if (requested > 0) {
            info.device_count = requested;
        } else {
            GGML_LOG_WARN("%s: ignoring invalid GGML_CUDA_DEVICES=\"%s\"\n", __func__, devices_env);
        }
    }

    if (info.device_count > GGML_CUDA_MAX_DEVICES) {
        GGML_LOG_WARN("%s: requested %d devices, clamping to GGML_CUDA_MAX_DEVICES=%d\n",
                      __func__, info.device_count, GGML_CUDA_MAX_DEVICES);
        info.device_count = GGML_CUDA_MAX_DEVICES;
    }

    // map each (virtual) device to a backing physical device (round-robin), assign each its index
    // among the (virtual) devices sharing that physical GPU, and store the per-physical share count
    int physical_share_count[GGML_CUDA_MAX_DEVICES] = {};
    GGML_ASSERT(info.device_count == 0 || info.physical_device_count > 0);
    for (int id = 0; id < info.device_count; ++id) {
        info.devices[id].physical_device = id % info.physical_device_count;
        info.devices[id].virtual_index  = physical_share_count[info.devices[id].physical_device]++;
    }

    int64_t total_vram = 0;
    for (int id = 0; id < info.physical_device_count; ++id) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
        total_vram += prop.totalGlobalMem;
    }
    GGML_LOG_INFO("%s: found %d " GGML_CUDA_NAME " devices (Total VRAM: %zu MiB):\n",
                  __func__, info.physical_device_count, (size_t)(total_vram / (1024 * 1024)));
    if (info.device_count != info.physical_device_count) {
        GGML_LOG_INFO("%s: emulating %d virtual device(s) on %d physical device(s) (GGML_CUDA_DEVICES)\n",
                      __func__, info.device_count, info.physical_device_count);
    }
    total_vram = 0;

    std::vector<std::pair<int, std::string>> turing_devices_without_mma;
    for (int id = 0; id < info.device_count; ++id) {
        const int physical_id = info.devices[id].physical_device;

        int device_vmm = 0;

#if defined(GGML_USE_VMM)
        CUdevice device;
        CU_CHECK(cuDeviceGet(&device, physical_id));
        CU_CHECK(cuDeviceGetAttribute(&device_vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device));

        if (device_vmm) {
            CUmemAllocationProp alloc_prop = {};
            alloc_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            alloc_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            alloc_prop.location.id = physical_id;
            CU_CHECK(cuMemGetAllocationGranularity(&info.devices[id].vmm_granularity, &alloc_prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        }
#endif // defined(GGML_USE_VMM)
        info.devices[id].vmm = !!device_vmm;

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, physical_id));

        // a virtual device owns only a share of its physical GPU's memory; report that share so the
        // logged per-device VRAM sums to the physical total above.
        GGML_ASSERT(physical_share_count[physical_id] > 0);
        info.devices[id].physical_share_count = physical_share_count[physical_id];
        const size_t device_vram = prop.totalGlobalMem / info.devices[id].physical_share_count;
        const size_t device_vram_mib = device_vram / (1024 * 1024);

        info.default_tensor_split[id] = total_vram;
        total_vram += device_vram;
        info.devices[id].integrated = false; // Temporarily disabled due to issues with corrupted output (e.g. #15034)
        info.devices[id].nsm        = prop.multiProcessorCount;
        info.devices[id].smpb       = prop.sharedMemPerBlock;
        info.devices[id].warp_size  = prop.warpSize;

#ifndef GGML_USE_MUSA
        int supports_coop_launch = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&supports_coop_launch, cudaDevAttrCooperativeLaunch, physical_id));
        info.devices[id].supports_cooperative_launch = !!supports_coop_launch;
#else
        info.devices[id].supports_cooperative_launch = false;
#endif // !(GGML_USE_MUSA)

#if defined(GGML_USE_HIP)
        info.devices[id].smpbo = prop.sharedMemPerBlock;

        info.devices[id].cc = ggml_cuda_parse_id(prop.gcnArchName);
        if ((info.devices[id].cc & 0xff00) == 0x0) {
            GGML_LOG_WARN("invalid architecture ID received for device %d %s: %s  cc %d.%d\n",
                            id, prop.name, prop.gcnArchName, prop.major, prop.minor);

            // Fallback to prop.major and prop.minor
            if (prop.major > 0) {
                info.devices[id].cc = GGML_CUDA_CC_OFFSET_AMD + prop.major * 0x100;
                info.devices[id].cc += prop.minor * 0x10;
            }
        }
        GGML_LOG_INFO("  Device %d: %s, %s (0x%x), VMM: %s, Wave Size: %d, VRAM: %zu MiB\n",
                      id, prop.name, prop.gcnArchName, info.devices[id].cc & 0xffff,
                      device_vmm ? "yes" : "no", prop.warpSize,
                      device_vram_mib);
#elif defined(GGML_USE_MUSA)
        // FIXME: Ensure compatibility with varying warp sizes across different MUSA archs.
        info.devices[id].warp_size = 32;
        info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
        info.devices[id].cc = GGML_CUDA_CC_OFFSET_MTHREADS + prop.major * 0x100;
        info.devices[id].cc += prop.minor * 0x10;
        GGML_LOG_INFO("  Device %d: %s, compute capability %d.%d, VMM: %s, VRAM: %zu MiB\n",
                      id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no",
                      device_vram_mib);
#else
        info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
        info.devices[id].cc = 100*prop.major + 10*prop.minor;
        GGML_LOG_INFO("  Device %d: %s, compute capability %d.%d, VMM: %s, VRAM: %zu MiB\n",
                      id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no",
                      device_vram_mib);
        std::string device_name(prop.name);
        if (device_name == "NVIDIA GeForce MX450") {
            turing_devices_without_mma.push_back({ id, device_name });
        } else if (device_name == "NVIDIA GeForce MX550") {
            turing_devices_without_mma.push_back({ id, device_name });
        } else if (device_name.substr(0, 21) == "NVIDIA GeForce GTX 16") {
            turing_devices_without_mma.push_back({ id, device_name });
        }

        // Temporary performance fix:
        // Setting device scheduling strategy for iGPUs with cc121 to "spinning" to avoid delays in cuda synchronize calls.
        // TODO: Check for future drivers the default scheduling strategy and
        // remove this call again when cudaDeviceScheduleSpin is default.
        if (prop.major == 12 && prop.minor == 1) {
            CUDA_CHECK(cudaSetDevice(physical_id));
            CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
        }

#endif  // defined(GGML_USE_HIP)
    }

    if (ggml_cuda_highest_compiled_arch(GGML_CUDA_CC_TURING) >= GGML_CUDA_CC_TURING && !turing_devices_without_mma.empty()) {
        GGML_LOG_INFO("The following devices will have suboptimal performance due to a lack of tensor cores:\n");
        for (size_t device_pos = 0; device_pos < turing_devices_without_mma.size(); device_pos++) {
            GGML_LOG_INFO(
                "  Device %d: %s\n", turing_devices_without_mma[device_pos].first, turing_devices_without_mma[device_pos].second.c_str());
        }
        GGML_LOG_INFO(
            "Consider compiling with CMAKE_CUDA_ARCHITECTURES=61-virtual;80-virtual and DGGML_CUDA_FORCE_MMQ to force the use of the Pascal code for Turing.\n");
    }

    for (int id = 0; id < info.device_count; ++id) {
        info.default_tensor_split[id] /= total_vram;
    }

    // configure logging to stdout
    // CUBLAS_CHECK(cublasLoggerConfigure(1, 1, 0, nullptr));

    if (getenv("GGML_CUDA_P2P") != nullptr) {
        for (int id = 0; id < info.physical_device_count; ++id) {
            CUDA_CHECK(cudaSetDevice(id));
            for (int id_other = 0; id_other < info.physical_device_count; ++id_other) {
                if (id == id_other) {
                    continue;
                }
                int can_access_peer;
                CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, id, id_other));
                if (can_access_peer) {
                    CUDA_CHECK(cudaDeviceEnablePeerAccess(id_other, 0));
                }
            }
        }
    }

    return info;
}

const ggml_cuda_device_info & ggml_cuda_info() {
    static ggml_cuda_device_info info = ggml_cuda_init();
    return info;
}

// #define DEBUG_CUDA_MALLOC

// buffer pool for cuda (legacy)
struct ggml_cuda_pool_leg : public ggml_cuda_pool {
    static const int MAX_BUFFERS = 256;

    int device;
    struct ggml_cuda_buffer {
        void * ptr = nullptr;
        size_t size = 0;
    };

    ggml_cuda_buffer buffer_pool[MAX_BUFFERS] = {};
    size_t pool_size = 0;

    explicit ggml_cuda_pool_leg(int device) :
        device(device) {
    }

    ~ggml_cuda_pool_leg() {
        clear_pool();
        GGML_ASSERT(pool_size == 0);
    }

    void clear_pool() {
        ggml_cuda_set_device(device);
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer & b = buffer_pool[i];
            if (b.ptr != nullptr) {
                CUDA_CHECK(cudaFree(b.ptr));
                pool_size -= b.size;
                b.ptr  = nullptr;
                b.size = 0;
            }
        }
    }

    void * alloc(size_t size, size_t * actual_size) override {
#ifdef DEBUG_CUDA_MALLOC
        int nnz = 0;
        size_t max_size = 0;
#endif
        size_t best_diff = 1ull << 36;
        int ibest = -1;
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr != nullptr) {
#ifdef DEBUG_CUDA_MALLOC
                ++nnz;
                if (b.size > max_size) max_size = b.size;
#endif
                if (b.size >= size) {
                    size_t diff = b.size - size;
                    if (diff < best_diff) {
                        best_diff = diff;
                        ibest = i;
                        if (!best_diff) {
                            void * ptr = b.ptr;
                            *actual_size = b.size;
                            b.ptr = nullptr;
                            b.size = 0;
                            return ptr;
                        }
                    }
                }
            }
        }
        if (ibest >= 0) {
            ggml_cuda_buffer& b = buffer_pool[ibest];
            void * ptr = b.ptr;
            *actual_size = b.size;
            b.ptr = nullptr;
            b.size = 0;
            return ptr;
        }
        void * ptr;
        size_t look_ahead_size = (size_t) (1.05 * size);
        look_ahead_size = 256 * ((look_ahead_size + 255)/256);
        ggml_cuda_set_device(device);
        cudaError_t err = ggml_cuda_device_malloc(&ptr, look_ahead_size, device);
        if (err == cudaErrorMemoryAllocation) {
            (void)cudaGetLastError();
            const size_t cached_bytes = pool_size;
            GGML_LOG_DEBUG(GGML_CUDA_NAME " pool[%d]: alloc of %.2f MiB failed, flushing %.2f MiB of cached buffers and retrying\n",
                           device, look_ahead_size/1024.0/1024.0, cached_bytes/1024.0/1024.0);
            CUDA_CHECK(cudaDeviceSynchronize());
            clear_pool();
            err = ggml_cuda_device_malloc(&ptr, look_ahead_size, device);
            if (err == cudaSuccess) {
                GGML_LOG_DEBUG(GGML_CUDA_NAME " pool[%d]: retry succeeded\n", device);
            }
        }
        CUDA_CHECK(err);
        *actual_size = look_ahead_size;
        pool_size += look_ahead_size;
#ifdef DEBUG_CUDA_MALLOC
        GGML_LOG_INFO("%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, requested %u MB\n", __func__, device, nnz,
                           (uint32_t)(max_size / 1024 / 1024), (uint32_t)(pool_size / 1024 / 1024), (uint32_t)(size / 1024 / 1024));
#endif
        return ptr;
    }

    void free(void * ptr, size_t size) override {
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr == nullptr) {
                b.ptr = ptr;
                b.size = size;
                return;
            }
        }
        GGML_LOG_DEBUG(GGML_CUDA_NAME " buffer pool full, increase MAX_CUDA_BUFFERS\n");
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaFree(ptr));
        pool_size -= size;
    }
};

#if defined(GGML_USE_VMM)
// VMM mappings are not covered by cudaDeviceEnablePeerAccess: grant access to the owning device and, with P2P, to its peers
static void ggml_cuda_vmm_set_access(CUdeviceptr ptr, size_t size, int physical_device) {
    // VMM Bug fix for P2P access if GGML_CUDA_P2P is set, or if NCCL build
    bool use_peer_access = getenv("GGML_CUDA_P2P") != nullptr;
#if defined(GGML_USE_NCCL)
    use_peer_access = true;
#endif // defined(GGML_USE_NCCL)

    if (use_peer_access) {
        // NCCL implicitly enables peer access (cudaDeviceEnablePeerAccess), and
        // GGML_CUDA_P2P enables it explicitly. Unlike cudaMalloc buffers, VMM
        // allocations do not become peer-accessible from that alone, so access
        // must be granted explicitly here. With virtual devices, grant access
        // on the backing *physical* devices (deduplicated, since several
        // virtual devices can map to the same physical GPU).
        std::vector<CUmemAccessDesc> access_descs;
        bool physical_seen[GGML_CUDA_MAX_DEVICES] = {};
        const int device_count = ggml_cuda_info().device_count;
        for (int id = 0; id < device_count; ++id) {
            const int id_physical = ggml_cuda_get_physical_device(id);
            if (id_physical != physical_device) {
                int can_access_peer = 0;
                CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, id_physical, physical_device));
                if (!can_access_peer) {
                    continue;
                }
            }
            if (physical_seen[id_physical]) {
                continue;
            }
            physical_seen[id_physical] = true;
            CUmemAccessDesc access = {};
            access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            access.location.id = id_physical;
            access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            access_descs.push_back(access);
        }
        CU_CHECK(cuMemSetAccess(ptr, size, access_descs.data(), access_descs.size()));
    } else {
        // set access for non P2P
        CUmemAccessDesc access = {};
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = physical_device;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CU_CHECK(cuMemSetAccess(ptr, size, &access, 1));
    }
}

// pool with virtual memory
struct ggml_cuda_pool_vmm : public ggml_cuda_pool {
    static const size_t CUDA_POOL_VMM_MAX_SIZE = 1ull << 35; // 32 GB

    int device;
    int physical_device;
    CUdeviceptr pool_addr = 0;
    size_t pool_used = 0;
    size_t pool_size = 0;
    size_t granularity;
#if defined(GGML_USE_HIP)
    std::vector<std::pair<CUdeviceptr, size_t>> mappings;
#endif

    explicit ggml_cuda_pool_vmm(int device) :
        device(device),
        physical_device(ggml_cuda_get_physical_device(device)),
        granularity(ggml_cuda_info().devices[device].vmm_granularity) {
    }

    ~ggml_cuda_pool_vmm() {
        if (pool_addr != 0) {
#if defined(GGML_USE_HIP)
            // Workaround for https://github.com/ROCm/ROCR-Runtime/issues/285
            for (std::pair<CUdeviceptr, size_t> & mapping : mappings) {
                CU_CHECK(cuMemUnmap(mapping.first, mapping.second));
            }
#else
            CU_CHECK(cuMemUnmap(pool_addr, pool_size));
#endif
            CU_CHECK(cuMemAddressFree(pool_addr, CUDA_POOL_VMM_MAX_SIZE));
        }
    }

    void * alloc(size_t size, size_t * actual_size) override {
        // round up the allocation size to the alignment to ensure that all allocations are aligned for all data types
        const size_t alignment = 128;
        size = alignment * ((size + alignment - 1) / alignment);

        size_t avail = pool_size - pool_used;

        if (size > avail) {
            // round up to the next multiple of the granularity
            size_t reserve_size = size - avail;
            reserve_size = granularity * ((reserve_size + granularity - 1) / granularity);

            GGML_ASSERT(pool_size + reserve_size <= CUDA_POOL_VMM_MAX_SIZE);

            // allocate more physical memory
            CUmemAllocationProp prop = {};
            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = physical_device;
            CUmemGenericAllocationHandle handle;
            CU_CHECK(cuMemCreate(&handle, reserve_size, &prop, 0));

            // reserve virtual address space (if not already reserved)
            if (pool_addr == 0) {
                CU_CHECK(cuMemAddressReserve(&pool_addr, CUDA_POOL_VMM_MAX_SIZE, 0, 0, 0));
            }

            // map at the end of the pool
            CUdeviceptr start_ptr = (CUdeviceptr)((char *)(pool_addr) + pool_size);
            CU_CHECK(cuMemMap(start_ptr, reserve_size, 0, handle, 0));
#if defined(GGML_USE_HIP)
            mappings.push_back({start_ptr, reserve_size});
#endif

            // the memory allocation handle is no longer needed after mapping
            CU_CHECK(cuMemRelease(handle));

            ggml_cuda_vmm_set_access(start_ptr, reserve_size, physical_device);

            // add to the pool
            pool_size += reserve_size;

            //printf("cuda pool[%d]: size increased to %llu MB (reserved %llu MB)\n",
            //       device, (unsigned long long) (pool_size/1024/1024),
            //       (unsigned long long) (reserve_size/1024/1024));
        }

        GGML_ASSERT(pool_addr != 0);

        void * ptr = (void *) ((CUdeviceptr)((char *)(pool_addr) + pool_used));
        *actual_size = size;
        pool_used += size;

#ifdef DEBUG_CUDA_MALLOC
        printf("cuda pool[%d]: allocated %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

        return ptr;
    }

    void free(void * ptr, size_t size) override {
#ifdef DEBUG_CUDA_MALLOC
        printf("cuda pool[%d]: freed %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

        pool_used -= size;

        // all deallocations must be in reverse order of the allocations
        GGML_ASSERT(ptr == (void *) ((char *)(pool_addr) + pool_used));
    }
};
#endif // defined(GGML_USE_VMM)

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int                  device,
                                                                               [[maybe_unused]] int stream_no) {
#if defined(GGML_USE_VMM)
    if (ggml_cuda_info().devices[device].vmm) {
        return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_vmm(device));
    }
#endif // defined(GGML_USE_VMM)
    return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_leg(device));
}

// destroying a cuBLAS handle while a graph is being captured in a different thread can result in a CUDA error
// this lock is used to ensure that no cuBLAS handle is destroyed while a graph is being captured

static std::mutex ggml_cuda_lock;
static std::condition_variable ggml_cuda_lock_cv;
static std::atomic<int> ggml_cuda_lock_counter;

// Sample the first evaluated graph of each node count; this is not a replay census.
struct ggml_cuda_ft_graph_census {
    struct registry {
        std::mutex mutex;
        std::map<const void *, std::map<int, bool>> seen;
    };
    static bool enabled() {
        static const bool value = [] {
            const char * env = std::getenv("GGML_FT_GRAPH_CENSUS");
            return env && std::strcmp(env, "1") == 0;
        }();
        return value;
    }
    static registry & get_registry() {
        static registry value;
        return value;
    }
    static void drop(const void * ctx) {
        if (!enabled()) { return; }
        auto & r = get_registry();
        std::lock_guard<std::mutex> lock(r.mutex);
        r.seen.erase(ctx);
    }
    const void * ctx;
    int nodes;
    bool sample = false;
    std::map<std::string, int> counts;

    ggml_cuda_ft_graph_census(const void * ctx, int nodes, bool evaluating) : ctx(ctx), nodes(nodes) {
        if (!enabled() || !evaluating) { return; }
        auto & r = get_registry();
        std::lock_guard<std::mutex> lock(r.mutex);
        auto & seen = r.seen[ctx];
        if (seen.size() < 64 && !seen.count(nodes)) {
            seen[nodes] = true;
            sample = true;
        }
    }
    void record(const ggml_tensor * node, int skipped) {
        if (!sample) { return; }
        std::string family;
        for (const char * p = node->name; *p; ++p) {
            if (*p >= '0' && *p <= '9') {
                if (family.empty() || family.back() != '#') { family += '#'; }
            } else { family += *p; }
        }
        std::string key = std::string(ggml_op_desc(node)) + " family=" + family;
        for (int d = 0; d < 4; ++d) { key += " ne" + std::to_string(d) + "=" + std::to_string(node->ne[d]); }
        key += " skipped=" + std::to_string(skipped);
        ++counts[key];
    }
    ~ggml_cuda_ft_graph_census() {
        if (!sample) { return; }
        for (const auto & entry : counts) {
            fprintf(stderr, "FreeToken graph census ctx=%p nodes=%d dispatches=%d op=%s\n",
                    ctx, nodes, entry.second, entry.first.c_str());
        }
    }
};

// Host durations can include GPU waits; launch time is nested in graph time.
static bool ggml_cuda_ft_host_timing_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_FT_HOST_TIMING");
        return value && std::atoi(value) != 0;
    }();
    return enabled;
}

struct ggml_cuda_ft_host_timing {
    struct counter {
        int64_t calls = 0;
        int64_t us = 0;
        int64_t nodes = 0;
    };
    std::mutex mutex;
    std::map<const void *, std::array<counter, 8>> contexts;
};

static ggml_cuda_ft_host_timing & ggml_cuda_ft_host_stats() {
    static ggml_cuda_ft_host_timing stats;
    return stats;
}

static void ggml_cuda_ft_host_record(const void * ctx, int kind, int64_t start, int nodes = 0) {
    if (!ggml_cuda_ft_host_timing_enabled()) {
        return;
    }
    const int64_t elapsed = ggml_time_us() - start;
    auto & stats = ggml_cuda_ft_host_stats();
    std::lock_guard<std::mutex> lock(stats.mutex);
    auto & count = stats.contexts[ctx][kind];
    ++count.calls;
    count.us += elapsed;
    count.nodes += nodes;
    static const int64_t period = [] {
        const char * value = std::getenv("GGML_FT_HOST_SNAPSHOT_PERIOD");
        return value && std::atoll(value) > 0 ? std::atoll(value) : 0;
    }();
    const auto & all = stats.contexts[ctx];
    const int64_t passes = all[0].calls + all[1].calls + all[2].calls;
    if (period && kind <= 2 && passes % period == 0) {
        fprintf(stderr, "FreeToken host snapshot ctx=%p passes=%" PRId64 " eager=%" PRId64 " capture=%" PRId64 " replay=%" PRId64 " instantiate=%" PRId64 " exec_update=%" PRId64 " reinstantiate=%" PRId64 "\n",
                ctx, passes, all[0].calls, all[1].calls, all[2].calls,
                all[5].calls, all[6].calls, all[7].calls);
    }
}

static void ggml_cuda_ft_host_report(const void * ctx) {
    if (!ggml_cuda_ft_host_timing_enabled()) {
        return;
    }
    auto & stats = ggml_cuda_ft_host_stats();
    std::lock_guard<std::mutex> lock(stats.mutex);
    const auto it = stats.contexts.find(ctx);
    if (it == stats.contexts.end()) {
        return;
    }
    const char * names[] = { "eager", "capture", "replay", "launch", "synchronize", "instantiate", "exec_update", "reinstantiate" };
    for (int i = 0; i < 8; ++i) {
        const auto & count = it->second[i];
        fprintf(stderr, "FreeToken host ctx=%p kind=%s calls=%" PRId64 " total_us=%" PRId64 " nodes=%" PRId64 "\n",
                      ctx, names[i], count.calls, count.us, count.nodes);
    }
    stats.contexts.erase(it);
}

#ifdef USE_CUDA_GRAPH
static thread_local int ggml_cuda_prefill_capture_hint = -1;

// V436 is private, default-off, and never prepares a graph by eager execution.
#if defined(__linux__)
#include <dlfcn.h>
#endif
#if defined(__linux__) && CUDART_VERSION >= 12000 && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
static thread_local bool ggml_cuda_prepare_only_v436 = false;
static thread_local bool ggml_cuda_prepare_selected_v436 = false;
namespace ggml_cuda_prepare_v436 {
using prepare_fn = cudaError_t (*)(const void *, const void *, cudaGraph_t, cudaStream_t, int, int *);
using can_prepare_fn = cudaError_t (*)(const void *, const void *, int, int *);
struct api {
    void * handle = nullptr;
    prepare_fn prepare = nullptr;
    can_prepare_fn can_prepare = nullptr;
    api() {
        const char * path = std::getenv("GGML_CUDA_GRAPH_DEVICE_TAIL_LIBRARY");
        if (!path || !*path) return;
        // Retain this extra reference until process exit: uploaded graph handles
        // and their launch kernels must outlive every backend owner.
        handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (!handle) return;
        prepare = reinterpret_cast<prepare_fn>(dlsym(handle, "ac922_tail_bridge_v3_prepare_captured"));
        can_prepare = reinterpret_cast<can_prepare_fn>(dlsym(handle, "ac922_tail_bridge_v3_can_prepare"));
    }
    bool available() const { return prepare && can_prepare; }
};
static api & get_api() { static api value; return value; }
}
#endif


#if defined(__linux__) && CUDART_VERSION >= 12000 && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
static bool ggml_cuda_graph_capture_only_requested(const ggml_backend_cuda_context * ctx) {
    return ctx != nullptr && ggml_cuda_prefill_capture_hint == 1 &&
            ggml_cuda_info().devices[ctx->device].cc == GGML_CUDA_CC_VOLTA &&
            ggml_cuda_device_tail_client::requested() &&
            ggml_cuda_device_tail_capture_only_client::requested();
}

static bool ggml_cuda_graph_capture_only_ready(
        const ggml_backend_cuda_context * ctx, const ggml_cuda_graph * graph) {
    if (!graph || graph->instance != nullptr || graph->graph == nullptr ||
            !graph->warmup_complete || !graph->is_enabled() ||
            !ggml_cuda_graph_capture_only_requested(ctx)) return false;
    int ready = 0;
    CUDA_CHECK(ggml_cuda_device_tail_capture_only_client::ready(
            ctx, graph, graph->graph, ctx->device, &ready));
    return ready == 1;
}
#else
static bool ggml_cuda_graph_capture_only_requested(const ggml_backend_cuda_context *) {
    return false;
}
static bool ggml_cuda_graph_capture_only_ready(
        const ggml_backend_cuda_context *, const ggml_cuda_graph *) {
    return false;
}
#endif

static bool ggml_cuda_graph_has_executable(
        const ggml_backend_cuda_context * ctx, const ggml_cuda_graph * graph) {
    return graph != nullptr && (graph->instance != nullptr ||
            ggml_cuda_graph_capture_only_ready(ctx, graph));
}

static bool ggml_cuda_ft_graph_cache_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_FT_GRAPH_CACHE");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

struct ggml_cuda_ft_graph_variants {
    struct entry {
        const void * key;
        std::unique_ptr<ggml_cuda_graph> graph;
    };
    struct context {
        std::vector<entry> entries;
        size_t hits = 0, saves = 0, evictions = 0;
    };
    std::mutex mutex;
    std::map<const void *, context> contexts;
};

static ggml_cuda_ft_graph_variants & ggml_cuda_ft_graph_variants_get() {
    static ggml_cuda_ft_graph_variants variants;
    return variants;
}

static void ggml_cuda_ft_graph_cache_drop(const void * ctx) {
    if (!ggml_cuda_ft_graph_cache_enabled()) { return; }
    auto & variants = ggml_cuda_ft_graph_variants_get();
    std::lock_guard<std::mutex> lock(variants.mutex);
    const auto it = variants.contexts.find(ctx);
    if (it != variants.contexts.end()) {
        const auto & c = it->second;
        fprintf(stderr, "FreeToken graph cache ctx=%p hits=%zu saves=%zu evictions=%zu retained=%zu\n",
                ctx, c.hits, c.saves, c.evictions, c.entries.size());
        variants.contexts.erase(it);
    }
}
#endif

#ifdef USE_CUDA_GRAPH
struct ggml_cuda_graph_structural_stats {
    size_t passes = 0;
    size_t attempts = 0;
    size_t builds = 0;
    size_t reject_too_large = 0;
    size_t reject_graph = 0;
    size_t reject_external = 0;
    size_t reject_missing = 0;
    std::array<size_t, size_t(ggml_cuda_graph_structural::build_failure::count)> failures = {};
    size_t active_matches = 0;
    size_t variant_matches = 0;
    size_t lookup_misses = 0;
    size_t update_layout_candidates = 0;
    size_t update_layout_misses = 0;
    size_t cross_descriptor_replays = 0;
    size_t flat_snapshot_hits = 0;
    size_t flat_snapshot_misses = 0;
    int64_t signature_build_us = 0;
    int64_t lookup_us = 0;
    size_t diagnostic_miss_reports = 0;
    size_t selected_stale_touches = 0;
    size_t binding_keys_created = 0;
    size_t binding_keys_reused = 0;
    size_t binding_key_cap_fallbacks = 0;
    std::array<size_t, 3> path_calls = {}; // eager, capture, replay
    std::array<size_t, 5> fallback_calls = {}; // warmup, changed, incompatible, env, architecture
    bool path_config_logged = false;
};

struct ggml_cuda_graph_structural_record {
    const ggml_backend_cuda_context * owner = nullptr;
    ggml_cuda_graph_structural::graph_signature signature;
};

struct ggml_cuda_graph_structural_registry {
    std::mutex mutex;
    std::unordered_map<const ggml_cuda_graph *, ggml_cuda_graph_structural_record> graphs;
    std::unordered_map<const ggml_backend_cuda_context *, ggml_cuda_graph_structural_stats> contexts;
    std::unordered_map<const ggml_backend_cuda_context *, int64_t> last_prune_us;
    // Opaque identities, never tensor pointers. Keep tokens until context teardown
    // so a retired CUDA graph key can never alias a later binding in this context.
    std::unordered_map<const ggml_backend_cuda_context *, std::vector<std::unique_ptr<uint8_t>>> binding_keys;
};

static ggml_cuda_graph_structural_registry & ggml_cuda_graph_structural_registry_get() {
    static ggml_cuda_graph_structural_registry registry;
    return registry;
}

#ifdef USE_CUDA_GRAPH
static int ggml_cuda_graph_lease_retain_ms() {
    static const int retain_ms = ggml_cuda_graph_lease::parse_retain_ms(
            std::getenv("GGML_CUDA_GRAPH_RETAIN_MS"));
    return retain_ms;
}

static void ggml_cuda_graph_lease_refresh_context(ggml_backend_cuda_context * cuda_ctx) {
    const int retain_ms = ggml_cuda_graph_lease_retain_ms();
    if (retain_ms <= 0 || cuda_ctx == nullptr) return;
    const int64_t now_us = ggml_time_us();
    const size_t refreshed = ggml_cuda_graph_lease::get_registry().refresh_context(cuda_ctx, now_us, retain_ms,
            [cuda_ctx](const void * key) -> const void * {
                const auto active = cuda_ctx->cuda_graphs.find(key);
                if (active == cuda_ctx->cuda_graphs.end() || !active->second->is_enabled() ||
                        !ggml_cuda_graph_has_executable(cuda_ctx, active->second.get())) return nullptr;
                return active->second.get();
            },
            [cuda_ctx](const void * key, const void * graph, int64_t stamp_us) {
                const auto active = cuda_ctx->cuda_graphs.find(key);
                if (active != cuda_ctx->cuda_graphs.end() && active->second.get() == graph) {
                    active->second->last_used_time = stamp_us;
                }
            });
    if (refreshed != 0) {
        static std::array<std::atomic<size_t>, GGML_CUDA_MAX_DEVICES> refresh_notices{};
        const int device = cuda_ctx->device;
        if (device >= 0 && device < GGML_CUDA_MAX_DEVICES &&
                refresh_notices[size_t(device)].fetch_add(1, std::memory_order_relaxed) < 8) {
            GGML_LOG_INFO("CUDA graph lease refresh ctx=%p device=%d graphs=%zu ttl_ms=%d actual_expiry_unchanged=1\n",
                    (void *) cuda_ctx, device, refreshed, retain_ms);
        }
    }
}

static void ggml_cuda_graph_lease_touch_actual_compute(
        ggml_backend_cuda_context * cuda_ctx, const void * key, ggml_cuda_graph * graph,
        int64_t actual_use_us, int retain_ms) {
    if (retain_ms <= 0 || cuda_ctx == nullptr || key == nullptr || graph == nullptr) return;
    const auto result = ggml_cuda_graph_lease::get_registry().touch_actual(
            cuda_ctx, key, graph, actual_use_us, retain_ms,
            [cuda_ctx](const void * candidate_key) -> const void * {
                const auto active = cuda_ctx->cuda_graphs.find(candidate_key);
                if (active == cuda_ctx->cuda_graphs.end() || !active->second->is_enabled() ||
                        !ggml_cuda_graph_has_executable(cuda_ctx, active->second.get())) return nullptr;
                return active->second.get();
            });
    const auto active = cuda_ctx->cuda_graphs.find(key);
    if ((result == ggml_cuda_graph_lease::touch_result::created ||
            result == ggml_cuda_graph_lease::touch_result::refreshed) &&
            active != cuda_ctx->cuda_graphs.end() && active->second.get() == graph) {
        graph->last_used_time = actual_use_us;
        static std::array<std::atomic<size_t>, GGML_CUDA_MAX_DEVICES> admission_notices{};
        const int device = cuda_ctx->device;
        if (device >= 0 && device < GGML_CUDA_MAX_DEVICES &&
                admission_notices[size_t(device)].fetch_add(1, std::memory_order_relaxed) < 8) {
            GGML_LOG_INFO("CUDA graph lease admitted=1 result=%s ctx=%p device=%d key=%p graph=%p ttl_ms=%d actual_prefill_compute=1\n",
                    result == ggml_cuda_graph_lease::touch_result::created ? "created" : "renewed",
                    (void *) cuda_ctx, device, key, (void *) graph, retain_ms);
        }
    } else if (result == ggml_cuda_graph_lease::touch_result::capacity_full ||
            result == ggml_cuda_graph_lease::touch_result::inactive_generation) {
        static std::array<std::atomic<size_t>, GGML_CUDA_MAX_DEVICES> decline_notices{};
        const int device = cuda_ctx->device;
        if (device >= 0 && device < GGML_CUDA_MAX_DEVICES &&
                decline_notices[size_t(device)].fetch_add(1, std::memory_order_relaxed) < 8) {
            GGML_LOG_INFO("CUDA graph lease admitted=0 reason=%s ctx=%p device=%d key=%p graph=%p cap=%zu\n",
                    result == ggml_cuda_graph_lease::touch_result::capacity_full ? "context_cap" : "generation_mismatch",
                    (void *) cuda_ctx, device, key, (void *) graph,
                    ggml_cuda_graph_lease::kMaxLeasesPerContext);
        }
    }
}
#endif // USE_CUDA_GRAPH

static bool ggml_cuda_graph_binding_keys_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_CUDA_GRAPH_BINDING_KEYS");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

// Pure identity policy: reuse only keys absent from both active and retained
// graph sets. Neither choosing a key nor recycling it authorizes CUDA replay.
static const void * ggml_cuda_graph_acquire_binding_key(
        std::vector<std::unique_ptr<uint8_t>> & keys,
        const std::unordered_set<const void *> & live_keys,
        size_t cap, bool & reused) {
    reused = false;
    if (cap < 1 || cap > 64) return nullptr;
    for (const auto & key : keys) {
        if (live_keys.count(key.get()) == 0) {
            reused = true;
            return key.get();
        }
    }
    if (keys.size() >= cap) return nullptr;
    keys.push_back(std::make_unique<uint8_t>(0));
    return keys.back().get();
}

template <typename F>
static void ggml_cuda_graph_structural_stats_update(const ggml_backend_cuda_context * ctx, F && update) {
    auto & registry = ggml_cuda_graph_structural_registry_get();
    std::lock_guard<std::mutex> lock(registry.mutex);
    update(registry.contexts[ctx]);
}

enum class ggml_cuda_graph_structural_match { unknown, equal, different };

static ggml_cuda_graph_structural_match ggml_cuda_graph_structural_compare(
        const ggml_backend_cuda_context * ctx, const ggml_cuda_graph * graph,
        const ggml_cuda_graph_structural::graph_signature & candidate) {
    auto & registry = ggml_cuda_graph_structural_registry_get();
    std::lock_guard<std::mutex> lock(registry.mutex);
    const auto it = registry.graphs.find(graph);
    if (it == registry.graphs.end() || it->second.owner != ctx) {
        return ggml_cuda_graph_structural_match::unknown;
    }
    // A graph-map/variant eviction may have freed this graph object and its
    // address been reused. A fresh graph has no saved node properties yet.
    if (graph->node_props.size() != it->second.signature.nodes.size()) {
        registry.graphs.erase(it);
        return ggml_cuda_graph_structural_match::unknown;
    }
    return it->second.signature == candidate ? ggml_cuda_graph_structural_match::equal :
        ggml_cuda_graph_structural_match::different;
}

// Diagnostic only: this never selects a key, declares readiness, or permits
// replay. A real update path additionally needs buffer-class/alias validation
// and a complete recapture followed by transactional cudaGraphExecUpdate.
static bool ggml_cuda_graph_structural_has_update_layout(
        const ggml_backend_cuda_context * ctx, const ggml_cuda_graph * graph,
        const ggml_cuda_graph_structural::graph_signature & candidate) {
    if (!ggml_cuda_graph_has_executable(ctx, graph) ||
            !graph->warmup_complete || !graph->is_enabled()) return false;
    auto & registry = ggml_cuda_graph_structural_registry_get();
    std::lock_guard<std::mutex> lock(registry.mutex);
    const auto it = registry.graphs.find(graph);
    return it != registry.graphs.end() && it->second.owner == ctx &&
           graph->node_props.size() == it->second.signature.nodes.size() &&
           ggml_cuda_graph_structural::same_update_layout(it->second.signature, candidate);
}

static void ggml_cuda_graph_structural_store(const ggml_backend_cuda_context * ctx,
        const ggml_cuda_graph * graph, ggml_cuda_graph_structural::graph_signature signature) {
    auto & registry = ggml_cuda_graph_structural_registry_get();
    std::lock_guard<std::mutex> lock(registry.mutex);
    registry.graphs[graph] = {ctx, std::move(signature)};
}

static void ggml_cuda_graph_structural_erase(const ggml_backend_cuda_context * ctx, const ggml_cuda_graph * graph) {
    auto & registry = ggml_cuda_graph_structural_registry_get();
    std::lock_guard<std::mutex> lock(registry.mutex);
    const auto it = registry.graphs.find(graph);
    if (it != registry.graphs.end() && it->second.owner == ctx) {
        registry.graphs.erase(it);
    }
}

static void ggml_cuda_graph_log_signature_miss(const ggml_backend_cuda_context * ctx,
        const void * raw_key, const ggml_cuda_graph_structural::graph_signature & candidate) {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_CUDA_GRAPH_MISS_TRACE");
        return value && std::strcmp(value, "1") == 0;
    }();
    if (!enabled) return;
    auto & registry = ggml_cuda_graph_structural_registry_get();
    std::lock_guard<std::mutex> lock(registry.mutex);
    auto & stats = registry.contexts[ctx];
    static const size_t report_limit = [] {
        const char * text = std::getenv("GGML_CUDA_GRAPH_MISS_TRACE_LIMIT");
        if (!text || !*text) return size_t(8);
        char * end = nullptr;
        const long value = std::strtol(text, &end, 10);
        return end != text && *end == '\0' && value >= 1 && value <= 128 ? size_t(value) : size_t(8);
    }();
    if (stats.diagnostic_miss_reports >= report_limit) return;
    const ggml_cuda_graph_structural_record * reference = nullptr;
    const char * reference_kind = "none";
    const auto raw = ctx->cuda_graphs.find(raw_key);
    if (raw != ctx->cuda_graphs.end()) {
        const auto stored = registry.graphs.find(raw->second.get());
        if (stored != registry.graphs.end() && stored->second.owner == ctx) {
            reference = &stored->second;
            reference_kind = "raw_key_signature";
        }
    }
    if (!reference) {
        for (const auto & item : registry.graphs) {
            const auto & stored = item.second;
            if (stored.owner == ctx && stored.signature.nodes.size() == candidate.nodes.size() &&
                    stored.signature.external_bindings.size() == candidate.external_bindings.size()) {
                reference = &stored;
                reference_kind = "same_size_owned_signature";
                break;
            }
        }
    }
    // A first graph has no comparison. Save the bounded reports for misses
    // which can actually identify a changed field.
    if (!reference) return;
    ++stats.diagnostic_miss_reports;
    const auto diff = ggml_cuda_graph_diagnostic::first_difference(reference->signature, candidate);
    int same_data = -1, same_buffer = -1, same_shape = -1, same_params = -1;
    const auto * old_nodes = std::strcmp(diff.section, "node") == 0 ? &reference->signature.nodes :
            std::strcmp(diff.section, "external") == 0 ? &reference->signature.external_bindings : nullptr;
    const auto * new_nodes = std::strcmp(diff.section, "node") == 0 ? &candidate.nodes :
            std::strcmp(diff.section, "external") == 0 ? &candidate.external_bindings : nullptr;
    if (old_nodes && new_nodes && diff.index < old_nodes->size() && diff.index < new_nodes->size()) {
        const auto & a = (*old_nodes)[diff.index].descriptor;
        const auto & b = (*new_nodes)[diff.index].descriptor;
        same_data = a.data == b.data;
        same_buffer = a.buffer == b.buffer;
        same_shape = a.ne == b.ne && a.nb == b.nb;
        same_params = a.op == b.op && a.op_params == b.op_params;
        GGML_LOG_INFO("CUDA graph signature detail ctx=%p device=%d raw_key=%p section=%s index=%zu"
                " old_name=%.64s new_name=%.64s old_ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]"
                " new_ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]"
                " old_data=%p new_data=%p old_view_offs=%zu new_view_offs=%zu diagnostic_only=1\n",
                (const void *) ctx, ctx->device, raw_key, diff.section, diff.index, a.name.data(), b.name.data(),
                a.ne[0], a.ne[1], a.ne[2], a.ne[3], b.ne[0], b.ne[1], b.ne[2], b.ne[3],
                (const void *) a.data, (const void *) b.data, a.view_offs, b.view_offs);
    }
    GGML_LOG_INFO("CUDA graph signature miss ctx=%p device=%d reference=%s section=%s index=%zu"
            " field=%s same_data=%d same_buffer=%d same_shape_strides=%d same_op_params=%d"
            " diagnostic_only=1 decision_unchanged=1 report=%zu/%zu\n",
            (const void *) ctx, ctx->device, reference_kind, diff.section, diff.index, diff.field,
            same_data, same_buffer, same_shape, same_params, stats.diagnostic_miss_reports, report_limit);
}

static void ggml_cuda_graph_structural_prune(const ggml_backend_cuda_context * ctx) {
    auto & registry = ggml_cuda_graph_structural_registry_get();
    const int64_t now = ggml_time_us();
    {
        std::lock_guard<std::mutex> lock(registry.mutex);
        const auto last = registry.last_prune_us.find(ctx);
        if (last != registry.last_prune_us.end() && now - last->second < 5'000'000) {
            return;
        }
    }

    std::unordered_set<const ggml_cuda_graph *> live;
    live.reserve(ctx->cuda_graphs.size());
    for (const auto & [key, graph] : ctx->cuda_graphs) {
        GGML_UNUSED(key);
        live.insert(graph.get());
    }

    auto & variants = ggml_cuda_ft_graph_variants_get();
    std::unique_lock<std::mutex> variants_lock(variants.mutex, std::defer_lock);
    if (ggml_cuda_ft_graph_cache_enabled()) {
        variants_lock.lock();
        const auto context = variants.contexts.find(ctx);
        if (context != variants.contexts.end()) {
            for (const auto & entry : context->second.entries) {
                live.insert(entry.graph.get());
            }
        }
    }

    std::lock_guard<std::mutex> lock(registry.mutex);
    const auto last = registry.last_prune_us.find(ctx);
    if (last != registry.last_prune_us.end() && now - last->second < 5'000'000) {
        return;
    }
    registry.last_prune_us[ctx] = now;
    for (auto it = registry.graphs.begin(); it != registry.graphs.end(); ) {
        if (it->second.owner == ctx && live.count(it->first) == 0) {
            it = registry.graphs.erase(it);
        } else {
            ++it;
        }
    }
}

static bool ggml_cuda_graph_structural_enabled();

static bool ggml_cuda_graph_path_logging_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_CUDA_GRAPH_PATH_LOG");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

static void ggml_cuda_graph_log_path(ggml_backend_cuda_context * ctx, const ggml_cgraph * cgraph,
                                    int mode, int reason, const ggml_tensor * incompatible) {
    if (!ggml_cuda_graph_path_logging_enabled()) return;
    ggml_cuda_graph_structural_stats_update(ctx, [&](auto & stats) {
        static const char * modes[] = {"eager", "capture", "replay"};
        static const char * reasons[] = {"initial_or_changed_warmup", "shape_or_binding_changed",
                                        "mul_mat_id_requires_sync", "disabled_by_environment", "unsupported_architecture"};
        const auto flag = [](const char * name) {
            const char * value = std::getenv(name);
            return value && std::strcmp(value, "1") == 0;
        };
        if (!stats.path_config_logged) {
            stats.path_config_logged = true;
            fprintf(stderr, "CUDA graph path config ctx=%p device=%d sm=%d compiled=1 disable_env_present=%d structural_requested=%d structural_active=%d test_policy=%d variant_cache_requested=%d graph_opt_requested=%d\n",
                    (void *) ctx, ctx->device, ggml_cuda_info().devices[ctx->device].cc,
                    getenv("GGML_CUDA_DISABLE_GRAPHS") != nullptr, flag("GGML_CUDA_GRAPH_STRUCTURAL"),
                    ggml_cuda_graph_structural_enabled(), ggml_hardware_test_policy(), flag("GGML_FT_GRAPH_CACHE"), flag("GGML_CUDA_GRAPH_OPT"));
        }
        const size_t mode_count = ++stats.path_calls[mode];
        if (mode == 0) {
            GGML_ASSERT(reason >= 0 && reason < 5);
            const size_t reason_count = ++stats.fallback_calls[reason];
            if (reason_count <= 2) {
                fprintf(stderr, "CUDA graph fallback ctx=%p device=%d uid=%" PRIu64 " nodes=%d mode=eager reason=%s node=%.*s op=%s count=%zu\n",
                        (void *) ctx, ctx->device, uint64_t(cgraph->uid), cgraph->n_nodes, reasons[reason],
                        incompatible ? GGML_MAX_NAME : 0, incompatible ? incompatible->name : "",
                        incompatible ? ggml_op_name(incompatible->op) : "none", reason_count);
            }
        } else if (mode_count <= 2) {
            fprintf(stderr, "CUDA graph path active ctx=%p device=%d uid=%" PRIu64 " nodes=%d mode=%s count=%zu\n",
                    (void *) ctx, ctx->device, uint64_t(cgraph->uid), cgraph->n_nodes, modes[mode], mode_count);
        }
        const size_t passes = stats.path_calls[0] + stats.path_calls[1] + stats.path_calls[2];
        if (passes % 256 == 0) {
            fprintf(stderr, "CUDA graph path summary ctx=%p device=%d passes=%zu eager=%zu capture=%zu replay=%zu warmup=%zu changed=%zu incompatible=%zu env_disabled=%zu arch_disabled=%zu test_policy=%d structural_active=%d actual_replay_seen=%d\n",
                    (void *) ctx, ctx->device, passes, stats.path_calls[0], stats.path_calls[1], stats.path_calls[2],
                    stats.fallback_calls[0], stats.fallback_calls[1], stats.fallback_calls[2], stats.fallback_calls[3],
                    stats.fallback_calls[4], ggml_hardware_test_policy(), ggml_cuda_graph_structural_enabled(), stats.path_calls[2] != 0);
        }
    });
}

static void ggml_cuda_graph_structural_drop_context(const ggml_backend_cuda_context * ctx) {
    ggml_cuda_graph_structural_probe::context_snapshot_cache_get().drop(ctx);
    auto & registry = ggml_cuda_graph_structural_registry_get();
    std::lock_guard<std::mutex> lock(registry.mutex);
    const char * trace = std::getenv("GGML_CUDA_GRAPH_STRUCTURAL_TRACE");
    if (trace && std::strcmp(trace, "1") == 0) {
        const auto stats_it = registry.contexts.find(ctx);
        const ggml_cuda_graph_structural_stats stats = stats_it == registry.contexts.end() ?
            ggml_cuda_graph_structural_stats{} : stats_it->second;
        GGML_LOG_INFO("CUDA graph structural stats device=%d attempts=%zu built=%zu reject_too_large=%zu reject_graph=%zu reject_external=%zu reject_missing=%zu active_matches=%zu variant_matches=%zu lookup_misses=%zu cross_descriptor_replays=%zu flat_snapshot_hits=%zu flat_snapshot_misses=%zu signature_build_us=%" PRId64 " lookup_us=%" PRId64 "\n",
                ctx->device, stats.attempts, stats.builds, stats.reject_too_large, stats.reject_graph,
                stats.reject_external, stats.reject_missing, stats.active_matches, stats.variant_matches,
                stats.lookup_misses, stats.cross_descriptor_replays, stats.flat_snapshot_hits, stats.flat_snapshot_misses,
                stats.signature_build_us, stats.lookup_us);
    }
    for (auto it = registry.graphs.begin(); it != registry.graphs.end(); ) {
        if (it->second.owner == ctx) {
            it = registry.graphs.erase(it);
        } else {
            ++it;
        }
    }
    registry.contexts.erase(ctx);
    registry.last_prune_us.erase(ctx);
}
#endif

ggml_backend_cuda_context::~ggml_backend_cuda_context() {
    if (ggml_cuda_cublas_role_audit::dispatch_scope::requested()) {
        ggml_cuda_cublas_role_audit::dispatch_scope::drop_context(this);
    }
#ifdef USE_CUDA_GRAPH
    if (ggml_cuda_ft_graph_cache_enabled()) {
        ggml_cuda_set_device(device);
        ggml_cuda_ft_graph_cache_drop(this);
    }
    ggml_cuda_graph_structural_drop_context(this);
#endif
    ggml_cuda_ft_host_report(this);
    ggml_cuda_ft_graph_census::drop(this);
    std::unique_lock<std::mutex> lock(ggml_cuda_lock);
    ggml_cuda_lock_cv.wait(lock, []{ return ggml_cuda_lock_counter.load(std::memory_order_relaxed) == 0; });

    // These direct cudaMalloc arenas are not owned by the ordinary scratch
    // pool. Release them after workers retire, just like the workspaces below.
    ggml_cuda_set_device(device);
#if defined(USE_CUDA_GRAPH) && defined(__linux__) && CUDART_VERSION >= 12000 && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    // The helper's last parent-stream event also covers its tail child. Drain
    // and retire those handles before any stream or referenced buffer is freed.
    CUDA_CHECK(ggml_cuda_device_tail_capture_only_client::drop_context(this));
    CUDA_CHECK(ggml_cuda_device_tail_client::drop_context(this));
#endif
    size_t q8_cache_freed_bytes = 0;
    for (auto & cache : q8_1_cache) {
        if (cache.buf != nullptr) {
            CUDA_CHECK(cudaFree(cache.buf));
            q8_cache_freed_bytes += cache.cap;
            cache.buf = nullptr;
            cache.cap = 0;
            cache.src = nullptr;
        }
    }
    if (q8_cache_freed_bytes != 0) GGML_LOG_INFO(
            "cuda-context-cleanup device=%d q8_cache_freed_bytes=%zu\n", device, q8_cache_freed_bytes);

    if (copy_event != nullptr) {
        CUDA_CHECK(cudaEventDestroy(copy_event));
    }
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        for (int j = 0; j < GGML_CUDA_MAX_STREAMS; ++j) {
            if (streams[i][j] != nullptr) {
                CUDA_CHECK(cudaStreamDestroy(streams[i][j]));
            }
            if (cublas_handles[i][j] != nullptr) {
                CUBLAS_CHECK(cublasDestroy(cublas_handles[i][j]));
            }
            if (cublas_workspaces[i][j] != nullptr) {
                CUDA_CHECK(cudaFree(cublas_workspaces[i][j]));
            }
        }
    }
#ifdef USE_CUDA_GRAPH
    // Context destruction is after worker drain; drop CPU lease records before
    // any active CUDA graph object can be freed below.
    ggml_cuda_graph_lease::get_registry().drop_context(this);
    // Preserve opaque key identity through the existing capture wait and all
    // graph destruction. No key address is ever passed as a CUDA tensor pointer.
    auto & registry = ggml_cuda_graph_structural_registry_get();
    bool owns_binding_keys = false;
    {
        std::lock_guard<std::mutex> registry_lock(registry.mutex);
        owns_binding_keys = registry.binding_keys.count(this) != 0;
    }
    if (owns_binding_keys) {
        cuda_graphs.clear();
        std::lock_guard<std::mutex> registry_lock(registry.mutex);
        registry.binding_keys.erase(this);
    }
#endif
}


// cuda buffer

#if defined(GGML_USE_VMM) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#define GGML_CUDA_EVICT
#endif

// Evictable weights, GGML_CUDA_EVICT_MIB=N: a buffer of at least 2N MiB is mapped through the VMM API, with its last N MiB
// as a separate physical allocation. ggml_backend_cuda_evict_weights() releases that tail of every weight buffer on a
// device for a while (the vision encoder borrows the memory); ggml_backend_cuda_restore_weights() maps new memory at the
// same addresses and copies the bytes back from a pinned host copy. Tensors keep their pointers throughout, so nothing
// else needs to know, as long as no graph runs on the device in between.
struct ggml_cuda_vmm_buffer {
    uintptr_t base      = 0;
    size_t    size      = 0;       // reserved and mapped
    size_t    tail_off  = 0;
    size_t    tail_size = 0;
    bool      evicted   = false;
    void    * shadow    = nullptr; // host copy of the tail, taken on the first eviction
    bool      shadow_pinned = false;
};

#if defined(GGML_CUDA_EVICT)
static size_t ggml_cuda_evict_tail_size() {
    static const size_t size = [] {
        const char * env = getenv("GGML_CUDA_EVICT_MIB");
        return env ? (size_t) std::max(0, atoi(env)) << 20 : 0;
    }();
    return size;
}

static bool ggml_cuda_vmm_map(uintptr_t ptr, size_t size, int device) {
    const int physical_device = ggml_cuda_get_physical_device(device);
    CUmemAllocationProp prop = {};
    prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id   = physical_device;
    CUmemGenericAllocationHandle handle;
    if (cuMemCreate(&handle, size, &prop, 0) != CUDA_SUCCESS) {
        return false;
    }
    CU_CHECK(cuMemMap((CUdeviceptr) ptr, size, 0, handle, 0));
    // the mapping keeps the memory alive: unmapping it frees it
    CU_CHECK(cuMemRelease(handle));
    ggml_cuda_vmm_set_access((CUdeviceptr) ptr, size, physical_device);
    return true;
}

// nullptr when the buffer is too small to give up the tail, or the memory is not there
static ggml_cuda_vmm_buffer * ggml_cuda_vmm_buffer_alloc(int device, size_t size) {
    const size_t gran = ggml_cuda_info().devices[device].vmm_granularity;
    if (!ggml_cuda_info().devices[device].vmm || gran == 0) {
        return nullptr;
    }
    const size_t tail  = GGML_PAD(ggml_cuda_evict_tail_size(), gran);
    const size_t total = GGML_PAD(size, gran);
    if (tail == 0 || 2*tail > total) {
        return nullptr;
    }
    CUdeviceptr base = 0;
    if (cuMemAddressReserve(&base, total, 0, 0, 0) != CUDA_SUCCESS) {
        return nullptr;
    }
    if (!ggml_cuda_vmm_map(base, total - tail, device)) {
        CU_CHECK(cuMemAddressFree(base, total));
        return nullptr;
    }
    if (!ggml_cuda_vmm_map(base + total - tail, tail, device)) {
        CU_CHECK(cuMemUnmap(base, total - tail));
        CU_CHECK(cuMemAddressFree(base, total));
        return nullptr;
    }
    ggml_cuda_vmm_buffer * vmm = new ggml_cuda_vmm_buffer;
    vmm->base      = base;
    vmm->size      = total;
    vmm->tail_off  = total - tail;
    vmm->tail_size = tail;
    return vmm;
}

static void ggml_cuda_vmm_buffer_free(ggml_cuda_vmm_buffer * vmm) {
    CU_CHECK(cuMemUnmap((CUdeviceptr) vmm->base, vmm->evicted ? vmm->tail_off : vmm->size));
    CU_CHECK(cuMemAddressFree((CUdeviceptr) vmm->base, vmm->size));
    if (vmm->shadow != nullptr) {
        if (vmm->shadow_pinned) {
            CUDA_CHECK(cudaFreeHost(vmm->shadow));
        } else {
            free(vmm->shadow);
        }
    }
    delete vmm;
}

static std::mutex                         ggml_cuda_evict_mutex;
static std::vector<ggml_backend_buffer_t> ggml_cuda_evict_buffers;
#endif // defined(GGML_CUDA_EVICT)

struct ggml_backend_cuda_buffer_context {
    int device;
    void * dev_ptr = nullptr;
    std::string name;
    ggml_cuda_vmm_buffer * vmm = nullptr; // evictable (GGML_CUDA_EVICT_MIB), dev_ptr is its base

    ggml_backend_cuda_buffer_context(int device, void * dev_ptr, ggml_cuda_vmm_buffer * vmm = nullptr) :
        device(device), dev_ptr(dev_ptr),
        name(GGML_CUDA_NAME + std::to_string(device)), vmm(vmm) {
    }

    ~ggml_backend_cuda_buffer_context() {
#if defined(GGML_CUDA_EVICT)
        if (vmm != nullptr) {
            ggml_cuda_set_device(device);
            CUDA_CHECK(cudaDeviceSynchronize());
            ggml_cuda_vmm_buffer_free(vmm);
            return;
        }
#endif // defined(GGML_CUDA_EVICT)
        CUDA_CHECK(cudaFree(dev_ptr));
    }
};

static void ggml_backend_cuda_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
#if defined(GGML_CUDA_EVICT)
    if (ctx->vmm != nullptr) {
        std::lock_guard<std::mutex> lock(ggml_cuda_evict_mutex);
        auto & v = ggml_cuda_evict_buffers;
        v.erase(std::remove(v.begin(), v.end(), buffer), v.end());
    }
#endif // defined(GGML_CUDA_EVICT)
    delete ctx;
}

static bool ggml_backend_buffer_is_cuda(ggml_backend_buffer_t buffer) {
    return buffer->iface.free_buffer == ggml_backend_cuda_buffer_free_buffer;
}

static void * ggml_backend_cuda_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    return ctx->dev_ptr;
}

static enum ggml_status ggml_backend_cuda_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    if (tensor->view_src != NULL) {
        assert(tensor->view_src->buffer->buft == buffer->buft);
        return GGML_STATUS_SUCCESS;
    }

    if (ggml_is_quantized(tensor->type) && tensor->view_src == nullptr && ggml_backend_buffer_get_usage(buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        // initialize padding to 0 to avoid possible NaN values
        const size_t original_size = ggml_nbytes(tensor);
        const size_t padded_size = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);

        if (padded_size > original_size) {
            ggml_cuda_set_device(ctx->device);
            CUDA_CHECK(cudaMemset((char *)tensor->data + original_size, 0, padded_size - original_size));
        }
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_cuda_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemsetAsync((char *) tensor->data + offset, value, size, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpyAsync((char *) tensor->data + offset, data, size, cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpyAsync(data, (const char *) tensor->data + offset, size, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_set_tensor_2d(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpy2DAsync(
        (char *) tensor->data + offset, stride_tensor, data, stride_data, size, n_copies, cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_get_tensor_2d(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpy2DAsync(
        data, stride_data, (const char *) tensor->data + offset, stride_tensor, size, n_copies, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static bool ggml_backend_cuda_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_cuda(src->buffer)) {
        ggml_backend_cuda_buffer_context * src_ctx = (ggml_backend_cuda_buffer_context *)src->buffer->context;
        ggml_backend_cuda_buffer_context * dst_ctx = (ggml_backend_cuda_buffer_context *)dst->buffer->context;
        // compare the backing physical devices: distinct virtual devices may share one physical GPU,
        // in which case a same-device copy (not a peer copy) is required
        const int src_physical = ggml_cuda_get_physical_device(src_ctx->device);
        const int dst_physical = ggml_cuda_get_physical_device(dst_ctx->device);
        if (src_physical == dst_physical) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(src), cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, dst_physical, src->data, src_physical, ggml_nbytes(src), cudaStreamPerThread));
#endif
        }
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_cuda_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemsetAsync(ctx->dev_ptr, value, buffer->size, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static const ggml_backend_buffer_i ggml_backend_cuda_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_cuda_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cuda_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_cuda_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_cuda_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cuda_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cuda_buffer_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_cuda_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_cuda_buffer_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_cuda_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cuda_buffer_clear,
    /* .reset           = */ NULL,
};

// cuda buffer type
struct ggml_backend_cuda_buffer_type_context {
    int device;
    std::string name;
};

static const char * ggml_backend_cuda_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_buffer_type_context * ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    return ctx->name.c_str();
}

static bool ggml_backend_buft_is_cuda(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_buffer_type_get_name;
}

static ggml_backend_buffer_t ggml_backend_cuda_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    ggml_cuda_set_device(buft_ctx->device);

#if defined(GGML_CUDA_EVICT)
    if (ggml_cuda_evict_tail_size() > 0 && getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY") == nullptr) {
        ggml_cuda_vmm_buffer * vmm = ggml_cuda_vmm_buffer_alloc(buft_ctx->device, size);
        if (vmm != nullptr) {
            ggml_backend_cuda_buffer_context * ctx = new ggml_backend_cuda_buffer_context(buft_ctx->device, (void *) vmm->base, vmm);
            ggml_backend_buffer_t buffer = ggml_backend_buffer_init(buft, ggml_backend_cuda_buffer_interface, ctx, size);
            std::lock_guard<std::mutex> lock(ggml_cuda_evict_mutex);
            ggml_cuda_evict_buffers.push_back(buffer);
            return buffer;
        }
    }
#endif // defined(GGML_CUDA_EVICT)

    void * dev_ptr;
    cudaError_t err = ggml_cuda_device_malloc(&dev_ptr, size, buft_ctx->device);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
        GGML_LOG_ERROR("%s: allocating %.2f MiB on device %d: cudaMalloc failed: %s\n", __func__, size / 1024.0 / 1024.0, buft_ctx->device, cudaGetErrorString(err));
        return nullptr;
    }

    ggml_backend_cuda_buffer_context * ctx = new ggml_backend_cuda_buffer_context(buft_ctx->device, dev_ptr);

    return ggml_backend_buffer_init(buft, ggml_backend_cuda_buffer_interface, ctx, size);
}

static size_t ggml_backend_cuda_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_cuda_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    if (tensor->op == GGML_OP_MONOMOE_SM70) return ggml_cuda_monomoe_alloc_size(tensor);
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *) buft->context;

    size_t size = tensor->op == GGML_OP_FLASH_ATTN_EXT
        ? ggml_cuda_flash_attn_ext_get_alloc_size(buft_ctx->device, tensor)
        : ggml_nbytes(tensor);
    int64_t ne0 = tensor->ne[0];

    // [TAG_ALLOC_SIZE_EXPAND]
    if (ggml_is_quantized(tensor->type)) {
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            GGML_ASSERT(tensor->nb[0] == ggml_element_size(tensor));
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return size;
}

static const ggml_backend_buffer_type_i ggml_backend_cuda_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_cuda_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_cuda_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_cuda_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size   = */ ggml_backend_cuda_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (device >= ggml_backend_cuda_get_device_count()) {
        return nullptr;
    }

    static ggml_backend_buffer_type ggml_backend_cuda_buffer_types[GGML_CUDA_MAX_DEVICES];

    static bool ggml_backend_cuda_buffer_type_initialized = false;

    if (!ggml_backend_cuda_buffer_type_initialized) {
        for (int i = 0; i < ggml_backend_cuda_get_device_count(); i++) {
            ggml_backend_cuda_buffer_types[i] = {
                /* .iface    = */ ggml_backend_cuda_buffer_type_interface,
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), i),
                /* .context  = */ new ggml_backend_cuda_buffer_type_context{i, GGML_CUDA_NAME + std::to_string(i)},
            };
        }
        ggml_backend_cuda_buffer_type_initialized = true;
    }

    return &ggml_backend_cuda_buffer_types[device];
}

// Communication context for multi-GPU AllReduce during tensor parallelism.
//
// Created once per meta backend instance.  Resources for the selected mode
// (NCCL communicators or the internal AllReduce pipeline) are initialised
// eagerly during comm_init so any init failure surfaces at startup rather
// than mid-run.
struct ggml_backend_cuda_comm_context {
    using try_allreduce_fn = bool(*)(ggml_backend_cuda_comm_context *, struct ggml_tensor **);

    std::vector<ggml_backend_t> backends;
    std::vector<int>            dev_ids;

    // Set by the init chain (comm_init_{nccl, internal, none}) to one of
    // try_allreduce_{nccl, internal, butterfly}.  nccl needs `comms`,
    // internal needs `ar_pipeline`, butterfly needs nothing.  Per-call
    // failures return false; the meta backend's generic implementation then
    // handles that call.
    try_allreduce_fn            try_allreduce = nullptr;

    ggml_cuda_ar_pipeline *     ar_pipeline = nullptr;

#ifdef GGML_USE_NCCL
    std::vector<ncclComm_t>     comms;
#endif // GGML_USE_NCCL

    ~ggml_backend_cuda_comm_context() {
#ifdef GGML_USE_NCCL
        for (ncclComm_t comm : comms) {
            NCCL_CHECK(ncclCommDestroy(comm));
        }
#endif // GGML_USE_NCCL
        ggml_cuda_ar_pipeline_free(ar_pipeline);
    }
};

#ifdef GGML_USE_NCCL
static bool ggml_cuda_comm_nccl_force_f32() {
    static const bool requested = [] {
        const char * env = getenv("GGML_CUDA_NCCL_F32");
        return env != nullptr && atoi(env) != 0;
    }();
    return requested;
}

// AllReduce via NCCL. Reduces as FP32 for small tensors and BF16 for large
// tensors (bandwidth-bound), then converts back to FP32.
static bool ggml_backend_cuda_comm_allreduce_nccl(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    const int64_t ne = ggml_nelements(tensors[0]);
    // FIXME the input of llm_graph_context::build_in_out_ids can produce a tensor with 0 elements if n_outputs == 0
    // This then causes a crash in this function
    if (ne == 0) {
        return true;
    }

    const size_t n_backends = comm_ctx->backends.size();

    for (size_t i = 0; i < n_backends; ++i) {
        // This provider always uses ncclFloat (or explicitly compresses F32).
        // Never reinterpret a half or quantized partial as a float payload.
        if (tensors[i] == nullptr || tensors[i]->type != GGML_TYPE_F32 ||
                ggml_nelements(tensors[i]) != ne || tensors[i]->data == nullptr ||
                !ggml_is_contiguously_allocated(tensors[i])) return false;
    }

    // For small tensors, simply reduce them as FP32.
    // The following heuristic for how "small" a tensor should be is based on RTX 4090s connected via 16x PCIe 4.0.
    // Row-sharded gathers have exactly one nonzero contributor per row; retain its exact value.
    const bool exact_rows = tensors[0]->op == GGML_OP_GET_ROWS && ggml_get_op_params_i32(tensors[0], 1) != 0;
    const bool force_f32 = ggml_cuda_comm_nccl_force_f32();
    if (force_f32) {
        auto * lead = (ggml_backend_cuda_context *) comm_ctx->backends[0]->context;
        ggml_cuda_optimization_path_log(lead->device, "nccl_wire_f32", true, true,
                "explicit_f32_no_bf16_scratch", -1, ne, n_backends);
    }
    // Keep fallback arithmetic consistent with the F32 in-graph P2P reduction.
    if (force_f32 || exact_rows || (n_backends <= 2 && ne < 32768) || (n_backends == 3 && ne < 131072) || (n_backends >= 4 && ne < 262144)) {
        for (size_t i = 0; i < n_backends; ++i) {
            if ((tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
                ggml_cuda_set_device(cuda_ctx->device);
                CUDA_CHECK(cudaMemsetAsync(tensors[i]->data, 0, ggml_nbytes(tensors[i]), cuda_ctx->stream()));
            }
        }
        NCCL_CHECK(ncclGroupStart());
        for (size_t i = 0; i < n_backends; ++i) {
            ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
            NCCL_CHECK(ncclAllReduce(tensors[i]->data, tensors[i]->data, ne, ncclFloat, ncclSum, comm_ctx->comms[i], cuda_ctx->stream()));
        }
        NCCL_CHECK(ncclGroupEnd());
        return true;
    }

    // For large tensors it's faster to compress them to BF16 for the reduction:
    to_bf16_cuda_t to_bf16 = ggml_get_to_bf16_cuda(GGML_TYPE_F32);
    to_fp32_cuda_t to_fp32 = ggml_get_to_fp32_cuda(GGML_TYPE_BF16);

    ggml_cuda_pool_alloc<nv_bfloat16> tmp[GGML_CUDA_MAX_DEVICES];
    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
        tmp[i].pool = &cuda_ctx->pool();
        tmp[i].alloc(ne);

        ggml_cuda_set_device(cuda_ctx->device);
        if (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) {
            to_bf16(tensors[i]->data, tmp[i].get(), ne, cuda_ctx->stream());
        } else {
            CUDA_CHECK(cudaMemsetAsync(tmp[i].get(), 0, ne * sizeof(nv_bfloat16), cuda_ctx->stream()));
        }
        CUDA_CHECK(cudaGetLastError());
    }

    NCCL_CHECK(ncclGroupStart());
    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
        NCCL_CHECK(ncclAllReduce(tmp[i].get(), tmp[i].get(), ne, ncclBfloat16, ncclSum, comm_ctx->comms[i], cuda_ctx->stream()));
    }
    NCCL_CHECK(ncclGroupEnd());

    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;

        ggml_cuda_set_device(cuda_ctx->device);
        to_fp32(tmp[i].get(), (float *) tensors[i]->data, ne, cuda_ctx->stream());
        CUDA_CHECK(cudaGetLastError());
    }

    return true;
}
#endif // GGML_USE_NCCL

// Run the internal AR pipeline.  Returns false on unsupported / failed input
// -- the caller decides whether to abort (env-forced) or fall back silently.
static bool ggml_backend_cuda_comm_allreduce_internal(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    GGML_ASSERT(comm_ctx->ar_pipeline != nullptr);

    const size_t n_backends = comm_ctx->backends.size();
    GGML_ASSERT(n_backends == 2);
    GGML_ASSERT(tensors[0] != nullptr);

    const int64_t   ne   = ggml_nelements(tensors[0]);
    const ggml_type type = tensors[0]->type;

    if (type != GGML_TYPE_F32 && type != GGML_TYPE_F16 && type != GGML_TYPE_BF16) {
        GGML_LOG_DEBUG("%s: internal unsupported: type=%d\n", __func__, (int) type);
        return false;
    }

    if (ne == 0) {
        return true;
    }

    for (size_t i = 0; i < n_backends; ++i) {
        if (tensors[i] == nullptr) {
            GGML_LOG_ERROR("%s: internal failed: tensor[%zu] is null\n", __func__, i);
            return false;
        }
        if (ggml_nelements(tensors[i]) != ne || tensors[i]->type != type) {
            GGML_LOG_ERROR("%s: internal failed: tensor[%zu] ne=%" PRId64 " type=%d expected ne=%" PRId64 " type=%d\n",
                           __func__, i, ggml_nelements(tensors[i]), (int) tensors[i]->type, ne, (int) type);
            return false;
        }
        if (!ggml_is_contiguously_allocated(tensors[i])) {
            GGML_LOG_DEBUG("%s: internal unsupported: tensor[%zu] is not contiguously allocated: ne=%" PRId64 " nbytes=%zu packed=%zu type=%d\n",
                           __func__, i, ne, ggml_nbytes(tensors[i]),
                           (size_t) ne * ggml_type_size(type) / ggml_blck_size(type), (int) type);
            return false;
        }
        if (((uintptr_t) tensors[i]->data & 0xF) != 0) {
            GGML_LOG_DEBUG("%s: internal unsupported: tensor[%zu] data pointer is not 16-byte aligned: %p type=%d ne=%" PRId64 "\n",
                           __func__, i, tensors[i]->data, (int) type, ne);
            return false;
        }
        GGML_ASSERT((ggml_nbytes(tensors[i]) & 0xF) == 0);
    }

    return ggml_cuda_ar_allreduce(comm_ctx->ar_pipeline, comm_ctx->backends.data(), tensors);
}

// ---------------------------------------------------------------------------
// Per-call dispatch -- three variants, one per backend.  Each is set as
// comm_ctx->try_allreduce by the matching init step.  Per-call failure
// returns false; the meta backend's generic implementation handles that call.
// ---------------------------------------------------------------------------

#ifdef GGML_USE_NCCL
static bool ggml_backend_cuda_comm_try_allreduce_nccl(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    return ggml_backend_cuda_comm_allreduce_nccl(comm_ctx, tensors);
}
#endif // GGML_USE_NCCL

static bool ggml_backend_cuda_comm_try_allreduce_internal(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    return ggml_backend_cuda_comm_allreduce_internal(comm_ctx, tensors);
}

static bool ggml_backend_cuda_comm_try_allreduce_butterfly(
        ggml_backend_cuda_comm_context *, struct ggml_tensor **) {
    return false;
}

// in-graph P2P AllReduce, see allreduce-p2p.cuh
static void * ggml_backend_cuda_p2p_ar_init(ggml_backend_t * backends, size_t n_backends) {
    std::vector<int> devices;
    for (size_t i = 0; i < n_backends; ++i) {
        if (!ggml_backend_is_cuda(backends[i])) {
            return nullptr;
        }
        devices.push_back(((ggml_backend_cuda_context *) backends[i]->context)->device);
    }
    ggml_cuda_p2p_ar * ar = ggml_cuda_p2p_ar_init(devices.data(), devices.size());
    if (ar != nullptr) {
        // cublasCreate can block while a peer GPU spins in the AllReduce: create the handles now
        //     (stream 0 only: the meta backend has no graph_optimize, so no concurrent streams)
        for (size_t i = 0; i < n_backends; ++i) {
            ggml_backend_cuda_context * ctx = (ggml_backend_cuda_context *) backends[i]->context;
            GGML_ASSERT(ctx->curr_stream_no == 0);
            ctx->cublas_handle();
        }
    }
    return ar;
}
// true if computing cgraph will only replay a captured CUDA graph: no capture, instantiation or eager launches
#ifdef USE_CUDA_GRAPH
static bool ggml_cuda_graph_structural_enabled();
static bool ggml_cuda_graph_make_structural_signature(
        ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph,
        ggml_cuda_graph_structural::graph_signature * signature,
        ggml_cuda_graph_structural_probe::flat_snapshot * snapshot);
static const void * ggml_cuda_graph_get_key(
        ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph,
        ggml_cuda_graph_structural::graph_signature * signature, bool * signature_valid,
        bool allow_new_binding_key = false);
static bool ggml_cuda_ft_graph_cached_ready(
        ggml_backend_cuda_context * cuda_ctx,
        const ggml_cuda_graph_structural::graph_signature * signature, bool signature_valid);
#endif

static bool ggml_backend_cuda_graph_ready(ggml_backend_t backend, const struct ggml_cgraph * cgraph) {
    // without CUDA graphs a chunk is plain kernel launches: the meta backend merges a graph shape only after running
    // it unmerged once, so its kernels are loaded and its pool allocations exist, and nothing on the host can block
    // while a peer spins in an AllReduce. No lockstep needed (it would idle the GPUs at every chunk boundary).
    static const bool graphs_off = getenv("GGML_CUDA_DISABLE_GRAPHS") != nullptr;
    if (graphs_off && cgraph->n_nodes > 0) {
        GGML_UNUSED(backend);
        return true;
    }
#ifdef USE_CUDA_GRAPH
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    if (cgraph->n_nodes == 0 || cgraph->uid == 0) {
        return false;
    }
    ggml_cuda_graph_structural::graph_signature signature;
    bool signature_valid = false;
    const void * graph_key = ggml_cuda_graph_get_key(cuda_ctx, cgraph, &signature, &signature_valid, false);
    const auto it = cuda_ctx->cuda_graphs.find(graph_key);
    if (it == cuda_ctx->cuda_graphs.end()) {
        return ggml_cuda_graph_structural_enabled() &&
            ggml_cuda_ft_graph_cached_ready(cuda_ctx, &signature, signature_valid);
    }
    const ggml_cuda_graph & graph = *it->second;
    const bool fresh = ggml_time_us() - graph.last_used_time < 5'000'000; // not about to be evicted
    if (!graph.is_enabled() || !ggml_cuda_graph_has_executable(cuda_ctx, &graph) ||
            !graph.warmup_complete || !fresh) {
        return ggml_cuda_graph_structural_enabled() &&
            ggml_cuda_ft_graph_cached_ready(cuda_ctx, &signature, signature_valid);
    }
    if (ggml_cuda_graph_structural_enabled()) {
        if (signature_valid) {
            if (ggml_cuda_graph_structural_compare(cuda_ctx, &graph, signature) ==
                    ggml_cuda_graph_structural_match::equal) {
                return true;
            }
            return ggml_cuda_ft_graph_cached_ready(cuda_ctx, &signature, signature_valid);
        }
    }
    return graph.uid == cgraph->uid;
#else
    GGML_UNUSED(backend);
    GGML_UNUSED(cgraph);
    return false;
#endif // USE_CUDA_GRAPH
}

#ifdef USE_CUDA_GRAPH
#include "graph-compose-private.cuh"
#endif

static void ggml_backend_cuda_p2p_ar_free(void * ar) {
    ggml_cuda_p2p_ar_free((ggml_cuda_p2p_ar *) ar);
}
static bool ggml_backend_cuda_p2p_ar_reserve(void * ar, size_t nbytes) {
    return ggml_cuda_p2p_ar_reserve((ggml_cuda_p2p_ar *) ar, nbytes);
}
static void ggml_backend_cuda_p2p_ar_set_params(void * ar, size_t rank, struct ggml_tensor * node) {
    ggml_cuda_p2p_ar_set_params((ggml_cuda_p2p_ar *) ar, rank, node);
}

// Cold graphs use the same rank-ordered kernel as captured graphs.
static bool ggml_backend_cuda_p2p_ar_allreduce_impl(
        void * ar_v, ggml_backend_t * backends, size_t n_backends, ggml_tensor ** tensors, bool drain) {
    if (ar_v == nullptr || n_backends < 2 || n_backends > GGML_CUDA_MAX_DEVICES || tensors[0] == nullptr) {
        return false;
    }
    const int64_t ne = ggml_nelements(tensors[0]);
    std::vector<ggml_backend_cuda_context *> contexts(n_backends);
    std::vector<ggml_tensor> nodes(n_backends);
    for (size_t i = 0; i < n_backends; ++i) {
        if (!ggml_backend_is_cuda(backends[i]) || tensors[i] == nullptr ||
                tensors[i]->type != GGML_TYPE_F32 || ggml_nelements(tensors[i]) != ne ||
                !ggml_is_contiguous(tensors[i]) || (ne > 0 && tensors[i]->data == nullptr)) {
            return false;
        }
        contexts[i] = (ggml_backend_cuda_context *) backends[i]->context;
        ggml_cuda_set_device(contexts[i]->device);
        contexts[i]->stream();
    }
    if (ne == 0) {
        return true;
    }
    auto * ar = (ggml_cuda_p2p_ar *) ar_v;
    if (!ggml_cuda_p2p_ar_reserve(ar, ggml_nbytes(tensors[0]))) {
        return false;
    }
    for (size_t i = 0; i < n_backends; ++i) {
        nodes[i] = *tensors[i];
        nodes[i].op = GGML_OP_ALLREDUCE;
        nodes[i].src[0] = tensors[i];
        nodes[i].flags |= GGML_TENSOR_FLAG_COMPUTE;
        ggml_format_name(&nodes[i], "allreduce_0");
        ggml_cuda_p2p_ar_set_params(ar, i, &nodes[i]);
    }
    // All ranks must be enqueued before any stream is drained.
    for (size_t i = 0; i < n_backends; ++i) {
        ggml_cuda_set_device(contexts[i]->device);
        ggml_cuda_op_allreduce(*contexts[i], &nodes[i]);
    }
    // Cold callers drain before driver allocation/capture. The opt-in nested
    // hot path instead guards its next cold preparation and keeps work ordered
    // on these exact physical streams. All ranks were enqueued above.
    if (drain) {
        for (size_t i = 0; i < n_backends; ++i) {
            ggml_cuda_set_device(contexts[i]->device);
            CUDA_CHECK(cudaStreamSynchronize(contexts[i]->stream()));
        }
    }
    return true;
}

static bool ggml_backend_cuda_p2p_ar_allreduce(
        void * ar, ggml_backend_t * backends, size_t n, ggml_tensor ** tensors) {
    return ggml_backend_cuda_p2p_ar_allreduce_impl(ar, backends, n, tensors, true);
}

static bool ggml_backend_cuda_p2p_ar_allreduce_async(
        void * ar, ggml_backend_t * backends, size_t n, ggml_tensor ** tensors) {
    return ggml_backend_cuda_p2p_ar_allreduce_impl(ar, backends, n, tensors, false);
}

static void ggml_backend_cuda_comm_free(void * comm_ctx_v) {
    if (comm_ctx_v == nullptr) {
        return;
    }
    delete static_cast<ggml_backend_cuda_comm_context *>(comm_ctx_v);
}

// ---------------------------------------------------------------------------
// Init -- chained nccl -> internal -> none.  Each step tries to bring up its
// resource; on failure it warns and recurses into the next step.
// ---------------------------------------------------------------------------
static void ggml_backend_cuda_comm_init_none(ggml_backend_cuda_comm_context * ret) {
    ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_butterfly;
}

static void ggml_backend_cuda_comm_init_internal(ggml_backend_cuda_comm_context * ret) {
    ret->ar_pipeline = ggml_cuda_ar_pipeline_init(ret->dev_ids.data(), ret->dev_ids.size());
    if (ret->ar_pipeline) {
        ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_internal;
        return;
    }

    // Clear sticky CUDA error from the failed init.
    (void) cudaGetLastError();
    GGML_LOG_WARN("internal AllReduce init failed (n_devices != 2?); "
                  "falling back to meta-backend butterfly\n");
    ggml_backend_cuda_comm_init_none(ret);
}

static void ggml_backend_cuda_comm_init_nccl(ggml_backend_cuda_comm_context * ret) {
#ifdef GGML_USE_NCCL
    // Disabling NCCL path when CUDA virtual devices are in use since NCCL requires one distinct physical GPU per rank.
    const ggml_cuda_device_info & info = ggml_cuda_info();
    if (info.device_count > info.physical_device_count) {
        GGML_LOG_WARN("NCCL disabled: virtual devices in use; "
                      "falling back to internal AllReduce\n");
        ggml_backend_cuda_comm_init_internal(ret);
        return;
    }

    const size_t n = ret->dev_ids.size();
    ret->comms.resize(n);
    ncclResult_t rc = ncclCommInitAll(ret->comms.data(), (int) n, ret->dev_ids.data());
    if (rc == ncclSuccess) {
        ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_nccl;
        return;
    }

    ret->comms.clear();
    GGML_LOG_WARN("NCCL init failed (%s); falling back to internal AllReduce\n",
                  ncclGetErrorString(rc));
#else // GGML_USE_NCCL
#ifndef GGML_USE_HIP
    GGML_LOG_WARN("NCCL not compiled in; falling back to internal AllReduce.  "
                  "Recompile with -DGGML_CUDA_NCCL=ON for best multi-GPU performance.\n");
#endif // !GGML_USE_HIP
#endif // GGML_USE_NCCL

    ggml_backend_cuda_comm_init_internal(ret);
}

// Top-level init.  Picks one of the three init paths based on
// GGML_CUDA_ALLREDUCE (or the platform default) and lets the chain handle
// any fallback.  Unrecognised env values warn and fall through to the
// platform default.
static void * ggml_backend_cuda_comm_init(ggml_backend_t * backends, size_t n_backends) {
    for (size_t i = 0; i < n_backends; i++) {
        if (!ggml_backend_is_cuda(backends[i])) {
            return nullptr;
        }
    }

    auto * ret = new ggml_backend_cuda_comm_context;
    ret->backends.assign(backends, backends + n_backends);
    ret->dev_ids.reserve(n_backends);
    for (size_t i = 0; i < n_backends; i++) {
        ret->dev_ids.push_back(static_cast<ggml_backend_cuda_context *>(backends[i]->context)->device);
    }

    const char * env = getenv("GGML_CUDA_ALLREDUCE");
    if (!env) {
        // Platform default: Linux uses NCCL, otherwise (generally Windows) internal
#if defined(__linux__)
        ggml_backend_cuda_comm_init_nccl(ret);
#else
        ggml_backend_cuda_comm_init_internal(ret);
#endif // defined(__linux__)
    } else {
        std::string env_str(env);
        if (env_str == "nccl") {
            ggml_backend_cuda_comm_init_nccl(ret);
        } else if (env_str == "internal") {
            ggml_backend_cuda_comm_init_internal(ret);
        } else if (env_str == "none") {
            ggml_backend_cuda_comm_init_none(ret);
        } else {
            GGML_LOG_WARN("unknown GGML_CUDA_ALLREDUCE value: %s\n", env);
            ggml_backend_cuda_comm_init_none(ret);
        }
    }

    return ret;
}

// Top-level dispatch -- calls the function pointer chosen by comm_init.
// Returns false to let the meta-backend's butterfly run.
static bool ggml_backend_cuda_comm_allreduce_tensor(void * comm_ctx_v, struct ggml_tensor ** tensors) {
    if (comm_ctx_v == nullptr) {
        return false;
    }
    auto * comm_ctx = static_cast<ggml_backend_cuda_comm_context *>(comm_ctx_v);
    return comm_ctx->try_allreduce(comm_ctx, tensors);
}

// Private capability query: the Meta large-payload route must never assume
// that a requested NCCL provider actually initialized, or use compressed BF16.
static bool ggml_backend_cuda_comm_is_nccl_f32(void * comm_ctx_v) {
#ifdef GGML_USE_NCCL
    if (comm_ctx_v == nullptr) return false;
    const auto * comm_ctx = static_cast<const ggml_backend_cuda_comm_context *>(comm_ctx_v);
    return comm_ctx->try_allreduce == ggml_backend_cuda_comm_try_allreduce_nccl &&
           !comm_ctx->comms.empty() && ggml_cuda_comm_nccl_force_f32();
#else
    GGML_UNUSED(comm_ctx_v);
    return false;
#endif
}

// host buffer type

static const char * ggml_backend_cuda_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_CUDA_NAME "_Host";

    GGML_UNUSED(buft);
}

static bool ggml_backend_buft_is_cuda_host(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_host_buffer_type_name;
}

static void ggml_backend_cuda_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    CUDA_CHECK(cudaFreeHost(buffer->context));
}

static void * ggml_cuda_host_malloc(size_t size) {
    if (getenv("GGML_CUDA_NO_PINNED") != nullptr) {
        return nullptr;
    }

    void * ptr = nullptr;
    cudaError_t err = cudaMallocHost((void **) &ptr, size);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
        GGML_LOG_DEBUG("%s: failed to allocate %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return nullptr;
    }

    return ptr;
}

static ggml_backend_buffer_t ggml_backend_cuda_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * ptr = ggml_cuda_host_malloc(size);

    if (ptr == nullptr) {
        // fallback to cpu buffer
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.free_buffer = ggml_backend_cuda_host_buffer_free_buffer;

    return buffer;
}

ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_cuda_buffer_type_host = {
        /* .iface    = */ {
            /* .get_name         = */ ggml_backend_cuda_host_buffer_type_name,
            /* .alloc_buffer     = */ ggml_backend_cuda_host_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
            /* .is_host          = */ ggml_backend_cpu_buffer_type()->iface.is_host,
        },
        /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), 0),
        /* .context  = */ nullptr,
    };

    return &ggml_backend_cuda_buffer_type_host;
}

//static bool ggml_backend_buffer_is_cuda_host(ggml_backend_buffer_t buffer) {
//    return buffer->buft->iface.get_name == ggml_backend_cuda_host_buffer_type_name;
//}

/// kernels

typedef void (*ggml_cuda_op_mul_mat_t)(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

static __global__ void k_compute_batched_ptrs(
        const void * src0_as_f16, const void * src1_as_f16, char * dst,
        const void ** ptrs_src, void ** ptrs_dst,
        int64_t ne12, int64_t ne13,
        int64_t ne23,
        size_t  nb02, size_t  nb03,
        size_t  nb12, size_t  nb13,
        size_t  nbd2, size_t  nbd3,
        int64_t r2,   int64_t r3) {
    const int64_t i13 = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t i12 = blockIdx.y * blockDim.y + threadIdx.y;

    if (i13 >= ne13 || i12 >= ne12) {
        return;
    }

    const int64_t i03 = i13 / r3;
    const int64_t i02 = i12 / r2;

    ptrs_src[0*ne23 + i12 + i13*ne12] = (const char *) src0_as_f16 + i02*nb02 + i03*nb03;
    ptrs_src[1*ne23 + i12 + i13*ne12] = (const char *) src1_as_f16 + i12*nb12 + i13*nb13;
    ptrs_dst[0*ne23 + i12 + i13*ne12] = (      char *)         dst + i12*nbd2 + i13*nbd3;
}

// Type traits for mapping ggml types to CUDA/cuBLAS types
template<ggml_type T>
struct batched_mul_mat_traits;

template<>
struct batched_mul_mat_traits<GGML_TYPE_F32> {
    using cuda_type = float;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    static inline const cudaDataType_t data_type = CUDA_R_32F;
    static inline const ggml_type ggml_type_val = GGML_TYPE_F32;
    static inline const float alpha = 1.0f;
    static inline const float beta = 0.0f;
    static inline const void* get_alpha() { static const float val = alpha; return &val; }
    static inline const void* get_beta() { static const float val = beta; return &val; }
    static inline auto convert(ggml_type src_type) { return ggml_get_to_fp32_cuda(src_type); }
    static inline auto convert_nc(ggml_type src_type) { return ggml_get_to_fp32_nc_cuda(src_type); }
};

template<>
struct batched_mul_mat_traits<GGML_TYPE_BF16> {
    using cuda_type = nv_bfloat16;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    static inline const cudaDataType_t data_type = CUDA_R_16BF;
    static inline const ggml_type ggml_type_val = GGML_TYPE_BF16;
    static inline const float alpha = 1.0f;
    static inline const float beta = 0.0f;
    static inline const void* get_alpha() { static const float val = alpha; return &val; }
    static inline const void* get_beta() { static const float val = beta; return &val; }
    static inline auto convert(ggml_type src_type) { return ggml_get_to_bf16_cuda(src_type); }
    static inline auto convert_nc(ggml_type src_type) { return ggml_get_to_bf16_nc_cuda(src_type); }
};

template<>
struct batched_mul_mat_traits<GGML_TYPE_F16> {
    using cuda_type = half;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
    static inline const cudaDataType_t data_type = CUDA_R_16F;
    static inline const ggml_type ggml_type_val = GGML_TYPE_F16;
    static inline const half alpha = 1.0;
    static inline const half beta = 0.0;
    static inline const void* get_alpha() { static const half val = alpha; return &val; }
    static inline const void* get_beta() { static const half val = beta; return &val; }
    // [SM70_BF16_PREFILL_F16 / strata #540] saturated operands: finite values beyond F16's range clamp
    // to +-65504 instead of turning Inf/NaN inside the product (see convert.cu); in-range values convert
    // bit-identically to the unsaturated kernels (512-token parity proven on this path).
    static inline auto convert(ggml_type src_type) { return ggml_get_to_fp16_sat_cuda(src_type); }
    static inline auto convert_nc(ggml_type src_type) { return ggml_get_to_fp16_sat_nc_cuda(src_type); }
};

static void ggml_cuda_cublas_role_audit_record(
        const ggml_backend_cuda_context & ctx, const ggml_tensor * dst,
        const ggml_tensor * source, ggml_cuda_cublas_role_audit::operand_role role,
        ggml_type target_type, const char * layout) {
    if (!ggml_cuda_cublas_role_audit::dispatch_scope::active_on_thread()) return;

    ggml_cuda_cublas_role_audit::conversion_event event;
    event.role = role;
    event.source_type = ggml_type_name(source->type);
    event.target_type = ggml_type_name(target_type);
    event.expert_fallback = ggml_cuda_cublas_role_audit::dispatch_scope::expert_fallback_active();
    event.op = event.expert_fallback ? "MUL_MAT_ID_expert_fallback" : ggml_op_name(dst->op);
    event.source_op = ggml_op_name(source->op);
    event.tensor_name = source->name;
    event.layout = layout;
    event.source_bytes = ggml_nbytes(source);
    const int64_t n_elements = ggml_nelements(source);
    const size_t target_type_size = ggml_type_size(target_type);
    if (n_elements > 0 && (uint64_t) n_elements <= SIZE_MAX / target_type_size) {
        event.materialized_bytes = (size_t) n_elements * target_type_size;
    }

    const ggml_tensor * root = source;
    size_t view_offset = 0;
    bool view_offset_overflow = false;
    size_t view_hops = 0;
    for (; root->view_src != nullptr && view_hops < 32; ++view_hops) {
        if (root->view_offs > SIZE_MAX - view_offset) {
            view_offset_overflow = true;
            break;
        }
        view_offset += root->view_offs;
        root = root->view_src;
    }
    const std::string view_offset_text = view_offset_overflow ?
            std::string("overflow") : std::to_string(view_offset);
    event.root_name = root->name;
    if (source->buffer == nullptr) {
        event.buffer_usage = "NO_BUFFER";
    } else {
        const auto usage = ggml_backend_buffer_get_usage(source->buffer);
        event.buffer_usage = usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS ? "WEIGHTS" :
                usage == GGML_BACKEND_BUFFER_USAGE_COMPUTE ? "COMPUTE" : "OTHER";
    }

    std::ostringstream shape;
    std::ostringstream strides;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (i != 0) { shape << 'x'; strides << ','; }
        shape << source->ne[i];
        strides << source->nb[i];
    }
    event.shape = shape.str();
    event.strides = strides.str();

    std::ostringstream identity;
    identity << "device=" << ctx.device
             << ";buffer=" << static_cast<const void *>(source->buffer)
             << ";tensor=" << static_cast<const void *>(source) << ':' << source->name
             << ";root=" << static_cast<const void *>(root) << ':' << root->name
             << ";data=" << source->data
             << ";view_offset=" << view_offset_text
             << ";view_hops=" << view_hops
             << ";type=" << event.source_type << ";target=" << event.target_type
             << ";shape=" << event.shape << ";strides=" << event.strides;
    event.logical_source_identity = identity.str();
    ggml_cuda_cublas_role_audit::dispatch_scope::record(std::move(event));
}

template<ggml_type compute_type>
static void ggml_cuda_mul_mat_cublas_impl(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    using traits = batched_mul_mat_traits<compute_type>;
    using cuda_t = typename traits::cuda_type;

    GGML_ASSERT(ggml_is_contiguous(dst));

    // Byte offsets and tensor dimensions are currently used in an inconsistent way for dst.
    // As long as dst is contiguous this does not matter though.

    GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t ne_dst = ggml_nelements(dst);
    cudaStream_t main_stream = ctx.stream();
    cublasHandle_t cublas_h = ctx.cublas_handle();

    const size_t src0_ts = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == src0_ts);
    int64_t s01 = nb01 / src0_ts;
    int64_t s02 = nb02 / src0_ts;
    int64_t s03 = nb03 / src0_ts;

    const size_t src1_ts = ggml_type_size(src1->type);
    GGML_ASSERT(nb10 == src1_ts);
    int64_t s11 = nb11 / src1_ts;
    int64_t s12 = nb12 / src1_ts;
    int64_t s13 = nb13 / src1_ts;

    float * dst_ddf = (float *) dst->data;

    const cuda_t * src0_ptr = nullptr;
    const cuda_t * src1_ptr = nullptr;

    ggml_cuda_pool_alloc<cuda_t> src0_alloc(ctx.pool());
    ggml_cuda_pool_alloc<cuda_t> src1_alloc(ctx.pool());

    bool is_src0_cont_2 = ggml_is_contiguous_2(src0);
    bool is_src1_cont_2 = ggml_is_contiguous_2(src1);

    if (src0->type == compute_type) {
        src0_ptr = (const cuda_t *) src0->data;
    } else {
        src0_alloc.alloc(ggml_nelements(src0));

        if (ggml_is_contiguously_allocated(src0)) {
            const auto convert_func = traits::convert(src0->type);
            GGML_ASSERT(convert_func != nullptr);
            ggml_cuda_cublas_role_audit_record(ctx, dst, src0,
                    ggml_cuda_cublas_role_audit::operand_role::src0, compute_type, "contiguous");
            convert_func(src0->data, src0_alloc.get(), ggml_nelements(src0), main_stream);
            const size_t src0_bs = ggml_blck_size(src0->type);
            s01 *= src0_bs;
            s02 *= src0_bs;
            s03 *= src0_bs;
        } else {
            const auto convert_func = traits::convert_nc(src0->type);
            GGML_ASSERT(convert_func != nullptr);
            ggml_cuda_cublas_role_audit_record(ctx, dst, src0,
                    ggml_cuda_cublas_role_audit::operand_role::src0, compute_type, "strided");
            convert_func(src0->data, src0_alloc.get(), ne00, ne01, ne02, ne03, s01, s02, s03, main_stream);
            s01 = ne00;
            s02 = ne01*s01;
            s03 = ne02*s02;
            is_src0_cont_2 = true;
        }
        src0_ptr = src0_alloc.get();
    }

    if (src1->type == compute_type) {
        src1_ptr = (const cuda_t *) src1->data;
    } else {
        src1_alloc.alloc(ggml_nelements(src1));

        if (ggml_is_contiguously_allocated(src1)) {
            const auto convert_func = traits::convert(src1->type);
            GGML_ASSERT(convert_func != nullptr);
            ggml_cuda_cublas_role_audit_record(ctx, dst, src1,
                    ggml_cuda_cublas_role_audit::operand_role::src1, compute_type, "contiguous");
            convert_func(src1->data, src1_alloc.get(), ggml_nelements(src1), main_stream);
            const size_t src1_bs = ggml_blck_size(src1->type);
            s11 *= src1_bs;
            s12 *= src1_bs;
            s13 *= src1_bs;
        } else {
            const auto convert_func = traits::convert_nc(src1->type);
            GGML_ASSERT(convert_func != nullptr);
            ggml_cuda_cublas_role_audit_record(ctx, dst, src1,
                    ggml_cuda_cublas_role_audit::operand_role::src1, compute_type, "strided");
            convert_func(src1->data, src1_alloc.get(), ne10, ne11, ne12, ne13, s11, s12, s13, main_stream);
            s11 = ne10;
            s12 = ne11*s11;
            s13 = ne12*s12;
            is_src1_cont_2 = true;
        }
        src1_ptr = src1_alloc.get();
    }

    ggml_cuda_pool_alloc<cuda_t> dst_temp(ctx.pool());
    char * dst_ptr;
    size_t nbd2 = dst->nb[2];
    size_t nbd3 = dst->nb[3];

    cublasComputeType_t cu_compute_type = traits::compute_type;
    cudaDataType_t cu_data_type = traits::data_type;
    cudaDataType_t cu_data_type_a = traits::data_type;
    cudaDataType_t cu_data_type_b = traits::data_type;
    const void * alpha = traits::get_alpha();
    const void * beta = traits::get_beta();

    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    bool prefer_f32_output = false;
    if (compute_type == GGML_TYPE_F16) {
        prefer_f32_output = cc == GGML_CUDA_CC_VOLTA || GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_CDNA(cc);
    } else if (compute_type == GGML_TYPE_BF16) {
        prefer_f32_output = !GGML_CUDA_CC_IS_RDNA3(cc) && !GGML_CUDA_CC_IS_CDNA(cc);
    }

    if (prefer_f32_output) {
        dst_ptr = (char *) dst_ddf;
        cu_compute_type = batched_mul_mat_traits<GGML_TYPE_F32>::compute_type;
        cu_data_type = batched_mul_mat_traits<GGML_TYPE_F32>::data_type;
        alpha = batched_mul_mat_traits<GGML_TYPE_F32>::get_alpha();
        beta = batched_mul_mat_traits<GGML_TYPE_F32>::get_beta();
    } else {
        if constexpr (compute_type == GGML_TYPE_F32) {
            dst_ptr = (char *) dst_ddf;  // Direct F32 output
        } else {
            dst_ptr = (char *) dst_temp.alloc(ne_dst);
            nbd2 /= sizeof(float) / sizeof(cuda_t);
            nbd3 /= sizeof(float) / sizeof(cuda_t);
        }
    }

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    // broadcast factors
    const int64_t r2 = ne12/ne02;
    const int64_t r3 = ne13/ne03;

    // Theoretically cublasGemmStridedBatchedEx would always work, even for a single matrix.
    // However, for some old NVIDIA and AMD GPUs the strided/Ex GEMM is much slower,
    //     probably because the internal kernel selection logic is suboptimal.
    if (compute_type == GGML_TYPE_F32 && ne12 == 1 && ne13 == 1) {
        CUBLAS_CHECK(
            cublasSgemm(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                    ne01, ne11, ne10,
                    (const float *) alpha, (const float *) src0_ptr, s01,
                                           (const float *) src1_ptr, s11,
                    (const float *) beta,  (float       *)  dst_ptr, ne0));
    } else if (ne12 == 1 && ne13 == 1) {
        CUBLAS_CHECK(
            cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                    ne01, ne11, ne10,
                    alpha, src0_ptr, cu_data_type_a, s01,
                           src1_ptr, cu_data_type_b, s11,
                    beta,   dst_ptr, cu_data_type,   ne0,
                    cu_compute_type,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else if (r2 == 1 && r3 == 1 && is_src0_cont_2 && is_src1_cont_2) {
        // with a [0, 2, 1, 3] perm. and ne02==1 the matrix strides need to be determined from dim 3:
        const int64_t sma = ne02 == 1 ? s03 : s02;
        const int64_t smb = ne12 == 1 ? s13 : s12;

        // there is no broadcast and src0, src1 are contiguous across dims 2, 3
        // use cublasGemmStridedBatchedEx
        CUBLAS_CHECK(
        cublasGemmStridedBatchedEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, src0_ptr, cu_data_type_a, s01, sma,     // strideA
                       src1_ptr, cu_data_type_b, s11, smb,     // strideB
                beta,   dst_ptr, cu_data_type,   ne0, ne1*ne0, // strideC
                ne12*ne13,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else {
        // use cublasGemmBatchedEx
        const int64_t ne23 = ne12*ne13;

        ggml_cuda_pool_alloc<const void *> ptrs_src(ctx.pool(), 2*ne23);
        ggml_cuda_pool_alloc<      void *> ptrs_dst(ctx.pool(), 1*ne23);

        const size_t src_type_size = sizeof(cuda_t);

        const int threads_x = 16;
        const int threads_y = 16;
        const dim3 block_dims(threads_x, threads_y);

        const dim3 grid_dims(
            (ne13 + threads_x - 1) / threads_x,
            (ne12 + threads_y - 1) / threads_y
        );
        k_compute_batched_ptrs<<<grid_dims, block_dims, 0, main_stream>>>(
                src0_ptr, src1_ptr, dst_ptr,
                ptrs_src.get(), ptrs_dst.get(),
                ne12, ne13,
                ne23,
                s02*src_type_size, s03*src_type_size,
                s12*src_type_size, s13*src_type_size,
                nbd2, nbd3,
                r2, r3);

        CUDA_CHECK(cudaGetLastError());

        CUBLAS_CHECK(
        cublasGemmBatchedEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, (const void **) (ptrs_src.get() + 0*ne23), cu_data_type_a, s01,
                       (const void **) (ptrs_src.get() + 1*ne23), cu_data_type_b, s11,
                beta,  (      void **) (ptrs_dst.get() + 0*ne23), cu_data_type,   ne0,
                ne23,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    // Convert output back to F32 if needed
    if (cu_data_type != CUDA_R_32F) {
        const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(traits::ggml_type_val);
        to_fp32_cuda(dst_temp.get(), dst_ddf, ne_dst, main_stream);
    }
}

static void ggml_cuda_mul_mat_cublas(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    ggml_type compute_type = src0->type;
    if (ggml_is_quantized(compute_type)) {
        compute_type = fast_fp16_hardware_available(cc) ? GGML_TYPE_F16 : GGML_TYPE_F32;
    } else if (compute_type == GGML_TYPE_F16 && !fast_fp16_hardware_available(cc)) {
        compute_type = GGML_TYPE_F32;
    } else if (compute_type == GGML_TYPE_BF16 && !fast_bf16_hardware_available(cc)) {
        if (GGML_CUDA_CC_IS_AMD(cc) && src1->ne[1] > 32) {
            compute_type = GGML_TYPE_F32;
        }
        if (GGML_CUDA_CC_IS_NVIDIA(cc) && src1->ne[1] > (cc >= GGML_CUDA_CC_VOLTA ? 8 : 128)) {
            // [SM70_BF16_PREFILL_F16] strata-engine #655/#540 port, promoted to default 2026-10-06
            // (A/B in artifacts/strata-port: 8K prefill +6.5%, 64K +6.2%, TG unchanged, 512-token parity
            // bit-exact at 8Kx2 and 64K). Volta/Turing have no BF16 tensor cores, so the F32 path converts
            // the whole BF16 weight to F32 every call and runs cuBLAS on the CUDA cores (~13 TFLOPS measured
            // on V100, vs ~67 through the FP16 HMMA units). BF16 -> F16 is exact inside FP16's normal range
            // (the 7-bit mantissa fits in 10), so large-batch (prefill) products go through the F16 tensor
            // cores with F32 accumulate instead (the F16 traits below + prefer_f32_output on Volta). Operands
            // beyond F16's finite range saturate to +-65504 instead of turning Inf/NaN (strata #540, see
            // ggml_get_to_fp16_sat_cuda in convert.cu). Decode shapes (ne[1] <= 8) are untouched - they keep
            // the custom sm70 BF16 GEMV / stock path. GGML_SM70_BF16_PREFILL_F16=0 restores the stock F32
            // path bit for bit.
            static const bool sm70_prefill_f16 = [] {
                const char * env = getenv("GGML_SM70_BF16_PREFILL_F16");
                return env == nullptr || atoi(env) != 0;
            }();
            if (sm70_prefill_f16 && cc >= GGML_CUDA_CC_VOLTA && cc < GGML_CUDA_CC_AMPERE) {
                compute_type = GGML_TYPE_F16;
            } else {
                compute_type = GGML_TYPE_F32;
            }
        }
    }
    if (dst->op_params[0] == GGML_PREC_F32) {
        compute_type = GGML_TYPE_F32;
    }

    const char * env_c = getenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE");
    if (env_c != nullptr) {
        std::string env_cpp = env_c;
        for (char & c : env_cpp) {
            c = std::tolower(c);
        }
        if (env_cpp == "f32" || env_cpp == "fp32") {
            compute_type = GGML_TYPE_F32;
        } else if (env_cpp == "f16" || env_cpp == "fp16") {
            compute_type = GGML_TYPE_F16;
        } else if (env_cpp == "bf16") {
            compute_type = GGML_TYPE_BF16;
        } else if (env_cpp != "auto") {
            GGML_LOG_WARN("%s: unknown value for GGML_CUDA_CUBLAS_COMPUTE_TYPE: %s", __func__, env_cpp.c_str());
        }
    }

    switch (compute_type) {
        case GGML_TYPE_F32:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F32>(ctx, src0, src1, dst);
            break;
        case GGML_TYPE_BF16:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_BF16>(ctx, src0, src1, dst);
            break;
        case GGML_TYPE_F16:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F16>(ctx, src0, src1, dst);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

static bool ggml_cuda_should_fuse_mul_mat(const ggml_tensor * ffn_up,
                                          const ggml_tensor * ffn_gate,
                                          const ggml_tensor * glu,
                                          const ggml_tensor * ffn_up_bias = nullptr,
                                          const ggml_tensor * ffn_gate_bias = nullptr,
                                          const ggml_tensor * ffn_up_scale = nullptr,
                                          const ggml_tensor * ffn_gate_scale = nullptr) {
    const bool has_bias = ffn_up_bias != nullptr || ffn_gate_bias != nullptr;
    const bool has_scale = ffn_up_scale != nullptr || ffn_gate_scale != nullptr;

    if (has_bias && (!ffn_up_bias || !ffn_gate_bias)) {
        return false;
    }
    if (has_scale && (!ffn_up_scale || !ffn_gate_scale)) {
        return false;
    }

    const bool is_mul_mat     = ffn_up->op == GGML_OP_MUL_MAT     && ffn_gate->op == GGML_OP_MUL_MAT     && glu->op == GGML_OP_GLU;
    const bool is_mul_mat_id  = ffn_up->op == GGML_OP_MUL_MAT_ID  && ffn_gate->op == GGML_OP_MUL_MAT_ID  && glu->op == GGML_OP_GLU;

    GGML_ASSERT(ffn_up && ffn_gate && glu);

    if (!is_mul_mat && !is_mul_mat_id) {
        return false;
    }

    const ggml_op expected_bias_op = is_mul_mat ? GGML_OP_ADD : GGML_OP_ADD_ID;
    const ggml_tensor * ffn_up_bias_src   = has_scale ? ffn_up_scale   : ffn_up;
    const ggml_tensor * ffn_gate_bias_src = has_scale ? ffn_gate_scale : ffn_gate;
    const ggml_tensor * ffn_up_out        = has_bias ? ffn_up_bias     : ffn_up_bias_src;
    const ggml_tensor * ffn_gate_out      = has_bias ? ffn_gate_bias   : ffn_gate_bias_src;

    if (glu->src[0] != ffn_gate_out || glu->src[1] != ffn_up_out) {
        return false;
    }

    if (has_scale) {
        if (ffn_up_scale->op != GGML_OP_MUL || ffn_gate_scale->op != GGML_OP_MUL) {
            return false;
        }
        const bool up_has_mm   = ffn_up_scale->src[0] == ffn_up || ffn_up_scale->src[1] == ffn_up;
        const bool gate_has_mm = ffn_gate_scale->src[0] == ffn_gate || ffn_gate_scale->src[1] == ffn_gate;
        if (!up_has_mm || !gate_has_mm) {
            return false;
        }
    }

    if (has_bias) {
        if (ffn_up_bias->op != expected_bias_op || ffn_gate_bias->op != expected_bias_op) {
            return false;
        }

        if (expected_bias_op == GGML_OP_ADD) {
            const bool up_has_mul   = ffn_up_bias->src[0] == ffn_up_bias_src || ffn_up_bias->src[1] == ffn_up_bias_src;
            const bool gate_has_mul = ffn_gate_bias->src[0] == ffn_gate_bias_src || ffn_gate_bias->src[1] == ffn_gate_bias_src;
            if (!up_has_mul || !gate_has_mul) {
                return false;
            }
        } else { // GGML_OP_ADD_ID
            if (ffn_up_bias->src[0] != ffn_up_bias_src || ffn_gate_bias->src[0] != ffn_gate_bias_src) {
                return false;
            }
            if (ffn_up_bias->src[2] != ffn_up->src[2] || ffn_gate_bias->src[2] != ffn_gate->src[2]) {
                return false;
            }
        }
    }

    if (ffn_up->src[0]->type != ffn_gate->src[0]->type || !ggml_are_same_shape(ffn_up->src[0], ffn_gate->src[0]) ||
        !ggml_are_same_stride(ffn_up->src[0], ffn_gate->src[0])) {
        return false;
    }

    if (ffn_up->src[1] != ffn_gate->src[1]) {
        return false;
    }

    if (is_mul_mat_id && ffn_up->src[2] != ffn_gate->src[2]) {
        return false;
    }

    static constexpr std::array<ggml_glu_op, 4> valid_glu_ops = { GGML_GLU_OP_SWIGLU, GGML_GLU_OP_GEGLU, GGML_GLU_OP_SWIGLU_OAI, GGML_GLU_OP_SWIGLU_CLAMP };

    if (std::find(valid_glu_ops.begin(), valid_glu_ops.end(), ggml_get_glu_op(glu)) == valid_glu_ops.end()) {
        return false;
    }

    if (const bool swapped = ggml_get_op_params_i32(glu, 1); swapped) {
        return false;
    }

    return true;
}

static bool ggml_cuda_should_fuse_mul_mat_vec_f(const ggml_tensor * tensor) {
    ggml_tensor *       src0 = tensor->src[0];
    ggml_tensor *       src1 = tensor->src[1];
    const ggml_tensor * dst  = tensor;

    const bool is_mul_mat_id = tensor->op == GGML_OP_MUL_MAT_ID;

    bool use_mul_mat_vec_f =
        (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16) &&
        src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;

    const int cc      = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    use_mul_mat_vec_f = use_mul_mat_vec_f && ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, is_mul_mat_id ? src1->ne[2] : src1->ne[1]);

    //we only support fusion for ncols_dst = 1
    if (tensor->op == GGML_OP_MUL_MAT && dst->ne[1] != 1) {
        return false;
    }

    if (tensor->op == GGML_OP_MUL_MAT_ID && dst->ne[2] != 1) {
        return false;
    }


    return use_mul_mat_vec_f;
}

static bool ggml_cuda_moe_fused_t16_requested() {
    static const bool value = [] { const char * env = std::getenv("GGML_CUDA_MOE_FUSED_T16");
        return env && std::strcmp(env, "1") == 0; }();
    return value;
}

static bool ggml_cuda_moe_fused_t16_type(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q2_0: case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S: case GGML_TYPE_IQ3_XXS: case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_NL: return true;
        default: return false;
    }
}

static bool ggml_cuda_should_fuse_mul_mat_vec_q(const ggml_tensor * tensor, bool allow_batch = false) {
    ggml_tensor *       src0 = tensor->src[0];
    ggml_tensor *       src1 = tensor->src[1];
    const ggml_tensor * dst  = tensor;

    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
                                   ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) &&
                                   src0->view_src;

    bool use_mul_mat_vec_q = ggml_is_quantized(src0->type) && !bad_padding_clear && src1->type == GGML_TYPE_F32 &&
                             dst->type == GGML_TYPE_F32 && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;

    // fusion is not universally faster on Pascal
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (cc <= GGML_CUDA_CC_PASCAL) {
        return false;
    }
    // fusion is for ncols_dst = 1, except that Volta also takes small batches (speculative
    // verification) through the gate+up+GLU path when the caller has no biases (allow_batch)
    if (tensor->op == GGML_OP_MUL_MAT && dst->ne[1] != 1) {
        const bool volta = GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_VOLTA;
        if (!(allow_batch && volta && dst->ne[1] <= MMVQ_MAX_BATCH_SIZE && dst->ne[2] == 1 && dst->ne[3] == 1)) {
            return false;
        }
    }

    if (tensor->op == GGML_OP_MUL_MAT_ID && dst->ne[2] > get_mmvq_mmid_max_batch(src0->type, cc)) {
        std::array<ggml_moe_t16_private::slice, 2> slices;
        const bool chunked = allow_batch && cc == GGML_CUDA_CC_VOLTA &&
                ggml_cuda_moe_fused_t16_requested() && ggml_ep_private::enabled(tensor) &&
                ggml_cuda_moe_fused_t16_type(src0->type) && get_mmvq_mmid_max_batch(src0->type, cc) >= 8 &&
                ggml_moe_t16_private::prepare(tensor, src1, tensor->src[2], dst, slices);
        if (!chunked) return false;
    }

    return use_mul_mat_vec_q;
}

// Keep small SM70 learned projections on the same reduction kernel as single-lane
// decode. This changes linear operator dispatch only; no model tokens are padded.
static int ggml_sm70_decode_linear_chunks() {
    static const int width = [] {
        const char * env = getenv("GGML_SM70_DECODE_LINEAR_CHUNKS");
        return env && strcmp(env, "4") == 0 ? 4 : env && strcmp(env, "8") == 0 ? 8 : 0;
    }();
    return width;
}

static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS

    const int32_t hint = ggml_get_op_params_i32(dst, 1);
    if (hint == GGML_HINT_SRC0_IS_HADAMARD && ggml_cuda_op_fwht(ctx, src1, dst)) {
        return;
    }

    // If src0 is a temporary compute buffer it may have some padding that needs to be cleared for mul_mat_vec_q or mul_mat_q.
    // But if src0 is also a view of another tensor then this cannot be done safely because it may overwrite valid tensor data.
    // Therefore, in such cases use cuBLAS.
    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE
        && ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) && src0->view_src;
    if (bad_padding_clear || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst);
        return;
    }

    const int cc        = ggml_cuda_info().devices[ctx.device].cc;
    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;

    const int dense_decode_chunk = ggml_sm70_decode_linear_chunks();
    if (dense_decode_chunk && cc == GGML_CUDA_CC_VOLTA &&
            ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
            ggml_backend_buft_is_cuda(src0->buffer->buft) &&
            (ggml_is_quantized(src0->type) || src0->type == GGML_TYPE_BF16 || src0->type == GGML_TYPE_F16) &&
            nb00 == ggml_type_size(src0->type) && nb10 == sizeof(float) && nb0 == sizeof(float) &&
            ne11 > dense_decode_chunk && ne11 <= 64 && ne02 == 1 && ne03 == 1 && ne12 == 1 && ne13 == 1 && ne2 == 1 && ne3 == 1) {
        static thread_local uint64_t logged_shapes[GGML_TYPE_COUNT] = {};
        const uint64_t shape_bit = UINT64_C(1) << (ne11 - 1);
        if (!(logged_shapes[src0->type] & shape_bit)) {
            fprintf(stderr, "sm70 decode linear chunks active: dense columns=%lld chunk=%d type=%s\n",
                    (long long) ne11, dense_decode_chunk, ggml_type_name(src0->type));
            logged_shapes[src0->type] |= shape_bit;
        }
        // Broadcast the same weight matrix across independent four-column
        // channel groups. The MMVQ specialization/reduction stays T4, while
        // input quantization and kernel launch are shared by all groups.
        static const bool packed_chunks = [] {
            const char * env = getenv("GGML_SM70_DECODE_LINEAR_PACKED");
            return env && strcmp(env, "1") == 0;
        }();
        static const bool bf16_panel_requested = [] {
            const char * panel = getenv("GGML_SM70_BF16_DECODE_PANEL");
            const char * gemv = getenv("GGML_SM70_BF16_GEMV");
            return panel && strcmp(panel, "1") == 0 && (!gemv || atoi(gemv) != 0);
        }();
        if (bf16_panel_requested && dense_decode_chunk == 4 && src0->type == GGML_TYPE_BF16 &&
                (ne00 >= 10240 || ne01 >= 8192) &&
                ggml_is_contiguous(src1) && ggml_is_contiguous(dst)) {
            const bool selected = ggml_sm70_bf16_panel::dispatch(src0->data,
                    static_cast<const float *>(src1->data), static_cast<float *>(dst->data),
                    int(ne01), int(ne00), int(ne11), nb01/2, nb11/4, nb1/4, ctx.stream());
            ggml_cuda_optimization_path_log(ctx.device, "sm70_bf16_decode_panel", true, selected,
                    selected ? "original_fp32_reduction_wide_columns" : "unsupported_shape_or_alignment",
                    ne11, ne01, ne00);
            if (selected) return;
        }
        if (packed_chunks && ggml_is_quantized(src0->type) &&
                ne11 % dense_decode_chunk == 0 && ggml_is_contiguous(src1) && ggml_is_contiguous(dst) &&
                ggml_cuda_should_use_mmvq(src0->type, cc, dense_decode_chunk)) {
            ggml_tensor input = *src1;
            ggml_tensor output = *dst;
            input.ne[1] = output.ne[1] = dense_decode_chunk;
            input.ne[2] = output.ne[2] = ne11 / dense_decode_chunk;
            input.nb[2] = input.nb[1] * dense_decode_chunk;
            output.nb[2] = output.nb[1] * dense_decode_chunk;
            output.src[0] = const_cast<ggml_tensor *>(src0);
            output.src[1] = &input;
            GGML_ASSERT(ggml_is_contiguous(&input) && ggml_is_contiguous(&output));
            ggml_cuda_optimization_path_log(ctx.device, "dense_decode_packed_chunks", true, true,
                    "original_mmvq_reduction_broadcast_weight_channels", ne11, dense_decode_chunk, input.ne[2]);
            ggml_cuda_mul_mat_vec_q(ctx, src0, &input, nullptr, &output);
            return;
        }
        for (int64_t first = 0; first < ne11; first += dense_decode_chunk) {
            ggml_tensor input = *src1;
            ggml_tensor output = *dst;
            input.ne[1] = output.ne[1] = std::min<int64_t>(dense_decode_chunk, ne11 - first);
            input.data = (char *) src1->data + first * src1->nb[1];
            output.data = (char *) dst->data + first * dst->nb[1];
            output.src[0] = const_cast<ggml_tensor *>(src0);
            output.src[1] = &input;
            ggml_cuda_mul_mat(ctx, src0, &input, &output);
        }
        return;
    }

    // The stock F16 path switches MMVF -> Tensor Core at four columns. Keep
    // the tail of a decode chunk on the same vector reduction as its full chunk.
    if (ggml_sm70_decode_linear_chunks() && cc == GGML_CUDA_CC_VOLTA && src0->type == GGML_TYPE_F16 &&
            ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
            ggml_backend_buft_is_cuda(src0->buffer->buft) && ne11 >= 1 && ne11 <= 8 &&
            ne02 == 1 && ne03 == 1 && ne12 == 1 && ne13 == 1 && ne2 == 1 && ne3 == 1 &&
            ne00 % 2 == 0 && nb00 == sizeof(ggml_fp16_t) && nb01 % 4 == 0 && nb02 % 4 == 0 && nb03 % 4 == 0 &&
            nb10 == sizeof(float) && nb0 == sizeof(float)) {
        ggml_cuda_mul_mat_vec_f(ctx, src0, src1, nullptr, dst);
        return;
    }

    // PLE value is mirrored and shares an input with PLE key. Small shared
    // graphs use fused-projection's FP32 reduction, while larger decode graphs
    // used MMVF after the 8-column split. Use the matching standalone reduction
    // for this one weight so packing lanes cannot select a different sum order.
    static const bool canonical_ple_value_on = [] {
        const char * env = getenv("GGML_SM70_BF16_CANONICAL_PLE");
        return env != nullptr && atoi(env) != 0;
    }();
    if (canonical_ple_value_on && ggml_sm70_decode_linear_chunks() &&
            cc == GGML_CUDA_CC_VOLTA && src0->type == GGML_TYPE_BF16 &&
            strcmp(src0->name, "blk.1.ple_value.weight") == 0 &&
            ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
            ggml_backend_buft_is_cuda(src0->buffer->buft) && ne00 == 2560 && ne01 == 2560 &&
            ne11 >= 1 && ne11 <= 8 && ne02 == 1 && ne03 == 1 &&
            ne12 == 1 && ne13 == 1 && ne2 == 1 && ne3 == 1 &&
            nb00 == ggml_type_size(src0->type) && nb10 == sizeof(float) && nb0 == sizeof(float)) {
        const bool selected = gemv_sm70::gemv_sm70_bf16_launch_generic(
                src0->data, (const float *) src1->data, (float *) dst->data,
                (int) ne01, (int) ne00, (int) ne11, nb01/2, nb11/4, nb1/4, ctx.stream());
        ggml_cuda_optimization_path_log(ctx.device, "sm70_ple_value_gemv", true, selected,
                selected ? "canonical_fp32_reduction" : "kernel_guard_fallback", ne11, ne01, ne00);
        if (selected) {
            return;
        }
    }

    // [SM70_BF16_GEMV] V100-specific BF16 matvec for decode shapes (kernel_sm70_bf16.cuh): wins where
    // the weight stream is long (K >= 10240) or the matrix is tall at MTP-verify batch sizes, loses on
    // small shapes -> shape whitelist + all kernel-side guards (alignment/strides) keep the rest on the
    // stock path. Env GGML_SM70_BF16_GEMV=0 forces the stock path, =2 additionally logs every hit.
    static const bool sm70_bf16_gemv_on = [] {
        const char * env = getenv("GGML_SM70_BF16_GEMV");
        return env == nullptr || atoi(env) != 0;
    }();
    static const bool sm70_bf16_gemv_log = [] {
        const char * env = getenv("GGML_SM70_BF16_GEMV");
        return env != nullptr && atoi(env) == 2;
    }();
    if (sm70_bf16_gemv_on && src0->type == GGML_TYPE_BF16 && cc == GGML_CUDA_CC_VOLTA &&
            ne11 <= 8 && ne02 == 1 && ne03 == 1 && ne12 == 1 && ne13 == 1 && ne2 == 1 && ne3 == 1 &&
            nb00 == ggml_type_size(src0->type) && nb10 == sizeof(float) && nb0 == sizeof(float) &&
            (ne00 >= 10240 || ((ne11 >= 5 || (ggml_sm70_decode_linear_chunks() &&
                ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS)) && ne01 >= 8192)) &&
            gemv_sm70::gemv_sm70_bf16_launch_generic(src0->data, (const float *) src1->data, (float *) dst->data,
                (int) ne01, (int) ne00, (int) ne11, nb01/2, nb11/4, nb1/4, ctx.stream())) {
        if (sm70_bf16_gemv_log) {
            fprintf(stderr, "%s: sm70_bf16_gemv n_out=%lld k_dim=%lld ncols=%lld\n",
                    __func__, (long long) ne01, (long long) ne00, (long long) ne11);
        }
        return;
    }

#ifdef MONPACK_AVAILABLE
    // [P3 monpack] weights redirected to the sidecar blob: src0->data points into a packed
    // monpack region, so it must be consumed here before any stock path (MMVF/MMF/MMVQ/MMQ/
    // cuBLAS) interprets those bytes as a plain quantized layout. Placed after the SM70
    // special sections, before the general dispatch tier.
    { static int mp_dbg2 = 6; if (mp_dbg2 > 0) { fprintf(stderr, "[mp-dbg] mul_mat hook: type=%d ne11=%lld reg=%d ptr=%p\n", (int)src0->type, (long long)src1->ne[1], (int)monpack::is_registered(src0->data), src0->data); mp_dbg2--; } }
    if (monpack::is_registered(src0->data) && monpack::try_mul_mat(ctx, dst, src0)) {
        return;
    }
#endif

    if (ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, ne11)) {
        // The custom F16 vector kernel can be used over batched cuBLAS GEMM.
        // But this is only faster for GPUs without tensor cores or with a thin src0 matrix (particularly KQV in attention)
        ggml_cuda_mul_mat_vec_f(ctx, src0, src1, nullptr, dst);
        return;
    }
    // A transposed vector can still use MMVQ (i.e. ne01 == 1)
    if (ne01 == 1 && ne11 > MMVF_MAX_BATCH_SIZE && ne2 == 1 && ne3 == 1
            && src0->type == GGML_TYPE_F32
            && ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst)
            && ggml_cuda_should_use_mmvf(src1->type, cc, src1->ne, src1->nb, /*ne11 =*/ 1)) {
        ggml_tensor dst_vec = *dst;
        dst_vec.ne[0] = ne11;
        dst_vec.ne[1] = 1;
        dst_vec.nb[1] = dst_vec.nb[0]*ne11;
        dst_vec.nb[2] = dst_vec.nb[1];
        dst_vec.nb[3] = dst_vec.nb[1];
        ggml_cuda_mul_mat_vec_f(ctx, src1, src0, nullptr, &dst_vec);
        return;
    }
    if (ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, ne11, /*mul_mat_id =*/ false)) {
        ggml_cuda_mul_mat_f(ctx, src0, src1, nullptr, dst);
        return;
    }
    if (ggml_cuda_should_use_mmvq(src0->type, cc, ne11)) {
        ggml_cuda_mul_mat_vec_q(ctx, src0, src1, nullptr, dst);
        return;
    }
    if (ggml_cuda_should_use_mmq(src0->type, cc, ne11, /*n_experts =*/ 0)) {
        ggml_cuda_mul_mat_q(ctx, src0, src1, nullptr, dst);
        return;
    }
    ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst);
}

// returns true when ggml_cuda_mul_mat_id takes the fallback path that requires stream synchronization
// [TAG_MUL_MAT_ID_CUDA_GRAPHS]
static bool ggml_cuda_mul_mat_id_needs_sync(const ggml_tensor * dst, const int cc) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const int64_t n_experts_for_heuristic = ggml_ep_private::enabled(dst) ?
        ggml_ep_private::global_experts(dst) : src0->ne[2];

#ifdef MONPACK_AVAILABLE
    // [P3 monpack] registered (sidecar-packed) weights never reach the sync-requiring
    // fallback: the monpack entry point is stream-pure and CUDA-graph capturable
    if (monpack::is_registered(src0->data)) {
        return false;
    }
#endif

    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return true;
    }

    if (dst->ne[2] <= MMVQ_MAX_BATCH_SIZE) {
        if (ggml_is_quantized(src0->type)) {
            if (dst->ne[2] <= get_mmvq_mmid_max_batch(src0->type, cc)) {
                return false;
            }
        } else if (GGML_CUDA_CC_IS_AMD(cc)) {
            return false;
        }
    }

    if (ggml_cuda_should_use_mmq(src0->type, cc, src1->ne[2], n_experts_for_heuristic)) {
        return false;
    }

    if (ggml_cuda_should_use_mmf(src0->type, cc, WARP_SIZE, src0->ne, src0->nb, src1->ne[2], /*mul_mat_id=*/true)) {
        return false;
    }

    return true;
}

static void ggml_cuda_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const bool ep = ggml_ep_private::enabled(dst);
    const int64_t n_experts_for_heuristic = ep ? ggml_ep_private::global_experts(dst) : ne02;

    if (ggml_sm70_decode_linear_chunks() && cc == GGML_CUDA_CC_VOLTA && ggml_is_quantized(src0->type) &&
            ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
            ggml_backend_buft_is_cuda(src0->buffer->buft) &&
            nb00 == ggml_type_size(src0->type) && nb10 == sizeof(float) && nb0 == sizeof(float) && ids->nb[0] == sizeof(int32_t) &&
            ne2 > 8 && ne2 <= 64 && ne12 == ne2 && ne13 == 1 && ne3 == 1 && ids->ne[1] == ne2 &&
            ids->ne[2] == 1 && ids->ne[3] == 1) {
        const int chunk = std::min(8, get_mmvq_mmid_max_batch(src0->type, cc));
        if (chunk > 0) {
            if (ep) {
                ggml_cuda_optimization_path_log(ctx.device, "ep_mmvq_chunks", true, true,
                        "contiguous_expert_slice_preserved_in_each_chunk", ne2, ne01, ne02);
            }
            static thread_local uint64_t logged_shapes[GGML_TYPE_COUNT] = {};
            const uint64_t shape_bit = UINT64_C(1) << (ne2 - 1);
            if (!(logged_shapes[src0->type] & shape_bit)) {
                fprintf(stderr, "sm70 decode linear chunks active: moe tokens=%lld chunk=%d type=%s\n",
                        (long long) ne2, chunk, ggml_type_name(src0->type));
                logged_shapes[src0->type] |= shape_bit;
            }
            for (int64_t first = 0; first < ne2; first += chunk) {
                ggml_tensor input = *src1;
                ggml_tensor selected = *ids;
                ggml_tensor output = *dst;
                const int64_t count = std::min<int64_t>(chunk, ne2 - first);
                input.ne[2] = output.ne[2] = selected.ne[1] = count;
                input.data = (char *) src1->data + first * src1->nb[2];
                selected.data = (char *) ids->data + first * ids->nb[1];
                output.data = (char *) dst->data + first * dst->nb[2];
                output.src[0] = const_cast<ggml_tensor *>(src0);
                output.src[1] = &input;
                output.src[2] = &selected;
                ggml_cuda_mul_mat_vec_q(ctx, src0, &input, &selected, &output);
            }
            return;
        }
    }

#ifdef MONPACK_AVAILABLE
    // [P3 monpack] unified entry for sidecar-packed expert weights, after the SM70 decode
    // chunk path and before the grouped MMQ / MMVQ / MMF tier: small decode M (ne2) is
    // handled inside the pack by the GEMV-on-pack path, so no chunking is needed here.
    { static int mp_dbg = 6; if (mp_dbg > 0) { fprintf(stderr, "[mp-dbg] mul_mat_id hook: type=%d ne11=%lld reg=%d ptr=%p ne=[%lld,%lld,%lld]\n", (int)src0->type, (long long)src1->ne[1], (int)monpack::is_registered(src0->data), src0->data, (long long)src0->ne[0], (long long)src0->ne[1], (long long)src0->ne[2]); mp_dbg--; } }
    if (monpack::is_registered(src0->data) && ep) {
        fprintf(stderr, "EP unsupported: sidecar-packed MUL_MAT_ID weights cannot use the canonical contiguous-slice kernels\n");
        GGML_ABORT("expert parallelism cannot read monpack sidecar weights through stock MMVQ/MMQ");
    }
    if (!ep && monpack::is_registered(src0->data) && monpack::try_mul_mat_id(ctx, dst)) {
        return;
    }
#endif

    // [TAG_MUL_MAT_ID_CUDA_GRAPHS]
    if (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        static_assert(MMVQ_MAX_BATCH_SIZE == MMVF_MAX_BATCH_SIZE);
        if (ne2 <= MMVQ_MAX_BATCH_SIZE) {
            if (ggml_is_quantized(src0->type)) {
                const int mmvq_mmid_max = get_mmvq_mmid_max_batch(src0->type, cc);
                if (ne2 <= mmvq_mmid_max) {
                    if (ep) ggml_cuda_optimization_path_log(ctx.device, "ep_mmvq", true, true,
                            "contiguous_expert_route", ne2, ne01, ne02);
                    ggml_cuda_mul_mat_vec_q(ctx, src0, src1, ids, dst);
                    return;
                }
            } else {
                if (GGML_CUDA_CC_IS_AMD(cc)) {
                    ggml_cuda_mul_mat_vec_f(ctx, src0, src1, ids, dst);
                    return;
                }
            }
        }

        if (ggml_cuda_should_use_mmq(src0->type, cc, ne12, n_experts_for_heuristic)) {
            if (ep) ggml_cuda_optimization_path_log(ctx.device, "ep_mmq", true, true,
                    "contiguous_expert_route_grouped_or_regular", ne2, ne01, ne02);
            ggml_cuda_mul_mat_q(ctx, src0, src1, ids, dst);
            return;
        }

        if (ggml_cuda_should_use_mmf(src0->type, cc, WARP_SIZE, src0->ne, src0->nb, src1->ne[2], /*mul_mat_id=*/true)) {
            ggml_cuda_mul_mat_f(ctx, src0, src1, ids, dst);
            return;
        }
    }

    // note: this path should not be reached when recording CUDA graphs, because it requires stream synchronization
    GGML_ASSERT(ggml_cuda_mul_mat_id_needs_sync(dst, cc));
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const ggml_type type_src1_sorted = (src0->type == GGML_TYPE_F16 && !fast_fp16_hardware_available(cc))
        || ggml_is_quantized(src0->type) ? GGML_TYPE_F32 : src0->type;
    const ggml_type type_dst_sorted  = GGML_TYPE_F32;
    const size_t ts_src1_sorted = ggml_type_size(type_src1_sorted);
    const size_t ts_dst_sorted  = ggml_type_size(type_dst_sorted);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;

    std::vector<int32_t> ids_to_sorted_host;
    ids_to_sorted_host.reserve(2*ne_get_rows);
    std::vector<int32_t> ids_from_sorted_host(ne_get_rows);

    ggml_cuda_pool_alloc<int32_t> ids_buf_dev(ctx.pool(), 2*ne_get_rows);

    std::vector<int32_t> tokens_per_expert(ne02);

    ggml_cuda_pool_alloc<char> src1_sorted(ctx.pool(), ne12*n_expert_used*ne10*ts_src1_sorted);
    ggml_cuda_pool_alloc<char>  dst_sorted(ctx.pool(), ne2 *n_expert_used* ne0*ts_dst_sorted);

    std::vector<char> ids_host(ggml_nbytes(ids));
    CUDA_CHECK(cudaMemcpyAsync(ids_host.data(), ids->data, ggml_nbytes(ids), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int64_t i02 = 0; i02 < ne02; ++i02) { // expert matrices
        for (int64_t i12 = 0; i12 < ne12; ++i12) { // tokens
            for (int64_t iex = 0; iex < n_expert_used; ++iex) {
                const int32_t expert_to_use = *(const int32_t *)(ids_host.data() + i12*ids->nb[1] + iex*ids->nb[0]);
                assert(expert_to_use >= 0 && expert_to_use < ne02);
                if (expert_to_use == i02) {
                    ids_from_sorted_host[i12*n_expert_used + iex] = ids_to_sorted_host.size();
                    ids_to_sorted_host.push_back(i12*ne11 + iex % ne11);
                    tokens_per_expert[i02]++;
                    break;
                }
            }
        }
    }
    GGML_ASSERT(ids_to_sorted_host.size() == size_t(ne_get_rows));

    ids_to_sorted_host.insert(ids_to_sorted_host.end(), ids_from_sorted_host.begin(), ids_from_sorted_host.end());

    CUDA_CHECK(cudaMemcpyAsync(ids_buf_dev.ptr, ids_to_sorted_host.data(), 2*ne_get_rows*sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const int32_t * ids_to_sorted   = ids_buf_dev.ptr + 0*ne_get_rows;
    const int32_t * ids_from_sorted = ids_buf_dev.ptr + 1*ne_get_rows;

    get_rows_cuda(src1->data, src1->type, ids_to_sorted, src1_sorted.ptr, type_src1_sorted,
        ne10, nb11, nb12, nb13,
        ne_get_rows, 1, 1, sizeof(int32_t), ne_get_rows*sizeof(int32_t), ne_get_rows*sizeof(int32_t),
        ne10*ts_src1_sorted, ne_get_rows*ne10*ts_src1_sorted, ne_get_rows*ne10*ts_src1_sorted, stream);
    CUDA_CHECK(cudaGetLastError());

    char * src1_data_cur = (char *) src1_sorted.ptr;
    char *  dst_data_cur = (char *)  dst_sorted.ptr;
    for (int64_t i02 = 0; i02 < ne02; ++i02) {
        if (tokens_per_expert[i02] == 0) {
            continue;
        }

        ggml_tensor src0_slice = *src0;
        src0_slice.ne[2]    = 1;
        src0_slice.nb[3]    = src0_slice.nb[2];
        src0_slice.op       = GGML_OP_VIEW;
        src0_slice.view_src = dst->src[0]; // non-const pointer to src0
        src0_slice.data     = (char *) src0->data + i02*nb02;

        ggml_tensor src1_slice;
        memset(&src1_slice, 0, sizeof(src1_slice));
        src1_slice.buffer = src1->buffer;
        src1_slice.type   = type_src1_sorted;
        src1_slice.ne[0]  = ne10;
        src1_slice.ne[1]  = tokens_per_expert[i02];
        src1_slice.ne[2]  = 1;
        src1_slice.ne[3]  = 1;
        src1_slice.nb[0]  = ts_src1_sorted;
        src1_slice.nb[1]  = src1_slice.ne[0] * src1_slice.nb[0];
        src1_slice.nb[2]  = src1_slice.ne[1] * src1_slice.nb[1];
        src1_slice.nb[3]  = src1_slice.ne[2] * src1_slice.nb[2];
        src1_slice.data   = src1_data_cur;

        ggml_tensor dst_slice;
        memset(&dst_slice, 0, sizeof(dst_slice));
        dst_slice.buffer = dst->buffer;
        dst_slice.type   = type_dst_sorted;
        dst_slice.ne[0]  = ne0;
        dst_slice.ne[1]  = tokens_per_expert[i02];
        dst_slice.ne[2]  = 1;
        dst_slice.ne[3]  = 1;
        dst_slice.nb[0]  = ts_dst_sorted;
        dst_slice.nb[1]  = dst_slice.ne[0] * dst_slice.nb[0];
        dst_slice.nb[2]  = dst_slice.ne[1] * dst_slice.nb[1];
        dst_slice.nb[3]  = dst_slice.ne[2] * dst_slice.nb[2];
        dst_slice.data   = dst_data_cur;

        // A local src0_slice has ne[2] == 1, so preserve the expert-fallback
        // caller role explicitly instead of guessing from the view shape.
        ggml_cuda_cublas_role_audit::expert_fallback_scope expert_role_scope;
        ggml_cuda_mul_mat(ctx, &src0_slice, &src1_slice, &dst_slice);
        CUDA_CHECK(cudaGetLastError());

        src1_data_cur += src1_slice.nb[2];
        dst_data_cur  +=  dst_slice.nb[2];
    }

    get_rows_cuda(dst_sorted.ptr, type_dst_sorted, ids_from_sorted, dst->data, dst->type,
        ne0, ne0*ts_dst_sorted, ne_get_rows*ne0*ts_dst_sorted, ne_get_rows*ne0*ts_dst_sorted,
        ne_get_rows, 1, 1, sizeof(int32_t), ne_get_rows*sizeof(int32_t), ne_get_rows*sizeof(int32_t),
        nb1, nb2, nb3, stream);
}

static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    switch (dst->op) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        case GGML_OP_CUSTOM:
            if (ggml_cuda_ft_dense_supports(dst, ctx.device)) { ggml_cuda_ft_dense_compute(ctx, dst); break; }
            if (!ggml_cuda_ft_supports(dst, ctx.device)) { return false; }
            ggml_cuda_ft_compute(ctx, dst);
            break;
#endif
        case GGML_OP_ARGMAX:
            ggml_cuda_argmax(ctx, dst);
            break;
        case GGML_OP_ARGMAX_SPLIT:
            ggml_cuda_argmax_split(ctx, dst);
            break;
        case GGML_OP_HC_FUSED:
            ggml_cuda_op_hc_fused(ctx, dst);
            break;
        case GGML_OP_COUNT_EQUAL:
            ggml_cuda_count_equal(ctx, dst);
            break;
        case GGML_OP_REPEAT:
            ggml_cuda_op_repeat(ctx, dst);
            break;
        case GGML_OP_REPEAT_BACK:
            ggml_cuda_op_repeat_back(ctx, dst);
            break;
        case GGML_OP_GET_ROWS:
            ggml_cuda_op_get_rows(ctx, dst);
            break;
        case GGML_OP_GET_ROWS_BACK:
            ggml_cuda_op_get_rows_back(ctx, dst);
            break;
        case GGML_OP_SET_ROWS:
            ggml_cuda_op_set_rows(ctx, dst);
            break;
        case GGML_OP_SET:
            ggml_cuda_op_set(ctx, dst);
            break;
        case GGML_OP_DUP:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_CPY:
            ggml_cuda_cpy(ctx, dst->src[0], dst->src[1]);
            break;
        case GGML_OP_CONT:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_ADD:
        case GGML_OP_ADD1: // TODO: more efficient implementation
            ggml_cuda_op_add(ctx, dst);
            break;
        case GGML_OP_ADD_ID:
            ggml_cuda_op_add_id(ctx, dst);
            break;
        case GGML_OP_SUB:
            ggml_cuda_op_sub(ctx, dst);
            break;
        case GGML_OP_ACC:
            ggml_cuda_op_acc(ctx, dst);
            break;
        case GGML_OP_MUL:
            ggml_cuda_op_mul(ctx, dst);
            break;
        case GGML_OP_DIV:
            ggml_cuda_op_div(ctx, dst);
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_ABS:
                    ggml_cuda_op_abs(ctx, dst);
                    break;
                case GGML_UNARY_OP_SGN:
                    ggml_cuda_op_sgn(ctx, dst);
                    break;
                case GGML_UNARY_OP_NEG:
                    ggml_cuda_op_neg(ctx, dst);
                    break;
                case GGML_UNARY_OP_STEP:
                    ggml_cuda_op_step(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU:
                    ggml_cuda_op_gelu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SILU:
                    ggml_cuda_op_silu(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_ERF:
                    ggml_cuda_op_gelu_erf(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_QUICK:
                    ggml_cuda_op_gelu_quick(ctx, dst);
                    break;
                case GGML_UNARY_OP_TANH:
                    ggml_cuda_op_tanh(ctx, dst);
                    break;
                case GGML_UNARY_OP_RELU:
                    ggml_cuda_op_relu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SIGMOID:
                    ggml_cuda_op_sigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSIGMOID:
                    ggml_cuda_op_hardsigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSWISH:
                    ggml_cuda_op_hardswish(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXP:
                    ggml_cuda_op_exp(ctx, dst);
                    break;
                case GGML_UNARY_OP_ELU:
                    ggml_cuda_op_elu(ctx, dst);
                    break;
                case GGML_UNARY_OP_XIELU:
                    ggml_cuda_op_xielu(ctx, dst);
                    break;
                case GGML_UNARY_OP_FLOOR:
                    ggml_cuda_op_floor(ctx, dst);
                    break;
                case GGML_UNARY_OP_CEIL:
                    ggml_cuda_op_ceil(ctx, dst);
                    break;
                case GGML_UNARY_OP_ROUND:
                    ggml_cuda_op_round(ctx, dst);
                    break;
                case GGML_UNARY_OP_TRUNC:
                    ggml_cuda_op_trunc(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXPM1:
                    ggml_cuda_op_expm1(ctx, dst);
                    break;
                case GGML_UNARY_OP_SOFTPLUS:
                    ggml_cuda_op_softplus(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(dst)) {
                case GGML_GLU_OP_REGLU:
                    ggml_cuda_op_reglu(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU:
                    ggml_cuda_op_geglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU:
                    ggml_cuda_op_swiglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU_OAI:
                    ggml_cuda_op_swiglu_oai(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_ERF:
                    ggml_cuda_op_geglu_erf(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_QUICK:
                    ggml_cuda_op_geglu_quick(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    ggml_cuda_op_swiglu_clamp(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_NORM:
            ggml_cuda_op_norm(ctx, dst);
            break;
        case GGML_OP_GROUP_NORM:
            ggml_cuda_op_group_norm(ctx, dst);
            break;
        case GGML_OP_L2_NORM:
            ggml_cuda_op_l2_norm(ctx, dst);
            break;
        case GGML_OP_CONCAT:
            ggml_cuda_op_concat(ctx, dst);
            break;
        case GGML_OP_UPSCALE:
            ggml_cuda_op_upscale(ctx, dst);
            break;
        case GGML_OP_PAD:
            ggml_cuda_op_pad(ctx, dst);
            break;
        case GGML_OP_PAD_REFLECT_1D:
            ggml_cuda_op_pad_reflect_1d(ctx, dst);
            break;
        case GGML_OP_ARANGE:
            ggml_cuda_op_arange(ctx, dst);
            break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            ggml_cuda_op_timestep_embedding(ctx, dst);
            break;
        case GGML_OP_LEAKY_RELU:
            ggml_cuda_op_leaky_relu(ctx, dst);
            break;
        case GGML_OP_SILU_BACK:
            ggml_cuda_op_silu_back(ctx, dst);
            break;
        case GGML_OP_RMS_NORM:
            ggml_cuda_op_rms_norm(ctx, dst);
            break;
        case GGML_OP_RMS_NORM_BACK:
            ggml_cuda_op_rms_norm_back(ctx, dst);
            break;
        case GGML_OP_MUL_MAT:
            ggml_cuda_mul_mat(ctx, dst->src[0], dst->src[1], dst);
            break;
        case GGML_OP_MUL_MAT_ID:
            ggml_cuda_mul_mat_id(ctx, dst);
            break;
        case GGML_OP_OUT_PROD:
            ggml_cuda_out_prod(ctx, dst);
            break;
        case GGML_OP_SCALE:
            ggml_cuda_op_scale(ctx, dst);
            break;
        case GGML_OP_SQR:
            ggml_cuda_op_sqr(ctx, dst);
            break;
        case GGML_OP_SQRT:
            ggml_cuda_op_sqrt(ctx, dst);
            break;
        case GGML_OP_SIN:
            ggml_cuda_op_sin(ctx, dst);
            break;
        case GGML_OP_COS:
            ggml_cuda_op_cos(ctx, dst);
            break;
        case GGML_OP_CLAMP:
            ggml_cuda_op_clamp(ctx, dst);
            break;
        case GGML_OP_LOG:
            ggml_cuda_op_log(ctx, dst);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
                break;
        case GGML_OP_DIAG:
            ggml_cuda_op_diag(ctx, dst);
            break;
        case GGML_OP_DIAG_MASK_INF:
            ggml_cuda_op_diag_mask_inf(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX:
            ggml_cuda_op_soft_max(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX_BACK:
            ggml_cuda_op_soft_max_back(ctx, dst);
            break;
        case GGML_OP_ROPE:
            ggml_cuda_op_rope(ctx, dst);
            break;
        case GGML_OP_ROPE_BACK:
            ggml_cuda_op_rope_back(ctx, dst);
            break;
        case GGML_OP_ROLL:
            ggml_cuda_op_roll(ctx, dst);
            break;
        case GGML_OP_IM2COL:
            ggml_cuda_op_im2col(ctx, dst);
            break;
        case GGML_OP_IM2COL_3D:
            ggml_cuda_op_im2col_3d(ctx, dst);
            break;
        case GGML_OP_CONV_2D:
            ggml_cuda_op_conv2d(ctx, dst);
            break;
        case GGML_OP_CONV_2D_DW:
            ggml_cuda_op_conv2d_dw(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_2D:
            ggml_cuda_conv_2d_transpose_p0(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            ggml_cuda_op_conv_transpose_1d(ctx,dst);
            break;
        case GGML_OP_COL2IM_1D:
            ggml_cuda_op_col2im_1d(ctx, dst);
            break;
        case GGML_OP_POOL_2D:
            ggml_cuda_op_pool2d(ctx, dst);
            break;
        case GGML_OP_POOL_1D:
            ggml_cuda_op_pool1d(ctx, dst);
            break;
        case GGML_OP_SUM:
            ggml_cuda_op_sum(ctx, dst);
            break;
        case GGML_OP_CUMSUM:
            ggml_cuda_op_cumsum(ctx, dst);
            break;
        case GGML_OP_SUM_ROWS:
            ggml_cuda_op_sum_rows(ctx, dst);
            break;
        case GGML_OP_MEAN:
            ggml_cuda_op_mean(ctx, dst);
            break;
        case GGML_OP_SSM_CONV:
            ggml_cuda_op_ssm_conv(ctx, dst);
            break;
        case GGML_OP_SSM_SCAN:
            ggml_cuda_op_ssm_scan(ctx, dst);
            break;
        case GGML_OP_TOP_K:
            ggml_cuda_op_top_k(ctx, dst);
            break;
        case GGML_OP_ARGSORT:
            ggml_cuda_op_argsort(ctx, dst);
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            ggml_cuda_flash_attn_ext(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS:
            ggml_cuda_cross_entropy_loss(ctx, dst);
            break;
        case GGML_OP_TRI:
            ggml_cuda_op_tri(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV6:
            ggml_cuda_op_rwkv_wkv6(ctx, dst);
            break;
        case GGML_OP_GATED_LINEAR_ATTN:
            ggml_cuda_op_gated_linear_attn(ctx, dst);
            break;
        case GGML_OP_GATED_DELTA_NET:
            ggml_cuda_op_gated_delta_net(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_COMB:
            ggml_cuda_op_dsv4_hc_comb(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_PRE:
            ggml_cuda_op_dsv4_hc_pre(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_POST:
            ggml_cuda_op_dsv4_hc_post(ctx, dst);
            break;
        case GGML_OP_MONOMOE_SM70:
            ggml_cuda_op_monomoe(ctx, dst);
            break;
        case GGML_OP_PLE_CONV_SM70:
            ggml_cuda_op_ple_conv_sm70(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV7:
            ggml_cuda_op_rwkv_wkv7(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            ggml_cuda_cross_entropy_loss_back(ctx, dst);
            break;
        case GGML_OP_OPT_STEP_ADAMW:
            ggml_cuda_opt_step_adamw(ctx, dst);
            break;
        case GGML_OP_OPT_STEP_SGD:
            ggml_cuda_opt_step_sgd(ctx, dst);
            break;
        case GGML_OP_SOLVE_TRI:
            ggml_cuda_op_solve_tri(ctx, dst);
            break;
        case GGML_OP_FILL:
            ggml_cuda_op_fill(ctx, dst);
            break;
        case GGML_OP_LIGHTNING_INDEXER:
            ggml_cuda_lightning_indexer(ctx, dst);
            break;
        case GGML_OP_ALLREDUCE:
            ggml_cuda_op_allreduce(ctx, dst);
            break;
        default:
            return false;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("%s: %s failed\n", __func__, ggml_op_desc(dst));
        CUDA_CHECK(err);
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////

// backend

static const char * ggml_backend_cuda_get_name(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    return cuda_ctx->name.c_str();
}

static void ggml_backend_cuda_free(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    ggml_cuda_layer_pinned_upload_private::free(cuda_ctx);
#endif
    delete cuda_ctx;
    delete backend;
}

static void ggml_backend_cuda_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpyAsync((char *) tensor->data + offset, data, size, cudaMemcpyHostToDevice, cuda_ctx->stream()));
}

static void ggml_backend_cuda_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpyAsync(data, (const char *) tensor->data + offset, size, cudaMemcpyDeviceToHost, cuda_ctx->stream()));
}

static void ggml_backend_cuda_set_tensor_2d_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpy2DAsync(
        (char *) tensor->data + offset, stride_tensor, data, stride_data, size, n_copies, cudaMemcpyHostToDevice, cuda_ctx->stream()));
}

static void ggml_backend_cuda_get_tensor_2d_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpy2DAsync(
        data, stride_data, (const char *) tensor->data + offset, stride_tensor, size, n_copies, cudaMemcpyDeviceToHost, cuda_ctx->stream()));
}

// Optional stream spans; cross-device timestamps are approximate.
struct ggml_cuda_meta_span_slot {
    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    size_t sequence = 0;
    size_t nodes = 0;
    bool pending = false;
};

struct ggml_cuda_meta_span {
    ggml_backend_cuda_context * ctx = nullptr;
    cudaEvent_t origin = nullptr;
    int64_t origin_us = 0;
    int64_t error_us = 0;
    size_t next = 0;
    size_t dropped = 0;
    ggml_cuda_meta_span_slot slots[16];
};

static void ggml_cuda_meta_span_poll(ggml_cuda_meta_span * trace, bool drain) {
    ggml_cuda_set_device(trace->ctx->device);
    for (auto & slot : trace->slots) {
        if (!slot.pending) continue;
        cudaError_t ready = cudaEventQuery(slot.end);
        if (ready == cudaErrorNotReady && !drain) continue;
        if (drain) CUDA_CHECK(cudaEventSynchronize(slot.end));
        else CUDA_CHECK(ready);
        float offset_ms = 0, duration_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&offset_ms, trace->origin, slot.begin));
        CUDA_CHECK(cudaEventElapsedTime(&duration_ms, slot.begin, slot.end));
        GGML_LOG_INFO("meta-gpu-span device=%d trace=%p seq=%zu nodes=%zu approx_begin_us=%.0f duration_us=%.0f initial_calibration_bracket_us=%" PRId64 "\n",
                trace->ctx->device, (void *) trace, slot.sequence, slot.nodes,
                trace->origin_us + 1000.0 * offset_ms, 1000.0 * duration_ms, trace->error_us);
        slot.pending = false;
    }
}

static void * ggml_backend_cuda_meta_span_init(ggml_backend_t backend) {
    const char * env = getenv("GGML_META_GPU_SPANS");
    if (!env || atoi(env) != 1 || !ggml_backend_is_cuda(backend)) return nullptr;
    auto * trace = new ggml_cuda_meta_span();
    trace->ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_cuda_set_device(trace->ctx->device);
    CUDA_CHECK(cudaEventCreate(&trace->origin));
    const int64_t before = ggml_time_us();
    CUDA_CHECK(cudaEventRecord(trace->origin, trace->ctx->stream()));
    CUDA_CHECK(cudaEventSynchronize(trace->origin));
    const int64_t after = ggml_time_us();
    trace->origin_us = before + (after - before) / 2;
    trace->error_us = (after - before + 1) / 2;
    for (auto & slot : trace->slots) {
        CUDA_CHECK(cudaEventCreate(&slot.begin));
        CUDA_CHECK(cudaEventCreate(&slot.end));
    }
    return trace;
}

static void * ggml_backend_cuda_meta_span_begin(void * opaque, size_t sequence, size_t nodes) {
    auto * trace = (ggml_cuda_meta_span *) opaque;
    // Diagnostics only: leave the original 1-in-8 sampling as the default.
    // All-frame sampling helps locate pipeline gaps without CUPTI injection.
    static const size_t sample_period = [] {
        const char * value = std::getenv("GGML_META_GPU_SPANS_PERIOD");
        if (!value || !*value) return size_t(8);
        char * end = nullptr;
        const unsigned long parsed = std::strtoul(value, &end, 10);
        return end && *end == '\0' && parsed >= 1 && parsed <= 1024 ? size_t(parsed) : size_t(8);
    }();
    if (!trace || sequence % sample_period != 0) return nullptr;
    ggml_cuda_meta_span_poll(trace, false);
    auto & slot = trace->slots[trace->next++ % 16];
    if (slot.pending) {
        ++trace->dropped;
        return nullptr;
    }
    slot.sequence = sequence;
    slot.nodes = nodes;
    CUDA_CHECK(cudaEventRecord(slot.begin, trace->ctx->stream()));
    return &slot;
}

static void ggml_backend_cuda_meta_span_end(void * opaque, void * token) {
    auto * trace = (ggml_cuda_meta_span *) opaque;
    auto * slot = (ggml_cuda_meta_span_slot *) token;
    if (!trace || !slot) return;
    ggml_cuda_set_device(trace->ctx->device);
    CUDA_CHECK(cudaEventRecord(slot->end, trace->ctx->stream()));
    slot->pending = true;
}

static void ggml_backend_cuda_meta_span_free(void * opaque) {
    auto * trace = (ggml_cuda_meta_span *) opaque;
    if (!trace) return;
    ggml_cuda_meta_span_poll(trace, true);
    for (auto & slot : trace->slots) {
        CUDA_CHECK(cudaEventDestroy(slot.begin));
        CUDA_CHECK(cudaEventDestroy(slot.end));
    }
    CUDA_CHECK(cudaEventDestroy(trace->origin));
    GGML_LOG_INFO("meta-gpu-span device=%d trace=%p dropped=%zu\n", trace->ctx->device, (void *) trace, trace->dropped);
    delete trace;
}

// Host-staged bridge used only by Meta MIRRORED copies between disjoint islands.
// The Meta layer owns one state per source/destination backend pair; this state
// owns portable pinned slots until every destination H2D has completed.
static constexpr size_t GGML_CUDA_META_COPY_MEMBERS = 3;
static constexpr size_t GGML_CUDA_META_COPY_SLOTS   = 2;

struct ggml_backend_cuda_meta_copy_slot {
    void *      host            = nullptr;
    size_t      capacity        = 0;
    cudaEvent_t source_ready    = nullptr;
    cudaEvent_t dst_done[GGML_CUDA_META_COPY_MEMBERS] = {};
    void * gpu_stage = nullptr;
    bool        in_flight       = false;
};

struct ggml_backend_cuda_meta_copy_context {
    int src_devices[GGML_CUDA_META_COPY_MEMBERS] = {};
    int dst_devices[GGML_CUDA_META_COPY_MEMBERS] = {};
    ggml_backend_cuda_meta_copy_slot slots[GGML_CUDA_META_COPY_SLOTS];
    size_t next_slot = 0;
    size_t leader = 0;
    bool peer_fanout = false;
    bool gpu_wait = false;
    size_t gpu_reuses = 0;
    size_t host_reuses = 0;
    std::mutex mutex;
};

static void ggml_backend_cuda_meta_copy_wait_slot(
        ggml_backend_cuda_meta_copy_context * copy_ctx, ggml_backend_cuda_meta_copy_slot & slot) {
    if (!slot.in_flight) {
        return;
    }
    for (size_t i = 0; i < GGML_CUDA_META_COPY_MEMBERS; i++) {
        ggml_cuda_set_device(copy_ctx->dst_devices[i]);
        CUDA_CHECK(cudaEventSynchronize(slot.dst_done[i]));
    }
    slot.in_flight = false;
}

static void ggml_backend_cuda_meta_copy_destroy_slot(
        ggml_backend_cuda_meta_copy_context * copy_ctx, ggml_backend_cuda_meta_copy_slot & slot) {
    ggml_backend_cuda_meta_copy_wait_slot(copy_ctx, slot);
    for (size_t i = 0; i < GGML_CUDA_META_COPY_MEMBERS; i++) {
        if (slot.dst_done[i] != nullptr) {
            ggml_cuda_set_device(copy_ctx->dst_devices[i]);
            CUDA_CHECK(cudaEventDestroy(slot.dst_done[i]));
            slot.dst_done[i] = nullptr;
        }
    }
    if (slot.gpu_stage != nullptr) {
        ggml_cuda_set_device(copy_ctx->dst_devices[copy_ctx->leader]);
        CUDA_CHECK(cudaFree(slot.gpu_stage));
        slot.gpu_stage = nullptr;
        slot.capacity = 0;
    }
    if (slot.source_ready != nullptr) {
        ggml_cuda_set_device(copy_ctx->src_devices[0]);
        CUDA_CHECK(cudaEventDestroy(slot.source_ready));
        slot.source_ready = nullptr;
    }
    if (slot.host != nullptr) {
        CUDA_CHECK(cudaFreeHost(slot.host));
        slot.host = nullptr;
        slot.capacity = 0;
    }
}

static bool ggml_backend_cuda_meta_copy_create_slot(
        ggml_backend_cuda_meta_copy_context * copy_ctx, ggml_backend_cuda_meta_copy_slot & slot) {
    ggml_cuda_set_device(copy_ctx->src_devices[0]);
    cudaError_t err = cudaEventCreateWithFlags(&slot.source_ready, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    for (size_t i = 0; i < GGML_CUDA_META_COPY_MEMBERS; i++) {
        ggml_cuda_set_device(copy_ctx->dst_devices[i]);
        err = cudaEventCreateWithFlags(&slot.dst_done[i], cudaEventDisableTiming);
        if (err != cudaSuccess) {
            (void) cudaGetLastError();
            return false;
        }
    }
    return true;
}

static bool ggml_backend_cuda_meta_copy_peer_route(ggml_backend_cuda_meta_copy_context * ctx) {
    const int source = ggml_cuda_get_physical_device(ctx->src_devices[0]);
    const char * leader_env = getenv("GGML_CUDA_META_COPY_LEADER");
    const int requested = leader_env ? atoi(leader_env) : -1;
    int best_rank = std::numeric_limits<int>::max();
    size_t best = GGML_CUDA_META_COPY_MEMBERS;
    for (size_t lead = 0; lead < GGML_CUDA_META_COPY_MEMBERS; ++lead) {
        if (requested >= 0 && requested < (int) GGML_CUDA_META_COPY_MEMBERS && lead != (size_t) requested) continue;
        const int device = ggml_cuda_get_physical_device(ctx->dst_devices[lead]);
        int supported = 0;
        if (cudaDeviceCanAccessPeer(&supported, source, device) != cudaSuccess || !supported) {
            (void) cudaGetLastError();
            continue;
        }
        bool fanout = true;
        for (size_t i = 0; i < GGML_CUDA_META_COPY_MEMBERS; ++i) {
            if (i == lead) continue;
            const int target = ggml_cuda_get_physical_device(ctx->dst_devices[i]);
            if (cudaDeviceCanAccessPeer(&supported, device, target) != cudaSuccess || !supported) {
                (void) cudaGetLastError();
                fanout = false;
                break;
            }
        }
        if (!fanout) continue;
        int rank = 0;
        if (cudaDeviceGetP2PAttribute(&rank, cudaDevP2PAttrPerformanceRank, source, device) != cudaSuccess) {
            (void) cudaGetLastError();
            rank = std::numeric_limits<int>::max() - 1;
        }
        if (rank < best_rank) {
            best_rank = rank;
            best = lead;
        }
    }
    if (best == GGML_CUDA_META_COPY_MEMBERS) return false;
    auto enable = [](int from, int to) {
        CUDA_CHECK(cudaSetDevice(from));
        const cudaError_t err = cudaDeviceEnablePeerAccess(to, 0);
        if (err == cudaSuccess) return true;
        (void) cudaGetLastError();
        return err == cudaErrorPeerAccessAlreadyEnabled;
    };
    const int leader = ggml_cuda_get_physical_device(ctx->dst_devices[best]);
    if (!enable(source, leader)) return false;
    for (size_t i = 0; i < GGML_CUDA_META_COPY_MEMBERS; ++i) {
        if (i == best) continue;
        const int target = ggml_cuda_get_physical_device(ctx->dst_devices[i]);
        if (!enable(leader, target) || !enable(target, leader)) return false;
    }
    ctx->leader = best;
    return true;
}

static void * ggml_backend_cuda_meta_copy_init(
        ggml_backend_t * src_backends, ggml_backend_t * dst_backends, size_t n_backends) {
    if (n_backends != GGML_CUDA_META_COPY_MEMBERS) {
        return nullptr;
    }

    auto * copy_ctx = new ggml_backend_cuda_meta_copy_context();
    for (size_t i = 0; i < GGML_CUDA_META_COPY_MEMBERS; i++) {
        if (src_backends[i] == nullptr || dst_backends[i] == nullptr ||
                !ggml_backend_is_cuda(src_backends[i]) || !ggml_backend_is_cuda(dst_backends[i])) {
            delete copy_ctx;
            return nullptr;
        }
        copy_ctx->src_devices[i] = ((ggml_backend_cuda_context *) src_backends[i]->context)->device;
        copy_ctx->dst_devices[i] = ((ggml_backend_cuda_context *) dst_backends[i]->context)->device;

        const int src_physical = ggml_cuda_get_physical_device(copy_ctx->src_devices[i]);
        const int dst_physical = ggml_cuda_get_physical_device(copy_ctx->dst_devices[i]);
        for (size_t j = 0; j < i; j++) {
            if (src_physical == ggml_cuda_get_physical_device(copy_ctx->src_devices[j]) ||
                    dst_physical == ggml_cuda_get_physical_device(copy_ctx->dst_devices[j])) {
                delete copy_ctx;
                return nullptr;
            }
        }
    }

    const char * topology_env = getenv("GGML_CUDA_META_COPY_TOPOLOGY");
    const bool topology = topology_env && atoi(topology_env) == 1;
    for (size_t i = 0; i < GGML_CUDA_META_COPY_MEMBERS; i++) {
        const int src_physical = ggml_cuda_get_physical_device(copy_ctx->src_devices[i]);
        for (size_t j = 0; j < GGML_CUDA_META_COPY_MEMBERS; j++) {
            const int dst_physical = ggml_cuda_get_physical_device(copy_ctx->dst_devices[j]);
            if (src_physical == dst_physical) {
                delete copy_ctx;
                return nullptr;
            }
            int can_access_peer = 0;
            const cudaError_t err = cudaDeviceCanAccessPeer(&can_access_peer, src_physical, dst_physical);
            if (err != cudaSuccess) {
                (void) cudaGetLastError();
                delete copy_ctx;
                return nullptr;
            }
            if (can_access_peer && !topology) {
                delete copy_ctx;
                return nullptr;
            }
        }
    }

    copy_ctx->peer_fanout = topology && ggml_backend_cuda_meta_copy_peer_route(copy_ctx);
    const char * gpu_wait_env = getenv("GGML_CUDA_META_COPY_GPU_WAIT");
    copy_ctx->gpu_wait = gpu_wait_env && atoi(gpu_wait_env) == 1;

    for (auto & slot : copy_ctx->slots) {
        if (!ggml_backend_cuda_meta_copy_create_slot(copy_ctx, slot)) {
            for (auto & allocated : copy_ctx->slots) {
                ggml_backend_cuda_meta_copy_destroy_slot(copy_ctx, allocated);
            }
            delete copy_ctx;
            return nullptr;
        }
    }
    if (copy_ctx->peer_fanout) {
        GGML_LOG_INFO("meta copy topology: peer source=%d staging leader=%d parallel local fanout=2\n",
                      copy_ctx->src_devices[0], copy_ctx->dst_devices[copy_ctx->leader]);
    }
    GGML_LOG_INFO("meta copy slot reuse: %s; allocation growth still drains on host\n",
                  copy_ctx->gpu_wait ? "GPU event waits" : "host event waits");
    return copy_ctx;
}

static void ggml_backend_cuda_meta_copy_free(void * opaque) {
    auto * copy_ctx = (ggml_backend_cuda_meta_copy_context *) opaque;
    if (copy_ctx == nullptr) {
        return;
    }
    for (auto & slot : copy_ctx->slots) {
        ggml_backend_cuda_meta_copy_destroy_slot(copy_ctx, slot);
    }
    GGML_LOG_INFO("meta copy reuse totals: gpu=%zu host=%zu\n", copy_ctx->gpu_reuses, copy_ctx->host_reuses);
    delete copy_ctx;
}

static bool ggml_backend_cuda_meta_copy_async(
        void * opaque, ggml_backend_t src_backend, const ggml_tensor * src,
        ggml_backend_t * dst_backends, ggml_tensor ** dst, size_t n_backends) {
    auto * copy_ctx = (ggml_backend_cuda_meta_copy_context *) opaque;
    if (copy_ctx == nullptr || n_backends != GGML_CUDA_META_COPY_MEMBERS ||
            src_backend == nullptr || !ggml_backend_is_cuda(src_backend) ||
            !ggml_is_contiguous(src) || ggml_nbytes(src) == 0) {
        return false;
    }
    const int test_policy = ggml_hardware_test_policy();
    if (copy_ctx->peer_fanout && test_policy >= 0 && test_policy != 1 && test_policy != 3) return false;

    auto * src_cuda_ctx = (ggml_backend_cuda_context *) src_backend->context;
    if (src_cuda_ctx->device != copy_ctx->src_devices[0]) {
        return false;
    }
    const size_t nbytes = ggml_nbytes(src);
    for (size_t i = 0; i < GGML_CUDA_META_COPY_MEMBERS; i++) {
        if (dst_backends[i] == nullptr || dst[i] == nullptr || !ggml_backend_is_cuda(dst_backends[i]) ||
                ((ggml_backend_cuda_context *) dst_backends[i]->context)->device != copy_ctx->dst_devices[i] ||
                !ggml_is_contiguous(dst[i]) || ggml_nbytes(dst[i]) != nbytes || !ggml_are_same_layout(src, dst[i])) {
            return false;
        }
    }

    if (copy_ctx->peer_fanout) {
        const auto source_buf = src->view_src ? src->view_src->buffer : src->buffer;
        if (!source_buf || !ggml_backend_buffer_is_cuda(source_buf) ||
                ((ggml_backend_cuda_buffer_context *) source_buf->context)->device != src_cuda_ctx->device) return false;
        for (size_t i = 0; i < GGML_CUDA_META_COPY_MEMBERS; ++i) {
            const auto target_buf = dst[i]->view_src ? dst[i]->view_src->buffer : dst[i]->buffer;
            if (!target_buf || !ggml_backend_buffer_is_cuda(target_buf) ||
                    ((ggml_backend_cuda_buffer_context *) target_buf->context)->device != copy_ctx->dst_devices[i]) return false;
        }
    }

    std::lock_guard<std::mutex> lock(copy_ctx->mutex);
    ggml_backend_cuda_meta_copy_slot & slot = copy_ctx->slots[copy_ctx->next_slot++ % GGML_CUDA_META_COPY_SLOTS];
    if (slot.in_flight) {
        if (copy_ctx->gpu_wait && slot.capacity >= nbytes) {
            // Order the overwrite after all readers, without blocking the host.
            ggml_cuda_set_device(src_cuda_ctx->device);
            for (size_t i = 0; i < GGML_CUDA_META_COPY_MEMBERS; ++i) {
                CUDA_CHECK(cudaStreamWaitEvent(src_cuda_ctx->stream(), slot.dst_done[i], 0));
            }
            ++copy_ctx->gpu_reuses;
        } else {
            ggml_backend_cuda_meta_copy_wait_slot(copy_ctx, slot);
            ++copy_ctx->host_reuses;
        }
    }

    if (copy_ctx->peer_fanout) {
        constexpr size_t max_stage = 64 * 1024 * 1024;
        if (nbytes > max_stage) return false;
        const int leader_device = copy_ctx->dst_devices[copy_ctx->leader];
        if (slot.capacity < nbytes) {
            ggml_cuda_set_device(leader_device);
            size_t free_bytes = 0, total_bytes = 0;
            CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
            if (free_bytes < nbytes + 512 * 1024 * 1024) return false;
            if (slot.gpu_stage != nullptr) CUDA_CHECK(cudaFree(slot.gpu_stage));
            slot.gpu_stage = nullptr;
            slot.capacity = 0;
            const cudaError_t err = cudaMalloc(&slot.gpu_stage, nbytes);
            if (err != cudaSuccess) {
                (void) cudaGetLastError();
                return false;
            }
            slot.capacity = nbytes;
        }
        // Staging is separate from live inputs, so source work can overlap prior destination work.
        ggml_cuda_set_device(src_cuda_ctx->device);
        CUDA_CHECK(cudaMemcpyPeerAsync(slot.gpu_stage, ggml_cuda_get_physical_device(leader_device),
                src->data, ggml_cuda_get_physical_device(src_cuda_ctx->device), nbytes, src_cuda_ctx->stream()));
        CUDA_CHECK(cudaEventRecord(slot.source_ready, src_cuda_ctx->stream()));
        for (size_t i = 0; i < GGML_CUDA_META_COPY_MEMBERS; ++i) {
            auto * target_ctx = (ggml_backend_cuda_context *) dst_backends[i]->context;
            ggml_cuda_set_device(target_ctx->device);
            CUDA_CHECK(cudaStreamWaitEvent(target_ctx->stream(), slot.source_ready, 0));
            if (i == copy_ctx->leader) {
                CUDA_CHECK(cudaMemcpyAsync(dst[i]->data, slot.gpu_stage, nbytes,
                        cudaMemcpyDeviceToDevice, target_ctx->stream()));
            } else {
                CUDA_CHECK(cudaMemcpyPeerAsync(dst[i]->data, ggml_cuda_get_physical_device(target_ctx->device),
                        slot.gpu_stage, ggml_cuda_get_physical_device(leader_device), nbytes, target_ctx->stream()));
            }
            CUDA_CHECK(cudaEventRecord(slot.dst_done[i], target_ctx->stream()));
        }
        slot.in_flight = true;
        return true;
    }

    if (slot.capacity < nbytes) {
        if (slot.host != nullptr) {
            CUDA_CHECK(cudaFreeHost(slot.host));
            slot.host = nullptr;
            slot.capacity = 0;
        }
        const cudaError_t err = cudaHostAlloc(&slot.host, nbytes, cudaHostAllocPortable);
        if (err != cudaSuccess) {
            (void) cudaGetLastError();
            return false;
        }
        slot.capacity = nbytes;
    }

    // Read one canonical replica, then fan out to all members of the receiving
    // island. The source event orders D2H before each independent destination H2D.
    ggml_cuda_set_device(src_cuda_ctx->device);
    ggml_backend_tensor_get_async(src_backend, src, slot.host, 0, nbytes);
    CUDA_CHECK(cudaEventRecord(slot.source_ready, src_cuda_ctx->stream()));

    for (size_t i = 0; i < GGML_CUDA_META_COPY_MEMBERS; i++) {
        auto * dst_cuda_ctx = (ggml_backend_cuda_context *) dst_backends[i]->context;
        ggml_cuda_set_device(dst_cuda_ctx->device);
        CUDA_CHECK(cudaStreamWaitEvent(dst_cuda_ctx->stream(), slot.source_ready, 0));
        ggml_backend_tensor_set_async(dst_backends[i], dst[i], slot.host, 0, nbytes);
        CUDA_CHECK(cudaEventRecord(slot.dst_done[i], dst_cuda_ctx->stream()));
    }
    slot.in_flight = true;
    return true;
}

static bool ggml_backend_cuda_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    if (!ggml_backend_is_cuda(backend_src) || !ggml_backend_is_cuda(backend_dst)) {
        return false;
    }

    if (!ggml_backend_buffer_is_cuda(buf_src) || !ggml_backend_buffer_is_cuda(buf_dst)) {
        return false;
    }

    // device -> device copy
    ggml_backend_cuda_context * cuda_ctx_src = (ggml_backend_cuda_context *) backend_src->context;
    ggml_backend_cuda_context * cuda_ctx_dst = (ggml_backend_cuda_context *) backend_dst->context;

    ggml_backend_cuda_buffer_context * buf_ctx_src = (ggml_backend_cuda_buffer_context *) buf_src->context;
    ggml_backend_cuda_buffer_context * buf_ctx_dst = (ggml_backend_cuda_buffer_context *) buf_dst->context;

    if (cuda_ctx_src->device != buf_ctx_src->device || cuda_ctx_dst->device != buf_ctx_dst->device) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: backend and buffer devices do not match\n", __func__);
#endif // NDEBUG
        return false;
    }

    if (backend_src != backend_dst) {
        // copy on src stream
        // compare the backing physical devices: distinct virtual devices may share one physical GPU,
        // in which case a same-device copy (not a peer copy) is required
        const int src_physical = ggml_cuda_get_physical_device(cuda_ctx_src->device);
        const int dst_physical = ggml_cuda_get_physical_device(cuda_ctx_dst->device);
        if (src_physical == dst_physical) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream()));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, dst_physical, src->data, src_physical, ggml_nbytes(dst), cuda_ctx_src->stream()));
#endif // GGML_CUDA_NO_PEER_COPY
        }

        // record event on src stream after the copy
        if (!cuda_ctx_src->copy_event) {
            ggml_cuda_set_device(cuda_ctx_src->device);
            CUDA_CHECK(cudaEventCreateWithFlags(&cuda_ctx_src->copy_event, cudaEventDisableTiming));
        }

        CUDA_CHECK(cudaEventRecord(cuda_ctx_src->copy_event, cuda_ctx_src->stream()));

        // wait on dst stream for the copy to complete
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx_dst->stream(), cuda_ctx_src->copy_event, 0));
    } else {
        // src and dst are on the same backend
        CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream()));
    }
    return true;
}

static void ggml_backend_cuda_synchronize(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    const int64_t start = ggml_cuda_ft_host_timing_enabled() ? ggml_time_us() : 0;
    CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    ggml_cuda_layer_pinned_upload_private::synchronize(cuda_ctx);
#endif
    ggml_cuda_ft_host_record(cuda_ctx, 4, start);

    GGML_UNUSED(backend);
}

static bool ggml_cuda_is_view_or_noop(const ggml_tensor * t) {
    return ggml_is_empty(t) || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE ||
           t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_NONE;
}

#ifdef USE_CUDA_GRAPH
static bool ggml_cuda_graph_check_compability(const ggml_cgraph * cgraph, ggml_tensor ** incompatible = nullptr) {

    bool use_cuda_graph = true;
    // Loop over nodes in GGML graph to obtain info needed for CUDA graph

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        if (ggml_cuda_is_view_or_noop(node)) {
            continue;
        }

        // [TAG_MUL_MAT_ID_CUDA_GRAPHS]
        if (node->op == GGML_OP_MUL_MAT_ID) {
            const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
            if (ggml_cuda_mul_mat_id_needs_sync(node, cc)) {
                // the mul_mat_id fallback path synchronizes the stream, so we cannot use CUDA graphs
                // ref: https://github.com/ggml-org/llama.cpp/pull/18958
                use_cuda_graph = false;
                if (incompatible) *incompatible = node;
#ifndef NDEBUG
                GGML_LOG_DEBUG("%s: disabling CUDA graphs due to unsupported node type\n", __func__);
#endif
            }
        }

        if (!use_cuda_graph) {
            break;
        }
    }

    return use_cuda_graph;
}

static bool ggml_cuda_graph_structural_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_CUDA_GRAPH_STRUCTURAL");
        return value && std::strcmp(value, "1") == 0;
    }();
    static const bool private_policy = [] {
        const char * value = std::getenv("GGML_CUDA_GRAPH_STRUCTURAL_TEST_POLICY");
        return value && std::strcmp(value, "1") == 0;
    }();
    if (enabled && private_policy) {
        const int policy = ggml_hardware_test_policy();
        if (policy >= 0) return policy == 3 || policy == 4;
    }
    return enabled;
}

static ggml_cuda_graph_structural::tensor_input ggml_cuda_graph_tensor_input(const ggml_tensor * tensor) {
    using namespace ggml_cuda_graph_structural;
    tensor_input result;
    result.host_identity = reinterpret_cast<uintptr_t>(tensor);
    result.descriptor.type = tensor->type;
    result.descriptor.buffer = reinterpret_cast<uintptr_t>(tensor->buffer);
    std::memcpy(result.descriptor.ne.data(), tensor->ne, sizeof(tensor->ne));
    std::memcpy(result.descriptor.nb.data(), tensor->nb, sizeof(tensor->nb));
    result.descriptor.op = tensor->op;
    std::memcpy(result.descriptor.op_params.data(), tensor->op_params, sizeof(tensor->op_params));
    result.descriptor.flags = tensor->flags;
    result.descriptor.view_offs = tensor->view_offs;
    result.descriptor.data = reinterpret_cast<uintptr_t>(tensor->data);
    std::memcpy(result.descriptor.name.data(), tensor->name, sizeof(tensor->name));
    result.descriptor.extra = reinterpret_cast<uintptr_t>(tensor->extra);
    std::memcpy(result.descriptor.padding.data(), tensor->padding, sizeof(tensor->padding));
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        result.sources[i] = reinterpret_cast<uintptr_t>(tensor->src[i]);
    }
    result.view_source = reinterpret_cast<uintptr_t>(tensor->view_src);

    if (tensor->extra != nullptr) {
        // A buffer's backend does not define the type of an arbitrary tensor
        // extra. In particular, the ordinary CUDA buffer does not construct a
        // ggml_tensor_extra_gpu; materialized external producers may retain an
        // opaque extra from a different descriptor generation. Never dereference
        // such a value while inspecting a graph. Decline structural reuse and
        // leave execution on the existing UID/capture path instead.
        result.extra_state_valid = false;
    }
    return result;
}

static size_t ggml_cuda_graph_structural_max_nodes() {
    static const size_t max_nodes = [] {
        const char * value = std::getenv("GGML_CUDA_GRAPH_STRUCTURAL_MAX_NODES");
        if (value != nullptr) {
            const long long parsed = std::atoll(value);
            if (parsed > 0) {
                return size_t(parsed);
            }
        }
        return size_t(2048);
    }();
    return max_nodes;
}

static bool ggml_cuda_graph_make_structural_signature(
        ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph,
        ggml_cuda_graph_structural::graph_signature * signature,
        ggml_cuda_graph_structural_probe::flat_snapshot * snapshot) {
    using namespace ggml_cuda_graph_structural;
    if (cuda_ctx == nullptr) {
        return false;
    }
    ggml_cuda_graph_structural_stats_update(cuda_ctx, [](auto & stats) { ++stats.attempts; });
    const int64_t start = ggml_time_us();
    static const bool trace = [] {
        const char * env = std::getenv("GGML_CUDA_GRAPH_STRUCTURAL_TRACE");
        return env && std::strcmp(env, "1") == 0;
    }();
    const auto reject = [&](build_failure failure, size_t external_count = 0) {
        if (trace) {
            static std::mutex trace_mutex;
            static std::map<std::pair<const void *, build_failure>, size_t> printed;
            std::lock_guard<std::mutex> lock(trace_mutex);
            if (printed[{cuda_ctx, failure}]++ < 2) {
                fprintf(stderr, "CUDA graph structural rejected graph device=%d reason=%s nodes=%d external=%zu uid=%" PRIu64 "\n",
                        cuda_ctx->device, failure_name(failure), cgraph ? cgraph->n_nodes : 0,
                        external_count, cgraph ? uint64_t(cgraph->uid) : uint64_t(0));
            }
        }
        const int64_t elapsed = ggml_time_us() - start;
        ggml_cuda_graph_structural_stats_update(cuda_ctx, [failure, elapsed](auto & stats) {
            ++stats.failures[size_t(failure)];
            if (failure == build_failure::missing_external_binding) ++stats.reject_missing;
            else if (failure == build_failure::unsupported_external_binding) ++stats.reject_external;
            else ++stats.reject_graph;
            stats.signature_build_us += elapsed;
        });
        return false;
    };
    if (cgraph == nullptr || cgraph->n_nodes <= 0 || signature == nullptr) {
        return reject(build_failure::empty_graph);
    }
    if (size_t(cgraph->n_nodes) > ggml_cuda_graph_structural_max_nodes()) {
        const int64_t elapsed = ggml_time_us() - start;
        ggml_cuda_graph_structural_stats_update(cuda_ctx, [elapsed](auto & stats) {
            ++stats.reject_too_large;
            stats.signature_build_us += elapsed;
        });
        return false;
    }

    static const bool binding_boundaries = [] {
        const char * env = std::getenv("GGML_CUDA_GRAPH_STRUCTURAL_BOUNDARIES");
        return env && std::strcmp(env, "1") == 0;
    }();
    std::unordered_set<const ggml_tensor *> graph_tensors;
    std::unordered_set<uintptr_t> graph_identities;
    graph_tensors.reserve(cgraph->n_nodes);
    if (binding_boundaries) graph_identities.reserve(cgraph->n_nodes);
    std::vector<tensor_input> nodes;
    nodes.reserve(cgraph->n_nodes);
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (node == nullptr || !graph_tensors.insert(node).second) {
            return reject(build_failure::duplicate_or_invalid_node);
        }
        nodes.push_back(ggml_cuda_graph_tensor_input(node));
        if (binding_boundaries) graph_identities.insert(reinterpret_cast<uintptr_t>(node));
        if (!nodes.back().extra_state_valid) {
            if (trace) {
                static std::mutex opaque_mutex;
                static std::map<const void *, size_t> opaque_printed;
                std::lock_guard<std::mutex> lock(opaque_mutex);
                if (opaque_printed[cuda_ctx]++ < 2) {
                    GGML_LOG_INFO("CUDA graph structural opaque node device=%d name=%.*s op=%d\n",
                            cuda_ctx->device, GGML_MAX_NAME, node->name, int(node->op));
                }
            }
            return reject(build_failure::unsupported_node_extra);
        }
    }

    std::vector<tensor_input> external;
    std::unordered_set<const ggml_tensor *> seen_external;
    bool closure_too_large = false;
    const auto add_external = [&](const ggml_tensor * tensor) {
        if (tensor != nullptr && graph_tensors.count(tensor) == 0 && seen_external.insert(tensor).second) {
            if (external.size() == max_external_bindings) {
                closure_too_large = true;
                return;
            }
            auto input = ggml_cuda_graph_tensor_input(tensor);
            if (trace && !input.extra_state_valid) {
                static std::mutex opaque_external_mutex;
                static std::map<const void *, size_t> opaque_external_printed;
                std::lock_guard<std::mutex> lock(opaque_external_mutex);
                if (opaque_external_printed[cuda_ctx]++ < 4) {
                    GGML_LOG_INFO("CUDA graph structural opaque external device=%d name=%.*s op=%d extra=%p buffer=%p uid=%" PRIu64 "\n",
                            cuda_ctx->device, GGML_MAX_NAME, tensor->name, int(tensor->op),
                            tensor->extra, (void *) tensor->buffer, uint64_t(cgraph->uid));
                }
            }
            if (binding_boundaries) input = project_external_binding(std::move(input), graph_identities);
            external.push_back(std::move(input));
        }
    };
    for (int n = 0; n < cgraph->n_nodes; ++n) {
        const ggml_tensor * node = cgraph->nodes[n];
        for (int i = 0; i < GGML_MAX_SRC; ++i) add_external(node->src[i]);
        add_external(node->view_src);
        if (closure_too_large) return reject(build_failure::too_many_external_bindings, external.size());
    }
    // Close the external dependency/view DAG in deterministic first-use order.
    for (size_t e = 0; e < external.size(); ++e) {
        // Copy only scalar edges before vector growth can invalidate references.
        const auto sources = external[e].sources;
        const auto view_source = external[e].view_source;
        for (const auto source : sources) add_external(reinterpret_cast<const ggml_tensor *>(source));
        add_external(reinterpret_cast<const ggml_tensor *>(view_source));
        if (closure_too_large) return reject(build_failure::too_many_external_bindings, external.size());
    }
    build_failure failure = build_failure::none;
    cycle_edge cycle;
    if (!build_signature(nodes, external, signature, &failure, &cycle)) {
        if (trace && cycle.from && cycle.to) {
            static std::mutex cycle_mutex;
            static std::map<const void *, size_t> cycle_printed;
            std::lock_guard<std::mutex> lock(cycle_mutex);
            if (cycle_printed[cuda_ctx]++ < 2) {
                const auto * from = reinterpret_cast<const ggml_tensor *>(cycle.from);
                const auto * to = reinterpret_cast<const ggml_tensor *>(cycle.to);
                fprintf(stderr, "CUDA graph structural cycle edge device=%d from=%.*s/%s/graph%d to=%.*s/%s/graph%d\n",
                        cuda_ctx->device, GGML_MAX_NAME, from->name, ggml_op_name(from->op), int(graph_tensors.count(from)),
                        GGML_MAX_NAME, to->name, ggml_op_name(to->op), int(graph_tensors.count(to)));
            }
        }
        if (trace && failure == build_failure::unsupported_external_binding) {
            static std::mutex trace_mutex;
            static std::map<const void *, size_t> printed;
            std::lock_guard<std::mutex> lock(trace_mutex);
            for (const auto & input : external) {
                if (!is_external_bound(input) && printed[cuda_ctx] < 4) {
                    ++printed[cuda_ctx];
                    fprintf(stderr, "CUDA graph structural rejected binding device=%d name=%.*s op=%d buffer=%d data=%d extra_valid=%d view=%d ne=%lld,%lld,%lld,%lld\n",
                            cuda_ctx->device, GGML_MAX_NAME, input.descriptor.name.data(), input.descriptor.op,
                            input.descriptor.buffer != 0, input.descriptor.data != 0, input.extra_state_valid,
                            input.view_source != 0, (long long) input.descriptor.ne[0], (long long) input.descriptor.ne[1],
                            (long long) input.descriptor.ne[2], (long long) input.descriptor.ne[3]);
                }
            }
        }
        return reject(failure, external.size());
    }
    const int64_t elapsed = ggml_time_us() - start;
    ggml_cuda_graph_structural_stats_update(cuda_ctx, [elapsed](auto & stats) {
        ++stats.builds;
        stats.signature_build_us += elapsed;
    });
    if (snapshot != nullptr) {
        snapshot->raw_key = reinterpret_cast<uintptr_t>(cgraph->nodes[0]);
        snapshot->uid = cgraph->uid;
        snapshot->graph_identities.reserve(nodes.size());
        for (const auto & node : nodes) snapshot->graph_identities.push_back(node.host_identity);
        snapshot->graph_inputs = std::move(nodes);
        snapshot->external_inputs = std::move(external);
        snapshot->signature = *signature;
    }
    return true;
}

static const void * ggml_cuda_graph_get_key(
        ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph,
        ggml_cuda_graph_structural::graph_signature * signature, bool * signature_valid,
        bool allow_new_binding_key) {
    const void * const raw_key = (cgraph != nullptr && cgraph->n_nodes > 0) ? cgraph->nodes[0] : nullptr;
    if (signature_valid != nullptr) {
        *signature_valid = false;
    }
    if (cuda_ctx != nullptr) {
        // Readiness may refresh the normal sweep clock only while the independent
        // actual-use lease is alive; it never updates that lease's expiry time.
        ggml_cuda_graph_lease_refresh_context(cuda_ctx);
    }
    if (!ggml_cuda_graph_structural_enabled() || raw_key == nullptr || cuda_ctx == nullptr) {
        return raw_key;
    }

    // Preserve the existing UID fast path for the same live graph object. A
    // different first-node host address must pass through full canonical
    // comparison even if its UID happens to match.
    ggml_cuda_graph_structural_prune(cuda_ctx);
    const auto raw = cuda_ctx->cuda_graphs.find(raw_key);
    if (cgraph->uid != 0 && raw != cuda_ctx->cuda_graphs.end() && raw->second->uid == cgraph->uid) {
        return raw_key;
    }

    using namespace ggml_cuda_graph_structural_probe;
    static const bool snapshot_enabled = [] {
        const char * value = std::getenv("GGML_CUDA_GRAPH_SNAPSHOT_CACHE");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    std::vector<uintptr_t> graph_ids;
    if (snapshot_enabled) {
        graph_ids.reserve(cgraph->n_nodes);
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            graph_ids.push_back(reinterpret_cast<uintptr_t>(cgraph->nodes[i]));
        }
    }
    auto cached_snapshot = snapshot_enabled ? context_snapshot_cache_get().lookup(
            cuda_ctx, reinterpret_cast<uintptr_t>(raw_key), cgraph->uid) :
            std::shared_ptr<const flat_snapshot>{};
    const auto reader = [](uintptr_t identity) {
        // Only current cgraph nodes or descendants from current validated parent
        // edges reach this callback; no cached pointer is dereferenced.
        return ggml_cuda_graph_tensor_input(reinterpret_cast<const ggml_tensor *>(identity));
    };
    bool cache_hit = cached_snapshot && validate_flat_snapshot(
            *cached_snapshot, reinterpret_cast<uintptr_t>(raw_key), cgraph->uid, graph_ids, reader);
    static const bool snapshot_cross_uid = [] {
        const char * value = std::getenv("GGML_CUDA_GRAPH_SNAPSHOT_CROSS_UID");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    if (!cache_hit && snapshot_enabled && snapshot_cross_uid && cgraph->uid != 0) {
        // UID changes on scheduler allocation. Search bounded memo candidates,
        // then read every live descriptor/parent-derived edge before reusing
        // the signature. CUDA graph selection and readiness below are unchanged.
        for (const auto & candidate_snapshot : context_snapshot_cache_get().lookup_candidates(
                cuda_ctx, reinterpret_cast<uintptr_t>(raw_key), 8)) {
            if (candidate_snapshot == cached_snapshot) continue;
            if (!validate_flat_snapshot(*candidate_snapshot, reinterpret_cast<uintptr_t>(raw_key),
                    cgraph->uid, graph_ids, reader, true)) continue;
            cached_snapshot = candidate_snapshot;
            cache_hit = true;
            static thread_local unsigned cross_uid_logs = 0;
            if (cross_uid_logs++ < 8) {
                fprintf(stderr, "graph-snapshot-cross-uid: device=%d old_uid=%llu live_uid=%llu "
                        "live_descriptor_closure_exact=1 replay_authorized=0\n", cuda_ctx->device,
                        (unsigned long long) cached_snapshot->uid, (unsigned long long) cgraph->uid);
            }
            break;
        }
    }

    ggml_cuda_graph_structural::graph_signature candidate;
    if (cache_hit) {
        candidate = cached_snapshot->signature;
        ggml_cuda_graph_structural_stats_update(cuda_ctx, [](auto & stats) { ++stats.flat_snapshot_hits; });
    } else {
        flat_snapshot fresh_snapshot;
        if (!ggml_cuda_graph_make_structural_signature(cuda_ctx, cgraph, &candidate, snapshot_enabled ? &fresh_snapshot : nullptr)) {
            return raw_key;
        }
        if (snapshot_enabled && cgraph->uid != 0) {
            context_snapshot_cache_get().publish(cuda_ctx, std::move(fresh_snapshot));
        }
        if (snapshot_enabled) ggml_cuda_graph_structural_stats_update(cuda_ctx, [](auto & stats) { ++stats.flat_snapshot_misses; });
    }

    const int64_t lookup_start = ggml_time_us();
    const auto finish = [&](const void * key) {
        const int64_t elapsed = ggml_time_us() - lookup_start;
        ggml_cuda_graph_structural_stats_update(cuda_ctx, [elapsed](auto & stats) { stats.lookup_us += elapsed; });
        if (signature != nullptr) {
            *signature = std::move(candidate);
        }
        if (signature_valid != nullptr) {
            *signature_valid = true;
        }
        return key;
    };

    // Lookup does not extend graph lifetime; the actual cuda_graph() call
    // refreshes last_used.
    if (raw != cuda_ctx->cuda_graphs.end() && ggml_cuda_graph_structural_compare(
            cuda_ctx, raw->second.get(), candidate) == ggml_cuda_graph_structural_match::equal) {
        ggml_cuda_graph_structural_stats_update(cuda_ctx, [](auto & stats) { ++stats.active_matches; });
        return finish(raw_key);
    }
    for (const auto & [key, graph] : cuda_ctx->cuda_graphs) {
        if (key != raw_key && ggml_cuda_graph_structural_compare(
                cuda_ctx, graph.get(), candidate) == ggml_cuda_graph_structural_match::equal) {
            ggml_cuda_graph_structural_stats_update(cuda_ctx, [](auto & stats) { ++stats.active_matches; });
            return finish(key);
        }
    }

    // Previously captured variants have no active graph under their own key;
    // selection below will swap a matching retained instance into that live
    // active slot. Require the slot to exist so an expired key cannot be
    // resurrected outside the existing map lifetime.
    if (ggml_cuda_ft_graph_cache_enabled()) {
        auto & variants = ggml_cuda_ft_graph_variants_get();
        std::lock_guard<std::mutex> lock(variants.mutex);
        const auto context = variants.contexts.find(cuda_ctx);
        if (context != variants.contexts.end()) {
            for (const auto & entry : context->second.entries) {
                if (cuda_ctx->cuda_graphs.count(entry.key) != 0 && ggml_cuda_graph_structural_compare(
                        cuda_ctx, entry.graph.get(), candidate) == ggml_cuda_graph_structural_match::equal) {
                    ggml_cuda_graph_structural_stats_update(cuda_ctx, [](auto & stats) { ++stats.variant_matches; });
                    return finish(entry.key);
                }
            }
        }
    }
    static const bool layout_diagnostic = [] {
        const char * value = std::getenv("GGML_CUDA_GRAPH_UPDATE_LAYOUT_TRACE");
        return value && std::strcmp(value, "1") == 0;
    }();
    if (layout_diagnostic) {
        bool possible = false;
        for (const auto & [key, graph] : cuda_ctx->cuda_graphs) {
            if (ggml_cuda_graph_structural_has_update_layout(cuda_ctx, graph.get(), candidate)) {
                possible = true;
                break;
            }
        }
        ggml_cuda_graph_structural_stats_update(cuda_ctx, [possible](auto & stats) {
            if (possible) ++stats.update_layout_candidates;
            else ++stats.update_layout_misses;
        });
    }
    ggml_cuda_graph_log_signature_miss(cuda_ctx, raw_key, candidate);
    ggml_cuda_graph_structural_stats_update(cuda_ctx, [](auto & stats) { ++stats.lookup_misses; });
    const bool binding_keys_requested = ggml_cuda_graph_binding_keys_enabled();
    static const size_t binding_key_cap = [] {
        const char * text = std::getenv("GGML_CUDA_GRAPH_BINDING_KEYS_CAP");
        if (!text || !*text) return size_t(32);
        char * end = nullptr;
        const long value = std::strtol(text, &end, 10);
        return end != text && *end == '\0' && value >= 1 && value <= 64 ? size_t(value) : size_t(32);
    }();
    if (binding_keys_requested && allow_new_binding_key && ggml_cuda_graph_check_compability(cgraph)) {
        std::unordered_set<const void *> live_keys;
        for (const auto & entry : cuda_ctx->cuda_graphs) live_keys.insert(entry.first);
        // Keep the established variant-cache -> registry lock order. A key
        // retained only by an FT variant is not available for recycling.
        auto & variants = ggml_cuda_ft_graph_variants_get();
        std::unique_lock<std::mutex> variants_lock(variants.mutex, std::defer_lock);
        if (ggml_cuda_ft_graph_cache_enabled()) {
            variants_lock.lock();
            const auto found = variants.contexts.find(cuda_ctx);
            if (found != variants.contexts.end()) {
                for (const auto & entry : found->second.entries) live_keys.insert(entry.key);
            }
        }
        const void * binding_key = nullptr;
        bool reused = false;
        size_t pool_size = 0;
        size_t cap_fallbacks = 0;
        {
            auto & registry = ggml_cuda_graph_structural_registry_get();
            std::lock_guard<std::mutex> lock(registry.mutex);
            auto & keys = registry.binding_keys[cuda_ctx];
            auto & stats = registry.contexts[cuda_ctx];
            binding_key = ggml_cuda_graph_acquire_binding_key(keys, live_keys, binding_key_cap, reused);
            pool_size = keys.size();
            if (binding_key) {
                if (reused) ++stats.binding_keys_reused;
                else ++stats.binding_keys_created;
            } else {
                cap_fallbacks = ++stats.binding_key_cap_fallbacks;
            }
        }
        if (variants_lock.owns_lock()) variants_lock.unlock();
        if (binding_key) {
            if (reused || pool_size <= 2 || (pool_size & (pool_size - 1)) == 0) {
                GGML_LOG_INFO("CUDA graph binding key requested=1 selected=1 device=%d raw_key=%p key=%p"
                        " pool_size=%zu cap=%zu reused_retired_key=%d exact_binding_check_retained=1 compute_only=1\n",
                        cuda_ctx->device, raw_key, binding_key, pool_size, binding_key_cap, int(reused));
            }
            return finish(binding_key);
        }
        if (cap_fallbacks <= 2 || cap_fallbacks % 256 == 0) {
            GGML_LOG_INFO("CUDA graph binding key requested=1 selected=0 device=%d reason=context_key_cap"
                    " cap=%zu original_raw_key_retained=1 fallbacks=%zu\n",
                    cuda_ctx->device, binding_key_cap, cap_fallbacks);
        }
    }
    return finish(raw_key);
}

static bool ggml_cuda_ft_graph_matches_raw(const ggml_cuda_graph & graph, const ggml_cgraph * cgraph) {
    if (graph.node_props.size() != size_t(cgraph->n_nodes)) { return false; }
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        ggml_cuda_graph::node_properties prop = {};
        memcpy(&prop.node, cgraph->nodes[i], sizeof(ggml_tensor));
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            if (cgraph->nodes[i]->src[j]) {
                prop.node_src_data_ptrs[j] = cgraph->nodes[i]->src[j]->data;
                memcpy(prop.node_src_ne[j], cgraph->nodes[i]->src[j]->ne, sizeof(prop.node_src_ne[j]));
                memcpy(prop.node_src_nb[j], cgraph->nodes[i]->src[j]->nb, sizeof(prop.node_src_nb[j]));
            }
        }
        if (memcmp(&graph.node_props[i], &prop, sizeof(prop)) != 0) { return false; }
    }
    return true;
}

static bool ggml_cuda_ft_graph_matches(const ggml_backend_cuda_context * ctx,
        const ggml_cuda_graph & graph, const ggml_cgraph * cgraph,
        const ggml_cuda_graph_structural::graph_signature * signature, bool signature_valid) {
    if (signature_valid) {
        const auto match = ggml_cuda_graph_structural_compare(ctx, &graph, *signature);
        if (match != ggml_cuda_graph_structural_match::unknown) {
            return match == ggml_cuda_graph_structural_match::equal;
        }
    }
    // Keep the original exact-descriptor matcher as the fallback for graphs
    // outside the deliberately strict structural subset.
    return ggml_cuda_ft_graph_matches_raw(graph, cgraph);
}

static bool ggml_cuda_ft_graph_cached_ready(
        ggml_backend_cuda_context * cuda_ctx,
        const ggml_cuda_graph_structural::graph_signature * signature, bool signature_valid) {
    if (!ggml_cuda_graph_structural_enabled() || !ggml_cuda_ft_graph_cache_enabled() || !signature_valid) {
        return false;
    }

    auto & variants = ggml_cuda_ft_graph_variants_get();
    std::lock_guard<std::mutex> lock(variants.mutex);
    const auto context = variants.contexts.find(cuda_ctx);
    if (context == variants.contexts.end()) {
        return false;
    }
    const int64_t now = ggml_time_us();
    for (const auto & entry : context->second.entries) {
        const auto active = cuda_ctx->cuda_graphs.find(entry.key);
        const ggml_cuda_graph & graph = *entry.graph;
        if (active != cuda_ctx->cuda_graphs.end() && graph.is_enabled() &&
            ggml_cuda_graph_has_executable(cuda_ctx, &graph) && graph.warmup_complete &&
            ggml_cuda_graph_structural_compare(cuda_ctx, &graph, *signature) ==
                ggml_cuda_graph_structural_match::equal && now - graph.last_used_time < 5'000'000) {
            return true;
        }
    }
    return false;
}

static void ggml_cuda_ft_graph_select(ggml_backend_cuda_context * ctx, ggml_cgraph * cgraph, const void * key,
        const ggml_cuda_graph_structural::graph_signature * signature, bool signature_valid) {
    // GGML_FT_GRAPH_CACHE_MIN_NODES: smallest graph kept as a variant. Small graphs that alternate
    // shapes (an MTP draft head's catch-up and draft passes) otherwise lose their capture each switch.
    static const int min_nodes = [] {
        const char * value = std::getenv("GGML_FT_GRAPH_CACHE_MIN_NODES");
        return value ? std::atoi(value) : 1000;
    }();
    if (!ggml_cuda_ft_graph_cache_enabled() || cgraph->n_nodes <= min_nodes) { return; }
    auto & active = ctx->cuda_graphs.at(key);
    const auto active_match = signature_valid ?
        ggml_cuda_graph_structural_compare(ctx, active.get(), *signature) : ggml_cuda_graph_structural_match::unknown;
    const bool structural_candidate = active_match == ggml_cuda_graph_structural_match::equal;
    const bool uid_matches = cgraph->uid != 0 && active->uid == cgraph->uid;
    if (!active->is_enabled() || (uid_matches && (!signature_valid || structural_candidate)) ||
        structural_candidate || ggml_cuda_ft_graph_matches(ctx, *active, cgraph, signature, signature_valid)) { return; }
    auto & variants = ggml_cuda_ft_graph_variants_get();
    std::lock_guard<std::mutex> lock(variants.mutex);
    auto & c = variants.contexts[ctx];
    for (auto it = c.entries.begin(); it != c.entries.end(); ++it) {
        if (it->key == key && ggml_cuda_ft_graph_matches(ctx, *it->graph, cgraph, signature, signature_valid)) {
            std::swap(active, it->graph);
            active->uid = cgraph->uid;
            active->last_used_time = ggml_time_us();
            if (!ggml_cuda_graph_has_executable(ctx, it->graph.get()) ||
                    !it->graph->warmup_complete) { c.entries.erase(it); }
            ++c.hits;
            return;
        }
    }
    if (ggml_cuda_graph_has_executable(ctx, active.get()) && active->warmup_complete) {
        static const size_t cap = [] { const char * e = getenv("GGML_CUDA_GRAPH_VARIANTS_CAP");
            return (e && atoll(e) > 0) ? (size_t) atoll(e) : (size_t) 4; }();
        if (c.entries.size() >= cap) {
            const auto oldest = std::min_element(c.entries.begin(), c.entries.end(), [](const auto & a, const auto & b) {
                return a.graph->last_used_time < b.graph->last_used_time;
            });
            c.entries.erase(oldest);
            ++c.evictions;
        }
        auto fresh = std::make_unique<ggml_cuda_graph>();
        fresh->disable_due_to_gpu_arch = active->disable_due_to_gpu_arch;
        fresh->last_used_time = ggml_time_us();
        c.entries.push_back({key, std::move(active)});
        active = std::move(fresh);
        ++c.saves;
    }
}

static bool ggml_cuda_graph_update_required(
        ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, const void * graph_key,
        ggml_cuda_graph_structural::graph_signature * structural_candidate,
        bool structural_candidate_valid) {
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

    const auto structural_match = structural_candidate_valid ?
        ggml_cuda_graph_structural_compare(cuda_ctx, graph, *structural_candidate) :
        ggml_cuda_graph_structural_match::unknown;
    if (structural_match == ggml_cuda_graph_structural_match::equal) {
        // The captured launch graph contains identical ordered operations and
        // exact same CUDA bindings. The new host descriptors therefore need no
        // update or recapture; only the graph UID follows the current cgraph.
        GGML_ASSERT(graph->node_props.size() == size_t(cgraph->n_nodes));
        graph->uid = cgraph->uid;
        return false;
    }

    // Preserve the old UID fast path only when the structural path is outside
    // its supported subset. An eligible but changed signature must not bypass
    // the exact property comparison merely because a UID was reused.
    if (!structural_candidate_valid && cgraph->uid != 0 && cgraph->uid == graph->uid) {
        GGML_LOG_DEBUG("CUDA Graph id %zu reused\n", cgraph->uid);
        GGML_ASSERT((int)graph->node_props.size() == cgraph->n_nodes);
        return false;
    }

    graph->uid = cgraph->uid;

    // A changed structural signature is itself a property change, including
    // extra/event contents which the legacy raw node snapshot cannot observe.
    bool res = structural_candidate_valid && structural_match != ggml_cuda_graph_structural_match::equal;

    if ((int)graph->node_props.size() != cgraph->n_nodes) {
        res = true;
        graph->node_props.resize(cgraph->n_nodes);
    }

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_cuda_graph::node_properties prop = {};
        memcpy(&prop.node, cgraph->nodes[i], sizeof(ggml_tensor));

        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            if (cgraph->nodes[i]->src[j]) {
                prop.node_src_data_ptrs[j] = cgraph->nodes[i]->src[j]->data;
                memcpy(prop.node_src_ne[j], cgraph->nodes[i]->src[j]->ne, sizeof(prop.node_src_ne[j]));
                memcpy(prop.node_src_nb[j], cgraph->nodes[i]->src[j]->nb, sizeof(prop.node_src_nb[j]));
            }
        }

        if (res || memcmp(&graph->node_props[i], &prop, sizeof(prop)) != 0) {
            graph->node_props[i] = prop;
            res = true;
        }
    }

    if (structural_candidate_valid) {
        ggml_cuda_graph_structural_store(cuda_ctx, graph, std::move(*structural_candidate));
    } else {
        ggml_cuda_graph_structural_erase(cuda_ctx, graph);
    }
    return res;
}

// V251: thread-local request scoped by the actual llama prefill submission.
static int ggml_cuda_set_prefill_capture_hint(int value) {
    GGML_ASSERT(value >= -1 && value <= 1);
    const int previous = ggml_cuda_prefill_capture_hint;
    ggml_cuda_prefill_capture_hint = value;
    return previous;
}

// Independent ordinary-MTP decode hint: never changes graph capture policy.
static thread_local int ggml_cuda_device_tail_decode_hint = 0;
static int ggml_cuda_set_device_tail_decode_hint(int value) {
    GGML_ASSERT(value == 0 || value == 1);
    const int previous = ggml_cuda_device_tail_decode_hint;
    ggml_cuda_device_tail_decode_hint = value;
    return previous;
}

static void ggml_cuda_graph_update_executable(ggml_backend_cuda_context * cuda_ctx, const void * graph_key) {
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
    GGML_ASSERT(graph->graph != nullptr && graph->instance != nullptr);
    static const bool fresh_exec = [] {
        const char * value = std::getenv("GGML_CUDA_GRAPH_FRESH_EXEC");
        return value && std::strcmp(value, "1") == 0;
    }();
    // Graph APIs may report previous asynchronous errors. Do not clear an
    // unrelated kernel error while diagnosing/rebuilding an executable.
    const cudaError_t pending = cudaPeekAtLastError();
    if (pending != cudaSuccess) {
        GGML_LOG_ERROR("CUDA graph pending error ctx=%p device=%d status=%d error=%s before_update=1\n",
                (void *) cuda_ctx, cuda_ctx->device, int(pending), cudaGetErrorString(pending));
        CUDA_CHECK(pending);
    }
    const int64_t update_start = ggml_cuda_ft_host_timing_enabled() ? ggml_time_us() : 0;
    cudaError_t stat = cudaSuccess;
    int update_result = -1;
    if (!fresh_exec) {
#if CUDART_VERSION >= 12000
        cudaGraphExecUpdateResultInfo result_info{};
        stat = cudaGraphExecUpdate(graph->instance, graph->graph, &result_info);
        if (stat == cudaErrorGraphExecUpdateFailure) update_result = int(result_info.result);
#else
        cudaGraphNode_t error_node = nullptr;
        cudaGraphExecUpdateResult result_info{};
        stat = cudaGraphExecUpdate(graph->instance, graph->graph, &error_node, &result_info);
        if (stat == cudaErrorGraphExecUpdateFailure) update_result = int(result_info);
#endif
    }
    ggml_cuda_ft_host_record(cuda_ctx, 6, update_start);
    if (fresh_exec || stat == cudaErrorGraphExecUpdateFailure) {
        ggml_cuda_optimization_path_log(cuda_ctx->device, "cuda_graph_fresh_exec", fresh_exec, fresh_exec,
                fresh_exec ? "optin_fresh_captured_graph_executable" : "normal_update_constraint_fallback",
                0, 0, 0);
        GGML_LOG_WARN("CUDA graph executable rebuild ctx=%p device=%d reason=%s status=%d error=%s"
                " update_result=%d action=reinstantiate graph=%p instance=%p\n",
                (void *) cuda_ctx, cuda_ctx->device,
                fresh_exec ? "explicit_fresh_exec" : "exec_update_failed", int(stat),
                cudaGetErrorString(stat), update_result, (void *) graph->graph, (void *) graph->instance);
        // The documented update-constraint error is nonfatal. The explicit
        // fresh-exec route never clears an unrelated launch error.
        if (!fresh_exec) (void) cudaGetLastError();
        CUDA_CHECK(cudaGraphExecDestroy(graph->instance));
        graph->instance = nullptr;
        const int64_t start = ggml_cuda_ft_host_timing_enabled() ? ggml_time_us() : 0;
        CUDA_CHECK(cudaGraphInstantiate(&graph->instance, graph->graph, nullptr, nullptr, 0));
        ggml_cuda_ft_host_record(cuda_ctx, 7, start);
    } else if (stat != cudaSuccess) {
        // Keep the actual status: V54's bare assertion discarded this evidence.
        GGML_LOG_ERROR("CUDA graph update error ctx=%p device=%d status=%d error=%s graph=%p instance=%p\n",
                (void *) cuda_ctx, cuda_ctx->device, int(stat), cudaGetErrorString(stat),
                (void *) graph->graph, (void *) graph->instance);
        CUDA_CHECK(stat);
    }
}
#endif // USE_CUDA_GRAPH

static bool ggml_cuda_should_fuse_rope_set_rows(const ggml_tensor * rope,
                                                const ggml_tensor * view,
                                                const ggml_tensor * set_rows) {

    if (rope->op != GGML_OP_ROPE || view->op != GGML_OP_VIEW || set_rows->op != GGML_OP_SET_ROWS) {
        return false;
    }
    // ne3 not tested
    if (rope->src[0]->ne[3] != 1) {
        return false;
    }

    if (set_rows->type != GGML_TYPE_F32 && set_rows->type != GGML_TYPE_F16) {
        return false;
    }

    if (set_rows->src[1]->type != GGML_TYPE_I64) {
        return false;
    }

    // The view should flatten two dims of rope into one dim
    if (!ggml_is_contiguous(view) || view->ne[0] != rope->ne[0] * rope->ne[1]) {
        return false;
    }

    // Only norm/neox shaders have the fusion code
    const int mode = ((const int32_t *) rope->op_params)[2];
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX) {
        return false;
    }

    return true;
}

static bool ggml_cuda_should_fuse_rms_norm_mul_rope(const ggml_tensor * rms_norm,
                                                    const ggml_tensor * mul,
                                                    const ggml_tensor * rope) {
    if (rms_norm->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL || rope->op != GGML_OP_ROPE) {
        return false;
    }

    if (rms_norm->src[0]->type != GGML_TYPE_F32 || rms_norm->type != GGML_TYPE_F32 ||
        mul->src[0]->type != GGML_TYPE_F32 || mul->src[1]->type != GGML_TYPE_F32 ||
        mul->type != GGML_TYPE_F32 || rope->type != GGML_TYPE_F32) {
        return false;
    }

    if (rope->src[0] != mul) {
        return false;
    }

    //if rms norm is the B operand, then we don't handle broadcast
    if (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
        return false;
    }

    if (!ggml_are_same_shape(rms_norm, mul)) {
        return false;
    }

    //rms_norm kernel assumes contiguous rows
    if (!ggml_is_contiguous_rows(rms_norm->src[0]) ||
        !ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
        return false;
    }

    // the fused kernel handles the norm/neox rope modes only
    const int mode = ((const int32_t *) rope->op_params)[2];
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX) {
        return false;
    }

    const int n_dims = ((const int32_t *) rope->op_params)[1];
    if (n_dims % 2 != 0 || rope->src[0]->ne[0] % 2 != 0) {
        return false;
    }

    // ggml_rope_set_offset is not yet supported in the fused kernel
    const int n_offs = ((const int32_t *) rope->op_params)[15];
    if (n_offs != 0) {
        return false;
    }

    return true;
}

// match gated_delta_net + the strided cpy that scatters its state snapshots into the cache
// (slot i -> rollback group i, slot 0 newest), so the kernel can write them and skip the cpy.
static int ggml_cuda_try_gdn_cache_fusion(
        const ggml_cgraph * cgraph, int node_idx, ggml_cuda_gated_delta_net_fused_cache & fused_state_cpy) {
    const ggml_tensor * gdn = cgraph->nodes[node_idx];
    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    if (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->type != GGML_TYPE_F32 ||
        (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }

    const ggml_tensor * src_v     = gdn->src[2];
    const int64_t       S_v       = src_v->ne[0];
    const int64_t       H         = src_v->ne[1];
    const int64_t       n_tokens  = src_v->ne[2];
    const int64_t       n_seqs    = src_v->ne[3];
    const int64_t       D         = S_v * S_v * H;
    const int64_t       K         = ggml_get_op_params_i32(gdn, 0); // snapshot slot count
    const int64_t       n_written = std::min<int64_t>(n_tokens, K); // newest n_written slots are written

    // snapshot tail starts right after the attention scores
    const size_t tail_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    // snapshot cpy is the first real node after the gdn (skip views/no-ops)
    const ggml_tensor * cpy  = nullptr;
    int                 skip = 0;
    for (int j = node_idx + 1; j < cgraph->n_nodes && cpy == nullptr; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(n)) {
            continue;
        }
        if (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return 0;
        }
        cpy  = n;
        skip = j - node_idx;
    }
    if (cpy == nullptr) {
        return 0;
    }

    const ggml_tensor * src = cpy->src[0]; // view of the gdn snapshot tail
    const ggml_tensor * dst = cpy->src[1]; // cache view the kernel writes to

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    if (src->op != GGML_OP_VIEW || src->view_src != gdn || src->view_offs != tail_off ||
        !ggml_is_contiguous(src)) {
        return 0;
    }

    // dst is the [D, n_seqs, n_written] cache view; require nb[1] == D (the per-seq stride the kernel
    // assumes). ggml_cpy pins src to the same element count.
    const std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    if (dst->op != GGML_OP_VIEW || dst->type != GGML_TYPE_F32 || dst->data == nullptr ||
        !std::equal(expected_ne.begin(), expected_ne.end(), dst->ne) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) || dst->nb[1] != (size_t) ggml_row_size(GGML_TYPE_F32, D)) {
        return 0;
    }

    fused_state_cpy.data        = (float *) dst->data; // rollback group 0 (newest)
    fused_state_cpy.slot_stride = K > 1 ? (int64_t) (dst->nb[2] / sizeof(float)) : 0;
    return skip;
}

static bool ggml_cuda_topk_moe_fusion(const struct ggml_cgraph * cgraph, int node_idx, ggml_cuda_topk_moe_args & args) {
    args.sigmoid         = false;
    args.sqrt_softplus   = false;
    args.softmax         = false;
    args.delayed_softmax = false;
    args.prob_bias       = false;
    args.norm            = false;
    args.scale           = false;

    const int      n_nodes = cgraph->n_nodes;
    ggml_tensor ** nodes   = cgraph->nodes;

    if (nodes[node_idx]->op == GGML_OP_SOFT_MAX) {
        args.softmax = true;
    }

    if (nodes[node_idx]->op == GGML_OP_UNARY) {
        const ggml_unary_op unary_op = ggml_get_unary_op(nodes[node_idx]);
        if (unary_op == GGML_UNARY_OP_SIGMOID) {
            args.sigmoid = true;
        } else if (unary_op == GGML_UNARY_OP_SOFTPLUS && node_idx + 1 < n_nodes &&
                   nodes[node_idx + 1]->op == GGML_OP_SQRT && nodes[node_idx + 1]->src[0] == nodes[node_idx]) {
            // sqrt(softplus(x)) scoring (DeepSeek-V4)
            args.sqrt_softplus = true;
            node_idx++;
        } else {
            return false;
        }
    }

    if (nodes[node_idx]->op == GGML_OP_ARGSORT) {
        args.delayed_softmax = true;
    }

    node_idx++;

    if (args.sigmoid || args.sqrt_softplus || args.softmax) {
        // SOFTMAX -> RESHAPE
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_RESHAPE ||
                nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        ggml_tensor * probs_reshaped = nodes[node_idx];
        node_idx++;

        if (node_idx >= n_nodes) {
            return false;
        }

        // src of bias add is the unreshaped probs (-2 instead of -1)
        if (nodes[node_idx]->op == GGML_OP_ADD && nodes[node_idx]->src[0] == nodes[node_idx - 2]) {
            args.prob_bias = true;
            node_idx++;
        }
        // RESHAPE/ADD -> ARGSORT
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_ARGSORT) {
            return false;
        }

        if (args.prob_bias && nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        } else if (!args.prob_bias && nodes[node_idx]->src[0] != nodes[node_idx - 2]) {
            return false;
        }

        node_idx++;

        // ARGSORT-> VIEW
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
                nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;

        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_GET_ROWS) {
            return false;
        }

        // GET_ROWS
        if (nodes[node_idx]->src[0] != probs_reshaped || nodes[node_idx]->src[1] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;
    } else if (args.delayed_softmax) {
        if (node_idx - 2 < 0) {
            return false;
        }
        ggml_tensor * probs_reshaped = nodes[node_idx - 2];

        // VIEW->ARGSORT
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
            nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;

        // GET_ROWS
        if (node_idx >= n_nodes || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
                nodes[node_idx]->src[0] != probs_reshaped) {
            return false;
        }
        node_idx++;

        static const std::vector<ggml_op> remaining_ops = { GGML_OP_RESHAPE, GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

        for (const ggml_op op : remaining_ops) {
            if (node_idx >= n_nodes || nodes[node_idx]->op != op || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
                return false;
            }
            node_idx++;
        }
    }

    // At this point we can check for norm + scale. Everything is now at least valid till the norm
    if (node_idx >= n_nodes) {
        return true;
    }

    if (nodes[node_idx]->op == GGML_OP_RESHAPE) {
        //check RESHAPE->SUM_ROWS->CLAMP->DIV->RESHAPE
        static const std::vector<ggml_op> norm_ops = { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP };

        args.norm = true;
        for (const ggml_op op : norm_ops) {
            if (node_idx < n_nodes && nodes[node_idx]->op == op && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
                node_idx++;
            } else {
                args.norm = false;
                return true;
            }
        }

        // DIV <- CLAMP, RESHAPE
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_DIV || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
            nodes[node_idx]->src[0] != nodes[node_idx - 3]) {
            args.norm = false;
            return true;
        }
        node_idx++;

        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_RESHAPE || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            args.norm = false;
            return true;
        }

        node_idx++;
    }

    if (node_idx < n_nodes && nodes[node_idx]->op == GGML_OP_SCALE && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
        args.scale = true;
    }

    return true;
}

// returns whether the write (out) nodes overwrite the read nodes in operation
static bool ggml_cuda_check_fusion_memory_ranges(const ggml_cgraph * cgraph,
                                                 const int           node_idx,
                                                 const int           node_count,
                                                 const int *         out_nodes,
                                                 const int           out_count,
                                                 const bool          is_topk_moe = false) {
    auto nodes_overlap = [&](const ggml_tensor * a, const ggml_tensor * b) {
        const int64_t a_start = (int64_t) a->data;
        const int64_t a_end   = a_start + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);

        const int64_t b_start = (int64_t) b->data;
        const int64_t b_end   = b_start + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);

        if ((b_start <= a_start && a_start < b_end) || (a_start <= b_start && b_start < a_end)) {
            return true;
        }

        return false;
    };

    bool is_ok = true;
    // one block reads all logits before it writes, so logits may alias the out nodes
    const ggml_tensor * logits_may_alias = nullptr;
    if (is_topk_moe && ggml_nrows(cgraph->nodes[node_idx]) <= TOPK_MOE_ROWS_PER_BLOCK) {
        logits_may_alias = cgraph->nodes[node_idx]->src[0];
    }

    for (int i = 0; i < out_count; ++i) {
        const ggml_tensor * dst = cgraph->nodes[out_nodes[i]];

        for (int j = node_idx; j < node_idx + node_count; ++j) {
            // Loop over all srcs of all nodes in the fusion. If the src overlaps
            // the destination and the src is not an intermediate node that's being
            // elided, then disable fusion.

            for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                const ggml_tensor * src = cgraph->nodes[j]->src[src_idx];

                if (!src || src->op == GGML_OP_NONE || src == logits_may_alias) {
                    continue;
                }

                if (nodes_overlap(dst, src)) {
                    bool found = false;

                    for (int k = node_idx; k < j; ++k) {
                        if (cgraph->nodes[k] == src) {
                            found = true;
                            break;
                        }
                    }

                    if (!found) {
                        is_ok = false;
                        break;
                    }
                }
            }
        }
    }

    return is_ok;
}

// The long form spans 2*k + 1 nodes. ggml_can_fuse_subgraph() accepts at most
// 31 nodes, so k <= 15; larger values use the per-operation path.
static constexpr int MOE_WEIGHTED_REDUCTION_MAX_EXPERTS = 15;

struct ggml_cuda_moe_weighted_reduction_match {
    const ggml_tensor * experts      = nullptr;
    const ggml_tensor * expert_scale = nullptr;
    const ggml_tensor * weights      = nullptr;
    ggml_tensor *       dst          = nullptr;
    int                 node_count   = 0;
};

static bool ggml_cuda_match_moe_weighted_reduction(
        const ggml_cgraph * cgraph,
        int node_idx,
        ggml_cuda_moe_weighted_reduction_match & match) {
    const ggml_tensor * first = cgraph->nodes[node_idx];
    if (first->op != GGML_OP_MUL || first->type != GGML_TYPE_F32 || !ggml_is_contiguous(first)) {
        return false;
    }

    auto split_mul = [](const ggml_tensor * mul, const ggml_tensor *& full, const ggml_tensor *& broadcast) {
        auto is_weights = [mul](const ggml_tensor * tensor) {
            return tensor && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) && tensor->ne[0] == 1 &&
                tensor->ne[1] == mul->ne[1] && tensor->ne[2] == mul->ne[2] && tensor->ne[3] == mul->ne[3];
        };
        auto is_experts = [mul](const ggml_tensor * tensor) {
            return tensor && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) &&
                ggml_are_same_shape(tensor, mul);
        };

        if (is_experts(mul->src[0]) && is_weights(mul->src[1])) {
            full      = mul->src[0];
            broadcast = mul->src[1];
            return true;
        }
        if (is_experts(mul->src[1]) && is_weights(mul->src[0])) {
            full      = mul->src[1];
            broadcast = mul->src[0];
            return true;
        }
        return false;
    };

    const ggml_tensor * weighted     = first;
    const ggml_tensor * experts      = nullptr;
    const ggml_tensor * expert_scale = nullptr;
    const ggml_tensor * weights      = nullptr;
    int                 mul_count    = 1;

    // Match both structural forms:
    //   (experts * expert_scale) * router_weight
    //   experts * router_weight
    // The matcher does not depend on the model or quantization type.
    if (node_idx + 1 < cgraph->n_nodes) {
        const ggml_tensor * second = cgraph->nodes[node_idx + 1];
        const ggml_tensor * scaled = nullptr;
        const ggml_tensor * route  = nullptr;
        const ggml_tensor * raw    = nullptr;
        const ggml_tensor * scale  = nullptr;
        if (second->op == GGML_OP_MUL && second->type == GGML_TYPE_F32 && ggml_is_contiguous(second) &&
                split_mul(second, scaled, route) && scaled == first && split_mul(first, raw, scale)) {
            weighted     = second;
            experts      = raw;
            expert_scale = scale;
            weights      = route;
            mul_count    = 2;
        }
    }

    if (experts == nullptr && !split_mul(first, experts, weights)) {
        return false;
    }

    const int     n_expert_used = (int) weighted->ne[1];
    const int64_t n_tokens      = weighted->ne[2] * weighted->ne[3];
    if (n_expert_used < 2 || n_expert_used > MOE_WEIGHTED_REDUCTION_MAX_EXPERTS || n_tokens <= 0) {
        return false;
    }

    const int node_count = 2 * n_expert_used + mul_count - 1;
    if (node_idx + node_count > cgraph->n_nodes) {
        return false;
    }

    std::vector<ggml_op> ops(node_count, GGML_OP_VIEW);
    ops[0] = GGML_OP_MUL;
    if (mul_count == 2) {
        ops[1] = GGML_OP_MUL;
    }
    std::vector<const ggml_tensor *> views;
    views.reserve(n_expert_used);
    const ggml_tensor * previous = nullptr;
    int n_adds = 0;
    for (int offset = mul_count; offset < node_count; ++offset) {
        const ggml_tensor * candidate = cgraph->nodes[node_idx + offset];
        ops[offset] = candidate->op;

        if (candidate->op == GGML_OP_VIEW) {
            const int expert = (int) views.size();
            if (expert >= n_expert_used || candidate->src[0] != weighted || candidate->view_src != weighted ||
                    candidate->type != GGML_TYPE_F32 || candidate->ne[0] != weighted->ne[0] ||
                    candidate->ne[1] != n_tokens || candidate->ne[2] != 1 || candidate->ne[3] != 1 ||
                    candidate->nb[0] != weighted->nb[0] || candidate->nb[1] != weighted->nb[2] ||
                    candidate->view_offs != (size_t) expert * weighted->nb[1]) {
                return false;
            }
            views.push_back(candidate);
            continue;
        }

        if (candidate->op != GGML_OP_ADD || views.size() < 2 || n_adds + 1 >= (int) views.size()) {
            return false;
        }
        const ggml_tensor * lhs = n_adds == 0 ? views[0] : previous;
        const ggml_tensor * rhs = views[n_adds + 1];
        if (candidate->src[0] != lhs || candidate->src[1] != rhs || candidate->type != GGML_TYPE_F32) {
            return false;
        }
        previous = candidate;
        ++n_adds;
    }

    if ((int) views.size() != n_expert_used || n_adds != n_expert_used - 1 || previous == nullptr) {
        return false;
    }
    if (!ggml_is_contiguous(previous) || previous->ne[0] != weighted->ne[0] ||
            previous->ne[1] != n_tokens || previous->ne[2] != 1 || previous->ne[3] != 1) {
        return false;
    }

    const int output_idx = node_idx + node_count - 1;
    if (!ggml_can_fuse_subgraph(cgraph, node_idx, node_count, ops.data(), &output_idx, 1)) {
        return false;
    }

    match.experts      = experts;
    match.expert_scale = expert_scale;
    match.weights      = weights;
    match.dst          = cgraph->nodes[output_idx];
    match.node_count   = node_count;
    return true;
}


// like ggml_can_fuse, but lets a broadcast operand flow into the chain (see ggml_cuda_op_elem_chain)
static bool ggml_cuda_can_fuse_elem_chain(const struct ggml_cgraph * cgraph, int node_idx, std::initializer_list<enum ggml_op> ops) {
    const int n = (int) ops.size();
    if (node_idx + n > cgraph->n_nodes) {
        return false;
    }
    for (int k = 0; k < n; ++k) {
        ggml_tensor * node = cgraph->nodes[node_idx + k];
        if (node->op != ops.begin()[k] || (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            return false;
        }
        if (k < n - 1 && !ggml_node_has_n_uses(cgraph, node_idx + k, 1)) {
            return false;
        }
        if (k > 0) {
            const ggml_tensor * prev = cgraph->nodes[node_idx + k - 1];
            if (node->src[0] != prev && node->src[1] != prev) {
                return false;
            }
        }
    }
    return ggml_cuda_elem_chain_supported(cgraph->nodes + node_idx, n);
}

static bool ggml_cuda_can_fuse(const struct ggml_cgraph *                cgraph,
                               int                                       node_idx,
                               std::initializer_list<enum ggml_op>       ops,
                               std::initializer_list<enum ggml_unary_op> unary_ops) {
#ifndef NDEBUG
    const size_t num_unary = std::count(ops.begin(), ops.end(), GGML_OP_UNARY);
    GGML_ASSERT(unary_ops.size() == num_unary);
#endif

    const auto is_equal = [](const std::initializer_list<enum ggml_op> & list1,
                             const std::initializer_list<enum ggml_op> & list2) {
        return std::equal(list1.begin(), list1.end(), list2.begin(), list2.end());
    };

    std::initializer_list<enum ggml_op> mul_mat_bias_glu_ops    = { GGML_OP_MUL_MAT,    GGML_OP_ADD,    GGML_OP_MUL_MAT,    GGML_OP_ADD,    GGML_OP_GLU };
    std::initializer_list<enum ggml_op> mul_mat_id_bias_glu_ops = { GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID, GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID, GGML_OP_GLU };

    std::initializer_list<enum ggml_op> mul_mat_id_glu_ops = { GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU };
    std::initializer_list<enum ggml_op> mul_mat_glu_ops    = { GGML_OP_MUL_MAT,    GGML_OP_MUL_MAT,    GGML_OP_GLU };

    if ((is_equal(mul_mat_bias_glu_ops, ops) || is_equal(mul_mat_id_bias_glu_ops, ops)) &&
        ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 4 })) {
        const ggml_tensor * ffn_gate      = cgraph->nodes[node_idx];
        const ggml_tensor * ffn_gate_bias = cgraph->nodes[node_idx + 1];
        const ggml_tensor * ffn_up        = cgraph->nodes[node_idx + 2];
        const ggml_tensor * ffn_up_bias   = cgraph->nodes[node_idx + 3];
        const ggml_tensor * glu           = cgraph->nodes[node_idx + 4];

        if (ggml_cuda_should_fuse_mul_mat(ffn_up, ffn_gate, glu, ffn_up_bias, ffn_gate_bias)) {
            int out_nodes[] = { node_idx + 4 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    if ((is_equal(mul_mat_id_glu_ops, ops) || is_equal(mul_mat_glu_ops, ops)) &&
        ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
        const ggml_tensor * ffn_gate = cgraph->nodes[node_idx];
        const ggml_tensor * ffn_up   = cgraph->nodes[node_idx + 1];
        const ggml_tensor * glu      = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_mul_mat(ffn_up, ffn_gate, glu)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    std::initializer_list<enum ggml_op> rms_norm_mul_rope_ops          = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE };
    std::initializer_list<enum ggml_op> rms_norm_mul_rope_set_rows_ops = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS };

    if (is_equal(rms_norm_mul_rope_set_rows_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 4 })) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];
        const ggml_tensor * rope     = cgraph->nodes[node_idx + 2];
        const ggml_tensor * view     = cgraph->nodes[node_idx + 3];
        const ggml_tensor * set_rows = cgraph->nodes[node_idx + 4];

        if (ggml_check_edges(cgraph, node_idx, {{1, 0, 0}, {2, 0, 1}, {3, 0, 2}, {4, 0, 3}}) &&
            ggml_cuda_should_fuse_rms_norm_mul_rope(rms_norm, mul, rope) &&
            ggml_cuda_should_fuse_rope_set_rows(rope, view, set_rows)) {
            int out_nodes[] = { node_idx + 4 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    if (is_equal(rms_norm_mul_rope_ops, ops) && ggml_can_fuse(cgraph, node_idx, ops)) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];
        const ggml_tensor * rope     = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_rms_norm_mul_rope(rms_norm, mul, rope)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
        return false;
    }

    std::initializer_list<enum ggml_op> rope_set_rows_ops = { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS };

    if (is_equal(rope_set_rows_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
        const ggml_tensor * rope     = cgraph->nodes[node_idx];
        const ggml_tensor * view     = cgraph->nodes[node_idx + 1];
        const ggml_tensor * set_rows = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_rope_set_rows(rope, view, set_rows)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    if (!ggml_can_fuse(cgraph, node_idx, ops)) {
        return false;
    }

    if ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        const ggml_tensor *rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor *mul      = cgraph->nodes[node_idx+1];
        const ggml_tensor *add      = nullptr;

        if (ops.size() == 3 && ops.begin()[2] == GGML_OP_ADD) {
            add = cgraph->nodes[node_idx+2];
        }

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);

        //rms norm only supports F32
        if (mul->src[0]->type != GGML_TYPE_F32 ||
            mul->src[1]->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32) {
            return false;
        }

        if (add && (add->src[0]->type != GGML_TYPE_F32 ||
            add->src[1]->type != GGML_TYPE_F32 ||
            add->type != GGML_TYPE_F32) ) {
            return false;
        }

        //if rms norm is the B operand, then we don't handle broadcast
        if (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
            return false;
        }

        //rms_norm kernel assumes contiguous rows
        if (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
            return false;
        }

        if (add && (!ggml_is_contiguous(add->src[0]) || !ggml_is_contiguous_rows(add->src[1]))) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_UNARY
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        const ggml_tensor * ssm_conv = cgraph->nodes[node_idx];
        const ggml_tensor * silu     = cgraph->nodes[node_idx+1];
        if (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            return false;
        }

        if (ssm_conv->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            return false;
        }

        return true;
    }

    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_ADD
     && ops.begin()[2] == GGML_OP_UNARY && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        const ggml_tensor * ssm_conv = cgraph->nodes[node_idx];
        const ggml_tensor * add      = cgraph->nodes[node_idx+1];
        const ggml_tensor * silu     = cgraph->nodes[node_idx+2];
        if (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            return false;
        }

        if (ssm_conv->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            return false;
        }

        // ADD must consume ssm_conv's output and broadcast a 1-D channel-wise bias.
        const ggml_tensor * bias = (add->src[0] == ssm_conv) ? add->src[1] : add->src[0];
        if (bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias)) {
            return false;
        }
        if (ggml_nelements(bias) != ssm_conv->ne[0] || bias->ne[0] != ssm_conv->ne[0]) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_MUL
     && unary_ops.size() == 1 && (unary_ops.begin()[0] == GGML_UNARY_OP_SILU || unary_ops.begin()[0] == GGML_UNARY_OP_SIGMOID || unary_ops.begin()[0] == GGML_UNARY_OP_SOFTPLUS)) {
        const ggml_tensor * unary = cgraph->nodes[node_idx];
        const ggml_tensor * mul   = cgraph->nodes[node_idx+1];

        if (ggml_get_unary_op(unary) != unary_ops.begin()[0]) {
            return false;
        }

        if (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
            return false;
        }

        if (unary->type != mul->type) {
            return false;
        }

        const ggml_tensor * other = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];
        if (other->type != unary->type) {
            return false;
        }
        if (!ggml_is_contiguous_1(other) || !ggml_is_contiguous_1(unary->src[0]) || !ggml_are_same_shape(other, unary)) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_SQR
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_RELU) {
        const ggml_tensor * unary = cgraph->nodes[node_idx];
        const ggml_tensor * sqr   = cgraph->nodes[node_idx+1];

        if (ggml_get_unary_op(unary) != GGML_UNARY_OP_RELU) {
            return false;
        }

        if (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
            return false;
        }

        if (unary->type != sqr->type) {
            return false;
        }

        if (!ggml_is_contiguous(unary->src[0])) {
            return false;
        }

        return true;
    }

    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_SCALE && ops.begin()[1] == GGML_OP_UNARY && ops.begin()[2] == GGML_OP_SCALE
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_TANH) {
        const ggml_tensor *scale  = cgraph->nodes[node_idx];
        const ggml_tensor *tanh   = cgraph->nodes[node_idx+1];
        const ggml_tensor *scale2 = cgraph->nodes[node_idx+2];

        GGML_ASSERT(scale->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(scale->type == GGML_TYPE_F32);

        if (ggml_get_unary_op(tanh) != GGML_UNARY_OP_TANH) {
            return false;
        }

        // Check for bias
        if (ggml_get_op_params_f32(scale, 1) != 0.0f || ggml_get_op_params_f32(scale2, 1) != 0.0f) {
            return false;
        }

        return true;
    }

    return false;
}

// try and fuse nodes and return the number of nodes to skip
// Deferred recurrent-state gathers: a GET_ROWS that only feeds the state input of a GATED_DELTA_NET a
// few nodes later (possibly through a reshape) is not computed; the GDN kernel reads the selected cache
// rows itself (3 MB per layer that would otherwise be copied first). Tensors are tracked per thread for
// the graph being evaluated.
static thread_local std::unordered_map<const ggml_tensor *, const int32_t *> ggml_cuda_gdn_deferred_gathers;

const int32_t * ggml_cuda_gdn_state_gather_rows(const ggml_tensor * get_rows) {
    const auto it = ggml_cuda_gdn_deferred_gathers.find(get_rows);
    return it == ggml_cuda_gdn_deferred_gathers.end() ? nullptr : it->second;
}

// persistent per-device slots for deferred row ids; each deferral takes the next slot, so a captured
// CUDA graph keeps its own addresses
static int32_t * ggml_cuda_gdn_rows_slot(const int n, cudaStream_t stream) {
    constexpr int n_slots = 1 << 16;
    static int32_t * buf[GGML_CUDA_MAX_DEVICES] = {};
    static int next[GGML_CUDA_MAX_DEVICES] = {};
    const int dev = ggml_cuda_get_device();
    if (buf[dev] == nullptr) {
        CUDA_CHECK(cudaMalloc(&buf[dev], n_slots*sizeof(int32_t)));
        CUDA_CHECK(cudaMemsetAsync(buf[dev], 0, n_slots*sizeof(int32_t), stream));
    }
    if (next[dev] + n > n_slots) {
        next[dev] = 0;
    }
    int32_t * p = buf[dev] + next[dev];
    next[dev] += n;
    return p;
}

static bool ggml_cuda_gdn_state_gather_deferred(ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph, int i) {
    static const bool disabled = getenv("GGML_CUDA_NO_GDN_DEFER_GATHER") != nullptr;
    const ggml_tensor * gr = cgraph->nodes[i];
    if (disabled || gr->op != GGML_OP_GET_ROWS || gr->type != GGML_TYPE_F32 || gr->src[0]->type != GGML_TYPE_F32 ||
            gr->src[1]->type != GGML_TYPE_I32 || gr->src[1]->ne[1] != 1 || gr->src[1]->ne[2] != 1 ||
            gr->src[0]->nb[0] != sizeof(float) || gr->src[0]->nb[1] % sizeof(float) != 0 || !ggml_is_contiguous(gr) ||
            gr->ne[0] != gr->src[0]->ne[0] || gr->src[0]->ne[2] != 1 || gr->src[0]->ne[3] != 1 ||
            !ggml_node_has_n_uses(cgraph, i, 1)) {
        return false;
    }
    // the GDN kernel dereferences the rows and the ids itself: both must be device memory
    if (!gr->src[0]->buffer || !gr->src[1]->buffer || ggml_backend_buffer_is_host(gr->src[0]->buffer) ||
            ggml_backend_buffer_is_host(gr->src[1]->buffer)) {
        return false;
    }
    const ggml_tensor * via = nullptr; // a reshape between the gather and the GDN
    for (int j = i + 1; j < cgraph->n_nodes && j <= i + 1024; ++j) {
        const ggml_tensor * node = cgraph->nodes[j];
        if (node->op == GGML_OP_RESHAPE && node->src[0] == gr && node->view_offs == 0) {
            // use counts are not reliable for views inside a split graph: count the readers directly
            int readers = 0;
            const ggml_tensor * reader = nullptr;
            for (int u = j + 1; u < cgraph->n_nodes && u <= i + 1024; ++u) {
                for (int q = 0; q < GGML_MAX_SRC; ++q) {
                    if (cgraph->nodes[u]->src[q] == node) {
                        ++readers;
                        reader = cgraph->nodes[u];
                    }
                }
            }
            if (readers != 1 || reader->op != GGML_OP_GATED_DELTA_NET || reader->src[5] != node) {
                return false;
            }
            via = node;
            continue;
        }
        if (node->op == GGML_OP_GATED_DELTA_NET) {
            const ggml_tensor * st = node->src[5];
            // the reshape in between may not be a node of this (split) graph: match through view_src
            const bool through_view = st->view_src == gr && st->view_offs == 0 && ggml_is_contiguous(st);
            if (st == gr || (via != nullptr && st == via) || through_view) {
                // the GDN kernel indexes sequences by row: one gathered row per sequence
                if (st->ne[3] != gr->ne[1] || ggml_nelements(st) != ggml_nelements(gr)) {
                    return false;
                }
                const int n = (int) gr->src[1]->ne[0];
                int32_t * rows = ggml_cuda_gdn_rows_slot(n, cuda_ctx->stream());
                CUDA_CHECK(cudaMemcpyAsync(rows, gr->src[1]->data, n*sizeof(int32_t), cudaMemcpyDeviceToDevice, cuda_ctx->stream()));
                ggml_cuda_gdn_deferred_gathers[gr] = rows;
                return true;
            }
        }
    }
    return false;
}

// the activation chain that follows a small f32/bf16 GEMV at node j (see the mmvf epilogue): 0 none, 1 sigmoid
// (not for a one-row gate, left to the sigmoid-mul-add elem chain), 2 softplus(x + pre[row]) * mul[row]. last is the
// index of the chain's last node (j without a chain); chain holds the chain's nodes.
struct ggml_cuda_gemv_chain {
    int post_op = 0;
    const ggml_tensor * pre = nullptr;
    const ggml_tensor * mul = nullptr;
    int last = -1;
    int n = 0;
    int idx[3];
};

static ggml_cuda_gemv_chain ggml_cuda_match_gemv_chain(const ggml_cgraph * cgraph, const int j) {
    ggml_cuda_gemv_chain c;
    c.last = j;
    const ggml_tensor * node = cgraph->nodes[j];
    const auto from_mm = [&](const ggml_tensor * t) {
        return (t == node || t->view_src == node) && ggml_is_contiguous(t) && ggml_nelements(t) == ggml_nelements(node);
    };
    const auto per_row = [&](const ggml_tensor * t, const ggml_tensor * like) {
        return t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && ggml_nelements(t) == node->ne[0] && like->ne[0] == node->ne[0];
    };
    int js[3];
    int n = 0;
    for (int k = j + 1; k < cgraph->n_nodes && k <= j + 8 && n < 3; ++k) {
        if (ggml_cuda_is_view_or_noop(cgraph->nodes[k])) {
            continue;
        }
        js[n++] = k;
    }
    if (n == 0) {
        return c;
    }
    const ggml_tensor * n0 = cgraph->nodes[js[0]];
    const ggml_tensor * roots[3] = { node, nullptr, nullptr };
    if (!(n0->flags & GGML_TENSOR_FLAG_COMPUTE)) {
        return c;
    }
    if (n0->op == GGML_OP_UNARY && ggml_get_unary_op(n0) == GGML_UNARY_OP_SIGMOID && from_mm(n0->src[0]) && node->ne[0] > 1 &&
            n0->type == GGML_TYPE_F32 && ggml_is_contiguous(n0) && ggml_cuda_group_private(cgraph, j, js[0], roots, 1)) {
        c.post_op = 1;
        c.last    = js[0];
        c.idx[c.n++] = js[0];
    } else if (n >= 3 && n0->op == GGML_OP_ADD) {
        const ggml_tensor * sp  = cgraph->nodes[js[1]];
        const ggml_tensor * mul = cgraph->nodes[js[2]];
        const ggml_tensor * pre = from_mm(n0->src[0]) ? n0->src[1] : from_mm(n0->src[1]) ? n0->src[0] : nullptr;
        const ggml_tensor * a   = mul->op != GGML_OP_MUL ? nullptr : mul->src[0] == sp ? mul->src[1] : mul->src[1] == sp ? mul->src[0] : nullptr;
        roots[1] = n0;
        roots[2] = sp;
        if (pre && per_row(pre, n0) && ggml_is_contiguous(n0) &&
                sp->op == GGML_OP_UNARY && ggml_get_unary_op(sp) == GGML_UNARY_OP_SOFTPLUS && sp->src[0] == n0 &&
                a && a != sp && per_row(a, mul) && (sp->flags & GGML_TENSOR_FLAG_COMPUTE) && (mul->flags & GGML_TENSOR_FLAG_COMPUTE) &&
                mul->type == GGML_TYPE_F32 && ggml_is_contiguous(mul) && ggml_nelements(mul) == ggml_nelements(node) &&
                ggml_cuda_group_private(cgraph, j, js[2], roots, 3)) {
            c.post_op = 2;
            c.pre = pre;
            c.mul = a;
            c.last = js[2];
            c.idx[c.n++] = js[0];
            c.idx[c.n++] = js[1];
            c.idx[c.n++] = js[2];
        }
    }
    return c;
}

static bool ggml_cuda_is_small_gemv_f(const ggml_tensor * n) {
    return n->op == GGML_OP_MUL_MAT && ggml_get_op_params_i32(n, 1) != GGML_HINT_SRC0_IS_HADAMARD &&
        (n->src[0]->type == GGML_TYPE_F32 || n->src[0]->type == GGML_TYPE_BF16) && n->src[0]->view_src == nullptr &&
        n->src[0]->op == GGML_OP_NONE && n->src[1]->type == GGML_TYPE_F32 && n->type == GGML_TYPE_F32 &&
        ggml_is_contiguous(n) && n->ne[2] == 1 && n->ne[3] == 1 && n->src[1]->ne[1] <= 4 && n->ne[0] <= 1024;
}

// small f32/bf16 GEMVs that read the same activations as node i (within 64 nodes) computed by one launch at node i,
// each with its activation chain. A later GEMV joins only if nothing between i and its place reads or writes the
// memory of its result (it is written early). Returns the nodes to skip after i (node i's chain), -1 if not used.
static int ggml_cuda_try_mmvf_multi(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, const int i) {
    // opt-in (GGML_CUDA_MMVF_MULTI=1): per verify step it measured even with the separate launches (22.2 ms), and the
    // changed summation order moves the greedy text
    static const bool off = getenv("GGML_CUDA_MMVF_MULTI") == nullptr;
    ggml_tensor * node = cgraph->nodes[i];
    if (off || !ggml_cuda_is_small_gemv_f(node)) {
        return -1;
    }
    const ggml_tensor * x = node->src[1];
    ggml_cuda_mmvf_multi_seg segs[MMVF_MULTI_MAX_SEG];
    ggml_cuda_gemv_chain chains[MMVF_MULTI_MAX_SEG];
    int seg_idx[MMVF_MULTI_MAX_SEG];
    int n_seg = 0;

    const auto add = [&](const int j) {
        ggml_tensor * mm = cgraph->nodes[j];
        const ggml_cuda_gemv_chain c = ggml_cuda_match_gemv_chain(cgraph, j);
        const ggml_tensor * out = cgraph->nodes[c.last];
        segs[n_seg]    = { mm->src[0], mm, out->data, c.post_op, c.pre, c.mul };
        chains[n_seg]  = c;
        seg_idx[n_seg] = j;
        n_seg++;
    };
    add(i);
    int total_rows = (int) node->ne[0];
    for (int j = chains[0].last + 1; j < cgraph->n_nodes && j <= i + 64 && n_seg < MMVF_MULTI_MAX_SEG; ++j) {
        ggml_tensor * mm = cgraph->nodes[j];
        if (!ggml_cuda_is_small_gemv_f(mm) || !(mm->flags & GGML_TENSOR_FLAG_COMPUTE) || cuda_ctx->precomputed.count(mm) ||
                !(mm->src[1] == x || (mm->src[1]->data == x->data && ggml_are_same_shape(mm->src[1], x) &&
                  ggml_are_same_stride(mm->src[1], x))) || total_rows + mm->ne[0] > 2048) {
            continue;
        }
        const ggml_cuda_gemv_chain c = ggml_cuda_match_gemv_chain(cgraph, j);
        const ggml_tensor * out = cgraph->nodes[c.last];
        // written at node i: no node between may read or write that memory, and the chain's per-row inputs are weights.
        // The activations must be the same then and at node j: no node between writes them (a buffer reused by the
        // allocator can hold another tensor at the same address).
        bool safe = (c.pre == nullptr || c.pre->op == GGML_OP_NONE) && (c.mul == nullptr || c.mul->op == GGML_OP_NONE);
        for (int k = i; safe && k < j; ++k) {
            const ggml_tensor * m = cgraph->nodes[k];
            if (ggml_cuda_is_view_or_noop(m)) {
                continue;
            }
            safe = !ggml_cuda_ranges_overlap(m, out) && (k == i || !ggml_cuda_ranges_overlap(m, x));
            for (int q = 0; safe && q < GGML_MAX_SRC; ++q) {
                safe = !ggml_cuda_ranges_overlap(m->src[q], out);
            }
        }
        for (int s = 0; safe && s < n_seg; ++s) {
            safe = !ggml_cuda_ranges_overlap(cgraph->nodes[chains[s].last], out);
        }
        if (!safe) {
            continue;
        }
        add(j);
        total_rows += (int) mm->ne[0];
    }
    // only as a combined launch: alone, on cold weights, mul_mat_vec_f is as fast or faster
    if (n_seg < 2) {
        return -1;
    }
    if (!ggml_cuda_mmvf_multi(*cuda_ctx, x, segs, n_seg)) {
        return -1;
    }
    for (int s = 1; s < n_seg; ++s) {
        cuda_ctx->precomputed.insert(cgraph->nodes[seg_idx[s]]);
        for (int q = 0; q < chains[s].n; ++q) {
            cuda_ctx->precomputed.insert(cgraph->nodes[chains[s].idx[q]]);
        }
    }
    return chains[0].last - i;
}

// [SM70_FUSED_PROJ] plain BF16 MUL_MAT that the fused per-wave projection launch (fused-projection-sm70.cuh)
// can take as a wave member: a leaf BF16 weight matrix times a small-batch F32 activation, 2D, no hadamard hint.
// Nodes whose mmvf epilogue chain matches (sigmoid / softplus(x+pre)*mul, ggml_cuda_match_gemv_chain) are NOT
// candidates: the existing epilogue fusion keeps those, so the fused wave only absorbs plain members.
static bool ggml_cuda_is_fused_proj_candidate(const ggml_tensor * n) {
    const ggml_tensor * w = n->src[0];
    const ggml_tensor * y = n->src[1];
    return n->op == GGML_OP_MUL_MAT && ggml_get_op_params_i32(n, 1) != GGML_HINT_SRC0_IS_HADAMARD &&
        (n->flags & GGML_TENSOR_FLAG_COMPUTE) && w->op == GGML_OP_NONE && w->type == GGML_TYPE_BF16 &&
        w->ne[2] == 1 && w->ne[3] == 1 && w->nb[0] == ggml_type_size(GGML_TYPE_BF16) &&
        y->type == GGML_TYPE_F32 && y->ne[1] >= 1 && y->ne[1] <= 8 && y->ne[2] == 1 && y->ne[3] == 1 &&
        y->nb[0] == sizeof(float) && n->type == GGML_TYPE_F32 && n->ne[2] == 1 && n->ne[3] == 1 &&
        n->nb[0] == sizeof(float);
}

// [SM70_FUSED_PROJ] a candidate that feeds a GLU node a few nodes later stays on the stock path: the fork's
// mul-mat pair + GLU fusion (ggml_cuda_should_fuse_mul_mat, later in ggml_cuda_try_fuse) computes such a pair
// in ONE launch, which fused-wave members plus a separate GLU kernel cannot beat. The GLU may read the member
// directly or through one wrapper (view / per-row scale MUL).
static bool ggml_cuda_fused_proj_feeds_glu(const ggml_cgraph * cgraph, const int j) {
    const ggml_tensor * mm = cgraph->nodes[j];
    const auto from_member = [&](const ggml_tensor * t) {
        if (t == nullptr) {
            return false;
        }
        const ggml_tensor * r = t->view_src != nullptr ? t->view_src : t;
        if (t == mm || r == mm) {
            return true;
        }
        for (int p = 0; p < GGML_MAX_SRC; ++p) {
            if (t->src[p] == mm || (t->src[p] != nullptr && t->src[p]->view_src == mm)) {
                return true;
            }
        }
        return false;
    };
    for (int k = j + 1; k < cgraph->n_nodes && k <= j + 8; ++k) {
        const ggml_tensor * n = cgraph->nodes[k];
        if (n->op != GGML_OP_GLU) {
            continue;
        }
        for (int q = 0; q < GGML_MAX_SRC; ++q) {
            if (from_member(n->src[q])) {
                return true;
            }
        }
    }
    return false;
}

// [SM70_FUSED_PROJ] BF16 MUL_MATs of one dependency wave (same activation y, no producer/consumer edges between
// them): ONE fused launch instead of one launch per matrix (DESIGN layer-fusion/DESIGN.md sections 3/8). Members
// may sit anywhere in the following 64-node window (they are skipped via cuda_ctx->precomputed, like try_mmvf_multi),
// but the skip span must reach at least one passable node: the anchor is only kept out of the per-matrix path by
// the returned span, so a live compute node at i+1 cancels the fusion (span 0 would recompute the anchor). Only
// waves with >= 2 tasks that have rows on this GPU fuse: single-matrix waves would only pay the block-claim
// overhead (DESIGN section 10). The static launch takes the task table by value in the kernel parameters, so it
// is CUDA-Graph-capturable verbatim with no table buffer and no cache to invalidate. Env GGML_SM70_FUSED_PROJ:
// 0 off, 1 on (default), 2 on + per-group stderr log (first 96 groups), 3 = 2 + decline diagnostics.
static int ggml_cuda_try_fused_proj(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, const int i) {
    static const int mode = [] {
        const char * env = getenv("GGML_SM70_FUSED_PROJ");
        return env == nullptr ? 1 : atoi(env);   // default on: 4/4 512-ID parity runs bitwise-identical (2026-10-03)
    }();
    if (mode <= 0) {
        return -1;
    }
    static std::atomic<int> logged_groups{0};
    ggml_tensor * node = cgraph->nodes[i];
    if (ggml_cuda_info().devices[cuda_ctx->device].cc != GGML_CUDA_CC_VOLTA ||
            !ggml_cuda_is_fused_proj_candidate(node) ||
            ggml_cuda_match_gemv_chain(cgraph, i).post_op != 0 ||
            ggml_cuda_fused_proj_feeds_glu(cgraph, i)) {
        if (mode >= 3) {
            static std::atomic<int> logged_declines{0};
            const int d = logged_declines.fetch_add(1, std::memory_order_relaxed);
            if (d < 240 && node->op == GGML_OP_MUL_MAT && node->src[0]->type == GGML_TYPE_BF16 &&
                    node->ne[2] == 1 && node->ne[3] == 1) {
                fprintf(stderr, "%s: DECLINE d=%d dev=%d w=%-44s op0=%d wop=%d vsrc=%d comp=%d y=%s "
                        "yT=%d ync=%lld ynb0=%zu chain=%d glu=%d hint=%d nb00=%zu dstT=%d nb0=%zu wne2=%lld\n",
                        __func__, d, ggml_cuda_get_device(), node->src[0]->name,
                        node->src[0]->op, node->src[0]->op == GGML_OP_NONE ? 1 : 0,
                        node->src[0]->view_src != nullptr ? 1 : 0,
                        (node->flags & GGML_TENSOR_FLAG_COMPUTE) ? 1 : 0,
                        node->src[1]->name, node->src[1]->type == GGML_TYPE_F32 ? 1 : 0,
                        (long long) node->src[1]->ne[1], node->src[1]->nb[0],
                        ggml_cuda_match_gemv_chain(cgraph, i).post_op,
                        ggml_cuda_fused_proj_feeds_glu(cgraph, i) ? 1 : 0,
                        ggml_get_op_params_i32(node, 1) != GGML_HINT_SRC0_IS_HADAMARD ? 0 : 1,
                        node->src[0]->nb[0], node->type == GGML_TYPE_F32 ? 1 : 0, node->nb[0],
                        (long long) node->src[0]->ne[2]);
            }
        }
        return -1;
    }
    const ggml_tensor * y = node->src[1];
    const int ncols = (int) y->ne[1];
    constexpr int max_tasks = 8;  // fused_projection_sm70_launch_static table bound (NT)
    fusedproj_sm70::fused_proj_desc descs[max_tasks];
    int idx[max_tasks];
    int n_task = 0;
    idx[n_task++] = i;
    const auto same_y = [&](const ggml_tensor * x2) {
        return x2 == y || (x2->data == y->data && ggml_are_same_shape(x2, y) && ggml_are_same_stride(x2, y));
    };
    for (int j = i + 1; j < cgraph->n_nodes && j <= i + 64 && n_task < max_tasks; ++j) {
        ggml_tensor * mm = cgraph->nodes[j];
        if (!ggml_cuda_is_fused_proj_candidate(mm) || cuda_ctx->precomputed.count(mm) ||
                (int) mm->src[1]->ne[1] != ncols || !same_y(mm->src[1]) ||
                ggml_cuda_match_gemv_chain(cgraph, j).post_op != 0 ||
                ggml_cuda_fused_proj_feeds_glu(cgraph, j)) {
            continue;
        }
        // the member output is written early, at node i: no node between may read or write that memory, and no
        // node between may write y (a buffer reused by the allocator); prior members' outputs must not overlap it
        bool safe = true;
        int unsafe_why = -1, unsafe_k = -1;
        for (int k = i; safe && k < j; ++k) {
            const ggml_tensor * m = cgraph->nodes[k];
            if (ggml_cuda_is_view_or_noop(m)) {
                continue;
            }
            safe = !ggml_cuda_ranges_overlap(m, mm) && (k == i || !ggml_cuda_ranges_overlap(m, y));
            if (!safe) { unsafe_why = k == i ? 1 : (ggml_cuda_ranges_overlap(m, mm) ? 2 : 3); unsafe_k = k; }
            for (int q = 0; safe && q < GGML_MAX_SRC; ++q) {
                safe = !ggml_cuda_ranges_overlap(m->src[q], mm);
                if (!safe) { unsafe_why = 4 + q; unsafe_k = k; }
            }
        }
        for (int s = 0; safe && s < n_task; ++s) {
            safe = !ggml_cuda_ranges_overlap(cgraph->nodes[idx[s]], mm);
            if (!safe) { unsafe_why = 20 + s; unsafe_k = idx[s]; }
        }
        if (!safe) {
            if (mode >= 3) {
                static std::atomic<int> logged_memb{0};
                const int md = logged_memb.fetch_add(1, std::memory_order_relaxed);
                if (md < 24) {
                    fprintf(stderr, "%s: MEMBREJ d=%d anchor w=%s member w=%s why=%d k=%d\n", __func__, md,
                            node->src[0]->name, mm->src[0]->name, unsafe_why, unsafe_k);
                }
            }
            continue;
        }
        idx[n_task++] = j;
    }
    int live = 0;
    for (int s = 0; s < n_task; ++s) {
        live += cgraph->nodes[idx[s]]->ne[0] > 0;
    }
    if (live < 2) {
        if (mode >= 3) {
            fprintf(stderr, "%s: DECLINE live=%d n_task=%d anchor w=%s dev=%d\n", __func__, live, n_task,
                    node->src[0]->name, ggml_cuda_get_device());
        }
        return -1;
    }
    // skip span (walked BEFORE launching: a span that reaches nothing means the anchor would fall through to
    // the per-matrix path and be recomputed on top of the fused result -> decline). The span may cross views /
    // no-ops / non-compute nodes, group members, and at most ONE trailing scale of the anchor: build_lora_mm
    // appends mul(mm, w_s) after every projection (wqkv/wqkv_gate/wk/wv/... all carry one), so without this
    // the first live node after the anchor -- the scale -- would strand every wave. That node is computed here
    // inline after the launch (the fork's own fusion contract: try_fuse executes every node it spans; see the
    // mm+scale GLU variants and ggml_cuda_compute_forward at the moe-tail call site).
    int span = 0;
    int scale_idx = -1;
    for (int p = i + 1, next = 1; p < cgraph->n_nodes; ++p) {
        ggml_tensor * m = cgraph->nodes[p];
        if (ggml_cuda_is_view_or_noop(m) || (m->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            span = p - i;
            continue;
        }
        if (next < n_task && idx[next] == p) {
            ++next;
            span = p - i;
            continue;
        }
        if (scale_idx < 0 && (m->op == GGML_OP_MUL || m->op == GGML_OP_SCALE)) {
            const ggml_tensor * a = m->src[0];
            const ggml_tensor * b = m->src[1];
            const bool a_from = a == node || (a != nullptr && a->view_src == node);
            const bool b_from = b == node || (b != nullptr && b->view_src == node);
            const ggml_tensor * other = a_from && !b_from ? b : a;
            const bool other_leaf = other == nullptr || other->op == GGML_OP_NONE ||
                (other->view_src != nullptr && other->view_src->op == GGML_OP_NONE);
            if (mode >= 3) {
                static std::atomic<int> logged_scales{0};
                const int sd = logged_scales.fetch_add(1, std::memory_order_relaxed);
                if (sd < 40) {
                    fprintf(stderr, "%s: SCALECHK d=%d anchor=%s m_op=%d a=%s(b->op %d) b=%s(b->op %d) "
                            "a_from=%d b_from=%d other_op=%d leaf=%d\n", __func__, sd, node->name, (int) m->op,
                            a ? a->name : "(null)", a ? (int) a->op : -1, b ? b->name : "(null)",
                            b ? (int) b->op : -1, (int) a_from, (int) b_from,
                            other ? (int) other->op : -2, (int) other_leaf);
                }
            }
            if ((a_from || b_from) && other_leaf) {
                // computed early at node i: nothing between may write its memory, and no between-node input
                // may alias it (same aliasing rules the member-joining check applies to member outputs)
                bool scale_safe = true;
                for (int k = i; scale_safe && k < p; ++k) {
                    const ggml_tensor * t = cgraph->nodes[k];
                    if (k == i || ggml_cuda_is_view_or_noop(t)) {
                        continue;  // the anchor produced m's input; its output range ALIASING m's output range is
                                   // the expected in-place elementwise pattern (the anchor's only consumer is m,
                                   // so the allocator reuses its buffer) and is safe: read, then write, in order
                    }
                    scale_safe = !ggml_cuda_ranges_overlap(t, m) && !ggml_cuda_ranges_overlap(t, y);
                    for (int q = 0; scale_safe && q < GGML_MAX_SRC; ++q) {
                        scale_safe = !ggml_cuda_ranges_overlap(t->src[q], m);
                    }
                }
                if (scale_safe) {
                    scale_idx = p;
                    span = p - i;
                    continue;
                }
            }
        }
        break;
    }
    if (span < 1) {
        if (mode >= 3) {
            fprintf(stderr, "%s: DECLINE span0 n_task=%d anchor w=%s dev=%d next_op=%d\n", __func__, n_task,
                    node->src[0]->name, ggml_cuda_get_device(),
                    i + 1 < cgraph->n_nodes ? (int) cgraph->nodes[i + 1]->op : -1);
        }
        return -1;
    }
    for (int s = 0; s < n_task; ++s) {
        const ggml_tensor * mm = cgraph->nodes[idx[s]];
        const ggml_tensor * w  = mm->src[0];
        descs[s] = fusedproj_sm70::fused_proj_desc{
            w->data, (float *) mm->data, (int) mm->ne[0], (int) w->ne[0],
            (int64_t) (w->nb[1] / ggml_type_size(GGML_TYPE_BF16)), (int64_t) (mm->nb[1] / sizeof(float))};
    }
    if (!fusedproj_sm70::fused_projection_sm70_launch_static<max_tasks>(
                descs, n_task, (const float *) y->data, (int) node->src[0]->ne[0], ncols,
                (int64_t) (y->nb[1] / sizeof(float)), cuda_ctx->stream())) {
        return -1;  // requirement violation (alignment/stride/k%8): per-matrix paths, nothing marked yet
    }
    if (mode >= 2) {
        const int g = logged_groups.fetch_add(1, std::memory_order_relaxed);
        if (g < 96 || (g % 100000) == 0) {
            fprintf(stderr, "%s: fused_proj group=%d n_task=%d k_dim=%lld ncols=%d dev=%d span=%d scale=%d\n",
                    __func__, g, n_task, (long long) node->src[0]->ne[0], ncols, ggml_cuda_get_device(), span,
                    scale_idx >= 0);
            for (int s = 0; s < n_task; ++s) {
                const ggml_tensor * mm = cgraph->nodes[idx[s]];
                fprintf(stderr, "%s:   [%d] %-40s w=%-40s n_out=%lld\n", __func__, s, mm->name,
                        mm->src[0]->name, (long long) mm->ne[0]);
            }
        }
    }
    // the anchor's trailing scale node is spanned: execute it now, ordered after the fused launch
    if (scale_idx >= 0 && !ggml_cuda_compute_forward(*cuda_ctx, cgraph->nodes[scale_idx])) {
        GGML_LOG_ERROR("%s: fused_proj scale node failed: %s (%s)\n", __func__,
                cgraph->nodes[scale_idx]->name, ggml_op_name(cgraph->nodes[scale_idx]->op));
    }
    for (int s = 1; s < n_task; ++s) {
        cuda_ctx->precomputed.insert(cgraph->nodes[idx[s]]);
    }
    // the fused launch wrote every member output now (some earlier than their node): drop cached quantizations
    for (int s = 0; s < n_task; ++s) {
        cuda_ctx->q8_1_cache_invalidate_range(cgraph->nodes[idx[s]]->data, ggml_nbytes(cgraph->nodes[idx[s]]), nullptr);
    }
    return span;
}

// A MoE weighted reduction whose only reader is the shared-expert tail (SIGMOID(g), MUL(sh), ADD(moe)) a few nodes
// later: it is computed by the tail's kernel instead (see ggml_cuda_op_moe_tail). Per thread, for the graph being run.
struct ggml_cuda_moe_tail_deferral {
    const ggml_cgraph * graph = nullptr;
    int                 sig_idx = -1;
    ggml_cuda_moe_weighted_reduction_match match;
};
static thread_local ggml_cuda_moe_tail_deferral ggml_cuda_moe_tail_deferred;

static bool ggml_cuda_moe_tail_defer(const ggml_cgraph * cgraph, int i, int output_idx, const ggml_cuda_moe_weighted_reduction_match & m) {
    static const bool disabled = getenv("GGML_CUDA_NO_MOE_TAIL") != nullptr;
    const ggml_tensor * red = m.dst;
    if (disabled || !ggml_node_has_n_uses(cgraph, output_idx, 1) || !ggml_is_contiguous(red) || red->ne[2] != 1 || red->ne[3] != 1 ||
            red->ne[0] != m.experts->ne[0] || red->ne[1] != m.experts->ne[2]*m.experts->ne[3]) {
        return false;
    }
    const auto overlaps = [](const ggml_tensor * x, const ggml_tensor * y) {
        const char * xb = (const char *) x->data;
        const char * yb = (const char *) y->data;
        return xb < yb + ggml_nbytes(y) && yb < xb + ggml_nbytes(x);
    };
    for (int a = output_idx + 1; a < cgraph->n_nodes && a <= output_idx + 40; ++a) {
        const ggml_tensor * n = cgraph->nodes[a];
        if (n->op == GGML_OP_UNARY && ggml_get_unary_op(n) == GGML_UNARY_OP_SIGMOID && a + 2 < cgraph->n_nodes &&
                ggml_cuda_can_fuse_elem_chain(cgraph, a, { GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_ADD })) {
            const ggml_tensor * mul = cgraph->nodes[a + 1];
            const ggml_tensor * add = cgraph->nodes[a + 2];
            const ggml_tensor * sh  = mul->src[0] == n ? mul->src[1] : mul->src[0];
            const ggml_tensor * mo  = add->src[0] == mul ? add->src[1] : add->src[0];
            const ggml_tensor * g   = n->src[0];
            if (mo != red || !ggml_are_same_shape(add, red) || !ggml_is_contiguous(add) || !ggml_are_same_shape(sh, add) ||
                    !ggml_is_contiguous(sh) || sh->type != GGML_TYPE_F32 || g->type != GGML_TYPE_F32 || !ggml_is_contiguous(g) ||
                    ggml_nelements(g) != add->ne[1]) {
                return false;
            }
            // the tail reads the expert results later and writes while other threads read them
            const ggml_tensor * inputs[3] = { m.experts, m.weights, m.expert_scale };
            for (const ggml_tensor * in : inputs) {
                if (in == nullptr) {
                    continue;
                }
                if (overlaps(add, in)) {
                    return false;
                }
                for (int k = output_idx + 1; k < a; ++k) {
                    const ggml_tensor * w = cgraph->nodes[k];
                    if (!ggml_cuda_is_view_or_noop(w) && w->data != nullptr && overlaps(w, in)) {
                        return false;
                    }
                }
            }
            if (overlaps(add, sh) && add->data != sh->data) {
                return false;
            }
            // the gate input is read per token while other threads write add
            if (overlaps(add, g)) {
                return false;
            }
            ggml_cuda_moe_tail_deferred.graph   = cgraph;
            ggml_cuda_moe_tail_deferred.sig_idx = a;
            ggml_cuda_moe_tail_deferred.match   = m;
            GGML_UNUSED(i);
            return true;
        }
        if (n == nullptr) {
            break;
        }
    }
    return false;
}

const void * ggml_cuda_stage_if_overlap(ggml_backend_cuda_context & ctx, const ggml_tensor * dst, const ggml_tensor * src,
        ggml_cuda_pool_alloc<char> & tmp) {
    if (src == nullptr || !ggml_cuda_ranges_overlap(dst, src)) {
        return src ? src->data : nullptr;
    }
    const size_t n = ggml_nbytes(src);
    tmp.alloc(n);
    CUDA_CHECK(cudaMemcpyAsync(tmp.get(), src->data, n, cudaMemcpyDeviceToDevice, ctx.stream()));
    return tmp.get();
}

// Fused epilogues write a GLU/bias node whose op_params describe the
// epilogue, not the routed matmul. Keep the original MMID expert range in a
// launch-local descriptor so fusion cannot index global IDs into a local bank.
static void ggml_cuda_mul_mat_vec_q_fused_route(
        ggml_backend_cuda_context & ctx, const ggml_tensor * route,
        const ggml_tensor * weights, const ggml_tensor * input, const ggml_tensor * ids,
        ggml_tensor * output, const ggml_cuda_mm_fusion_args_host * fusion) {
    if (ggml_cuda_moe_fused_t16_requested() && ggml_ep_private::enabled(route) &&
            (output->ne[2] == 12 || output->ne[2] == 16) && ggml_cuda_info().devices[ctx.device].cc == GGML_CUDA_CC_VOLTA &&
            ggml_cuda_moe_fused_t16_type(weights->type) && fusion &&
            !fusion->x_bias && !fusion->gate_bias && !fusion->x_scale && !fusion->gate_scale) {
        std::array<ggml_moe_t16_private::slice, 2> slices;
        GGML_ASSERT(ggml_moe_t16_private::prepare(route, input, ids, output, slices));
        for (auto & slice : slices) {
            ggml_cuda_mul_mat_vec_q_fused_route(ctx, &slice.route, weights,
                    &slice.input, &slice.ids, &slice.output, fusion);
        }
        ggml_cuda_optimization_path_log(ctx.device, "ep_fused_mmvq_t16", true, true,
                "original_T8_plus_T4_or_T8_same_stream_EP_range_preserved", output->ne[2], weights->ne[1], weights->ne[2]);
        return;
    }
    if (!ggml_ep_private::enabled(route)) {
        ggml_cuda_mul_mat_vec_q(ctx, weights, input, ids, output, fusion);
        return;
    }
    GGML_ASSERT(ids && route->src[2] == ids);
    ggml_tensor routed_output = *output;
    routed_output.op = GGML_OP_MUL_MAT_ID;
    routed_output.src[0] = const_cast<ggml_tensor *>(weights);
    routed_output.src[1] = const_cast<ggml_tensor *>(input);
    routed_output.src[2] = const_cast<ggml_tensor *>(ids);
    ggml_ep_private::set(&routed_output, ggml_ep_private::offset(route),
            ggml_ep_private::global_experts(route));
    ggml_cuda_optimization_path_log(ctx.device, "ep_fused_mmvq", true, true,
            "original_MMID_route_preserved_for_fused_epilogue", output->ne[2], weights->ne[1], weights->ne[2]);
    ggml_cuda_mul_mat_vec_q(ctx, weights, input, ids, &routed_output, fusion);
}

static int ggml_cuda_try_fuse(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {

    static bool disable_fusion = getenv("GGML_CUDA_DISABLE_FUSION") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_FUSION"));
    if (disable_fusion) {
        return 0;
    }

    ggml_tensor * node = cgraph->nodes[i];

    // Qwen4Exp prompt indexer: fuse RELU -> PERMUTE(head-first) -> CONT ->
    // SUM_ROWS for the marked four-head score reduction.  This is opt-in and
    // private-marked so the generic graph remains unchanged by default.
    {
        static const bool qsa_pp_reduce = [] {
            const char * env = getenv("GGML_CUDA_QSA_PP_REDUCE");
            return env != nullptr && std::atoi(env) != 0;
        }();
        static constexpr int32_t qsa_pp_marker = 0x51535050; // "QSPP"
        const int n_nodes = cgraph->n_nodes;

        if (qsa_pp_reduce && i + 3 < n_nodes &&
                node->op == GGML_OP_UNARY && ggml_get_unary_op(node) == GGML_UNARY_OP_RELU &&
                cgraph->nodes[i + 1]->op == GGML_OP_PERMUTE &&
                cgraph->nodes[i + 1]->src[0] == node &&
                cgraph->nodes[i + 2]->op == GGML_OP_CONT &&
                cgraph->nodes[i + 2]->src[0] == cgraph->nodes[i + 1] &&
                cgraph->nodes[i + 3]->op == GGML_OP_SUM_ROWS &&
                cgraph->nodes[i + 3]->src[0] == cgraph->nodes[i + 2] &&
                cgraph->nodes[i + 3]->op_params[0] == qsa_pp_marker &&
                node->src[0] != nullptr && node->src[0]->type == GGML_TYPE_F32 &&
                cgraph->nodes[i + 3]->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(node->src[0]) &&
                node->src[0]->ne[0] > 0 && node->src[0]->ne[1] == 4 &&
                node->src[0]->ne[2] > 1 && node->src[0]->ne[3] > 0 &&
                ggml_nelements(node->src[0]) ==
                    4*node->src[0]->ne[0]*node->src[0]->ne[2]*node->src[0]->ne[3] &&
                ggml_nelements(cgraph->nodes[i + 3]) ==
                    node->src[0]->ne[0]*node->src[0]->ne[2]*node->src[0]->ne[3] &&
                ggml_check_edges(cgraph, i, {{1, 0, 0}, {2, 0, 1}, {3, 0, 2}}) &&
                ggml_can_fuse_subgraph(cgraph, i,
                        { GGML_OP_UNARY, GGML_OP_PERMUTE, GGML_OP_CONT, GGML_OP_SUM_ROWS },
                        { i + 3 })) {
            int out_nodes[] = { i + 3 };
            if (ggml_cuda_check_fusion_memory_ranges(cgraph, i, 4, out_nodes, 1)) {
                static const bool trace = [] {
                    const char * env = getenv("GGML_CUDA_QSA_PP_TRACE");
                    return env != nullptr && std::atoi(env) != 0;
                }();
                static std::atomic<bool> trace_reported = false;
                bool expected = false;
                if (trace && trace_reported.compare_exchange_strong(expected, true)) {
                    GGML_LOG_INFO("QSA PP fusion hit: blocks=%" PRId64 " queries=%" PRId64 " streams=%" PRId64 "\n",
                            node->src[0]->ne[0], node->src[0]->ne[2], node->src[0]->ne[3]);
                }
                ggml_cuda_op_qsa_relu4_sum(*cuda_ctx, node, cgraph->nodes[i + 3]);
                return 3;
            }
        }
    }

    {
        ggml_cuda_moe_tail_deferral & d = ggml_cuda_moe_tail_deferred;
        if (d.graph != nullptr) {
            GGML_ASSERT(d.graph == cgraph && i <= d.sig_idx && "deferred MoE reduction was not consumed");
            if (i == d.sig_idx) {
                ggml_tensor * mul = cgraph->nodes[i + 1];
                ggml_tensor * add = cgraph->nodes[i + 2];
                const ggml_tensor * sh = mul->src[0] == node ? mul->src[1] : mul->src[0];
                ggml_cuda_op_moe_tail(*cuda_ctx, d.match.experts, d.match.expert_scale, d.match.weights, node->src[0], sh, add);
                d = {};
                return 2;
            }
        }
    }

    // CONCAT followed by copies of windows of it (conv state slots): one kernel. Only views may sit in between.
    if (node->op == GGML_OP_CONCAT) {
        ggml_tensor * copies[GGML_CUDA_CONCAT_MAX_COPIES];
        int n_copies = 0;
        int last = i;
        for (int j = i + 1; j < cgraph->n_nodes && n_copies < GGML_CUDA_CONCAT_MAX_COPIES; ++j) {
            ggml_tensor * nj = cgraph->nodes[j];
            if (nj->op == GGML_OP_CPY && nj->src[0]->view_src == node && (nj->flags & GGML_TENSOR_FLAG_COMPUTE)) {
                copies[n_copies++] = nj;
                last = j;
                continue;
            }
            if (ggml_cuda_is_view_or_noop(nj) && nj->view_src != node) {
                continue;
            }
            if (nj->op == GGML_OP_VIEW && nj->view_src == node) {
                continue; // the window a copy reads
            }
            break;
        }
        if (n_copies > 0 && ggml_cuda_concat_copies_match(node, copies, n_copies)) {
            ggml_cuda_op_concat_copies(*cuda_ctx, node, copies, n_copies);
            return last - i;
        }
    }

    // small f32/bf16 GEMVs reading the same activations: one launch (router + shared-expert gate, alpha + beta, ...)
    {
        const int n = ggml_cuda_try_mmvf_multi(cuda_ctx, cgraph, i);
        if (n >= 0) {
            return n;
        }
    }

    // [SM70_FUSED_PROJ] BF16 projection waves (same y): one fused launch per wave (V100, default on after parity)
    {
        const int n = ggml_cuda_try_fused_proj(cuda_ctx, cgraph, i);
        if (n >= 0) {
            return n;
        }
    }

    // small GEMV (mmvf) followed by its activation: sigmoid, or softplus(x + b[row]) * a[row] (GDN beta and gate),
    //     applied in the GEMV epilogue
    if (node->op == GGML_OP_MUL_MAT && ggml_get_op_params_i32(node, 1) != GGML_HINT_SRC0_IS_HADAMARD) {
        const ggml_tensor * src0 = node->src[0];
        const ggml_tensor * src1 = node->src[1];
        const int cc = ggml_cuda_info().devices[cuda_ctx->device].cc;
        const auto from_mm = [&](const ggml_tensor * t) {
            return (t == node || t->view_src == node) && ggml_is_contiguous(t) && ggml_nelements(t) == ggml_nelements(node);
        };
        const auto per_row = [&](const ggml_tensor * t, const ggml_tensor * like) {
            return t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && ggml_nelements(t) == node->ne[0] && like->ne[0] == node->ne[0];
        };
        if ((src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16) && src0->view_src == nullptr &&
                src1->type == GGML_TYPE_F32 && node->type == GGML_TYPE_F32 && ggml_is_contiguous(node) &&
                node->ne[2] == 1 && node->ne[3] == 1 && src1->ne[1] <= MMVF_MAX_BATCH_SIZE &&
                ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, src1->ne[1])) {
            int js[3];
            int n = 0;
            for (int j = i + 1; j < cgraph->n_nodes && j <= i + 8 && n < 3; ++j) {
                if (ggml_cuda_is_view_or_noop(cgraph->nodes[j])) {
                    continue;
                }
                js[n++] = j;
            }
            ggml_tensor * n0 = n > 0 ? cgraph->nodes[js[0]] : nullptr;
            ggml_cuda_mm_fusion_args_host fusion{};
            ggml_tensor * last = nullptr;
            int last_j = -1;
            const ggml_tensor * roots[3] = { node, nullptr, nullptr };
            int n_roots = 1;
            // (a one-row gate, as the shared expert's, is left to the sigmoid-mul-add elem chain that follows it)
            if (n0 && n0->op == GGML_OP_UNARY && ggml_get_unary_op(n0) == GGML_UNARY_OP_SIGMOID && from_mm(n0->src[0]) &&
                    node->ne[0] > 1) {
                fusion.post_op = 1;
                last   = n0;
                last_j = js[0];
            } else if (n >= 3 && n0->op == GGML_OP_ADD) {
                ggml_tensor * sp  = cgraph->nodes[js[1]];
                ggml_tensor * mul = cgraph->nodes[js[2]];
                const ggml_tensor * pre = from_mm(n0->src[0]) ? n0->src[1] : from_mm(n0->src[1]) ? n0->src[0] : nullptr;
                const ggml_tensor * a   = mul->op != GGML_OP_MUL ? nullptr : mul->src[0] == sp ? mul->src[1] : mul->src[1] == sp ? mul->src[0] : nullptr;
                if (pre && per_row(pre, n0) && ggml_is_contiguous(n0) &&
                        sp->op == GGML_OP_UNARY && ggml_get_unary_op(sp) == GGML_UNARY_OP_SOFTPLUS && sp->src[0] == n0 &&
                        a && a != sp && per_row(a, mul) && (sp->flags & GGML_TENSOR_FLAG_COMPUTE) &&
                        (mul->flags & GGML_TENSOR_FLAG_COMPUTE)) {
                    fusion.post_op  = 2;
                    fusion.post_pre = pre;
                    fusion.post_mul = a;
                    last   = mul;
                    last_j = js[2];
                    roots[n_roots++] = n0;
                    roots[n_roots++] = sp;
                }
            }
            if (last && last->type == GGML_TYPE_F32 && ggml_is_contiguous(last) && ggml_nelements(last) == ggml_nelements(node) &&
                    (n0->flags & GGML_TENSOR_FLAG_COMPUTE) && ggml_cuda_group_private(cgraph, i, last_j, roots, n_roots)) {
                ggml_tensor out = *node; // the GEMV layout, written where the activation result lives
                out.data = last->data;
                // the GEMV writes last while every block still reads the whole activation (and the per-row vectors); the
                // unfused graph is done with them by then, so the allocator may have put last on top of one: read a copy
                ggml_cuda_pool_alloc<char> t1(cuda_ctx->pool()), tp(cuda_ctx->pool()), tm(cuda_ctx->pool());
                ggml_tensor src1_s = *src1;
                src1_s.data = (void *) ggml_cuda_stage_if_overlap(*cuda_ctx, last, src1, t1);
                ggml_tensor pre_s, mul_s;
                if (fusion.post_pre) {
                    pre_s = *fusion.post_pre;
                    pre_s.data = (void *) ggml_cuda_stage_if_overlap(*cuda_ctx, last, fusion.post_pre, tp);
                    fusion.post_pre = &pre_s;
                }
                if (fusion.post_mul) {
                    mul_s = *fusion.post_mul;
                    mul_s.data = (void *) ggml_cuda_stage_if_overlap(*cuda_ctx, last, fusion.post_mul, tm);
                    fusion.post_mul = &mul_s;
                }
                ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, &src1_s, nullptr, &out, &fusion);
                return last_j - i;
            }
        }
    }

    // strided windows of a GET_ROWS (q8_0) or RELU output summed by an ADD chain, then SCALE or a bias ADD: one kernel
    static const bool no_strided_sum = getenv("GGML_CUDA_NO_STRIDED_SUM") != nullptr;
    static const bool no_expand_rows = getenv("GGML_CUDA_NO_EXPAND_ROWS") != nullptr;
    if (!no_strided_sum && (node->op == GGML_OP_GET_ROWS || (node->op == GGML_OP_UNARY && ggml_get_unary_op(node) == GGML_UNARY_OP_RELU))) {
        const int n = ggml_cuda_strided_sum_try(*cuda_ctx, cgraph, i);
        if (n > 0) {
            return n;
        }
    }

    if (!no_expand_rows && node->op == GGML_OP_CONT) {
        const int n = ggml_cuda_expand_rows_try(*cuda_ctx, cgraph, i);
        if (n > 0) {
            return n;
        }
    }

    // hc combine, then the next hc mix-down of its result: the combine joins the mix-down's quantize kernel
    if (node->op == GGML_OP_HC_FUSED && ggml_get_op_params_i32(node, 0) == 2) {
        int j = i + 1;
        while (j < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[j])) {
            ++j;
        }
        if (j < cgraph->n_nodes && cgraph->nodes[j]->op == GGML_OP_HC_FUSED && (cgraph->nodes[j]->flags & GGML_TENSOR_FLAG_COMPUTE) &&
                cgraph->nodes[j]->src[0] == node && ggml_cuda_op_hc_combine_mix_down(*cuda_ctx, node, cgraph->nodes[j])) {
            return j - i;
        }
    }

    // SSM_CONV [+ bias ADD] + SILU, then RMS_NORM + SCALE of its q and k head windows (GDN): one kernel
    if (node->op == GGML_OP_SSM_CONV) {
        ggml_tensor * bias_add = nullptr;
        ggml_tensor * silu     = nullptr;
        if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_ADD, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
            bias_add = cgraph->nodes[i + 1];
            silu     = cgraph->nodes[i + 2];
        } else if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
            silu = cgraph->nodes[i + 1];
        }
        const ggml_tensor * out = silu ? silu : node;
        int j = i + (bias_add ? 2 : silu ? 1 : 0);
        ggml_tensor * rms[2];
        ggml_tensor * scl[2];
        int n = 0;
        int end = j;
        while (n < 2) {
            int a = j + 1;
            while (a < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[a])) {
                ++a;
            }
            int b = a + 1;
            while (b < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[b])) {
                ++b;
            }
            if (b >= cgraph->n_nodes || cgraph->nodes[a]->op != GGML_OP_RMS_NORM || cgraph->nodes[b]->op != GGML_OP_SCALE ||
                    cgraph->nodes[b]->src[0] != cgraph->nodes[a] || cgraph->nodes[a]->src[0]->view_src != out ||
                    !(cgraph->nodes[a]->flags & GGML_TENSOR_FLAG_COMPUTE) || !(cgraph->nodes[b]->flags & GGML_TENSOR_FLAG_COMPUTE)) {
                break;
            }
            rms[n] = cgraph->nodes[a];
            scl[n] = cgraph->nodes[b];
            ++n;
            j = end = b;
        }
        if (n > 0) {
            const ggml_tensor * roots[2] = { rms[0], n > 1 ? rms[1] : rms[0] };
            if (ggml_cuda_group_private(cgraph, i, end, roots, n) &&
                    ggml_cuda_op_ssm_conv_norm(*cuda_ctx, node, bias_add, silu, rms, scl, n)) {
                return end - i;
            }
        }
    }

    // RMS_NORM, MUL(weight row), [reshape], multi-section ROPE: one kernel
    if (node->op == GGML_OP_RMS_NORM) {
        int j1 = i + 1;
        while (j1 < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[j1])) {
            ++j1;
        }
        int j2 = j1 + 1;
        while (j2 < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[j2])) {
            ++j2;
        }
        if (j2 < cgraph->n_nodes && j2 <= i + 8) {
            ggml_tensor * mul  = cgraph->nodes[j1];
            ggml_tensor * rope = cgraph->nodes[j2];
            const ggml_tensor * w = mul->op != GGML_OP_MUL ? nullptr : mul->src[0] == node ? mul->src[1] : mul->src[1] == node ? mul->src[0] : nullptr;
            if (w && w != node && ggml_are_same_shape(mul, node) && ggml_is_contiguous(mul) && node->type == GGML_TYPE_F32 &&
                    mul->type == GGML_TYPE_F32 && rope->op == GGML_OP_ROPE && (mul->flags & GGML_TENSOR_FLAG_COMPUTE) &&
                    (rope->flags & GGML_TENSOR_FLAG_COMPUTE) &&
                    (rope->src[0] == mul || (rope->src[0]->view_src == mul && rope->src[0]->view_offs == 0 &&
                        ggml_is_contiguous(rope->src[0]))) &&
                    // the output may reuse the MUL buffer (not read), but not overlap the norm input
                    ((const char *) rope->data + ggml_nbytes(rope) <= (const char *) node->src[0]->data ||
                     (const char *) node->src[0]->data + ggml_nbytes(node->src[0]) <= (const char *) rope->data)) {
                const ggml_tensor * roots[2] = { node, mul };
                if (ggml_cuda_group_private(cgraph, i, j2, roots, 2) &&
                        ggml_cuda_op_rms_norm_mul_rope_multi(*cuda_ctx, node, w, rope)) {
                    return j2 - i;
                }
            }
        }
    }

    // gated norm: RMS_NORM, MUL(weight), [MUL_MAT computing z, views], SIGMOID(z), MUL -> the GEMV if present, then one
    //     kernel for the rest, quantized for the next GEMV if there is one
    if (node->op == GGML_OP_RMS_NORM && i + 3 < cgraph->n_nodes) {
        ggml_tensor * mul = cgraph->nodes[i + 1];
        ggml_tensor * mm  = nullptr;
        int js = i + 2;
        for (; js < cgraph->n_nodes && js <= i + 6; ++js) {
            ggml_tensor * n = cgraph->nodes[js];
            if (n->op == GGML_OP_UNARY) {
                break;
            }
            if (n->op == GGML_OP_MUL_MAT && mm == nullptr) {
                mm = n;
                continue;
            }
            if (!ggml_cuda_is_view_or_noop(n)) {
                js = cgraph->n_nodes;
                break;
            }
        }
        if (js + 1 < cgraph->n_nodes) {
            ggml_tensor * sig = cgraph->nodes[js];
            ggml_tensor * out = cgraph->nodes[js + 1];
            const ggml_tensor * z = sig->src[0];
            const ggml_tensor * x = node->src[0];
            auto overlaps = [](const ggml_tensor * a, const ggml_tensor * b) {
                const char * ab = (const char *) a->data;
                const char * bb = (const char *) b->data;
                return ab < bb + ggml_nbytes(b) && bb < ab + ggml_nbytes(a);
            };
            const bool mm_ok = mm == nullptr || ((z == mm || z->view_src == mm) && (mm->flags & GGML_TENSOR_FLAG_COMPUTE) &&
                mm->src[1] != node && mm->src[1] != mul && !overlaps(mm, x));
            const bool compute = (mul->flags & sig->flags & out->flags & GGML_TENSOR_FLAG_COMPUTE) != 0;
            if (mm_ok && compute && ggml_node_has_n_uses(cgraph, i, 1) && ggml_node_has_n_uses(cgraph, i + 1, 1) &&
                    ggml_node_has_n_uses(cgraph, js, 1) && !(node->flags & GGML_TENSOR_FLAG_OUTPUT) &&
                    !(mul->flags & GGML_TENSOR_FLAG_OUTPUT) && !(sig->flags & GGML_TENSOR_FLAG_OUTPUT) &&
                    ggml_cuda_rms_norm_gate_match(node, mul, sig, out)) {
                if (mm != nullptr) {
                    cuda_ctx->eval_i = i;
                    GGML_ASSERT(ggml_cuda_compute_forward(*cuda_ctx, mm));
                }
                block_q8_1 * yq = nullptr;
                int64_t ne10 = 0;
                static const bool no_produce = getenv("GGML_CUDA_NO_Q8_PRODUCE") != nullptr || getenv("GGML_CUDA_NO_Q8_CACHE") != nullptr;
                cuda_ctx->eval_i = js + 1; // the consumer comes after the group
                const ggml_tensor * key = no_produce ? nullptr : cuda_ctx->q8_1_consumer(out);
                if (key != nullptr && key->ne[0] % MATRIX_ROW_PADDING == 0 && x->ne[0] % QK8_1 == 0) {
                    ne10 = key->ne[0];
                    yq = (block_q8_1 *) cuda_ctx->q8_1_cache_produce(key, out, ggml_nrows(key)*ne10/QK8_1*sizeof(block_q8_1));
                }
                ggml_cuda_op_rms_norm_gate(*cuda_ctx, node, mul, sig, out, yq, ne10);
                return js + 1 - i;
            }
        }
    }

    if (node->op == GGML_OP_MUL) {
        ggml_cuda_moe_weighted_reduction_match match;
        if (ggml_cuda_match_moe_weighted_reduction(cgraph, i, match)) {
            const int output_idx = i + match.node_count - 1;
            if (ggml_cuda_check_fusion_memory_ranges(cgraph, i, match.node_count, &output_idx, 1) &&
                    ggml_cuda_moe_tail_defer(cgraph, i, output_idx, match)) {
                return match.node_count - 1;
            }
            if (ggml_cuda_check_fusion_memory_ranges(cgraph, i, match.node_count, &output_idx, 1)) {
                ggml_cuda_op_moe_weighted_reduction(
                    *cuda_ctx, match.experts, match.expert_scale, match.weights, match.dst);
                return match.node_count - 1;
            }
        }
    }

    // gated_delta_net -> cpy: scatter recurrent-state snapshots into the cache
    if (node->op == GGML_OP_GATED_DELTA_NET) {
        ggml_cuda_gated_delta_net_fused_cache fused_state_cpy;
        const int nodes_to_skip = ggml_cuda_try_gdn_cache_fusion(cgraph, i, fused_state_cpy);
        if (nodes_to_skip > 0) {
#ifdef GGML_CUDA_DEBUG
            GGML_LOG_INFO("%s: fused gated_delta_net snapshot copies for %s (skipped %d nodes)\n",
                          __func__, node->name, nodes_to_skip);
#endif
            ggml_cuda_op_gated_delta_net_fused_cache(*cuda_ctx, node, fused_state_cpy);
            return nodes_to_skip;
        }
    }

    //topk-moe
    if (cgraph->nodes[i]->op == GGML_OP_UNARY || cgraph->nodes[i]->op == GGML_OP_SOFT_MAX ||
            cgraph->nodes[i]->op == GGML_OP_ARGSORT) {
        ggml_cuda_topk_moe_args args;
        const bool              can_fuse = ggml_cuda_topk_moe_fusion(cgraph, i, args);
        std::vector<ggml_op>    ops;
        ops.reserve(13);  // max ops; avoids gcc -Wstringop-overflow false positive

        if (can_fuse) {
            const ggml_tensor * logits  = node->src[0];
            ggml_tensor *       weights = nullptr;
            ggml_tensor *       ids     = nullptr;
            const ggml_tensor * bias    = nullptr;
            const ggml_tensor * clamp   = nullptr;
            const ggml_tensor * scale   = nullptr;

            if (!args.delayed_softmax) {
                int out_nodes[2];  // nodes which can't be elided

                if (args.sigmoid) {
                    ops.insert(ops.end(), { GGML_OP_UNARY });
                } else if (args.sqrt_softplus) {
                    ops.insert(ops.end(), { GGML_OP_UNARY, GGML_OP_SQRT });
                } else {
                    ops.insert(ops.end(), { GGML_OP_SOFT_MAX });
                }
                const int i_probs = i + (int) ops.size() - 1;  // last node of the gating activation

                if (args.prob_bias) {
                    bias = cgraph->nodes[i_probs + 2]->src[1];
                    ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW,
                                            GGML_OP_GET_ROWS });
                    out_nodes[0] = i_probs + 4;
                } else {
                    ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS });
                    out_nodes[0] = i_probs + 3;
                }
                ids = cgraph->nodes[out_nodes[0]];

                if (args.norm) {
                    ops.insert(ops.end(),
                               { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
                    clamp = cgraph->nodes[i + ops.size() - 3];
                }
                if (args.scale) {
                    ops.insert(ops.end(), { GGML_OP_SCALE });
                    scale = cgraph->nodes[i + ops.size() - 1];
                }

                weights      = cgraph->nodes[i + ops.size() - 1];
                out_nodes[1] = i + ops.size() - 1;

                if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                        ggml_cuda_should_use_topk_moe(node, logits, weights, ids) &&
                        ggml_cuda_check_fusion_memory_ranges(cgraph, i, ops.size(), out_nodes, 2, /*is_topk_moe=*/true)) {
                    ggml_cuda_op_topk_moe(*cuda_ctx, logits, weights, ids, clamp, scale, bias, args);
                    return ops.size() - 1;
                }
            } else if (!args.norm && !args.prob_bias) {
                //special case gpt-oss, no norm, no bias.
                ops.insert(ops.end(), { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_RESHAPE,
                                        GGML_OP_SOFT_MAX, GGML_OP_RESHAPE });
                weights                     = cgraph->nodes[i + 5];
                ids                         = cgraph->nodes[i + 1];
                const ggml_tensor * softmax = cgraph->nodes[i + 4];

                int out_nodes[2] = { i + 1, i + 5 };
                if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                        ggml_cuda_should_use_topk_moe(softmax, logits, weights, ids) &&
                        ggml_cuda_check_fusion_memory_ranges(cgraph, i, ops.size(), out_nodes, 2, /*is_topk_moe=*/true)) {
                    ggml_cuda_op_topk_moe(*cuda_ctx, logits, weights, ids, clamp, scale, bias, args);
                    return ops.size() - 1;
                }
            }
        }
    }

    //RoPE + view + set-rows
    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
        ggml_tensor * rope     = cgraph->nodes[i];
        ggml_tensor * set_rows = cgraph->nodes[i + 2];

        ggml_cuda_op_rope_fused(*cuda_ctx, rope, set_rows);
        return 2;
    }

    // Snake activation: y = x + sin(a*x)^2 * inv_b
    // Naive 5-op decomposition emitted by frontends: mul -> sin -> sqr -> mul -> add
    if (ggml_can_fuse_subgraph(cgraph, i,
            { GGML_OP_MUL, GGML_OP_SIN, GGML_OP_SQR, GGML_OP_MUL, GGML_OP_ADD },
            { i + 4 })) {
        const ggml_tensor * mul0 = cgraph->nodes[i];
        const ggml_tensor * sqr  = cgraph->nodes[i + 2];
        const ggml_tensor * mul1 = cgraph->nodes[i + 3];
        ggml_tensor *       add  = cgraph->nodes[i + 4];

        // x carries the full activation shape, a is the broadcast operand
        const ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
        const ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

        // mul1 reads sqr and inv_b in either operand order
        const ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

        // closure check: the trailing add must read the same x as the leading mul
        const ggml_tensor * x_in_add = (add->src[0] == mul1) ? add->src[1] : add->src[0];

        // Kernel iterates over total = T * C, so x and add must be 2D and
        // a / inv_b must collapse to [1, C, 1, 1]. Higher dims are not handled.
        const bool dim_ok   = (x->ne[2]   == 1 && x->ne[3]   == 1) &&
                              (add->ne[2] == 1 && add->ne[3] == 1) &&
                              (a->ne[2]   == 1 && a->ne[3]   == 1);
        const bool shape_ok = ggml_are_same_shape(a, inv_b) && a->ne[0] == 1 && a->ne[1] == x->ne[1];

        // x is in the supported whitelist and every chain intermediate shares
        // x's type. launch_snake reads a and inv_b as const float *, so they
        // stay F32.
        const ggml_tensor * sin1 = cgraph->nodes[i + 1];
        const bool types_ok = (x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_BF16) &&
                              (a->type    == GGML_TYPE_F32) && (inv_b->type == GGML_TYPE_F32) &&
                              (mul0->type == x->type) && (sin1->type  == x->type) &&
                              (sqr->type  == x->type) && (mul1->type  == x->type) &&
                              (add->type  == x->type);

        // kernel reads x[idx] and a[c] / inv_b[c] linearly, so every operand is contiguous
        const bool contig_ok = ggml_is_contiguous(x) && ggml_is_contiguous(add) &&
                               ggml_is_contiguous(a) && ggml_is_contiguous(inv_b);

        if (types_ok && shape_ok && dim_ok && contig_ok && x_in_add == x) {
            ggml_cuda_op_snake_fused(*cuda_ctx, x, a, inv_b, add);
            return 4;
        }
    }

    // multi-(add or mul)
    if (node->op == GGML_OP_ADD || node->op == GGML_OP_MUL) {
        int     n_fuse = 0;
        ggml_op ops[8];
        std::fill(ops, ops + 8, node->op);

        for (; n_fuse <= 6; ++n_fuse) {
            if (!ggml_can_fuse(cgraph, i + n_fuse, ops + n_fuse, 2)) {
                break;
            }
            if (cgraph->nodes[i + n_fuse] != cgraph->nodes[i + n_fuse + 1]->src[0]) {
                break;
            }
            if (!ggml_are_same_layout(cgraph->nodes[i + n_fuse]->src[1], cgraph->nodes[i + n_fuse + 1]->src[1])) {
                break;
            }
        }

        n_fuse++;

        if (n_fuse > 1) {
            ggml_tensor fused_node;
            memcpy(&fused_node, node, sizeof(ggml_tensor));
            for (int j = 0; j < n_fuse - 1; ++j) {
                fused_node.src[j + 2] = cgraph->nodes[i + j + 1]->src[1];
            }
            fused_node.data = cgraph->nodes[i + n_fuse - 1]->data;
            if (node->op == GGML_OP_ADD) {
                ggml_cuda_op_fused_add(*cuda_ctx, &fused_node, n_fuse);
            } else {
                ggml_cuda_op_fused_mul(*cuda_ctx, &fused_node, n_fuse);
            }
            return n_fuse - 1;
        }
    }

    bool fused_mul_mat_vec = false;
    int  fused_node_count  = 0;

    auto get_mul_mat_scale = [](const ggml_tensor * scale_node, const ggml_tensor * mm_node) -> const ggml_tensor * {
        const bool scale_lhs_mm = scale_node->src[0] == mm_node;
        const bool scale_rhs_mm = scale_node->src[1] == mm_node;
        if (!scale_lhs_mm && !scale_rhs_mm) {
            return nullptr;
        }

        const ggml_tensor * scale = scale_lhs_mm ? scale_node->src[1] : scale_node->src[0];
        if (mm_node->src[0]->type != GGML_TYPE_NVFP4 || scale_node->type != GGML_TYPE_F32 ||
                scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) || ggml_nelements(scale) != 1 ||
                !ggml_are_same_shape(scale_node, mm_node)) {
            return nullptr;
        }

        return scale;
    };

    auto get_mul_mat_id_scale = [](const ggml_tensor * reshape, const ggml_tensor * repeat, const ggml_tensor * getrows,
            const ggml_tensor * scale_node, const ggml_tensor * mm_node) -> const ggml_tensor * {
        if (repeat->src[0] != reshape || getrows->src[0] != repeat || getrows->src[1] != mm_node->src[2]) {
            return nullptr;
        }
        if (!((scale_node->src[0] == mm_node && scale_node->src[1] == getrows) ||
                (scale_node->src[0] == getrows && scale_node->src[1] == mm_node))) {
            return nullptr;
        }

        const ggml_tensor * scale = reshape->src[0];
        if (mm_node->src[0]->type != GGML_TYPE_NVFP4 || scale_node->type != GGML_TYPE_F32 ||
                scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) || ggml_nelements(scale) != mm_node->src[0]->ne[2] ||
                !ggml_are_same_shape(scale_node, mm_node)) {
            return nullptr;
        }

        return scale;
    };

    auto get_bias_tensor = [](const ggml_tensor * bias_node, const ggml_tensor * mul_node, ggml_op op_bias) -> const ggml_tensor * {
        if (op_bias == GGML_OP_ADD) {
            if (bias_node->src[0] == mul_node) {
                return bias_node->src[1];
            }
            if (bias_node->src[1] == mul_node) {
                return bias_node->src[0];
            }
            return nullptr;
        }
        GGML_ASSERT(op_bias == GGML_OP_ADD_ID);
        GGML_ASSERT(bias_node->src[0] == mul_node);
        return bias_node->src[1];
    };

    // gate + glu + up, with optional scale/bias on both lanes.
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        if (op == GGML_OP_MUL_MAT) {
            for (const bool with_bias : { false, true }) {
                const int gate_idx       = i;
                const int gate_scale_idx = i + 1;
                const int gate_bias_idx  = with_bias ? i + 2 : -1;
                const int up_idx         = with_bias ? i + 3 : i + 2;
                const int up_scale_idx   = up_idx + 1;
                const int up_bias_idx    = with_bias ? up_idx + 2 : -1;
                const int glu_idx        = with_bias ? up_idx + 3 : up_idx + 2;

                const int out_nodes[] = { glu_idx };
                ggml_op ops[7];
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = bias_op;
                    ops[3] = op;
                    ops[4] = GGML_OP_MUL;
                    ops[5] = bias_op;
                    ops[6] = GGML_OP_GLU;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = op;
                    ops[3] = GGML_OP_MUL;
                    ops[4] = GGML_OP_GLU;
                }
                const int n_ops = with_bias ? 7 : 5;

                if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                        !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                    continue;
                }

                ggml_tensor * gate_n       = cgraph->nodes[gate_idx];
                ggml_tensor * gate_scale_n = cgraph->nodes[gate_scale_idx];
                ggml_tensor * gate_out_n   = with_bias ? cgraph->nodes[gate_bias_idx] : gate_scale_n;
                ggml_tensor * up_n         = cgraph->nodes[up_idx];
                ggml_tensor * up_scale_n   = cgraph->nodes[up_scale_idx];
                ggml_tensor * up_out_n     = with_bias ? cgraph->nodes[up_bias_idx] : up_scale_n;
                const ggml_tensor * glu = cgraph->nodes[glu_idx];

                if (!ggml_cuda_should_fuse_mul_mat(up_n, gate_n, glu,
                        with_bias ? up_out_n : nullptr, with_bias ? gate_out_n : nullptr, up_scale_n, gate_scale_n)) {
                    continue;
                }

                const ggml_tensor * gate_scale = get_mul_mat_scale(gate_scale_n, gate_n);
                const ggml_tensor * up_scale   = get_mul_mat_scale(up_scale_n, up_n);
                if (!gate_scale || !up_scale) {
                    continue;
                }

                const ggml_tensor * up_bias   = with_bias ? get_bias_tensor(up_out_n, up_scale_n, bias_op) : nullptr;
                const ggml_tensor * gate_bias = with_bias ? get_bias_tensor(gate_out_n, gate_scale_n, bias_op) : nullptr;
                if (with_bias && (!ggml_are_same_shape(gate_out_n->src[0], gate_out_n->src[1]) ||
                        !ggml_are_same_shape(up_out_n->src[0], up_out_n->src[1]))) {
                    continue;
                }

                const ggml_tensor * src0 = up_n->src[0];
                const ggml_tensor * src1 = up_n->src[1];
                const ggml_tensor * ids  = up_n->src[2];

                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate       = gate_n->src[0];
                fusion_data.x_bias     = up_bias;
                fusion_data.gate_bias  = gate_bias;
                fusion_data.x_scale    = up_scale;
                fusion_data.gate_scale = gate_scale;
                fusion_data.glu_op     = ggml_get_glu_op(glu);
                fusion_data.glu_limit  = ggml_get_op_params_f32(glu, 3);

                if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n)) {
                    ggml_cuda_mul_mat_vec_q_fused_route(*cuda_ctx, node, src0, src1, ids, cgraph->nodes[glu_idx], &fusion_data);
                    fused_mul_mat_vec = true;
                    fused_node_count  = n_ops;
                    break;
                }
            }

            if (fused_mul_mat_vec) {
                break;
            }
        } else {
            for (const bool with_bias : { false, true }) {
                const int gate_idx       = i;
                const int gate_scale_idx = i + 4;
                const int gate_bias_idx  = with_bias ? i + 5 : -1;
                const int up_idx         = with_bias ? i + 6 : i + 5;
                const int up_scale_idx   = up_idx + 4;
                const int up_bias_idx    = with_bias ? up_idx + 5 : -1;
                const int glu_idx        = with_bias ? up_idx + 6 : up_idx + 5;

                const int out_nodes[] = { glu_idx };
                ggml_op ops[13];
                if (with_bias) {
                    ops[0]  = op;
                    ops[1]  = GGML_OP_RESHAPE;
                    ops[2]  = GGML_OP_REPEAT;
                    ops[3]  = GGML_OP_GET_ROWS;
                    ops[4]  = GGML_OP_MUL;
                    ops[5]  = bias_op;
                    ops[6]  = op;
                    ops[7]  = GGML_OP_RESHAPE;
                    ops[8]  = GGML_OP_REPEAT;
                    ops[9]  = GGML_OP_GET_ROWS;
                    ops[10] = GGML_OP_MUL;
                    ops[11] = bias_op;
                    ops[12] = GGML_OP_GLU;
                } else {
                    ops[0]  = op;
                    ops[1]  = GGML_OP_RESHAPE;
                    ops[2]  = GGML_OP_REPEAT;
                    ops[3]  = GGML_OP_GET_ROWS;
                    ops[4]  = GGML_OP_MUL;
                    ops[5]  = op;
                    ops[6]  = GGML_OP_RESHAPE;
                    ops[7]  = GGML_OP_REPEAT;
                    ops[8]  = GGML_OP_GET_ROWS;
                    ops[9]  = GGML_OP_MUL;
                    ops[10] = GGML_OP_GLU;
                }
                const int n_ops = with_bias ? 13 : 11;

                if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                        !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                    continue;
                }

                ggml_tensor * gate_n       = cgraph->nodes[gate_idx];
                ggml_tensor * gate_scale_n = cgraph->nodes[gate_scale_idx];
                ggml_tensor * gate_out_n   = with_bias ? cgraph->nodes[gate_bias_idx] : gate_scale_n;
                ggml_tensor * up_n         = cgraph->nodes[up_idx];
                ggml_tensor * up_scale_n   = cgraph->nodes[up_scale_idx];
                ggml_tensor * up_out_n     = with_bias ? cgraph->nodes[up_bias_idx] : up_scale_n;
                const ggml_tensor * glu = cgraph->nodes[glu_idx];

                if (!ggml_cuda_should_fuse_mul_mat(up_n, gate_n, glu,
                        with_bias ? up_out_n : nullptr, with_bias ? gate_out_n : nullptr, up_scale_n, gate_scale_n)) {
                    continue;
                }

                const ggml_tensor * gate_scale = get_mul_mat_id_scale(cgraph->nodes[gate_idx + 1], cgraph->nodes[gate_idx + 2],
                        cgraph->nodes[gate_idx + 3], gate_scale_n, gate_n);
                const ggml_tensor * up_scale = get_mul_mat_id_scale(cgraph->nodes[up_idx + 1], cgraph->nodes[up_idx + 2],
                        cgraph->nodes[up_idx + 3], up_scale_n, up_n);
                if (!gate_scale || !up_scale) {
                    continue;
                }

                const ggml_tensor * up_bias   = with_bias ? get_bias_tensor(up_out_n, up_scale_n, bias_op) : nullptr;
                const ggml_tensor * gate_bias = with_bias ? get_bias_tensor(gate_out_n, gate_scale_n, bias_op) : nullptr;

                const ggml_tensor * src0 = up_n->src[0];
                const ggml_tensor * src1 = up_n->src[1];
                const ggml_tensor * ids  = up_n->src[2];

                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate       = gate_n->src[0];
                fusion_data.x_bias     = up_bias;
                fusion_data.gate_bias  = gate_bias;
                fusion_data.x_scale    = up_scale;
                fusion_data.gate_scale = gate_scale;
                fusion_data.glu_op     = ggml_get_glu_op(glu);
                fusion_data.glu_limit  = ggml_get_op_params_f32(glu, 3);

                if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n)) {
                    ggml_cuda_mul_mat_vec_q_fused_route(*cuda_ctx, node, src0, src1, ids, cgraph->nodes[glu_idx], &fusion_data);
                    fused_mul_mat_vec = true;
                    fused_node_count  = n_ops;
                    break;
                }
            }

            if (fused_mul_mat_vec) {
                break;
            }
        }

        if (ggml_cuda_can_fuse(cgraph, i, { op, bias_op, op, bias_op, GGML_OP_GLU }, {})) {
            ggml_tensor * glu         = cgraph->nodes[i + 4];
            ggml_tensor * gate_bias_n = glu->src[0];
            ggml_tensor * up_bias_n   = glu->src[1];

            //we don't assume the order for {gate, up}. Instead infer it from the bias tensor
            ggml_tensor * gate_n = nullptr;
            ggml_tensor * up_n   = nullptr;

            if (gate_bias_n->src[0] == cgraph->nodes[i] || gate_bias_n->src[1] == cgraph->nodes[i]) {
                gate_n = cgraph->nodes[i];
                up_n   = cgraph->nodes[i + 2];
            } else if (gate_bias_n->src[0] == cgraph->nodes[i + 2] || gate_bias_n->src[1] == cgraph->nodes[i + 2]) {
                gate_n = cgraph->nodes[i + 2];
                up_n   = cgraph->nodes[i];
            } else {
                continue;
            }

            const ggml_tensor * up_bias_tensor   = get_bias_tensor(up_bias_n, up_n, bias_op);
            const ggml_tensor * gate_bias_tensor = get_bias_tensor(gate_bias_n, gate_n, bias_op);

            if (!up_bias_tensor || !gate_bias_tensor) {
                continue;
            }

            // we don't support repeating adds
            if (bias_op == GGML_OP_ADD && (!ggml_are_same_shape(gate_bias_n->src[0], gate_bias_n->src[1]) ||
                                           !ggml_are_same_shape(up_bias_n->src[0], up_bias_n->src[1]))) {
                continue;
            }

            const ggml_tensor * src0 = up_n->src[0];
            const ggml_tensor * src1 = up_n->src[1];
            const ggml_tensor * ids  = up_n->src[2];

            if (ggml_cuda_should_fuse_mul_mat_vec_f(up_n)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate_n->src[0];
                fusion_data.x_bias    = up_bias_tensor;
                fusion_data.gate_bias = gate_bias_tensor;
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 5;
                break;
            }

            if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate_n->src[0];
                fusion_data.x_bias    = up_bias_tensor;
                fusion_data.gate_bias = gate_bias_tensor;
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_q_fused_route(*cuda_ctx, node, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 5;
                break;
            }
        } else if (ggml_cuda_can_fuse(cgraph, i, { op, op, GGML_OP_GLU }, {})) {
            ggml_tensor * glu  = cgraph->nodes[i + 2];
            ggml_tensor * gate = glu->src[0];
            ggml_tensor * up   = glu->src[1];

            bool ok = (gate == cgraph->nodes[i] && up == cgraph->nodes[i + 1]) ||
                      (gate == cgraph->nodes[i + 1] && up == cgraph->nodes[i]);

            if (!ok) {
                continue;
            }

            const ggml_tensor * src0 = up->src[0];
            const ggml_tensor * src1 = up->src[1];
            const ggml_tensor * ids  = up->src[2];

            if (ggml_cuda_should_fuse_mul_mat_vec_f(up)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate->src[0];
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }

            // Volta: for small dense matrices two plain GEMVs and a GLU kernel beat the fused GEMV (16 us against ~12 us
            //     for the split 640-row shared expert)
            if (ids == nullptr && src0->ne[1] <= 512 &&
                    ggml_cuda_info().devices[cuda_ctx->device].cc == GGML_CUDA_CC_VOLTA) {
                continue;
            }
            const bool fuse_quantized_glu = ggml_cuda_should_fuse_mul_mat_vec_q(up, /*allow_batch =*/ true);
            if (ids != nullptr && ggml_is_quantized(src0->type)) {
                ggml_cuda_optimization_path_log(cuda_ctx->device, "moe_gate_up_glu_fusion", true, fuse_quantized_glu,
                        fuse_quantized_glu ? "eligible" : "shape_or_backend_ineligible",
                        up->ne[2], src0->ne[1], src0->type);
            }
            if (fuse_quantized_glu) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate->src[0];
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_q_fused_route(*cuda_ctx, node, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    fused_mul_mat_vec = false;
    fused_node_count  = 0;

    // mul_mat + scale + optional bias
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        for (const bool with_bias : { false, true }) {
            const int n_ops = op == GGML_OP_MUL_MAT ? (with_bias ? 3 : 2) : (with_bias ? 6 : 5);
            const int out_nodes[] = { i + n_ops - 1 };
            ggml_op ops[6];
            if (op == GGML_OP_MUL_MAT) {
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = bias_op;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                }
            } else {
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_RESHAPE;
                    ops[2] = GGML_OP_REPEAT;
                    ops[3] = GGML_OP_GET_ROWS;
                    ops[4] = GGML_OP_MUL;
                    ops[5] = bias_op;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_RESHAPE;
                    ops[2] = GGML_OP_REPEAT;
                    ops[3] = GGML_OP_GET_ROWS;
                    ops[4] = GGML_OP_MUL;
                }
            }

            if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                    !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                continue;
            }

            ggml_tensor * mm_node    = cgraph->nodes[i];
            ggml_tensor * scale_node = op == GGML_OP_MUL_MAT ? cgraph->nodes[i + 1] : cgraph->nodes[i + 4];
            ggml_tensor * out_node   = with_bias ? cgraph->nodes[i + n_ops - 1] : scale_node;

            const ggml_tensor * scale = nullptr;
            if (op == GGML_OP_MUL_MAT) {
                scale = get_mul_mat_scale(scale_node, mm_node);
            } else {
                scale = get_mul_mat_id_scale(cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 3], scale_node, mm_node);
            }
            if (!scale) {
                continue;
            }

            const ggml_tensor * bias = with_bias ? get_bias_tensor(out_node, scale_node, bias_op) : nullptr;
            if (with_bias && !bias) {
                continue;
            }
            if (with_bias && bias_op == GGML_OP_ADD && !ggml_are_same_shape(out_node->src[0], out_node->src[1])) {
                continue;
            }
            if (with_bias && bias_op == GGML_OP_ADD_ID && out_node->src[2] != mm_node->src[2]) {
                continue;
            }

            const ggml_tensor * src0 = mm_node->src[0];
            const ggml_tensor * src1 = mm_node->src[1];
            const ggml_tensor * ids  = mm_node->src[2];

            ggml_cuda_mm_fusion_args_host fusion_data{};
            fusion_data.x_bias  = bias;
            fusion_data.x_scale = scale;

            if (ggml_cuda_should_fuse_mul_mat_vec_q(mm_node)) {
                ggml_cuda_mul_mat_vec_q_fused_route(*cuda_ctx, node, src0, src1, ids, out_node, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = n_ops;
                break;
            }
        }
        if (fused_mul_mat_vec) {
            break;
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    // mul_mat + add
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        if (!ggml_can_fuse(cgraph, i, { op, bias_op })) {
            continue;
        }

        ggml_tensor * mm_node   = cgraph->nodes[i];
        ggml_tensor * bias_node = cgraph->nodes[i + 1];

        ggml_tensor * bias_tensor = nullptr;
        if (bias_op == GGML_OP_ADD) {
            if (bias_node->src[0] == mm_node) {
                bias_tensor = bias_node->src[1];
            } else if (bias_node->src[1] == mm_node) {
                bias_tensor = bias_node->src[0];
            } else {
                continue;
            }
        } else {
            if (bias_node->src[0] != mm_node) {
                continue;
            }
            bias_tensor = bias_node->src[1];
        }

        const ggml_tensor * src0 = mm_node->src[0];
        const ggml_tensor * src1 = mm_node->src[1];
        const ggml_tensor * ids  = mm_node->src[2];

        if (bias_op == GGML_OP_ADD_ID && bias_node->src[2] != ids) {
            continue;
        }

        if (bias_op == GGML_OP_ADD && !ggml_are_same_shape(bias_node->src[0], bias_node->src[1])) {
            continue;
        }

        ggml_cuda_mm_fusion_args_host fusion_data{};
        fusion_data.x_bias = bias_tensor;

        if (ggml_cuda_should_fuse_mul_mat_vec_f(mm_node)) {
            ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, bias_node, &fusion_data);
            fused_mul_mat_vec = true;
            fused_node_count  = 2;
            break;
        }

        if (ggml_cuda_should_fuse_mul_mat_vec_q(mm_node)) {
            ggml_cuda_mul_mat_vec_q_fused_route(*cuda_ctx, node, src0, src1, ids, bias_node, &fusion_data);
            fused_mul_mat_vec = true;
            fused_node_count  = 2;
            break;
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    static const bool elem_chain_off = getenv("GGML_CUDA_NO_ELEM_CHAIN") != nullptr;
    if (!elem_chain_off) {
        if (ggml_cuda_can_fuse_elem_chain(cgraph, i, { GGML_OP_ADD, GGML_OP_UNARY, GGML_OP_MUL }) ||
            ggml_cuda_can_fuse_elem_chain(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_ADD })) {
            ggml_cuda_op_elem_chain(*cuda_ctx, cgraph->nodes + i, 3);
            return 2;
        }
        if (ggml_cuda_can_fuse_elem_chain(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY })) {
            ggml_cuda_op_elem_chain(*cuda_ctx, cgraph->nodes + i, 2);
            return 1;
        }
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
        ggml_cuda_op_rms_norm_mul_rope_fused(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 4]);
        return 4;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE }, {})) {
        ggml_cuda_op_rms_norm_mul_rope_fused(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], nullptr);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD }, {})) {
        ggml_cuda_op_rms_norm_fused_add(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }, {})) {
        ggml_cuda_op_rms_norm_fused(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    // RMS_NORM + SCALE (no bias, contiguous output): one kernel
    if (node->op == GGML_OP_RMS_NORM && i + 1 < cgraph->n_nodes && cgraph->nodes[i + 1]->op == GGML_OP_SCALE &&
            (cgraph->nodes[i + 1]->flags & GGML_TENSOR_FLAG_COMPUTE) && ggml_node_has_n_uses(cgraph, i, 1) &&
            !(node->flags & GGML_TENSOR_FLAG_OUTPUT) && cgraph->nodes[i + 1]->src[0] == node &&
            ggml_get_op_params_f32(cgraph->nodes[i + 1], 1) == 0.0f && ggml_is_contiguous(cgraph->nodes[i + 1]) &&
            ggml_is_contiguous(node) && node->type == GGML_TYPE_F32) {
        ggml_cuda_op_rms_norm_scale(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_ADD, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
        ggml_cuda_op_ssm_conv(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
        ggml_cuda_op_ssm_conv(*cuda_ctx, node, /*bias_add_node=*/ nullptr, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SILU }) ||
        ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SIGMOID }) ||
        ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SOFTPLUS })) {
        ggml_cuda_op_unary_mul(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_SQR }, { GGML_UNARY_OP_RELU })) {
        ggml_cuda_op_relu_sqr(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE }, { GGML_UNARY_OP_TANH })) {
        ggml_cuda_op_softcap(*cuda_ctx, cgraph->nodes[i + 2], node);
        return 2;
    }

    return 0;
}

#ifdef USE_CUDA_GRAPH
struct ggml_cuda_cold_scope {
    ggml_backend_cuda_context * context = nullptr;
    ggml_meta_cold::hooks hooks;
};
static thread_local ggml_cuda_cold_scope ggml_cuda_cold;

static bool ggml_backend_cuda_cold_cohort_scope(void * opaque_backend, const ggml_meta_cold::hooks * hooks) {
    auto backend = (ggml_backend_t) opaque_backend;
    if (!ggml_backend_is_cuda(backend)) return false;
    auto ctx = (ggml_backend_cuda_context *) backend->context;
    if (!hooks) {
        if (ggml_cuda_cold.context != ctx) return false;
        ggml_cuda_cold = {};
        return true;
    }
    if (ggml_cuda_cold.context || ctx->curr_stream_no != 0 || !hooks->opaque ||
            !hooks->vote || !hooks->prepared || !hooks->launch) return false;
    ggml_cuda_cold = {ctx, *hooks};
    return true;
}
#endif

static void ggml_cuda_graph_evaluate_and_capture(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, const bool use_cuda_graph, const bool cuda_graph_update_required, const void * graph_key) {
    ggml_cuda_gdn_deferred_gathers.clear();
    cuda_ctx->q8_1_cache_clear(); // inputs may have been rewritten between graphs
    cuda_ctx->eval_graph = nullptr;
    cuda_ctx->precomputed.clear();
    bool graph_evaluated_or_captured = false;
    ggml_cuda_ft_graph_census census(cuda_ctx, cgraph->n_nodes, !use_cuda_graph || cuda_graph_update_required);

    // flag used to determine whether it is an integrated_gpu
    const bool integrated            = ggml_cuda_info().devices[cuda_ctx->device].integrated;

    ggml_cuda_stream_context & stream_ctx = cuda_ctx->stream_context();
    bool                         is_concurrent_event_active = false;
    ggml_cuda_concurrent_event * concurrent_event           = nullptr;
    bool                         should_launch_concurrent_events = false;

    const auto try_launch_concurrent_event = [&](const ggml_tensor * node) {
        if (stream_ctx.concurrent_events.find(node) != stream_ctx.concurrent_events.end()) {
            concurrent_event = &stream_ctx.concurrent_events[node];

            is_concurrent_event_active = true;

            GGML_LOG_DEBUG("Launching %d streams at %s\n", concurrent_event->n_streams, node->name);

            cudaStream_t main_stream = cuda_ctx->stream();  // this should be stream 0
            GGML_ASSERT(cuda_ctx->curr_stream_no == 0);
            CUDA_CHECK(cudaEventRecord(concurrent_event->fork_event, main_stream));

            for (int i = 1; i <= concurrent_event->n_streams; ++i) {
                cudaStream_t stream = cuda_ctx->stream(cuda_ctx->device, i);
                CUDA_CHECK(cudaStreamWaitEvent(stream, concurrent_event->fork_event));
            }
        }
    };

    while (!graph_evaluated_or_captured) {
        // Only perform the graph execution if CUDA graphs are not enabled, or we are capturing the graph.
        // With the use of CUDA graphs, the execution will be performed by the graph launch.
        if (!use_cuda_graph || cuda_graph_update_required) {
            [[maybe_unused]] int prev_i = 0;

            if (stream_ctx.concurrent_events.size() > 0) {
                should_launch_concurrent_events = true;
                for (const auto & [tensor, event] : stream_ctx.concurrent_events) {
                    should_launch_concurrent_events = should_launch_concurrent_events && event.is_valid();
                }
            }

            if (should_launch_concurrent_events) {
                // Restore original node order within each concurrent region to enable fusion within streams

                std::unordered_map<const ggml_tensor *, int> node_to_idx;
                node_to_idx.reserve(cgraph->n_nodes);
                for (int i = 0; i < cgraph->n_nodes; ++i) {
                    node_to_idx[cgraph->nodes[i]] = i;
                }

                for (auto & [fork_node, event] : stream_ctx.concurrent_events) {
                    // Find positions of all nodes from this event in the current graph
                    std::vector<int> positions;
                    positions.reserve(event.original_order.size());

                    bool all_found = true;
                    for (const ggml_tensor * orig_node : event.original_order) {
                        auto it = node_to_idx.find(orig_node);
                        if (it != node_to_idx.end()) {
                            positions.push_back(it->second);
                        } else {
                            all_found = false;
                            break;
                        }
                    }

                    if (!all_found || positions.size() != event.original_order.size()) {
                        continue;
                    }

                    // Sort positions to get contiguous range
                    std::vector<int> sorted_positions = positions;
                    std::sort(sorted_positions.begin(), sorted_positions.end());

                    bool is_contiguous = true;
                    for (size_t i = 1; i < sorted_positions.size(); ++i) {
                        if (sorted_positions[i] != sorted_positions[i-1] + 1) {
                            is_contiguous = false;
                            break;
                        }
                    }

                    if (!is_contiguous) {
                        continue;
                    }

                    // Restore original order at the sorted positions
                    int start_pos = sorted_positions[0];
                    for (size_t i = 0; i < event.original_order.size(); ++i) {
                        cgraph->nodes[start_pos + i] = const_cast<ggml_tensor *>(event.original_order[i]);
                    }
                }
            } else {
                stream_ctx.concurrent_events.clear();
            }

            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
                if (is_concurrent_event_active) {
                    GGML_ASSERT(concurrent_event);

                    if (node == concurrent_event->join_node) {
                        cuda_ctx->curr_stream_no = 0;
                        for (int i = 1; i <= concurrent_event->n_streams; ++i) {
                            // Wait on join events of forked streams in the main stream
                            CUDA_CHECK(cudaEventRecord(concurrent_event->join_events[i - 1],
                                                       cuda_ctx->stream(cuda_ctx->device, i)));
                            CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), concurrent_event->join_events[i - 1]));
                        }

                        is_concurrent_event_active = false;
                        concurrent_event           = nullptr;
                    } else {
                        GGML_ASSERT (concurrent_event->stream_mapping.find(node) != concurrent_event->stream_mapping.end());
                        cuda_ctx->curr_stream_no = concurrent_event->stream_mapping[node];
                        GGML_LOG_DEBUG("Setting stream no to %d for node %s\n", cuda_ctx->curr_stream_no, node->name);
                    }
                } else if (i - prev_i > 1) {
                    //the previous node was fused
                    const ggml_tensor * prev_node = cgraph->nodes[i - 1];
                    try_launch_concurrent_event(prev_node);

                    if (is_concurrent_event_active) {
                        cuda_ctx->curr_stream_no = concurrent_event->stream_mapping[node];
                        GGML_LOG_DEBUG("Setting stream no to %d for node %s\n", cuda_ctx->curr_stream_no, node->name);
                    }
                }

                prev_i = i;

                if (ggml_cuda_is_view_or_noop(node)) {
                    continue;
                }

                if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                    continue;
                }

                if (!cuda_ctx->precomputed.empty() && cuda_ctx->precomputed.count(node)) {
                    continue; // computed by an earlier combined launch
                }
                if (ggml_cuda_gdn_state_gather_deferred(cuda_ctx, cgraph, i)) {
                    continue;
                }

                cuda_ctx->eval_graph = cgraph;
                cuda_ctx->eval_i     = i;
                int nodes_to_skip = ggml_cuda_try_fuse(cuda_ctx, cgraph, i);
                census.record(node, nodes_to_skip);
                for (int k = i; k <= i + nodes_to_skip; ++k) {
                    // outputs written by this node or fused group: drop a cached quantization of them, except the one the
                    //     group just produced (its output is the last node)
                    cuda_ctx->q8_1_cache_invalidate_range(cgraph->nodes[k]->data, ggml_nbytes(cgraph->nodes[k]), cgraph->nodes[i + nodes_to_skip]);
                }

                if (nodes_to_skip != 0) {
#ifdef GGML_CUDA_DEBUG
                    const int last_fused = i + nodes_to_skip;
                    GGML_LOG_INFO("nodes_fused: %d, first: %s (%s), last: %s (%s)\n",
                            nodes_to_skip + 1, ggml_op_name(node->op), node->name,
                            ggml_op_name(cgraph->nodes[last_fused]->op), cgraph->nodes[last_fused]->name);
#endif
                    i += nodes_to_skip;
                    continue;
                }
#ifndef NDEBUG
                // On integrated GPUs (APUs, e.g. RDNA3.5) the scheduler may place a
                // node's output on the host-visible buffer, which the compute path
                // handles. Allow that here, mirroring the src-tensor check below.
                assert(node->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) ||
                       (integrated && ggml_backend_buft_is_cuda_host(node->buffer->buft)));
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    if (node->src[j] != nullptr) {
                        assert(node->src[j]->buffer);
                        assert(node->src[j]->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) ||
                               (integrated && ggml_backend_buft_is_cuda_host(node->src[j]->buffer->buft)));
                    }
                }
#else
                GGML_UNUSED(integrated);
#endif  // NDEBUG

                static const bool node_trace = getenv("GGML_CUDA_NODE_TRACE") != nullptr;
                if (node_trace) {
                    fprintf(stderr, "node-trace %" PRId64 " dev %d begin %s %s\n", ggml_time_us(), cuda_ctx->device, ggml_op_desc(node), node->name);
                }
                bool ok = ggml_cuda_compute_forward(*cuda_ctx, node);
                if (node_trace) {
                    fprintf(stderr, "node-trace %" PRId64 " dev %d end %s %s\n", ggml_time_us(), cuda_ctx->device, ggml_op_desc(node), node->name);
                }
                if (!ok) {
                    GGML_LOG_ERROR("%s: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
                }
                GGML_ASSERT(ok);

                if (!is_concurrent_event_active) {
                    try_launch_concurrent_event(node);
               }
            }
        }

        cuda_ctx->eval_graph = nullptr;

#ifdef USE_CUDA_GRAPH
        ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
        if (use_cuda_graph && cuda_graph_update_required) { // End CUDA graph capture
            if (graph->graph != nullptr) {
                CUDA_CHECK(cudaGraphDestroy(graph->graph));
                graph->graph = nullptr;
            }

            CUDA_CHECK(cudaStreamEndCapture(cuda_ctx->stream(), &graph->graph));
            graph_evaluated_or_captured = true; // CUDA graph has been captured

            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            if (ggml_cuda_lock_counter.fetch_sub(1, std::memory_order_relaxed) == 1) {
                ggml_cuda_lock_cv.notify_all();
            }
        } else {
            graph_evaluated_or_captured = true; // ggml graph has been directly evaluated
        }
    }

    if (use_cuda_graph) {
        ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
        const auto * cold = ggml_cuda_cold.context == cuda_ctx ? &ggml_cuda_cold.hooks : nullptr;
        const bool tail_prefill = ggml_cuda_prefill_capture_hint == 1;
        const bool tail_mtp_decode = ggml_cuda_device_tail_decode_hint == 1 && !cuda_graph_update_required;
        const bool capture_only_mode = tail_prefill &&
                ggml_cuda_graph_capture_only_requested(cuda_ctx);
        bool device_tail_launched = false;

#if defined(__linux__) && CUDART_VERSION >= 12000 && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        if (ggml_cuda_prepare_only_v436) {
            int prepared = 0;
            if (!cold && capture_only_mode && graph->instance == nullptr && graph->graph != nullptr) {
                auto & api = ggml_cuda_prepare_v436::get_api();
                if (api.available()) CUDA_CHECK(api.prepare(cuda_ctx, graph, graph->graph,
                        cuda_ctx->stream(), cuda_ctx->device, &prepared));
            }
            ggml_cuda_prepare_selected_v436 = prepared == 1;
            GGML_LOG_INFO("CUDA parallel-prepare requested=1 selected=%d device=%d graph_uid=%" PRIu64
                    " graph=%p helper_key=%p capture_changed=%d model_launches=0\n",
                    prepared == 1, cuda_ctx->device, cgraph->uid, (void *) graph->graph,
                    (void *) graph, int(cuda_graph_update_required));
            return; // Includes declines: no standard/eager launch is legal here.
        }
#endif

        // Captured-only is considered only before any standard executable exists. The
        // helper receives the actual captured graph, never a fabricated cudaGraphExec_t.
        // It may reuse a prior exact owner/key/captured entry or prepare a fresh child/controller.
        if (!cold && capture_only_mode && graph->instance == nullptr && graph->graph != nullptr) {
            int outcome = -100;
            CUDA_CHECK(ggml_cuda_device_tail_capture_only_client::launch(
                    cuda_ctx, graph, graph->graph, cuda_graph_update_required, cuda_ctx->stream(),
                    cuda_ctx->device, &outcome));
            device_tail_launched = outcome == 1;
            ggml_cuda_optimization_path_log(cuda_ctx->device, "cuda_graph_device_tail_captured_only",
                    true, device_tail_launched, device_tail_launched ?
                    "captured_graph_helper_owned_exec_no_standard_exec" : "captured_only_declined_standard_fallback",
                    cgraph->n_nodes, 0, 0);
            GGML_LOG_INFO("CUDA graph captured-only requested=1 selected=%d device=%d outcome=%d"
                    " invalidate=%d standard_exec_present=0 graph=%p graph_key=%p helper_key=%p\n",
                    int(device_tail_launched), cuda_ctx->device, outcome, int(cuda_graph_update_required),
                    (void *) graph->graph, graph_key, (void *) graph);
        }

        // Decline, non-Volta, cold collectives, and all default-off paths retain
        // the original host executable and graph launch behavior.
        if (!device_tail_launched) {
            bool created_standard_exec = false;
            if (graph->instance == nullptr) { // Create the ordinary executable on fallback only.
                const int64_t start = ggml_cuda_ft_host_timing_enabled() ? ggml_time_us() : 0;
                CUDA_CHECK(cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0));
                ggml_cuda_ft_host_record(cuda_ctx, 5, start);
                created_standard_exec = true;
            }
            if (cuda_graph_update_required && !created_standard_exec) { // A new exec already matches this capture.
                ggml_cuda_graph_update_executable(cuda_ctx, graph_key);
            }
        }

        // A cold collective cannot start while a peer is still uploading or
        // instantiating its graph. Capture-only is always excluded from this scope.
        if (cold) {
            CUDA_CHECK(cudaGraphUpload(graph->instance, cuda_ctx->stream()));
            CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));
            if (!cold->prepared(cold->opaque)) {
                GGML_ABORT("cold CUDA cohort aborted before graph launch");
            }
            cold->launch(cold->opaque, false);
        }
        const int64_t start = ggml_cuda_ft_host_timing_enabled() ? ggml_time_us() : 0;
#if defined(__linux__) && CUDART_VERSION >= 12000 && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        // Preserve ABI v1 for MTP decode and the legacy default-off prefill path.
        // An opted-in captured-only prefill declines directly to standard host launch.
        if (!device_tail_launched && !capture_only_mode && !cold &&
                (tail_prefill || tail_mtp_decode) &&
                ggml_cuda_info().devices[cuda_ctx->device].cc == GGML_CUDA_CC_VOLTA &&
                ggml_cuda_device_tail_client::requested()) {
            int outcome = -100;
            CUDA_CHECK(ggml_cuda_device_tail_client::launch(cuda_ctx, graph, graph->graph,
                    graph->instance, cuda_graph_update_required, cuda_ctx->stream(),
                    cuda_ctx->device, &outcome));
            device_tail_launched = outcome == 1;
            ggml_cuda_optimization_path_log(cuda_ctx->device, tail_mtp_decode ?
                    "cuda_graph_device_tail_mtp_decode" : "cuda_graph_device_tail", true,
                    device_tail_launched, device_tail_launched ?
                    "one_node_host_controller_tail_unchanged_graph" : "explicit_helper_admission_declined",
                    cgraph->n_nodes, 0, 0);
        }
#endif
        if (!device_tail_launched) CUDA_CHECK(cudaGraphLaunch(graph->instance, cuda_ctx->stream()));
        if (cold) cold->launch(cold->opaque, true);
        ggml_cuda_ft_host_record(cuda_ctx, 3, start);
#else
        GGML_UNUSED(graph_key);
        graph_evaluated_or_captured = true;
#endif  // USE_CUDA_GRAPH
    }
}

#ifdef USE_CUDA_GRAPH
static bool ggml_cuda_graph_set_enabled(ggml_backend_cuda_context * cuda_ctx, const void * graph_key) {
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

    if (graph->graph == nullptr) {
        if (ggml_cuda_info().devices[cuda_ctx->device].cc < GGML_CUDA_CC_VOLTA) {
            if (!graph->disable_due_to_gpu_arch) {
                GGML_LOG_DEBUG("%s: disabling CUDA graphs due to GPU architecture\n", __func__);
            }
            graph->disable_due_to_gpu_arch = true;
        }
    }

    return graph->is_enabled();
}
#endif // USE_CUDA_GRAPH

static enum ggml_status ggml_backend_cuda_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    const int64_t start = ggml_cuda_ft_host_timing_enabled() ? ggml_time_us() : 0;
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

    ggml_cuda_set_device(cuda_ctx->device);

    struct onecat_dispatch_restore {
        int previous = -1;
        bool active = false;
        void set(bool allow) {
            previous = ggml_onecat_v100::init_ready::set_dispatch_override(allow ? 1 : 0);
            active = true;
        }
        ~onecat_dispatch_restore() {
            if (active) ggml_onecat_v100::init_ready::set_dispatch_override(previous);
        }
    } onecat_scope;

    // This env gate avoids changing the default-off path. Admission reuses the
    // exact runtime selector; sparse-QSA and stock fallbacks are ignored.
    if (ggml_onecat_v100::init_ready::configured_enabled()) {
        int min_entries = 0;
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            int entries = 0;
            if (ggml_cuda_onecat_fattn_admitted(cuda_ctx->device, cgraph->nodes[i], &entries)) {
                min_entries = std::max(min_entries, entries);
            }
        }
        if (min_entries > 0) {
            const char * reason = nullptr;
            const bool ready = ggml_onecat_v100::init_ready::ensure_ready(
                    cuda_ctx->device, min_entries, cuda_ctx->stream(), &reason);
            if (ready && reason &&
                    (std::strcmp(reason, "initialized_ready") == 0 ||
                     std::strcmp(reason, "capacity_grown_ready") == 0)) {
                GGML_LOG_INFO("Onecat init-ready prepared device=%d entries=%d reason=%s\n",
                        cuda_ctx->device, min_entries, reason);
            }
            if (!ready) {
                size_t max_onecat_reserved = 0;
                size_t max_stock_required = 0;
                bool fallback_fits = true;
                for (int i = 0; i < cgraph->n_nodes; ++i) {
                    int entries = 0;
                    if (!ggml_cuda_onecat_fattn_admitted(cuda_ctx->device, cgraph->nodes[i], &entries)) continue;
                    size_t onecat_reserved = 0;
                    size_t stock_required = 0;
                    if (!ggml_cuda_onecat_fallback_workspace_fits(cuda_ctx->device,
                            cgraph->nodes[i], &onecat_reserved, &stock_required)) {
                        fallback_fits = false;
                    }
                    max_onecat_reserved = std::max(max_onecat_reserved, onecat_reserved);
                    max_stock_required = std::max(max_stock_required, stock_required);
                }
                if (!fallback_fits) {
                    GGML_LOG_ERROR("Onecat readiness declined device=%d reason=%s;"
                            " stock fallback workspace does not fit (Onecat=%zu stock=%zu);"
                            " refusing this compute\n",
                            cuda_ctx->device, reason ? reason : "unknown",
                            max_onecat_reserved, max_stock_required);
                    return GGML_STATUS_FAILED;
                }
            }
            onecat_scope.set(ready);
            if (!ready) {
                GGML_LOG_WARN("Onecat init readiness declined device=%d entries=%d reason=%s;"
                        " workspace fit checked; stock FlashAttention serves this compute\n",
                        cuda_ctx->device, min_entries, reason ? reason : "unknown");
            }
        }
    }

    bool use_cuda_graph             = false;
    bool cuda_graph_update_required = false;
    const void * graph_key = nullptr;
    ggml_cuda_graph_structural::graph_signature structural_signature;
    bool structural_signature_valid = false;
    int eager_reason = getenv("GGML_CUDA_DISABLE_GRAPHS") ? 3 : 4;
    ggml_tensor * incompatible_node = nullptr;

#ifdef USE_CUDA_GRAPH
    graph_key = ggml_cuda_graph_get_key(cuda_ctx, cgraph, &structural_signature, &structural_signature_valid, true);

    // This is an actual compute acquisition, not a readiness probe. Preserve
    // the selected live entry before cuda_graph() sweeps old entries; otherwise
    // it could evict the very exact-match graph get_key() just selected.
    const auto selected = cuda_ctx->cuda_graphs.find(graph_key);
    if (selected != cuda_ctx->cuda_graphs.end() && ggml_cuda_graph_diagnostic::touch_selected(
            ggml_time_us(), cuda_ctx->last_graph_eviction_sweep, selected->second->last_used_time)) {
        ggml_cuda_graph_structural_stats_update(cuda_ctx, [&](auto & stats) {
            if (++stats.selected_stale_touches <= 8) {
                GGML_LOG_INFO("CUDA graph selected-key touch ctx=%p device=%d would_evict_before_touch=1"
                        " actual_compute=1 readiness_lifetime_unchanged=1 count=%zu\n",
                        (void *) cuda_ctx, cuda_ctx->device, stats.selected_stale_touches);
            }
        });
    }
    ggml_cuda_graph_set_enabled(cuda_ctx, graph_key);
    ggml_cuda_ft_graph_select(cuda_ctx, cgraph, graph_key, &structural_signature, structural_signature_valid);

    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

#if defined(__linux__) && CUDART_VERSION >= 12000 && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (ggml_cuda_prepare_only_v436) {
        int can_prepare = 0;
        auto & api = ggml_cuda_prepare_v436::get_api();
        if (graph->instance == nullptr && api.available())
            CUDA_CHECK(api.can_prepare(cuda_ctx, graph, cuda_ctx->device, &can_prepare));
        if (can_prepare != 1) {
            GGML_LOG_INFO("CUDA parallel-prepare requested=1 selected=0 device=%d"
                    " reason=helper_unavailable_standard_exec_or_inflight model_launches=0\n", cuda_ctx->device);
            return GGML_STATUS_SUCCESS;
        }
    }
#endif
    if (graph->is_enabled()) {
        const bool graph_compatible = ggml_cuda_graph_check_compability(cgraph, &incompatible_node);
        if (!graph_compatible) eager_reason = 2;
        if (graph_compatible) {
            const bool properties_changed = ggml_cuda_graph_update_required(
                    cuda_ctx, cgraph, graph_key, &structural_signature, structural_signature_valid);
            // Never trust a captured-only helper entry after a descriptor change. New/changed
            // captures invalidate the same owner/key entry in the helper before its next launch.
            const bool executable_ready = graph->instance != nullptr ||
                    (!properties_changed && ggml_cuda_graph_capture_only_ready(cuda_ctx, graph));

            // Opt-in from llama.cpp-multigpu c8832a5f: a compatible changed
            // prefill graph need not wait for a second identical call. Keep all
            // compatibility, property, owner and graph-key checks above intact.
            static const bool graphs_force = [] {
                const char * env = getenv("GGML_CUDA_GRAPHS_FORCE");
                return env && strcmp(env, "1") == 0;
            }();
            const bool capture_requested = ggml_cuda_prefill_capture_hint < 0
                    ? graphs_force : ggml_cuda_prefill_capture_hint == 1;
            const bool force_binding_validated = capture_requested && structural_signature_valid;
            if (capture_requested && !force_binding_validated) {
                // The imported immediate-capture mode must not bypass the
                // stable warmup path when external ownership/bindings could
                // not be described (for example the bounded external cap).
                ggml_cuda_optimization_path_log(cuda_ctx->device, "cuda_graph_force_capture", true, false,
                        "structural_binding_signature_unavailable", cgraph->n_nodes, 0, 0);
            }
            if (force_binding_validated) {
                graph->warmup_complete = true;
                use_cuda_graph = true;
                cuda_graph_update_required = properties_changed || !executable_ready;
                ggml_cuda_optimization_path_log(cuda_ctx->device, "cuda_graph_force_capture", true, true,
                        cuda_graph_update_required ? "compatible_changed_graph_capture" : "compatible_graph_replay",
                        cgraph->n_nodes, 0, 0);
            } else if (!graph->warmup_complete) {
                eager_reason = 0;
                // Warmup: need at least 2 calls with no property change on the 2nd call
                if (!properties_changed) {
                    graph->warmup_complete = true;
                    GGML_LOG_DEBUG("%s: CUDA graph warmup complete\n", __func__);
                    use_cuda_graph = true;
                    cuda_graph_update_required = true;
                }
                // else: properties changed or first call - execute directly (use_cuda_graph stays false)
            } else {
                // Post-warmup: normal CUDA graph operation
                if (properties_changed) {
                    eager_reason = 1;
                    // Properties changed - reset warmup, execute directly until stable again
                    graph->warmup_complete = false;
                    GGML_LOG_DEBUG("%s: CUDA graph warmup reset\n", __func__);
                } else {
                    use_cuda_graph = true;
                    cuda_graph_update_required = !executable_ready;
                }
            }
        }
    }

    if (ggml_cuda_graph_structural_enabled() && use_cuda_graph && !cuda_graph_update_required &&
        graph_key != cgraph->nodes[0]) {
        ggml_cuda_graph_structural_stats_update(cuda_ctx, [](auto & stats) { ++stats.cross_descriptor_replays; });
    }
#endif // USE_CUDA_GRAPH

#ifdef USE_CUDA_GRAPH
    if (ggml_cuda_cold.context == cuda_ctx) {
        const auto & cold = ggml_cuda_cold.hooks;
        const bool capture_intent = use_cuda_graph && cuda_graph_update_required;
        const int decision = cold.vote(cold.opaque, use_cuda_graph);
        if (decision < 0) {
            GGML_ABORT("cold CUDA cohort vote aborted");
        }
        if (decision == 0) {
            // Property snapshots were already advanced above. If capture is
            // canceled, force normal stable warmup to recapture next time,
            // even when an old executable is still present in this key.
            if (capture_intent) cuda_ctx->cuda_graph(graph_key)->warmup_complete = false;
            // The existing path counters have five fixed reason categories.
            // Cohort-wide warmup is reported by Meta's explicit vote log;
            // preserve an eager member's original reason (e.g. env-disabled).
            if (use_cuda_graph) eager_reason = 0;
            use_cuda_graph = false;
            cuda_graph_update_required = false;
        }
    }
#endif


#if defined(USE_CUDA_GRAPH) && defined(__linux__) && CUDART_VERSION >= 12000 && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (ggml_cuda_prepare_only_v436 && !use_cuda_graph) {
        GGML_LOG_INFO("CUDA parallel-prepare requested=1 selected=0 device=%d"
                " reason=not_capturable model_launches=0\n", cuda_ctx->device);
        return GGML_STATUS_SUCCESS;
    }
#endif

    // A decode helper cannot observe an in-place exec update from pointer
    // identity alone. Retire all old children before recapture/update, while
    // preserving this call's standard executable path and exact binding checks.
    if (use_cuda_graph && cuda_graph_update_required &&
            ggml_cuda_device_tail_decode_hint == 1 &&
            ggml_cuda_info().devices[cuda_ctx->device].cc == GGML_CUDA_CC_VOLTA &&
            ggml_cuda_device_tail_client::requested()) {
        CUDA_CHECK(ggml_cuda_device_tail_client::drop_context(cuda_ctx));
        static thread_local unsigned decode_tail_invalidation_notices = 0;
        if (decode_tail_invalidation_notices++ < 8) {
            GGML_LOG_INFO("CUDA decode device-tail invalidation device=%d before_capture=1 standard_update_retained=1\n", cuda_ctx->device);
        }
    }

    if (use_cuda_graph && cuda_graph_update_required) {
        // Start CUDA graph capture
        {
            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            ggml_cuda_lock_counter.fetch_add(1, std::memory_order_relaxed);
        }

        CUDA_CHECK(cudaStreamBeginCapture(cuda_ctx->stream(), cudaStreamCaptureModeRelaxed));
    }

#ifdef USE_CUDA_GRAPH
    ggml_cuda_graph_log_path(cuda_ctx, cgraph, !use_cuda_graph ? 0 : cuda_graph_update_required ? 1 : 2,
                             eager_reason, incompatible_node);
#else
    GGML_UNUSED(eager_reason);
    GGML_UNUSED(incompatible_node);
#endif
#ifdef USE_CUDA_GRAPH
    ggml_cuda_compose_private::note_compute(cuda_ctx);
#endif
    const auto cublas_audit_mode = !use_cuda_graph ?
            ggml_cuda_cublas_role_audit::dispatch_mode::host_dispatch :
            cuda_graph_update_required ? ggml_cuda_cublas_role_audit::dispatch_mode::graph_capture :
            ggml_cuda_cublas_role_audit::dispatch_mode::graph_replay;
    const void * cublas_audit_graph_key = graph_key != nullptr ? graph_key : static_cast<const void *>(cgraph);
    ggml_cuda_cublas_role_audit::dispatch_scope cublas_role_scope(
            cuda_ctx, cuda_ctx->device, cublas_audit_graph_key, cublas_audit_mode);
    ggml_cuda_graph_evaluate_and_capture(cuda_ctx, cgraph, use_cuda_graph, cuda_graph_update_required, graph_key);
#ifdef USE_CUDA_GRAPH
    // Only completed target-prefill capture/replay computes create or renew a
    // lease. Test retention before asking the V2 client about captured-only
    // executable readiness, and never treat prepare-only completion as use.
    const int lease_retain_ms = ggml_cuda_graph_lease_retain_ms();
    if (lease_retain_ms > 0) {
        bool prepare_only = false;
#if defined(__linux__) && CUDART_VERSION >= 12000 && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        prepare_only = ggml_cuda_prepare_only_v436;
#endif
        const bool graph_enabled = graph != nullptr && graph->is_enabled();
        const bool prefill_graph_compute = !prepare_only && ggml_cuda_prefill_capture_hint == 1 &&
                use_cuda_graph && graph_enabled;
        if (prefill_graph_compute) {
            const bool has_executable = ggml_cuda_graph_has_executable(cuda_ctx, graph);
            if (ggml_cuda_graph_lease::should_admit_actual_compute(lease_retain_ms,
                    ggml_cuda_prefill_capture_hint, prepare_only, use_cuda_graph, graph_enabled, has_executable) &&
                    graph_key != nullptr) {
                ggml_cuda_graph_lease_touch_actual_compute(
                        cuda_ctx, graph_key, graph, ggml_time_us(), lease_retain_ms);
            }
        }
    }
#endif
    for (const std::string & line : cublas_role_scope.finish()) {
        GGML_LOG_INFO("%s\n", line.c_str());
    }
    ggml_cuda_ft_host_record(cuda_ctx, !use_cuda_graph ? 0 : cuda_graph_update_required ? 1 : 2, start, cgraph->n_nodes);
#ifdef USE_CUDA_GRAPH
    static const size_t snapshot_period = [] {
        const char * value = std::getenv("GGML_FT_HOST_SNAPSHOT_PERIOD");
        return value && std::atoll(value) > 0 ? (size_t) std::atoll(value) : 0;
    }();
    if (snapshot_period && ggml_cuda_ft_host_timing_enabled()) {
        ggml_cuda_graph_structural_stats_update(cuda_ctx, [&](auto & stats) {
            if (++stats.passes % snapshot_period == 0) {
                fprintf(stderr, "CUDA graph structural snapshot ctx=%p device=%d passes=%zu attempts=%zu built=%zu reject_external=%zu reject_graph=%zu active_matches=%zu variant_matches=%zu cross_descriptor_replays=%zu update_layout_candidates=%zu update_layout_misses=%zu reject_cap=%zu reject_cycle=%zu reject_node_extra=%zu reject_invalid=%zu signature_us=%" PRId64 " lookup_us=%" PRId64 "\n",
                        (void *) cuda_ctx, cuda_ctx->device, stats.passes, stats.attempts, stats.builds,
                        stats.reject_external, stats.reject_graph, stats.active_matches, stats.variant_matches,
                        stats.cross_descriptor_replays, stats.update_layout_candidates, stats.update_layout_misses,
                        stats.failures[size_t(ggml_cuda_graph_structural::build_failure::too_many_external_bindings)],
                        stats.failures[size_t(ggml_cuda_graph_structural::build_failure::cyclic_external_binding)],
                        stats.failures[size_t(ggml_cuda_graph_structural::build_failure::unsupported_node_extra)],
                        stats.failures[size_t(ggml_cuda_graph_structural::build_failure::duplicate_or_invalid_node)],
                        stats.signature_build_us, stats.lookup_us);
            }
        });
    }
#endif

    return GGML_STATUS_SUCCESS;
}


static int ggml_backend_cuda_prefill_hint_v436() {
#ifdef USE_CUDA_GRAPH
    return ggml_cuda_prefill_capture_hint;
#else
    return -1;
#endif
}

static enum ggml_status ggml_backend_cuda_prepare_graph_v436(
        ggml_backend_t backend, ggml_cgraph * cgraph, int * prepared) {
    if (!prepared) return GGML_STATUS_FAILED;
    *prepared = 0;
#if defined(USE_CUDA_GRAPH) && defined(__linux__) && CUDART_VERSION >= 12000 && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (!backend || !ggml_backend_is_cuda(backend) || !cgraph || cgraph->n_nodes <= 0)
        return GGML_STATUS_SUCCESS;
    auto * ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
    if (!ggml_cuda_prepare_v436::get_api().available()) return GGML_STATUS_SUCCESS;
    // A real Onecat node may initialize tables outside capture. Keep such
    // graphs on normal ordered compute; unrelated sparse/stock graphs need
    // no Onecat initializer and can use the existing prepare-only contract.
    // Query the same selector before changing any caller thread-local state.
    if (ggml_onecat_v100::init_ready::configured_enabled()) {
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            int min_entries = 0;
            if (ggml_cuda_onecat_fattn_admitted(ctx->device, cgraph->nodes[i], &min_entries)) {
                return GGML_STATUS_SUCCESS;
            }
        }
    }
    struct scope {
        int hint = ggml_cuda_prefill_capture_hint;
        bool mode = ggml_cuda_prepare_only_v436;
        bool selected = ggml_cuda_prepare_selected_v436;
        scope() { ggml_cuda_prefill_capture_hint = 1; ggml_cuda_prepare_only_v436 = true;
                  ggml_cuda_prepare_selected_v436 = false; }
        ~scope() { ggml_cuda_prefill_capture_hint = hint; ggml_cuda_prepare_only_v436 = mode;
                   ggml_cuda_prepare_selected_v436 = selected; }
    } restore;
    ggml_cuda_set_device(ctx->device);
    if (!ggml_cuda_graph_capture_only_requested(ctx)) return GGML_STATUS_SUCCESS;
    const auto status = ggml_backend_cuda_graph_compute(backend, cgraph);
    if (status == GGML_STATUS_SUCCESS) *prepared = int(ggml_cuda_prepare_selected_v436);
    return status;
#else
    GGML_UNUSED(backend); GGML_UNUSED(cgraph);
    return GGML_STATUS_SUCCESS;
#endif
}

static void ggml_backend_cuda_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    CUDA_CHECK(cudaEventRecord((cudaEvent_t)event->context, cuda_ctx->stream()));
}

static void ggml_backend_cuda_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    if (ggml_backend_is_cuda(backend)) {
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), (cudaEvent_t)event->context, 0));
    } else {
#if 0
        // untested
        auto wait_fn = [](void * user_data) {
            ggml_backend_event_t event = (ggml_backend_event_t)user_data;
            ggml_backend_event_synchronize(event);
        };

        CUDA_CHECK(cudaLaunchHostFunc(cuda_ctx->stream(), wait_fn, event));
#endif
        GGML_ABORT("fatal error");
    }
}

static void ggml_backend_cuda_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

    static const bool disable_fusion = getenv("GGML_CUDA_DISABLE_FUSION") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_FUSION"));

    auto add_alloc_deps = [&](size_t start, size_t last_node) {

        for (size_t i = start; i < last_node; ++i) {
            params->add_alloc_dep(params->user_data, cgraph->nodes[i], cgraph->nodes[last_node]);

            for (int j = 0; j < GGML_MAX_SRC; ++j) {
                if (cgraph->nodes[i]->src[j]) {
                    params->add_alloc_dep(params->user_data, cgraph->nodes[i]->src[j], cgraph->nodes[last_node]);
                }
            }
        }
    };

    if (!disable_fusion) {
        // add alloc deps for performance positive fusions. This may increase the overall compute buffer size.
        // TODO: consolidate fusion paths in graph_optimize and graph_compute
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            ggml_cuda_moe_weighted_reduction_match match;
            if (ggml_cuda_match_moe_weighted_reduction(cgraph, i, match)) {
                params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(match.experts), match.dst);
                params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(match.weights), match.dst);
                if (match.expert_scale != nullptr) {
                    params->add_alloc_dep(
                        params->user_data, const_cast<ggml_tensor *>(match.expert_scale), match.dst);
                }
                i += match.node_count - 1;
            }

            if (cgraph->nodes[i]->op == GGML_OP_UNARY || cgraph->nodes[i]->op == GGML_OP_SOFT_MAX ||
                    cgraph->nodes[i]->op == GGML_OP_ARGSORT) {
                ggml_cuda_topk_moe_args args;
                const bool              can_fuse = ggml_cuda_topk_moe_fusion(cgraph, i, args);
                std::vector<ggml_op>    ops;
                ops.reserve(13);  // max ops; avoids gcc -Wstringop-overflow false positive

                const ggml_tensor * node = cgraph->nodes[i];

                if (can_fuse) {
                    const ggml_tensor * logits  = node->src[0];
                    ggml_tensor *       weights = nullptr;
                    ggml_tensor *       ids     = nullptr;

                    if (!args.delayed_softmax) {
                        int out_nodes[2];  // nodes which can't be elided

                        if (args.sigmoid) {
                            ops.insert(ops.end(), { GGML_OP_UNARY });
                        } else if (args.sqrt_softplus) {
                            ops.insert(ops.end(), { GGML_OP_UNARY, GGML_OP_SQRT });
                        } else {
                            ops.insert(ops.end(), { GGML_OP_SOFT_MAX });
                        }
                        const int i_probs = i + (int) ops.size() - 1;  // last node of the gating activation

                        if (args.prob_bias) {
                            ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW,
                                                    GGML_OP_GET_ROWS });
                            out_nodes[0] = i_probs + 4;
                        } else {
                            ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS });
                            out_nodes[0] = i_probs + 3;
                        }
                        ids = cgraph->nodes[out_nodes[0]];

                        if (args.norm) {
                            ops.insert(ops.end(),
                                       { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
                        }
                        if (args.scale) {
                            ops.insert(ops.end(), { GGML_OP_SCALE });
                        }

                        weights      = cgraph->nodes[i + ops.size() - 1];
                        out_nodes[1] = i + ops.size() - 1;

                        if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                                ggml_cuda_should_use_topk_moe(node, logits, weights, ids)) {

                            add_alloc_deps(i, i + ops.size());
                            i += ops.size() - 1;
                        }
                    }
                }
            }
        }
    }

#ifdef USE_CUDA_GRAPH
    const void * graph_key = ggml_cuda_graph_get_key(cuda_ctx, cgraph, nullptr, nullptr, false);
    const bool use_cuda_graph = ggml_cuda_graph_set_enabled(cuda_ctx, graph_key);
#else
    const bool use_cuda_graph = false;
    GGML_UNUSED(cuda_ctx);
    GGML_UNUSED(cgraph);
#endif

    static bool enable_graph_optimization = [] {
        const char * env     = getenv("GGML_CUDA_GRAPH_OPT");
        return env != nullptr && atoi(env) == 1;
    }();

    if (!enable_graph_optimization) {
        return;
    }

    ggml_cuda_stream_context & stream_context = cuda_ctx->stream_context();
    stream_context.reset();

    if (!use_cuda_graph) {
        return;
    }

    ggml_cuda_set_device(cuda_ctx->device);

    // number of out-degrees for a particular node
    std::unordered_map<const ggml_tensor *, int> fan_out;
    // reverse mapping of node to index in the cgraph
    std::unordered_map<const ggml_tensor *, int> node_indices;

    const auto & is_noop = [](const ggml_tensor * node) -> bool {
        return ggml_is_empty(node) || node->op == GGML_OP_NONE || node->op == GGML_OP_RESHAPE ||
               node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE;
    };

    const auto & depends_on = [](const ggml_tensor * dst, const ggml_tensor * src) -> bool {
        for (uint32_t s = 0; s < GGML_MAX_SRC; ++s) {
            if (dst->src[s] == src) {
                return true;
            }
        }
        // implicit dependency if they view the same tensor
        const ggml_tensor * dst2 = dst->view_src ? dst->view_src : dst;
        const ggml_tensor * src2 = src->view_src ? src->view_src : src;
        if (dst2 == src2) {
            return true;
        }
        return false;
    };

    for (int node_idx = 0; node_idx < cgraph->n_nodes; node_idx++) {
        const ggml_tensor * node = cgraph->nodes[node_idx];
        node_indices[node]       = node_idx;

        if (is_noop(node)) {
            continue;
        }
        for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
            const ggml_tensor * src = cgraph->nodes[node_idx]->src[src_idx];
            //TODO: check why nrows > 1 fails
            if (node && !is_noop(node) && ggml_nrows(node) <= 1) {
                fan_out[src] += 1;
            }
        }
    }

    // Target Q, K, V for concurrency
    // this is a more general way to find nodes which can be candidates for concurrency (although it has not been tested for anything else):
    // 1. find fan-out (fork) nodes where the same input is used at least N times (in QKV, it would be "attn-norm")
    // 2. find the join node, where 2 or more of the outputs are required (in QKV, this would "KQ" or "flash-attn")
    // 3. account for all branches from the fork to the join
    // 4. To extend lifetimes of the tensors, we interleave the branches (see below for more details)
    // 5. save the original cgraph and restore it in graph_compute, to enable fusion within streams
    // See discussion: https://github.com/ggml-org/llama.cpp/pull/16991#issuecomment-3522620030

    const int min_fan_out = 3;
    const int max_fan_out = 3;

    // store {fork_idx, join_idx}
    std::vector<std::pair<int, int>> concurrent_node_ranges;

    for (const auto & [root_node, count] : fan_out) {
        if (count >= min_fan_out && count <= max_fan_out) {
            const int root_node_idx = node_indices[root_node];

            // only optimize for attn_norm
            // TODO: make this more generic
            if (!strstr(root_node->name, "attn_norm")) {
                continue;
            }

            bool is_part_of_event = false;
            for (const auto & [start, end] : concurrent_node_ranges) {
                if (root_node_idx >= start && root_node_idx <= end) {
                    is_part_of_event = true;
                }
            }

            if (is_part_of_event) {
                continue;
            }

            std::vector<std::vector<const ggml_tensor *>> nodes_per_branch;
            for (int i = root_node_idx + 1; i < cgraph->n_nodes; ++i) {
                const ggml_tensor * node = cgraph->nodes[i];
                if (!is_noop(node) && depends_on(node, root_node)) {
                    nodes_per_branch.push_back({ node });
                }
            }

            GGML_ASSERT(nodes_per_branch.size() == (size_t) count);

            //find the join point
            const ggml_tensor * join_node = nullptr;

            const auto & belongs_to_branch = [&](const ggml_tensor *                      node,
                                                 const std::vector<const ggml_tensor *> & branch) -> bool {
                for (const ggml_tensor * n : branch) {
                    if (depends_on(node, n)) {
                        return true;
                    }
                }
                return false;
            };

            for (int i = root_node_idx + 1; i < cgraph->n_nodes; ++i) {
                const ggml_tensor * curr_node = cgraph->nodes[i];

                int num_joins = 0;
                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    if (belongs_to_branch(curr_node, nodes_per_branch[branch_idx])) {
                        num_joins++;
                    }
                }

                if (num_joins >= 2) {
                    join_node = curr_node;
                    break;
                }

                bool found_branch = false;
                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    std::vector<const ggml_tensor *> & branch_vec = nodes_per_branch[branch_idx];
                    if (belongs_to_branch(curr_node, branch_vec)) {
                        //continue accumulating
                        if (std::find(branch_vec.begin(), branch_vec.end(), curr_node) == branch_vec.end()) {
                            branch_vec.push_back(curr_node);
                        }
                        found_branch = true;
                    }
                }

                if (!found_branch && is_noop(curr_node)) {
                    // we can put it in any branch because it will be ignored
                    nodes_per_branch[0].push_back({ curr_node });
                }
            }

            if (join_node) {
                //Create ggml_cuda_concurrent_event
                ggml_cuda_concurrent_event concurrent_event(nodes_per_branch.size());
                concurrent_event.join_node = join_node;

                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    for (const ggml_tensor * n : nodes_per_branch[branch_idx]) {
                        concurrent_event.stream_mapping[n] = branch_idx + 1;
                    }
                }

                int fork_node_idx = node_indices[root_node];
                int join_node_idx = node_indices[join_node];

                int       current_branch_idx = 0;
                int       current_node_idx   = fork_node_idx + 1;
                const int n_branches         = nodes_per_branch.size();

                int total_branch_nodes = 0;
                for (std::vector<const ggml_tensor *> branch_nodes : nodes_per_branch) {
                    total_branch_nodes += branch_nodes.size();
                }

                // there are other nodes in the middle which are unaccounted for
                // usually (cpy) nodes, then ignore this fork
                if (join_node_idx - fork_node_idx - 1 != total_branch_nodes) {
                    GGML_LOG_DEBUG(
                        "Skipping %s because the number of nodes in the middle is not equal to the total number of "
                        "branch nodes %d != %d\n",
                        root_node->name, join_node_idx - fork_node_idx - 1, total_branch_nodes);
                    continue;
                }

                // Save the original order of nodes in this region before interleaving
                // This is used later to restore grouping for fusion within streams
                concurrent_event.original_order.reserve(total_branch_nodes);
                for (int i = fork_node_idx + 1; i < join_node_idx; ++i) {
                    concurrent_event.original_order.push_back(cgraph->nodes[i]);
                }

                std::unordered_map<const ggml_tensor *, ggml_cuda_concurrent_event> & concurrent_events = cuda_ctx->stream_context().concurrent_events;
                GGML_ASSERT(concurrent_events.find(root_node) == concurrent_events.end());
                concurrent_events.emplace(root_node, std::move(concurrent_event));
                GGML_LOG_DEBUG("Adding stream at node %s %p\n", root_node->name, root_node);
                concurrent_node_ranges.emplace_back(fork_node_idx, join_node_idx);

                // interleave tensors to extend lifetimes so that ggml graph doesn't recycle them
                // example transformation:
                // [attn-norm, QMul, QNorm, QRope, KMul, KNorm, KRope, VMul, attn] ->
                // [attn-norm, QMul, KMul, VMul, QNorm, VNorm, QRope, KRope, attn]
                while (current_node_idx < join_node_idx) {
                    std::vector<const ggml_tensor *> & branch_nodes = nodes_per_branch[current_branch_idx];

                    bool has_node = false;
                    for (std::vector<const ggml_tensor *> branch_node : nodes_per_branch) {
                        has_node |= branch_node.size() > 0;
                    }

                    GGML_ASSERT(has_node);

                    if (branch_nodes.empty()) {
                        current_branch_idx = (current_branch_idx + 1) % n_branches;
                        continue;
                    }

                    cgraph->nodes[current_node_idx] = const_cast<ggml_tensor *>(branch_nodes.front());
                    current_node_idx++;
                    branch_nodes.erase(branch_nodes.begin());

                    // append all empty nodes
                    while (!branch_nodes.empty() && is_noop(branch_nodes.front())) {
                        cgraph->nodes[current_node_idx] = const_cast<ggml_tensor *>(branch_nodes.front());
                        current_node_idx++;
                        branch_nodes.erase(branch_nodes.begin());
                    }

                    current_branch_idx = (current_branch_idx + 1) % n_branches;
                }
            }
        }
    }
}

static const ggml_backend_i ggml_backend_cuda_interface = {
    /* .get_name                = */ ggml_backend_cuda_get_name,
    /* .free                    = */ ggml_backend_cuda_free,
    /* .set_tensor_async        = */ ggml_backend_cuda_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_cuda_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_cuda_set_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_cuda_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ggml_backend_cuda_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_cuda_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_cuda_graph_compute,
    /* .event_record            = */ ggml_backend_cuda_event_record,
    /* .event_wait              = */ ggml_backend_cuda_event_wait,
    /* .graph_optimize          = */ ggml_backend_cuda_graph_optimize,
};

static ggml_guid_t ggml_backend_cuda_guid() {
    static ggml_guid guid = { 0x2c, 0xdd, 0xe8, 0x1c, 0x65, 0xb3, 0x65, 0x73, 0x6a, 0x12, 0x88, 0x61, 0x1c, 0xc9, 0xdc, 0x25 };
    return &guid;
}

bool ggml_backend_is_cuda(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_cuda_guid());
}

int ggml_backend_cuda_get_device_count() {
    return ggml_cuda_info().device_count;
}

// V226_DEVICE_PROPERTIES_CACHE_BEGIN
// Hardware properties are immutable for a CUDA process. Keep memory availability
// queries live; this cache only replaces repeated hardware description/type reads.
static cudaError_t ggml_cuda_get_static_device_properties(cudaDeviceProp * prop, int physical_device) {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_CUDA_CACHE_DEVICE_PROPERTIES");
        return value && std::strcmp(value, "1") == 0;
    }();
    if (!enabled) return cudaGetDeviceProperties(prop, physical_device);

    static const std::vector<cudaDeviceProp> snapshots = [] {
        int count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&count));
        std::vector<cudaDeviceProp> values((size_t) count);
        for (int physical = 0; physical < count; ++physical) {
            CUDA_CHECK(cudaGetDeviceProperties(&values[(size_t) physical], physical));
        }
        GGML_LOG_INFO("CUDA static device properties cache requested=1 selected=1 physical_snapshots=%zu"
                " dynamic_memory_queries_unchanged=1\n", values.size());
        return values;
    }();
    if (physical_device < 0 || (size_t) physical_device >= snapshots.size()) {
        return cudaGetDeviceProperties(prop, physical_device); // preserve runtime error semantics
    }
    *prop = snapshots[(size_t) physical_device];
    return cudaSuccess;
}
// V226_DEVICE_PROPERTIES_CACHE_END

static std::string ggml_cuda_device_description(int device) {
    cudaDeviceProp prop;
    CUDA_CHECK(ggml_cuda_get_static_device_properties(&prop, ggml_cuda_get_physical_device(device)));

    const ggml_cuda_device_info & info = ggml_cuda_info();
    std::string description = prop.name;
    if (info.device_count > info.physical_device_count) {
        description += " (dev p" + std::to_string(info.devices[device].physical_device) +
                       "/v" + std::to_string(info.devices[device].virtual_index) + ")";
    }
    return description;
}

void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size) {
    snprintf(description, description_size, "%s", ggml_cuda_device_description(device).c_str());
}

static int ggml_cuda_physical_device_share_count(int device) {
    const ggml_cuda_device_info & info = ggml_cuda_info();
    GGML_ASSERT(device >= 0 && device < info.device_count);
    return info.devices[device].physical_share_count;
}

void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total) {
    ggml_cuda_set_device(device);

    CUDA_CHECK(cudaMemGetInfo(free, total));

    // virtual devices sharing one physical GPU share its memory pool; split it between them
    const int share_count = ggml_cuda_physical_device_share_count(device);
    *free  /= share_count;
    *total /= share_count;
}

bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size) {
    if (getenv("GGML_CUDA_REGISTER_HOST") == nullptr) {
        return false;
    }

#if CUDART_VERSION >= 11010 || defined(GGML_USE_MUSA) || defined(GGML_USE_HIP)
    cudaError_t err = cudaHostRegister(buffer, size, cudaHostRegisterPortable | cudaHostRegisterReadOnly);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();

        GGML_LOG_DEBUG("%s: failed to register %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return false;
    }
    return true;
#else
    GGML_UNUSED(buffer);
    GGML_UNUSED(size);
    return false;
#endif // CUDART_VERSION >= 11010 || defined(GGML_USE_MUSA)
}

void ggml_backend_cuda_unregister_host_buffer(void * buffer) {
    if (getenv("GGML_CUDA_REGISTER_HOST") == nullptr) {
        return;
    }

    cudaError_t err = cudaHostUnregister(buffer);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
    }
}


// backend device

struct ggml_backend_cuda_device_context {
    int device;
    std::string name;
    std::string description;
    std::string pci_bus_id;
    int op_offload_min_batch_size;
};

#if defined(GGML_CUDA_EVICT)
// "ggml_backend_evict_weights": release the evictable tail of every weight buffer on dev (GGML_CUDA_EVICT_MIB).
// Returns the bytes freed. No graph may run on dev until ggml_backend_restore_weights.
static size_t ggml_backend_cuda_evict_weights(ggml_backend_dev_t dev) {
    const int device = ((ggml_backend_cuda_device_context *) dev->context)->device;

    std::lock_guard<std::mutex> lock(ggml_cuda_evict_mutex);

    // nothing may still read the weights; under tensor split every device takes part in a graph
    for (int id = 0; id < ggml_cuda_info().device_count; ++id) {
        ggml_cuda_set_device(id);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    size_t freed = 0;
    for (ggml_backend_buffer_t buffer : ggml_cuda_evict_buffers) {
        ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;
        ggml_cuda_vmm_buffer * vmm = ctx->vmm;
        if (ctx->device != device || vmm->evicted || ggml_backend_buffer_get_usage(buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            continue;
        }
        ggml_cuda_set_device(device);
        void * tail = (void *) (vmm->base + vmm->tail_off);
        if (vmm->shadow == nullptr) {
            // weights do not change: one copy serves every later restore
            vmm->shadow_pinned = cudaMallocHost(&vmm->shadow, vmm->tail_size) == cudaSuccess;
            if (!vmm->shadow_pinned) {
                (void) cudaGetLastError();
                vmm->shadow = malloc(vmm->tail_size);
                GGML_ASSERT(vmm->shadow != nullptr);
            }
            CUDA_CHECK(cudaMemcpy(vmm->shadow, tail, vmm->tail_size, cudaMemcpyDeviceToHost));
        } else if (getenv("GGML_CUDA_EVICT_CHECK") != nullptr) {
            std::vector<uint8_t> now(vmm->tail_size);
            CUDA_CHECK(cudaMemcpy(now.data(), tail, vmm->tail_size, cudaMemcpyDeviceToHost));
            size_t n_diff = 0, first = SIZE_MAX;
            for (size_t i = 0; i < vmm->tail_size; ++i) {
                if (now[i] != ((const uint8_t *) vmm->shadow)[i]) {
                    n_diff++;
                    first = std::min(first, i);
                }
            }
            GGML_LOG_WARN("%s: device %d tail check: %zu bytes differ from the saved copy (first at +%zu of %zu)\n",
                __func__, device, n_diff, first, vmm->tail_size);
        }
        CU_CHECK(cuMemUnmap((CUdeviceptr) tail, vmm->tail_size));
        vmm->evicted = true;
        freed += vmm->tail_size;
    }
    return freed;
}

// "ggml_backend_restore_weights": map memory for every evicted tail on dev again and copy its bytes back.
// False if the memory is not free again.
static bool ggml_backend_cuda_restore_weights(ggml_backend_dev_t dev) {
    const int device = ((ggml_backend_cuda_device_context *) dev->context)->device;

    std::lock_guard<std::mutex> lock(ggml_cuda_evict_mutex);

    bool ok = true;
    for (ggml_backend_buffer_t buffer : ggml_cuda_evict_buffers) {
        ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;
        ggml_cuda_vmm_buffer * vmm = ctx->vmm;
        if (ctx->device != device || !vmm->evicted) {
            continue;
        }
        ggml_cuda_set_device(device);
        void * tail = (void *) (vmm->base + vmm->tail_off);
        if (!ggml_cuda_vmm_map((uintptr_t) tail, vmm->tail_size, device)) {
            GGML_LOG_ERROR("%s: cannot map %.2f MiB on device %d again: the memory is still in use\n",
                __func__, vmm->tail_size / 1024.0 / 1024.0, device);
            ok = false;
            continue;
        }
        CUDA_CHECK(cudaMemcpy(tail, vmm->shadow, vmm->tail_size, cudaMemcpyHostToDevice));
        vmm->evicted = false;
    }
    return ok;
}
#endif // defined(GGML_CUDA_EVICT)

static const char * ggml_backend_cuda_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_cuda_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ctx->description.c_str();
}

#if defined(__linux__)
// Helper function to get available memory from /proc/meminfo for UMA systems
static bool ggml_backend_cuda_get_available_uma_memory(long * available_memory_kb, long * free_swap_kb) {
    FILE * meminfo_file = nullptr;
    // 2KB buffer for reading /proc/meminfo since it does not report size info, should be enough
    const size_t BUFFER_SIZE = 2048;
    auto file_buffer = std::make_unique<char[]>(BUFFER_SIZE);
    size_t bytes_read = 0;
    long huge_tlb_total_pages = -1;
    long huge_tlb_free_pages = -1;
    long huge_tlb_page_size = -1;

    if (available_memory_kb == nullptr || free_swap_kb == nullptr) {
        return false;
    }

    meminfo_file = fopen("/proc/meminfo", "r");
    if (meminfo_file == nullptr) {
        GGML_LOG_ERROR("%s: failed to open /proc/meminfo\n", __func__);
        return false;
    }

    // Read file into buffer
    bytes_read = fread(file_buffer.get(), 1, BUFFER_SIZE - 1, meminfo_file);
    fclose(meminfo_file);

    if (bytes_read == 0) {
        GGML_LOG_ERROR("%s: failed to read from /proc/meminfo\n", __func__);
        return false;
    }
    file_buffer[bytes_read] = '\0';

    *available_memory_kb = -1;
    *free_swap_kb = -1;

    // Parse the file buffer line by line
    char * line = file_buffer.get();
    char * line_next;
    while (line < file_buffer.get() + bytes_read) {
        // Find the end of the current line
        line_next = strchr(line, '\n');
        if (line_next != nullptr) {
            *line_next = '\0';
            line_next++;
        } else {
            line_next = file_buffer.get() + bytes_read;
        }

        long value;
        if (sscanf(line, "MemAvailable: %ld kB", &value) == 1) {
            *available_memory_kb = value;
        } else if (sscanf(line, "SwapFree: %ld kB", &value) == 1) {
            *free_swap_kb = value;
        } else if (sscanf(line, "HugePages_Total: %ld", &value) == 1) {
            huge_tlb_total_pages = value;
        } else if (sscanf(line, "HugePages_Free: %ld", &value) == 1) {
            huge_tlb_free_pages = value;
        } else if (sscanf(line, "Hugepagesize: %ld kB", &value) == 1) {
            huge_tlb_page_size = value;
        }

        line = line_next;
    }

    if (huge_tlb_total_pages != 0 && huge_tlb_total_pages != -1) {
        *available_memory_kb = huge_tlb_free_pages * huge_tlb_page_size;

        // Hugetlbfs pages are not swappable.
        *free_swap_kb = 0;
    }

    GGML_LOG_DEBUG("%s: final available_memory_kb: %ld\n", __func__, *available_memory_kb);
    return true;
}
#endif // defined(__linux__)

static void ggml_backend_cuda_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    ggml_cuda_set_device(ctx->device);
    cudaError_t err = cudaMemGetInfo(free, total);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        GGML_LOG_WARN("%s: cudaMemGetInfo failed (%s), returning 0/0\n", __func__, cudaGetErrorString(err));
        *free = 0;
        *total = 0;
        return;
    }

// ref: https://github.com/ggml-org/llama.cpp/pull/17368
#if defined(__linux__) && !defined(GGML_USE_HIP)
    // Check if this is a UMA (Unified Memory Architecture) system
    cudaDeviceProp prop;
    CUDA_CHECK(ggml_cuda_get_static_device_properties(&prop, ggml_cuda_get_physical_device(ctx->device)));

    // Check if UMA is explicitly enabled via environment variable
    bool uma_env = getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY") != nullptr;
    bool is_uma = prop.integrated > 0 || uma_env;

    if (is_uma) {
        // For UMA systems (like DGX Spark), use system memory info
        long available_memory_kb = 0;
        long free_swap_kb = 0;

        if (ggml_backend_cuda_get_available_uma_memory(&available_memory_kb, &free_swap_kb) && available_memory_kb > 0) {
            *free = (size_t)available_memory_kb * 1024;
        } else {
            GGML_LOG_ERROR("%s: /proc/meminfo reading failed, using cudaMemGetInfo\n", __func__);
        }
    }
#endif // defined(__linux__) && !defined(GGML_USE_HIP)

    // virtual devices sharing one physical GPU share its memory pool; split it between them
    const int share_count = ggml_cuda_physical_device_share_count(ctx->device);
    *free  /= share_count;
    *total /= share_count;
}

static enum ggml_backend_dev_type ggml_backend_cuda_device_get_type(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *) dev->context;

    cudaDeviceProp prop;
    CUDA_CHECK(ggml_cuda_get_static_device_properties(&prop, ggml_cuda_get_physical_device(ctx->device)));

    return prop.integrated
        ? GGML_BACKEND_DEVICE_TYPE_IGPU
        : GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_cuda_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;

    props->name        = ggml_backend_cuda_device_get_name(dev);
    props->description = ggml_backend_cuda_device_get_description(dev);
    props->type        = ggml_backend_cuda_device_get_type(dev);
    props->device_id   = ctx->pci_bus_id.empty() ? nullptr : ctx->pci_bus_id.c_str();
    ggml_backend_cuda_device_get_memory(dev, &props->memory_free, &props->memory_total);

    bool host_buffer = getenv("GGML_CUDA_NO_PINNED") == nullptr;
#ifdef GGML_CUDA_NO_PEER_COPY
    bool events = false;
#else
    bool events = true;
#endif

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ host_buffer,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ events,
        /* .mmap_support          = */ props->type != GGML_BACKEND_DEVICE_TYPE_IGPU,
    };
}

static ggml_backend_t ggml_backend_cuda_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ggml_backend_cuda_init(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_cuda_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ggml_backend_cuda_buffer_type(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_cuda_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_cuda_host_buffer_type();
}

// TODO: move these functions here
static bool ggml_backend_cuda_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;

    // check if all the sources are allocated on this device
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op->src[i] && op->src[i]->buffer && ggml_backend_buft_is_cuda(op->src[i]->buffer->buft)) {
            ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)op->src[i]->buffer->buft->context;
            if (buft_ctx->device != dev_ctx->device) {
                return false;
            }
        }
    }

    switch (op->op) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        case GGML_OP_CUSTOM:
            return ggml_cuda_ft_supports(op, dev_ctx->device) || ggml_cuda_ft_dense_supports(op, dev_ctx->device);
#endif
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_EXPM1:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_XIELU:
                case GGML_UNARY_OP_FLOOR:
                case GGML_UNARY_OP_CEIL:
                case GGML_UNARY_OP_ROUND:
                case GGML_UNARY_OP_TRUNC:
                    // TODO: should become:
                    //return ggml_is_contiguous_rows(op->src[0]);
                    return ggml_is_contiguous(op->src[0]);
                default:
                    return false;
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    return ggml_is_contiguous_1(op->src[0]);
                default:
                    return false;
            }
            break;
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            {
                struct ggml_tensor * a = op->src[0];
                struct ggml_tensor * b = op->src[1];
                if (a->nb[0] != ggml_element_size(a) || b->nb[0] != ggml_element_size(b)) {
                    return false; // TODO this could in principle be implemented though currently there is no use case.
                }
                if (b->type == GGML_TYPE_F16 && a->type != GGML_TYPE_F16) {
                    return false;
                }
                if (op->op == GGML_OP_MUL_MAT_ID && ggml_get_op_params_i32(op, 3) == GGML_PREC_F32) {
                    return false;
                }
#ifdef GGML_USE_MUSA
                const int cc = ggml_cuda_info().devices[dev_ctx->device].cc;
                if (b->ne[2]*b->ne[3] > 1 && !ggml_is_transposed(a) && !ggml_is_transposed(b)) {
                    if (GGML_CUDA_CC_IS_QY1(cc) && op->op == GGML_OP_MUL_MAT &&
                            a->type == GGML_TYPE_F16 && b->type == GGML_TYPE_F16) {
                        return false;
                    }
                    if (GGML_CUDA_CC_IS_QY2(cc) && op->op == GGML_OP_MUL_MAT_ID &&
                            a->type == GGML_TYPE_Q2_K && b->type == GGML_TYPE_F32) {
                        return false;
                    }
                }
#endif // GGML_USE_MUSA
                switch (a->type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_Q2_0:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_MXFP4:
                    case GGML_TYPE_NVFP4:
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_Q8_K:
                    case GGML_TYPE_IQ1_M:
                    case GGML_TYPE_IQ1_S:
                    case GGML_TYPE_IQ2_S:
                    case GGML_TYPE_IQ2_XS:
                    case GGML_TYPE_IQ2_XXS:
                    case GGML_TYPE_IQ3_S:
                    case GGML_TYPE_IQ3_XXS:
                    case GGML_TYPE_IQ4_NL:
                    case GGML_TYPE_IQ4_XS:
                    case GGML_TYPE_BF16:
                        return true;
                    default:
                        return false;
                }
            } break;
        case GGML_OP_OUT_PROD:
            return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_GET_ROWS:
            {
                switch (op->src[0]->type) {
                    case GGML_TYPE_F16:
                    case GGML_TYPE_F32:
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_I32:
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_Q2_0:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_IQ2_XXS:
                    case GGML_TYPE_IQ2_XS:
                    case GGML_TYPE_IQ2_S:
                    case GGML_TYPE_IQ3_XXS:
                    case GGML_TYPE_IQ3_S:
                    case GGML_TYPE_IQ1_S:
                    case GGML_TYPE_IQ1_M:
                    case GGML_TYPE_IQ4_XS:
                        return true;
                    case GGML_TYPE_IQ4_NL:
                        return op->src[0]->ne[0] % 32 == 0;
                    case GGML_TYPE_MXFP4:
                        return op->src[0]->ne[0] % QK_K == 0;
                    default:
                        return false;
                }
            } break;
        case GGML_OP_GET_ROWS_BACK:
            {
                return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->ne[2] == 1 && op->ne[3] == 1;
            } break;
        case GGML_OP_SET_ROWS:
            {
                return (
                           (
                               (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_BF16 ||
                               op->type == GGML_TYPE_Q4_0 || op->type == GGML_TYPE_Q4_1 || op->type == GGML_TYPE_Q5_0 ||
                               op->type == GGML_TYPE_Q5_1 || op->type == GGML_TYPE_Q8_0 || op->type == GGML_TYPE_IQ4_NL) &&
                               op->src[0]->type == GGML_TYPE_F32
                           ) || (
                               op->type == GGML_TYPE_F16 && op->src[0]->type == GGML_TYPE_F16
                           )
                       ) &&
                       (op->src[1]->type == GGML_TYPE_I64 || op->src[1]->type == GGML_TYPE_I32);
            } break;
        case GGML_OP_SET:
            {
                const ggml_type t = op->type;
                return (t == GGML_TYPE_F32 || t == GGML_TYPE_I32) &&
                    t == op->src[0]->type &&
                    t == op->src[1]->type;
            } break;
        case GGML_OP_CPY:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if ((src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_BF16 || src0_type == GGML_TYPE_F16) &&
                    (src1_type == GGML_TYPE_F32 || src1_type == GGML_TYPE_BF16 || src1_type == GGML_TYPE_F16)
                ) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q8_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q8_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q4_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q4_1 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q5_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q5_1 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_IQ4_NL) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_I32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_I32) {
                    return true;
                }
                if (src0_type == src1_type && ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) {
                    return true;
                }
                return false;
            } break;
        case GGML_OP_DUP:
                return true;
        case GGML_OP_ARGMAX:
        case GGML_OP_COUNT_EQUAL:
            {
                return true;
            } break;
        case GGML_OP_REPEAT:
            {
                // the CUDA REPEAT path only implements F32/F16; other types assert at runtime
                ggml_type src0_type = op->src[0]->type;
                return src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_F16;
            } break;
        case GGML_OP_REPEAT_BACK:
                return op->type == GGML_TYPE_F32 && (op->src[0]->ne[2]*op->src[0]->ne[3]) <= (1 << 15);
        case GGML_OP_CONCAT:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                const int32_t dim = op->op_params[0];
                return src0_type == src1_type &&
                       src0_type == op->type &&
                       (
                           (
                               ggml_is_quantized(src0_type) &&
                               (
                                   (
                                       dim == 3 &&
                                       ggml_is_contiguous(op->src[0]) &&
                                       ggml_is_contiguous(op->src[1])
                                   ) || (
                                       dim != 3 &&
                                       ggml_is_contiguous_to_3(op->src[0]) &&
                                       ggml_is_contiguous_to_3(op->src[1])
                                   )
                               ) &&
                               op->src[0]->ne[0] % ggml_blck_size(src0_type) == 0 &&
                               op->src[1]->ne[0] % ggml_blck_size(src0_type) == 0
                           ) || (
                               !ggml_is_quantized(src0_type) &&
                               ggml_blck_size(src0_type) == 1 &&
                               (
                                   ggml_type_size(src0_type) == 1 ||
                                   ggml_type_size(src0_type) == 2 ||
                                   ggml_type_size(src0_type) == 4 ||
                                   ggml_type_size(src0_type) == 8
                               )
                           )
                       );
            } break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                return false;
            } break;
        case GGML_OP_COL2IM_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                return (src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_F16 || src0_type == GGML_TYPE_BF16) &&
                    op->type == src0_type &&
                    ggml_is_contiguous(op->src[0]) &&
                    ggml_is_contiguous(op);
            } break;
        case GGML_OP_SILU_BACK:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
            break;
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_L2_NORM:
            return ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_RMS_NORM_BACK:
            return ggml_is_contiguous(op->src[0]);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_ADD_ID:
        case GGML_OP_ADD1:
        case GGML_OP_SCALE:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_CLAMP:
        case GGML_OP_LOG:
            return true;
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) &&
                   (op->type         == GGML_TYPE_F32 || op->type         == GGML_TYPE_F16);
        case GGML_OP_SSM_SCAN: {
            const int32_t K = ggml_get_op_params_i32(op, 0);

            if (op->src[3]->ne[0] == 1) {
                // Mamba2
                // (kernel only supports (d_state == 128 || d_state == 256) && d_head % 16 == 0)
                return (op->src[0]->ne[0] == 128 || op->src[0]->ne[0] == 256) && op->src[0]->ne[1] % 16 == 0;
            } else {
                if (K > 1) {
                    return false;
                }

                // Mamba
                // (kernel only supports d_state == 16, d_head == 1, n_head % 128 == 0, n_group == 1)
                return op->src[0]->ne[0] == 16 && op->src[0]->ne[1] == 1 && op->src[0]->ne[2] % 128 == 0 && op->src[4]->ne[1] == 1;
            }
        }
        case GGML_OP_SSM_CONV: {
            // assumes d_inner % threads == 0
            return op->src[0]->ne[1] % 128 == 0;
        }
        case GGML_OP_CONT:
            return true;
        case GGML_OP_DIAG_MASK_INF:
            if (ggml_qsa_is_block_bias(op)) return ggml_qsa_block_bias_layout_valid(op);
            if (ggml_qsa_is_position_mask(op)) return ggml_qsa_position_mask_layout_valid(op);
            return true;
        case GGML_OP_SOFT_MAX:
            return true;
        case GGML_OP_SOFT_MAX_BACK: {
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(float));
            return max_bias == 0.0f;
        }
        case GGML_OP_ROLL:
            if(op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0])) {
                return true;
            }
            return false;
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK: {
            return op->src[0]->nb[0] == ggml_type_size(op->src[0]->type) && ggml_is_contiguous_2(op->src[0]);
        }
        case GGML_OP_IM2COL:
        case GGML_OP_IM2COL_3D:
        case GGML_OP_CONV_2D:
            return (ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]));
        case GGML_OP_CONV_2D_DW:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_CONV_TRANSPOSE_2D:
        case GGML_OP_POOL_1D:
        case GGML_OP_POOL_2D:
            return true;
        case GGML_OP_ACC:
            // TODO: extend support like so:
            //return ggml_is_contiguous_rows(op->src[0]) && ggml_is_contiguous_rows(op->src[1]);
            return ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]);
        case GGML_OP_SUM:
            return ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_TOP_K:
            if (ggml_qsa_virtual_topk_private::tagged(op)) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
                return ggml_qsa_virtual_topk_private::valid(op);
#else
                return false;
#endif
            }
#if defined(GGML_USE_HIP) || defined(GGML_CUDA_USE_CUB)
            return true;
#else
            return op->src[0]->ne[0] <= 1024;
#endif // defined(GGML_USE_HIP) || defined(GGML_CUDA_USE_CUB)
        case GGML_OP_ARGSORT:
#ifndef GGML_CUDA_USE_CUB
            return op->src[0]->ne[0] <= 1024;
#else
            return true;
#endif
        case GGML_OP_SUM_ROWS:
            return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_MEAN:
            return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_GROUP_NORM:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_PAD:
            return true;
        case GGML_OP_UPSCALE:
        case GGML_OP_PAD_REFLECT_1D:
        case GGML_OP_ARANGE:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_GATED_LINEAR_ATTN:
        case GGML_OP_RWKV_WKV7:
            return true;
        case GGML_OP_GATED_DELTA_NET:
            //TODO: enable once MUSA compiler is solved https://github.com/ggml-org/llama.cpp/pull/19504#issuecomment-4018634327
#ifdef GGML_USE_MUSA
            return false;
#else
            return true;
#endif // GGML_USE_MUSA
        case GGML_OP_DSV4_HC_COMB:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_MONOMOE_SM70:
            return ggml_cuda_monomoe_supported(dev_ctx->device, op);
        case GGML_OP_PLE_CONV_SM70:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F16 && op->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) &&
                ggml_is_contiguous(op->src[2]) && ggml_is_contiguous(op);
        case GGML_OP_DSV4_HC_PRE:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32;
        case GGML_OP_DSV4_HC_POST:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && (op->src[3] == nullptr || op->src[3]->type == GGML_TYPE_F32) &&
                op->type == GGML_TYPE_F32;
        case GGML_OP_FLASH_ATTN_EXT:
            return ggml_cuda_flash_attn_ext_supported(dev_ctx->device, op);
        case GGML_OP_CROSS_ENTROPY_LOSS:
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        case GGML_OP_OPT_STEP_ADAMW:
        case GGML_OP_OPT_STEP_SGD:
        case GGML_OP_FILL:
        case GGML_OP_CUMSUM:
        case GGML_OP_TRI:
        case GGML_OP_DIAG:
        case GGML_OP_SOLVE_TRI:
            return true;
        case GGML_OP_LIGHTNING_INDEXER:
            return ggml_cuda_lightning_indexer_supported(dev_ctx->device, op);
        case GGML_OP_ALLREDUCE:
            return op->type == GGML_TYPE_F32 && ggml_is_contiguous(op);
        case GGML_OP_ARGMAX_SPLIT:
            return op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0]);
        case GGML_OP_HC_FUSED:
            return ggml_cuda_hc_fused_supported(op);

        default:
            return false;
    }
}

static bool ggml_backend_cuda_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;
    const bool integrated = ggml_cuda_info().devices[dev_ctx->device].integrated;
    return (ggml_backend_buft_is_cuda(buft) && buft->device == dev) || (integrated && ggml_backend_buft_is_cuda_host(buft));
}

static int64_t get_op_batch_size(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_GET_ROWS:
            return 0;
        case GGML_OP_MUL_MAT:
            return op->ne[1];
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
            return op->ne[2];
        default:
            return ggml_nrows(op);
    }
}

static bool ggml_backend_cuda_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;

    return get_op_batch_size(op) >= dev_ctx->op_offload_min_batch_size;
}

static ggml_backend_event_t ggml_backend_cuda_device_event_new(ggml_backend_dev_t dev) {
#ifdef GGML_CUDA_NO_PEER_COPY
    GGML_UNUSED(dev);
    return nullptr;
#else
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *)dev->context;

    ggml_cuda_set_device(dev_ctx->device);

    cudaEvent_t event;
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));

    return new ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ event,
    };
#endif
}

static void ggml_backend_cuda_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);

    CUDA_CHECK(cudaEventDestroy((cudaEvent_t)event->context));
    delete event;
}

// Private asynchronous fence query used by the staged prefill executor.
static int ggml_backend_cuda_device_event_query(ggml_backend_event_t event) {
    if (!event || !event->device || !event->context) return -1;
    auto * dev_ctx=static_cast<ggml_backend_cuda_device_context *>(event->device->context);
    ggml_cuda_set_device(dev_ctx->device);
    const cudaError_t status=cudaEventQuery(static_cast<cudaEvent_t>(event->context));
    if (status==cudaErrorNotReady) return 0;
    CUDA_CHECK(status);
    return 1;
}

static void ggml_backend_cuda_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    CUDA_CHECK(cudaEventSynchronize((cudaEvent_t)event->context));
}

static const ggml_backend_device_i ggml_backend_cuda_device_interface = {
    /* .get_name                = */ ggml_backend_cuda_device_get_name,
    /* .get_description         = */ ggml_backend_cuda_device_get_description,
    /* .get_memory              = */ ggml_backend_cuda_device_get_memory,
    /* .get_type                = */ ggml_backend_cuda_device_get_type,
    /* .get_props               = */ ggml_backend_cuda_device_get_props,
    /* .init_backend            = */ ggml_backend_cuda_device_init_backend,
    /* .get_buffer_type         = */ ggml_backend_cuda_device_get_buffer_type,
    /* .get_host_buffer_type    = */ ggml_backend_cuda_device_get_host_buffer_type,
    /* .buffer_from_host_ptr    = */ NULL,
    /* .supports_op             = */ ggml_backend_cuda_device_supports_op,
    /* .supports_buft           = */ ggml_backend_cuda_device_supports_buft,
    /* .offload_op              = */ ggml_backend_cuda_device_offload_op,
    /* .event_new               = */ ggml_backend_cuda_device_event_new,
    /* .event_free              = */ ggml_backend_cuda_device_event_free,
    /* .event_synchronize       = */ ggml_backend_cuda_device_event_synchronize,
};

// backend reg

struct ggml_backend_cuda_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_cuda_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_CUDA_NAME;
}

static size_t ggml_backend_cuda_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_cuda_reg_context * ctx = (ggml_backend_cuda_reg_context *)reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_cuda_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_cuda_reg_context * ctx = (ggml_backend_cuda_reg_context *)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

static ggml_backend_feature * ggml_backend_cuda_get_features(ggml_backend_reg_t reg) {
    static std::vector<ggml_backend_feature> features = []() {
        std::vector<ggml_backend_feature> features;
    #define _STRINGIFY(...) #__VA_ARGS__
    #define STRINGIFY(...) _STRINGIFY(__VA_ARGS__)

    #ifdef __CUDA_ARCH_LIST__
        features.push_back({ "ARCHS", STRINGIFY(__CUDA_ARCH_LIST__) });
    #endif

    #ifdef GGML_CUDA_FORCE_MMQ
        features.push_back({ "FORCE_MMQ", "1" });
    #endif

    #ifdef GGML_CUDA_FORCE_CUBLAS
        features.push_back({ "FORCE_CUBLAS", "1" });
    #endif

    #ifndef GGML_USE_VMM
        features.push_back({ "NO_VMM", "1" });
    #endif

    #ifdef GGML_CUDA_NO_PEER_COPY
        features.push_back({ "NO_PEER_COPY", "1" });
    #endif

    #ifdef GGML_CUDA_USE_GRAPHS
        features.push_back({ "USE_GRAPHS", "1" });
    #endif

    #ifdef GGML_CUDA_FA_QUANTS
        features.push_back({ "FA_QUANTS", GGML_CUDA_FA_QUANTS });
    #endif

    {
        const auto & info = ggml_cuda_info();
        for (int id = 0; id < info.device_count; ++id) {
            if (blackwell_mma_available(info.devices[id].cc)) {
                features.push_back({ "BLACKWELL_NATIVE_FP4", "1"});
                break;
            }
        }
    }

    #undef _STRINGIFY
    #undef STRINGIFY

        features.push_back({ nullptr, nullptr });

        return features;
    }();

    return features.data();

    GGML_UNUSED(reg);
}

static void * ggml_backend_cuda_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (strcmp(name, "ggml_backend_cuda_layer_pinned_upload") == 0) return (void *) ggml_cuda_layer_pinned_upload_private::enqueue;
#endif
    if (strcmp(name, "ggml_backend_cuda_prepare_graph_v436") == 0) return (void *) ggml_backend_cuda_prepare_graph_v436;
    if (strcmp(name, "ggml_backend_cuda_prefill_hint_v436") == 0) return (void *) ggml_backend_cuda_prefill_hint_v436;
    if (strcmp(name, "ggml_backend_cuda_set_prefill_capture_hint") == 0) return (void *) ggml_cuda_set_prefill_capture_hint;
    if (strcmp(name, "ggml_backend_cuda_set_device_tail_decode_hint") == 0) return (void *) ggml_cuda_set_device_tail_decode_hint;
    if (strcmp(name, "ggml_backend_cuda_event_query") == 0) return (void *) ggml_backend_cuda_device_event_query;
    if (strcmp(name, "ggml_backend_cuda_meta_span_init") == 0) return (void *) ggml_backend_cuda_meta_span_init;
    if (strcmp(name, "ggml_backend_cuda_meta_span_begin") == 0) return (void *) ggml_backend_cuda_meta_span_begin;
    if (strcmp(name, "ggml_backend_cuda_meta_span_end") == 0) return (void *) ggml_backend_cuda_meta_span_end;
    if (strcmp(name, "ggml_backend_cuda_meta_span_free") == 0) return (void *) ggml_backend_cuda_meta_span_free;
    if (strcmp(name, "ggml_backend_cuda_meta_copy_init") == 0) {
        return (void *) ggml_backend_cuda_meta_copy_init;
    }
    if (strcmp(name, "ggml_backend_cuda_meta_copy_free") == 0) {
        return (void *) ggml_backend_cuda_meta_copy_free;
    }
    if (strcmp(name, "ggml_backend_cuda_meta_copy_async") == 0) {
        return (void *) ggml_backend_cuda_meta_copy_async;
    }
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (strcmp(name, "ggml_cuda_ft_dense_api_v4") == 0) { return const_cast<ggml_cuda_ft_dense_api *>(ggml_cuda_ft_dense_get_api()); }
    if (strcmp(name, "ggml_cuda_ft_dense_api_v3") == 0) { return const_cast<ggml_cuda_ft_dense_api *>(ggml_cuda_ft_dense_get_api()); }
    if (strcmp(name, "ggml_cuda_ft_dense_api_v2") == 0) { return const_cast<ggml_cuda_ft_dense_api *>(ggml_cuda_ft_dense_get_api()); }
    if (strcmp(name, "ggml_cuda_ft_api_v1") == 0) { return const_cast<ggml_cuda_ft_api *>(ggml_cuda_ft_get_api()); }
    if (strcmp(name, "ggml_cuda_ft_api_v2") == 0) { return const_cast<ggml_cuda_ft_api *>(ggml_cuda_ft_get_api()); }
    if (strcmp(name, "ggml_cuda_ft_api_v3") == 0) { return const_cast<ggml_cuda_ft_api *>(ggml_cuda_ft_get_api()); }
#endif
    if (strcmp(name, "ggml_backend_comm_init") == 0) {
        return (void *)ggml_backend_cuda_comm_init;
    }
    if (strcmp(name, "ggml_backend_comm_free") == 0) {
        return (void *)ggml_backend_cuda_comm_free;
    }
    if (strcmp(name, "ggml_backend_p2p_ar_init") == 0) {
        return (void *)ggml_backend_cuda_p2p_ar_init;
    }
    if (strcmp(name, "ggml_backend_p2p_ar_free") == 0) {
        return (void *)ggml_backend_cuda_p2p_ar_free;
    }
    if (strcmp(name, "ggml_backend_p2p_ar_reserve") == 0) {
        return (void *)ggml_backend_cuda_p2p_ar_reserve;
    }
    if (strcmp(name, "ggml_backend_p2p_ar_allreduce") == 0) {
        return (void *)ggml_backend_cuda_p2p_ar_allreduce;
    }
    if (strcmp(name, "ggml_backend_p2p_ar_allreduce_async") == 0) {
        return (void *)ggml_backend_cuda_p2p_ar_allreduce_async;
    }
#ifdef USE_CUDA_GRAPH
    if (strcmp(name, "ggml_backend_cuda_compose_begin") == 0) return (void *) ggml_backend_cuda_compose_begin;
    if (strcmp(name, "ggml_backend_cuda_compose_append_child") == 0) return (void *) ggml_backend_cuda_compose_append_child;
    if (strcmp(name, "ggml_backend_cuda_compose_append_ar") == 0) return (void *) ggml_backend_cuda_compose_append_ar;
    if (strcmp(name, "ggml_backend_cuda_compose_finish") == 0) return (void *) ggml_backend_cuda_compose_finish;
    if (strcmp(name, "ggml_backend_cuda_compose_replay") == 0) return (void *) ggml_backend_cuda_compose_replay;
    if (strcmp(name, "ggml_backend_cuda_compose_free") == 0) return (void *) ggml_backend_cuda_compose_free;
#endif
    if (strcmp(name, "ggml_backend_p2p_ar_set_params") == 0) {
        return (void *)ggml_backend_cuda_p2p_ar_set_params;
    }
#if defined(GGML_CUDA_EVICT)
    if (strcmp(name, "ggml_backend_evict_weights") == 0) {
        return (void *)ggml_backend_cuda_evict_weights;
    }
    if (strcmp(name, "ggml_backend_restore_weights") == 0) {
        return (void *)ggml_backend_cuda_restore_weights;
    }
#endif // defined(GGML_CUDA_EVICT)
    if (strcmp(name, "ggml_backend_graph_ready") == 0) {
        return (void *)ggml_backend_cuda_graph_ready;
    }
#ifdef USE_CUDA_GRAPH
    if (strcmp(name, "ggml_backend_cuda_cold_cohort_scope") == 0) {
        return (void *)ggml_backend_cuda_cold_cohort_scope;
    }
#endif
    if (strcmp(name, "ggml_backend_comm_is_nccl_f32") == 0) {
        return (void *)ggml_backend_cuda_comm_is_nccl_f32;
    }
    if (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        return (void *)ggml_backend_cuda_comm_allreduce_tensor;
    }
    if (strcmp(name, "ggml_backend_register_host_buffer") == 0) {
        return (void *)ggml_backend_cuda_register_host_buffer;
    }
    if (strcmp(name, "ggml_backend_unregister_host_buffer") == 0) {
        return (void *)ggml_backend_cuda_unregister_host_buffer;
    }
    if (strcmp(name, "ggml_backend_get_features") == 0) {
        return (void *)ggml_backend_cuda_get_features;
    }
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_cuda_reg_interface = {
    /* .get_name          = */ ggml_backend_cuda_reg_get_name,
    /* .get_device_count  = */ ggml_backend_cuda_reg_get_device_count,
    /* .get_device        = */ ggml_backend_cuda_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_cuda_reg_get_proc_address,
};

// backend registry
ggml_backend_reg_t ggml_backend_cuda_reg() {
    static ggml_backend_reg reg;
    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            ggml_backend_cuda_reg_context * ctx = new ggml_backend_cuda_reg_context;
            const int min_batch_size = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;

            const ggml_cuda_device_info & info = ggml_cuda_info();
            const bool virtual_devices = info.device_count > info.physical_device_count;

            for (int i = 0; i < info.device_count; i++) {
                const int physical_id = info.devices[i].physical_device;

                ggml_backend_cuda_device_context * dev_ctx = new ggml_backend_cuda_device_context;
                dev_ctx->device = i;
                dev_ctx->name = GGML_CUDA_NAME + std::to_string(i);
                dev_ctx->description = ggml_cuda_device_description(i);

                char pci_bus_id[32] = {};
                CUDA_CHECK(cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), physical_id));
                dev_ctx->pci_bus_id = pci_bus_id;
                if (virtual_devices) {
                    // make the pci bus id unique for virtual devices
                    dev_ctx->pci_bus_id += "-v" + std::to_string(i);
                }
                for (char & c : dev_ctx->pci_bus_id) {
                    c = std::tolower(c);
                }
                dev_ctx->op_offload_min_batch_size = min_batch_size;

                ggml_backend_dev_t dev = new ggml_backend_device {
                    /* .iface   = */ ggml_backend_cuda_device_interface,
                    /* .reg     = */ &reg,
                    /* .context = */ dev_ctx
                };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg {
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ ggml_backend_cuda_reg_interface,
                /* .context     = */ ctx
            };
        }

        initialized = true;
    }

    return &reg;
}

ggml_backend_t ggml_backend_cuda_init(int device) {
    if (device < 0 || device >= ggml_backend_cuda_get_device_count()) {
        GGML_LOG_ERROR("%s: invalid device %d\n", __func__, device);
        return nullptr;
    }

    ggml_backend_cuda_context * ctx = new ggml_backend_cuda_context(device);
    if (ctx == nullptr) {
        GGML_LOG_ERROR("%s: failed to allocate context\n", __func__);
        return nullptr;
    }

    ggml_backend_t cuda_backend = new ggml_backend {
        /* .guid    = */ ggml_backend_cuda_guid(),
        /* .iface   = */ ggml_backend_cuda_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), device),
        /* .context = */ ctx,
    };

    return cuda_backend;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cuda_reg)
