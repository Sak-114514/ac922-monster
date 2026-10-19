#include <cuda_runtime.h>
#include <dlfcn.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call) do { const auto status = (call); if (status != cudaSuccess) \
    throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(status)); } while (0)

__global__ void add_one(int * values, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) ++values[index];
}

template <typename T> T symbol(void * handle, const char * name) {
    auto result = reinterpret_cast<T>(dlsym(handle, name));
    if (!result) throw std::runtime_error(std::string("missing symbol: ") + name);
    return result;
}

int main(int argc, char ** argv) {
    if (argc != 7) return 2;
    bool tracing = false;
    using stop_fn = int (*)();
    stop_fn stop = nullptr;
    try {
        const int nodes = std::stoi(argv[4]);
        const int devices = std::stoi(argv[5]);
        const std::string mode = argv[6];
        const bool copies = mode == "memops" || mode == "memcpy";
        const bool sets = mode == "memops" || mode == "memset";
        const bool memops = copies || sets;
        if (nodes < 1 || nodes > 4096 || devices < 1 || devices > 6) return 2;
        auto collector = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
        if (!collector) throw std::runtime_error(dlerror());
        auto prepare = symbol<int (*)()>(collector, "ac922_trace_prepare");
        auto start = symbol<int (*)(const char *)>(collector, "ac922_trace_start");
        stop = symbol<stop_fn>(collector, "ac922_trace_stop");
        if (prepare() != 0) throw std::runtime_error("early collector preparation failed");
        auto bridge = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
        if (!bridge) throw std::runtime_error(dlerror());
        using launch_fn = cudaError_t (*)(const void *, const void *, cudaGraph_t,
                                         int, cudaStream_t, int, int *);
        auto launch = symbol<launch_fn>(bridge, "ac922_tail_bridge_v2_launch_captured");
        auto drop = symbol<cudaError_t (*)(const void *)>(bridge, "ac922_tail_bridge_drop_context");
        constexpr int count = 4096;
        std::vector<int> owners(devices), keys(devices);
        std::vector<cudaGraph_t> graphs(devices);
        std::vector<cudaStream_t> streams(devices);
        std::vector<cudaEvent_t> begins(devices), ends(devices);
        std::vector<int *> values(devices), scratch(devices);
        for (int device = 0; device < devices; ++device) {
            CUDA_CHECK(cudaSetDevice(device));
            CUDA_CHECK(cudaStreamCreateWithFlags(&streams[device], cudaStreamNonBlocking));
            CUDA_CHECK(cudaEventCreate(&begins[device]));
            CUDA_CHECK(cudaEventCreate(&ends[device]));
            CUDA_CHECK(cudaMalloc(&values[device], count * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&scratch[device], count * sizeof(int)));
            CUDA_CHECK(cudaMemsetAsync(values[device], 0, count * sizeof(int), streams[device]));
            CUDA_CHECK(cudaStreamSynchronize(streams[device]));
            CUDA_CHECK(cudaStreamBeginCapture(streams[device], cudaStreamCaptureModeGlobal));
            for (int index = 0; index < nodes; ++index) {
                add_one<<<32, 128, 0, streams[device]>>>(values[device], count);
                if (index % 32 == 0) {
                    if (copies) CUDA_CHECK(cudaMemcpyAsync(scratch[device], values[device], count * sizeof(int),
                                                          cudaMemcpyDeviceToDevice, streams[device]));
                    if (sets) CUDA_CHECK(cudaMemsetAsync(scratch[device], 0, count * sizeof(int), streams[device]));
                }
            }
            CUDA_CHECK(cudaStreamEndCapture(streams[device], &graphs[device]));
            int outcome = -1;
            std::cerr << "r5_prime device=" << device << " nodes=" << nodes
                      << " memops=" << memops << std::endl;
            CUDA_CHECK(launch(&owners[device], &keys[device], graphs[device], 1,
                              streams[device], device, &outcome));
            if (outcome != 1) throw std::runtime_error("R5 declined captured graph: " + std::to_string(outcome));
            CUDA_CHECK(cudaStreamSynchronize(streams[device]));
        }
        if (start(argv[3]) != 0) throw std::runtime_error("collector start failed");
        tracing = true;
        for (int device = 0; device < devices; ++device) {
            CUDA_CHECK(cudaSetDevice(device));
            int outcome = -1;
            CUDA_CHECK(cudaEventRecord(begins[device], streams[device]));
            CUDA_CHECK(launch(&owners[device], &keys[device], graphs[device], 0,
                              streams[device], device, &outcome));
            if (outcome != 1) throw std::runtime_error("R5 replay declined captured graph");
            CUDA_CHECK(cudaEventRecord(ends[device], streams[device]));
        }
        for (int device = 0; device < devices; ++device) {
            CUDA_CHECK(cudaSetDevice(device));
            CUDA_CHECK(cudaStreamSynchronize(streams[device]));
        }
        const int stopped = stop();
        tracing = false;
        if (stopped != 0) throw std::runtime_error("collector stop failed");
        for (int device = 0; device < devices; ++device) {
            CUDA_CHECK(cudaSetDevice(device));
            float elapsed_ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, begins[device], ends[device]));
            std::cerr << "r5_cuda_event device=" << device << " elapsed_ms=" << elapsed_ms << std::endl;
            std::vector<int> actual(count);
            CUDA_CHECK(cudaMemcpy(actual.data(), values[device], count * sizeof(int), cudaMemcpyDeviceToHost));
            for (const int value : actual) if (value != 2 * nodes) throw std::runtime_error("R5 child oracle mismatch");
            CUDA_CHECK(drop(&owners[device]));
            CUDA_CHECK(cudaGraphDestroy(graphs[device]));
            CUDA_CHECK(cudaFree(scratch[device]));
            CUDA_CHECK(cudaFree(values[device]));
            CUDA_CHECK(cudaEventDestroy(ends[device]));
            CUDA_CHECK(cudaEventDestroy(begins[device]));
            CUDA_CHECK(cudaStreamDestroy(streams[device]));
        }
        std::cout << "{\"status\":\"PASS\",\"production_r5_export\":true,\"nodes\":" << nodes
                  << ",\"devices\":" << devices << ",\"mode\":\"" << mode << "\""
                  << ",\"memops\":" << (memops ? "true" : "false")
                  << ",\"expected_value\":" << 2 * nodes << ",\"oracle_elements_per_device\":4096}\n";
        return 0;
    } catch (const std::exception & error) {
        if (tracing && stop) stop();
        std::cerr << "r5_graph_probe: " << error.what() << '\n';
        return 1;
    }
}
