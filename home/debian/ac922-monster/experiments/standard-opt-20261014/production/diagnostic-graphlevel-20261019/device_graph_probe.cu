#include <cuda_runtime.h>
#include <cupti.h>
#include <dlfcn.h>

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call) do { const auto status = (call); if (status != cudaSuccess) \
    throw std::runtime_error(cudaGetErrorString(status)); } while (0)

__global__ void add_one(int * values, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) ++values[index];
}

__global__ void launch_child(cudaGraphExec_t child, int * status) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *status = int(cudaGraphLaunch(child, cudaStreamGraphFireAndForget));
    }
}

int main(int argc, char ** argv) {
    if (argc != 3 && argc != 4) return 2;
    const bool parent_device = argc == 4 && std::string(argv[3]) == "device";
    void * library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!library) { std::cerr << dlerror() << '\n'; return 3; }
    const auto prepare = reinterpret_cast<int (*)()>(dlsym(library, "ac922_trace_prepare"));
    const auto start = reinterpret_cast<int (*)(const char *)>(dlsym(library, "ac922_trace_start"));
    const auto stop = reinterpret_cast<int (*)()>(dlsym(library, "ac922_trace_stop"));
    if (!prepare || !start || !stop) return 4;
    bool tracing = false;
    try {
        if (prepare() != 0) throw std::runtime_error("collector early prepare failed");
        CUDA_CHECK(cudaSetDevice(0));
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        int * values = nullptr;
        int * launch_status = nullptr;
        constexpr int count = 4096;
        CUDA_CHECK(cudaMalloc(&values, count * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&launch_status, sizeof(int)));
        CUDA_CHECK(cudaMemsetAsync(values, 0, count * sizeof(int), stream));
        CUDA_CHECK(cudaMemsetAsync(launch_status, 0xff, sizeof(int), stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        cudaGraph_t child = nullptr, parent = nullptr;
        cudaGraphExec_t child_exec = nullptr, parent_exec = nullptr;
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        for (int index = 0; index < 4; ++index) add_one<<<32, 128, 0, stream>>>(values, count);
        CUDA_CHECK(cudaStreamEndCapture(stream, &child));
        CUDA_CHECK(cudaGraphInstantiateWithFlags(&child_exec, child, cudaGraphInstantiateFlagDeviceLaunch));
        CUDA_CHECK(cudaGraphUpload(child_exec, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        launch_child<<<1, 1, 0, stream>>>(child_exec, launch_status);
        CUDA_CHECK(cudaStreamEndCapture(stream, &parent));
        CUDA_CHECK(cudaGraphInstantiateWithFlags(&parent_exec, parent,
                parent_device ? cudaGraphInstantiateFlagDeviceLaunch : 0));
        CUDA_CHECK(cudaGraphUpload(parent_exec, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        uint32_t child_id = 0, parent_id = 0, child_exec_id = 0, parent_exec_id = 0;
        if (cuptiGetGraphId(reinterpret_cast<CUgraph>(child), &child_id) != CUPTI_SUCCESS ||
            cuptiGetGraphId(reinterpret_cast<CUgraph>(parent), &parent_id) != CUPTI_SUCCESS)
            throw std::runtime_error("CUPTI graph identity unavailable");
        if (cuptiGetGraphExecId(reinterpret_cast<CUgraphExec>(child_exec), &child_exec_id) != CUPTI_SUCCESS ||
            cuptiGetGraphExecId(reinterpret_cast<CUgraphExec>(parent_exec), &parent_exec_id) != CUPTI_SUCCESS)
            throw std::runtime_error("CUPTI executable graph identity unavailable");
        if (start(argv[2]) != 0) throw std::runtime_error("collector start failed");
        tracing = true;
        CUDA_CHECK(cudaGraphLaunch(parent_exec, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const int stop_result = stop();
        tracing = false;
        if (stop_result != 0) throw std::runtime_error("collector stop failed");
        std::vector<int> result(count);
        int device_launch_result = -1;
        CUDA_CHECK(cudaMemcpy(result.data(), values, count * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&device_launch_result, launch_status, sizeof(int), cudaMemcpyDeviceToHost));
        if (device_launch_result != cudaSuccess) throw std::runtime_error("device graph launch failed");
        for (const auto value : result) if (value != 4) throw std::runtime_error("child output oracle failed");
        CUDA_CHECK(cudaGraphExecDestroy(parent_exec));
        CUDA_CHECK(cudaGraphExecDestroy(child_exec));
        CUDA_CHECK(cudaGraphDestroy(parent));
        CUDA_CHECK(cudaGraphDestroy(child));
        CUDA_CHECK(cudaFree(values));
        CUDA_CHECK(cudaFree(launch_status));
        CUDA_CHECK(cudaStreamDestroy(stream));
        std::cout << "{\"status\":\"PASS\",\"actual_child_graph_id\":" << child_id
                  << ",\"actual_parent_graph_id\":" << parent_id
                  << ",\"actual_child_exec_id\":" << child_exec_id
                  << ",\"actual_parent_exec_id\":" << parent_exec_id
                  << ",\"parent_device_launch_enabled\":" << (parent_device ? "true" : "false")
                  << ",\"oracle_elements\":4096,\"expected_value\":4,\"device\":0}\n";
        return 0;
    } catch (const std::exception & error) {
        if (tracing) stop();
        std::cerr << "device_graph_probe: " << error.what() << '\n';
        return 1;
    }
}
