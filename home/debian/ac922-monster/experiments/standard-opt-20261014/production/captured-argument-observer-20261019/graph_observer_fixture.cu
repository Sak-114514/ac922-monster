#include "captured-observer-private.hpp"
#include "bridge-api.h"
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK(call) do { auto error = (call); if (error != cudaSuccess) \
    throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(error)); } while (0)

__global__ void add_scalar(const int * input, int * output, int count, int value) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) output[i] = input[i] + value;
}

void require(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

cudaGraph_t make_graph(int * input, int * output, int delta, int grid = 32) {
    cudaGraph_t graph = nullptr;
    CHECK(cudaGraphCreate(&graph, 0));
    cudaKernelNodeParams params{};
    params.func = reinterpret_cast<void *>(add_scalar);
    params.gridDim = dim3(grid);
    params.blockDim = dim3(128);
    int count = 4096;
    void * args[] = {&input, &output, &count, &delta};
    params.kernelParams = args;
    cudaGraphNode_t node = nullptr;
    CHECK(cudaGraphAddKernelNode(&node, graph, nullptr, 0, &params));
    return graph;
}

cudaGraph_t make_empty_graph(bool connected) {
    cudaGraph_t graph = nullptr;
    CHECK(cudaGraphCreate(&graph, 0));
    cudaGraphNode_t first = nullptr, second = nullptr;
    CHECK(cudaGraphAddEmptyNode(&first, graph, nullptr, 0));
    CHECK(cudaGraphAddEmptyNode(&second, graph, connected ? &first : nullptr, connected ? 1 : 0));
    return graph;
}

cudaGraph_t make_mixed_graph(int * input, int * output, int * other,
                             unsigned value = 0, size_t copy_count = 4096) {
    cudaGraph_t graph = nullptr;
    CHECK(cudaGraphCreate(&graph, 0));
    cudaGraphNode_t fill = nullptr, copy = nullptr, kernel = nullptr;
    cudaMemsetParams memset_params{};
    memset_params.dst = other;
    memset_params.elementSize = sizeof(int);
    memset_params.width = 4096;
    memset_params.height = 1;
    memset_params.value = value;
    CHECK(cudaGraphAddMemsetNode(&fill, graph, nullptr, 0, &memset_params));
    CHECK(cudaGraphAddMemcpyNode1D(&copy, graph, &fill, 1, output, input,
                                  copy_count * sizeof(int), cudaMemcpyDeviceToDevice));
    cudaKernelNodeParams params{};
    params.func = reinterpret_cast<void *>(add_scalar);
    params.gridDim = dim3(32);
    params.blockDim = dim3(128);
    int count = 4096, delta = 7;
    void * args[] = {&output, &other, &count, &delta};
    params.kernelParams = args;
    CHECK(cudaGraphAddKernelNode(&kernel, graph, &copy, 1, &params));
    return graph;
}

bool equal(const ac922_captured_observer::observation & a,
           const ac922_captured_observer::observation & b) {
    return a.selected && b.selected && a.hash_low == b.hash_low &&
           a.hash_high == b.hash_high && a.serialized_bytes == b.serialized_bytes;
}

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    try {
        CHECK(cudaSetDevice(0));
        int * input = nullptr, * output = nullptr, * other = nullptr;
        CHECK(cudaMalloc(&input, 4096 * sizeof(int)));
        CHECK(cudaMalloc(&output, 4096 * sizeof(int)));
        CHECK(cudaMalloc(&other, 4096 * sizeof(int)));
        std::vector<int> host(4096, 19);
        CHECK(cudaMemcpy(input, host.data(), host.size()*sizeof(int), cudaMemcpyHostToDevice));
        std::array<cudaGraph_t, 5> graphs = {
            make_graph(input, output, 7), make_graph(input, output, 7),
            make_graph(input, output, 8), make_graph(input, other, 7),
            make_graph(input, output, 7, 64)};
        ac922_captured_observer::history history;
        std::array<ac922_captured_observer::observation, 5> seen;
        for (int i = 0; i < 5; ++i) {
            seen[i] = history.observe(graphs[i], 0);
            require(seen[i].selected, seen[i].reason.c_str());
            CHECK(cudaPeekAtLastError());
        }
        require(equal(seen[0], seen[1]) && seen[1].seen_hash_before,
                "identical captured scalar/pointer/launch metadata did not match");
        require(!equal(seen[0], seen[2]), "changed scalar failed to change fingerprint");
        require(!equal(seen[0], seen[3]), "changed output pointer failed to change fingerprint");
        require(!equal(seen[0], seen[4]), "changed launch geometry failed to change fingerprint");
        std::array<cudaGraph_t, 4> mixed = {
            make_mixed_graph(input, output, other), make_mixed_graph(input, output, other),
            make_mixed_graph(input, output, other, 1), make_mixed_graph(input, output, other, 0, 2048)};
        const auto mixed_base = history.observe(mixed[0], 0);
        require(equal(mixed_base, history.observe(mixed[1], 0)), "mixed graph metadata mismatch");
        const auto changed_fill = history.observe(mixed[2], 0), changed_copy = history.observe(mixed[3], 0);
        require(changed_fill.selected && !equal(mixed_base, changed_fill), "changed memset value not detected");
        require(changed_copy.selected && !equal(mixed_base, changed_copy), "changed memcpy size not detected");
        auto empty = make_empty_graph(false), connected = make_empty_graph(true);
        const auto independent_nodes = history.observe(empty, 0), dependent_nodes = history.observe(connected, 0);
        require(independent_nodes.selected && dependent_nodes.selected && !equal(independent_nodes, dependent_nodes),
                "dependency edge not detected");
        cudaGraph_t child = nullptr;
        cudaGraphNode_t child_node = nullptr;
        CHECK(cudaGraphCreate(&child, 0));
        CHECK(cudaGraphAddChildGraphNode(&child_node, child, nullptr, 0, graphs[0]));
        const auto child_observation = history.observe(child, 0);
        require(!child_observation.selected && child_observation.reason == "child_graph_unsupported_in_flat_observer",
                "unsupported child graph did not decline observation");
        CHECK(cudaPeekAtLastError());

        void * library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
        require(library != nullptr, dlerror());
        using launch_fn = cudaError_t (*)(const void *, const void *, cudaGraph_t, int,
                                         cudaStream_t, int, int *);
        auto launch = reinterpret_cast<launch_fn>(dlsym(library, "ac922_tail_bridge_v2_launch_captured"));
        auto drop = reinterpret_cast<cudaError_t (*)(const void *)>(dlsym(library, "ac922_tail_bridge_drop_context"));
        require(launch && drop, "original R5 ABI exports missing");
        cudaStream_t stream = nullptr;
        CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        int owner = 1;
        std::array<int, 5> keys{};
        for (int i = 0; i < 5; ++i) {
            int outcome = 0;
            CHECK(launch(&owner, &keys[i], graphs[i], 1, stream, 0, &outcome));
            require(outcome == AC922_TAIL_LAUNCHED, "R5 graph declined");
            CHECK(cudaStreamSynchronize(stream));
            int * source = i == 3 ? other : output;
            CHECK(cudaMemcpy(host.data(), source, host.size()*sizeof(int), cudaMemcpyDeviceToHost));
            for (int value : host) require(value == (i == 2 ? 27 : 26), "R5 mathematical oracle failed");
        }
        int mixed_key = 0, mixed_outcome = 0;
        CHECK(launch(&owner, &mixed_key, mixed[0], 1, stream, 0, &mixed_outcome));
        require(mixed_outcome == AC922_TAIL_LAUNCHED, "mixed R5 graph declined");
        CHECK(cudaStreamSynchronize(stream));
        CHECK(cudaMemcpy(host.data(), other, host.size()*sizeof(int), cudaMemcpyDeviceToHost));
        for (int value : host) require(value == 26, "mixed graph mathematical oracle failed");
        // A new input payload at the same address must not change launch
        // metadata; the original graph still reads the new live contents.
        std::fill(host.begin(), host.end(), 23);
        CHECK(cudaMemcpy(input, host.data(), host.size()*sizeof(int), cudaMemcpyHostToDevice));
        require(equal(seen[0], history.observe(graphs[0], 0)), "observer read GPU payload contents");
        int outcome = 0;
        CHECK(launch(&owner, &keys[0], graphs[0], 0, stream, 0, &outcome));
        CHECK(cudaStreamSynchronize(stream));
        CHECK(cudaMemcpy(host.data(), output, host.size()*sizeof(int), cudaMemcpyDeviceToHost));
        for (int value : host) require(value == 30, "R5 graph did not read changed payload");
        CHECK(drop(&owner));
        CHECK(cudaGraphDestroy(child));
        CHECK(cudaGraphDestroy(empty));
        CHECK(cudaGraphDestroy(connected));
        for (auto graph : mixed) CHECK(cudaGraphDestroy(graph));
        for (auto graph : graphs) CHECK(cudaGraphDestroy(graph));
        CHECK(cudaStreamDestroy(stream));
        CHECK(cudaFree(other));
        CHECK(cudaFree(output));
        CHECK(cudaFree(input));
        std::cout << "{\"status\":\"PASS\",\"device\":0,\"oracle_elements\":4096,"
                     "\"duplicate_detected\":true,\"changed_scalar_pointer_grid_detected\":true,"
                     "\"mixed_memcpy_memset_oracle\":true,\"dependency_edge_detected\":true,"
                     "\"unsupported_child_declined\":true,"
                     "\"changed_gpu_payload_read_live\":true,\"replay_authorized\":false}\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "graph_observer_fixture: " << error.what() << '\n';
        return 1;
    }
}
