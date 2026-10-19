#include "helper/bridge-api.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace {

struct invocation {
    int * input = nullptr;
    int * scratch1 = nullptr;
    int * scratch2 = nullptr;
    int * output = nullptr;
    int p1 = 0;
    int p2 = 0;
    int p3 = 0;
    int expected = 0;
    char topology = '?';
    cudaGraph_t graph = nullptr;
};

struct key_token { int value; };

__global__ void kernel_a(const int * input, int * output, int bias) {
    if (blockIdx.x == 0 && threadIdx.x == 0) *output = *input + bias;
}

__global__ void kernel_b1(const int * input, int * scratch, int scale, int bias) {
    if (blockIdx.x == 0 && threadIdx.x == 0) *scratch = *input * scale + bias;
}

__global__ void kernel_b2(const int * scratch, int * output, int add) {
    if (blockIdx.x == 0 && threadIdx.x == 0) *output = *scratch + add;
}

__global__ void kernel_c1(const int * input, int * scratch, int scale, int bias) {
    if (blockIdx.x == 0 && threadIdx.x == 0) *scratch = *input * scale + bias;
}

__global__ void kernel_c2(const int * scratch, int * next, int add) {
    if (blockIdx.x == 0 && threadIdx.x == 0) *next = *scratch + add;
}

__global__ void kernel_c3(const int * next, int * output, int scale, int bias) {
    if (blockIdx.x == 0 && threadIdx.x == 0) *output = *next * scale + bias;
}

void check_cuda(cudaError_t status, const char * expression) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(expression) + ": " + cudaGetErrorString(status));
    }
}

#define CUDA_CHECK(expression) check_cuda((expression), #expression)

void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

void allocate_one(int ** pointer, int initial) {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(pointer), sizeof(int)));
    CUDA_CHECK(cudaMemcpy(*pointer, &initial, sizeof(initial), cudaMemcpyHostToDevice));
}

cudaGraphNode_t add_kernel(cudaGraph_t graph, void * function,
        void ** arguments, cudaGraphNode_t * dependency) {
    cudaKernelNodeParams params{};
    params.func = function;
    params.gridDim = dim3(1);
    params.blockDim = dim3(1);
    params.kernelParams = arguments;
    cudaGraphNode_t node = nullptr;
    CUDA_CHECK(cudaGraphAddKernelNode(&node, graph, dependency,
            dependency ? 1 : 0, &params));
    return node;
}

void initialize(invocation & call, int device, char topology, int seed, int ordinal) {
    call.topology = topology;
    const int initial = seed + device * 97 + ordinal * 19;
    allocate_one(&call.input, initial);
    allocate_one(&call.scratch1, -100 - ordinal);
    allocate_one(&call.scratch2, -200 - ordinal);
    allocate_one(&call.output, -300 - ordinal);
    CUDA_CHECK(cudaGraphCreate(&call.graph, 0));

    if (topology == 'A') {
        call.p1 = 3 + ordinal;
        void * args[] = { &call.input, &call.output, &call.p1 };
        add_kernel(call.graph, reinterpret_cast<void *>(kernel_a), args, nullptr);
        call.expected = initial + call.p1;
        return;
    }

    if (topology == 'B') {
        call.p1 = 2 + (ordinal % 3);
        call.p2 = 5 + ordinal;
        call.p3 = 7 + ordinal;
        void * first_args[] = { &call.input, &call.scratch1, &call.p1, &call.p2 };
        cudaGraphNode_t first = add_kernel(call.graph,
                reinterpret_cast<void *>(kernel_b1), first_args, nullptr);
        void * second_args[] = { &call.scratch1, &call.output, &call.p3 };
        add_kernel(call.graph, reinterpret_cast<void *>(kernel_b2), second_args, &first);
        call.expected = initial * call.p1 + call.p2 + call.p3;
        return;
    }

    require(topology == 'C', "unknown fixture topology");
    call.p1 = 2 + (ordinal % 2);
    call.p2 = 11 + ordinal;
    call.p3 = 3 + (ordinal % 4);
    int bias = 13 + ordinal;
    void * first_args[] = { &call.input, &call.scratch1, &call.p1, &bias };
    cudaGraphNode_t first = add_kernel(call.graph,
            reinterpret_cast<void *>(kernel_c1), first_args, nullptr);
    void * second_args[] = { &call.scratch1, &call.scratch2, &call.p2 };
    cudaGraphNode_t second = add_kernel(call.graph,
            reinterpret_cast<void *>(kernel_c2), second_args, &first);
    void * third_args[] = { &call.scratch2, &call.output, &call.p3, &call.p2 };
    add_kernel(call.graph, reinterpret_cast<void *>(kernel_c3), third_args, &second);
    call.expected = (initial * call.p1 + bias + call.p2) * call.p3 + call.p2;
}

int launch(const void * owner, const void * key, const invocation & call,
        cudaStream_t stream, int device) {
    int outcome = 0;
    CUDA_CHECK(ac922_tail_bridge_v2_launch_captured(owner, key, call.graph,
            0, stream, device, &outcome));
    return outcome;
}

void destroy_call(invocation & call) {
    if (call.graph) CUDA_CHECK(cudaGraphDestroy(call.graph));
    if (call.output) CUDA_CHECK(cudaFree(call.output));
    if (call.scratch2) CUDA_CHECK(cudaFree(call.scratch2));
    if (call.scratch1) CUDA_CHECK(cudaFree(call.scratch1));
    if (call.input) CUDA_CHECK(cudaFree(call.input));
    call = {};
}

int read_output(const invocation & call) {
    int actual = 0;
    CUDA_CHECK(cudaMemcpy(&actual, call.output, sizeof(actual), cudaMemcpyDeviceToHost));
    return actual;
}

void run_device(int device, int device_count) {
    CUDA_CHECK(cudaSetDevice(device));
    int owner_token = 44100 + device;
    const void * owner = &owner_token;
    std::array<key_token, 8> keys{};
    for (int i = 0; i < int(keys.size()); ++i) keys[size_t(i)].value = 44200 + device * 16 + i;
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    constexpr std::array<char, 8> topology_sequence{'A','B','B','B','B','C','B','B'};
    std::array<invocation, 8> first_wave{};
    invocation same_key_b{};
    invocation same_key_a{};
    for (int i = 0; i < int(first_wave.size()); ++i) {
        initialize(first_wave[size_t(i)], device, topology_sequence[size_t(i)],
                1000 + i * 31, i + 1);
    }
    initialize(same_key_b, device, 'B', 5301, 101);
    initialize(same_key_a, device, 'A', 6301, 102);

    // Queue distinct owner keys in A/B/B/B/B/C/B/B order without an
    // inter-launch stream/device synchronization.
    for (size_t i = 0; i < first_wave.size(); ++i) {
        require(launch(owner, &keys[i], first_wave[i], stream, device) == AC922_TAIL_LAUNCHED,
                "topology-bucket first-wave launch was declined");
    }

    // Readiness is metadata-only. With six devices, query every owner from a
    // different current device and verify the helper leaves that device alone.
    const int ready_device = device_count == 6 ? (device == 0 ? 5 : 0) : device;
    CUDA_CHECK(cudaSetDevice(ready_device));
    int current_before = -1;
    int current_after = -1;
    CUDA_CHECK(cudaGetDevice(&current_before));
    int ready = 0;
    CUDA_CHECK(ac922_tail_bridge_v2_ready(owner, &keys[0], first_wave[0].graph,
            device, &ready));
    CUDA_CHECK(cudaGetDevice(&current_after));
    require(ready == 1, "old key zero is not ready after the initial topology sequence");
    require(current_before == ready_device && current_after == current_before,
            "ready changed the caller's current CUDA device");
    CUDA_CHECK(cudaSetDevice(device));

    // The same key transitions A -> B -> A. Parameters and output pointers
    // differ for all three bindings; the captured topology hint only selects
    // a bucket and never grants replay without the helper's driver update.
    require(launch(owner, &keys[0], same_key_b, stream, device) == AC922_TAIL_LAUNCHED,
            "same-key topology B rebind was declined");
    require(launch(owner, &keys[0], same_key_a, stream, device) == AC922_TAIL_LAUNCHED,
            "same-key topology A rebind was declined");

    // This is the only explicit stream synchronization in the test. It occurs
    // after every queued launch, so every result is checked against its oracle.
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (const auto & call : first_wave) {
        require(read_output(call) == call.expected, "first-wave exact output oracle failed");
    }
    require(read_output(same_key_b) == same_key_b.expected,
            "same-key topology B exact output oracle failed");
    require(read_output(same_key_a) == same_key_a.expected,
            "same-key topology A exact output oracle failed");

    CUDA_CHECK(ac922_tail_bridge_drop_context(owner));
    for (auto & call : first_wave) destroy_call(call);
    destroy_call(same_key_b);
    destroy_call(same_key_a);
    CUDA_CHECK(cudaStreamDestroy(stream));

    const int cross_device = ready_device != device;
    std::printf("V441_TOPOLOGY_BUCKETS_PASS device=%d owner=%p stream=%p keys=8 "
            "first_sequence=ABBBBCBB same_key_rebind=ABA ready=1 ready_device=%d "
            "current_unchanged=1 cross_device=%d outputs_exact=1 no_inter_launch_sync=1\n",
            device, owner, static_cast<void *>(stream), ready_device, cross_device);
}

} // namespace

int main(int argc, char ** argv) {
    try {
        require(argc <= 2, "usage: test_topology_buckets [1|6]");
        const int requested = argc == 2 ? std::atoi(argv[1]) : 1;
        require(requested == 1 || requested == 6, "device count must be 1 or 6");
        int available = 0;
        CUDA_CHECK(cudaGetDeviceCount(&available));
        require(available >= requested, "requested CUDA device count is unavailable");
        for (int device = 0; device < requested; ++device) run_device(device, requested);
        std::printf("V441_TOPOLOGY_BUCKETS_COMPLETE devices=%d slots=%d "
                "distinct_buckets_per_device=3 outputs_exact=1 middle_updates_required=1\n",
                requested, 3 * requested);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "V441_TOPOLOGY_BUCKETS_FAIL %s\n", error.what());
        return 1;
    }
}
