# Captured graph argument observer

This private C++17 header produces bounded, non-cryptographic metadata fingerprints for a raw captured CUDA graph. It dynamically resolves CUDA 12.4 driver metadata functions from `libcuda.so.1`; it does not call `cuInit`, launch, synchronize, allocate device memory, clear errors, mutate a graph, access a `CUgraphExec`, or replay anything.

`selected` means only that the supported flat graph metadata was read successfully. `seen_hash_before` reports a prior match in a 128-entry in-memory hash history. A match is only a potential duplicate; graph attributes, user-object lifetimes, and all other semantic state are not proven equivalent. It is never authorization to update, cache, or replay an executable graph.

The implementation supports kernel, memcpy, memset, and empty nodes plus sorted dependency edges. Child-graph and all other node types decline with a reason. This keeps the first implementation small and fail-closed. It deep-copies host argument bytes using `cuFuncGetParamInfo` sizes, rejects `extra` packed arguments, caps each parameter blob at 32 KiB and 128 parameters, each graph at 2 MiB, nodes at 4096, and edges at 16384. Function parameter layouts are shared only within a single `observe()` call (at most 256 entries); only small hash pairs persist between calls.

The caller must hold the graph owner's lock while invoking `observe()`, so the graph and node-owned `kernelParams` storage remain stable during metadata reads. CUDA Driver API documentation for the relevant [CUDA 12.4 graph metadata functions](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-driver-api/group__CUDA__GRAPH.html), [parameter layouts](https://docs.nvidia.com/cuda/archive/12.4.0/pdf/CUDA_Driver_API.pdf), and [memcpy node fields](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-driver-api/group__CUDA__MEM.html) is the reference for this surface.

## CPU-only syntax check on AC922

Copy only this header, `captured-observer-syntax.cpp`, and this README into the matching private directory. Compile the stub without linking or executing it:

```sh
g++ -std=c++17 -Wall -Wextra -Werror \
  -I/usr/local/cuda-12.4/include \
  -c captured-observer-syntax.cpp -o /tmp/captured-observer-syntax.o
```

This is a compile-only check. Do not execute the stub or call `history::observe()` against a live graph during this verification step.
