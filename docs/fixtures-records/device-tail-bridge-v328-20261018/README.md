# V328 bounded device-tail-bridge fixture

This fixture consumes the root-owned ABI in bridge-api.h. It does not edit or link against engine internals. The source is prepared for one visible SM70 device; it is not a model test or performance claim.

## ABI outcomes

| Return | Meaning | Fixture action |
|---|---|---|
| CUDA success and outcome 1 | Controller plus unchanged captured child was enqueued | Wait on the original stream's completion fence |
| CUDA success and a negative outcome | No model computation was enqueued | Launch the original cudaGraphExec_t on the requested stream and check the same exact oracle |
| CUDA error | A runtime error occurred | Log the CUDA error and stop; do not silently fall back |

The fixture checks the root header's exact outcomes: unsupported host node (-1), DeviceLaunch instantiation decline (-2), cache capacity (-3), and stream changed (-4). Only the host-node, capacity, and stream-change outcomes are required by this fixture. A legitimate instantiation decline is logged and uses the original graph; it does not count as helper selection.

## Cases

1. Four pinned input/output slots each carry their own unchanged host graph generation. Four priming launches prepare the bridge entries before the bounded pending gate.
2. Twenty frame inputs repeat the A/B/A family pattern, with frame-specific integer data. Each original stream sequence is H2D, helper or host fallback compute graph, D2H, then a per-slot completion event. The child graph contains only the exact integer transform. Four-slot batches wait for the D2H completion event before reusing a pinned source or destination.
3. With the original slot-0 generation queued behind a roughly 10 ms single-thread delay gate, the fixture checks that the slot event after external D2H is still pending, then calls the helper with the same owner/key and a replacement compute-only generation with invalidate=1. A separate pinned replacement source/destination prevents overwriting the old DMA source while invalidation drains. It verifies both generations' outputs and the D2H slot fence before releasing the old graph and source buffers.
4. A compute graph containing a CUDA host callback must return AC922_TAIL_UNSUPPORTED_NODE. The fixture then performs the external H2D, launches the original host graph, performs the external D2H and checks the callback count and exact integer output.
5. A selected slot-0 entry is submitted on the original stream behind another short gate, including its external D2H and slot event. The same owner/key/generation is presented on a different stream. The helper must drain the original completion event, return AC922_TAIL_STREAM_CHANGED, and allow exact external-copy plus original-graph fallback on the alternate stream.
6. After drop_context, a fresh context using the same owner/key identities launches and verifies a new graph. A separate empty cache lifetime checks that eight distinct eligible entries can be selected and the ninth distinct key returns AC922_TAIL_CACHE_CAPACITY; the ninth original host graph must still produce the exact result.

Each helper call logs outcome and CUDA error. Each fallback logs its result, and every frame/fallback has a checksum and element-wise integer oracle. If the 10 ms gate expires before a pending event is observed, the run records that the pending-drain gate was not proved; it does not lengthen the gate.

## Guarded runtime

run-bounded.py refuses to start while any active root-v* user service or target-GPU compute process exists. It samples the selected GPU before and during execution, stopping if temperature reaches 58 C or free memory falls below 512 MiB. The fixture process has an 82-second command timeout inside an 85-second host deadline. It uses one visible GPU and never calls cudaDeviceReset.

Target the first V100 by default or set V328_GPU_UUID to another physical GPU UUID. The helper library defaults to build-r1/libac922-tail-bridge.so and can be overridden with V328_HELPER_SO. The fixture binary defaults to build-fixture-r2-fixed/bridge-fixture and can be overridden with V328_FIXTURE_BIN.

## R1 and R2 evidence

R1 is preserved at build-fixture-r1. Its bounded run stopped before any helper call because cudaGraphAddMemcpyNode1D was passed a null output-node pointer; the as-run log is results/v328-20261017t192734z.log. That failure was in the fixture graph-construction API call, not evidence about the helper. R2 removes H2D/D2H nodes from the child graph and moves both copies to the original host stream around helper/host-graph execution; the child now contains only the integer kernel. R1 source/binary/log artifacts remain untouched.

The first R2 CPU compile is preserved at build-fixture-r2; it linked but nvcc reported multi-character newline-literal warnings in diagnostic output. R2 source then normalized those log escapes and was rebuilt at build-fixture-r2-fixed. The fixed R2 compile uses CUDA 12.4, target SM70, shared cudart, and NUMA-bound CPU compilation. Its ldd resolves the helper DSO from build-r1 and libcudart.so.12 from CUDA 12.4 with no missing libraries. The fixed compile exited 0 with an empty compile log. Its binary SHA256 is `3b47e66ff24f27a50a02d8b6ae4bb61db5cf2b4249b224387d2e8b5ebf5b2411`; the as-built fixture source SHA256 is `ff9a0330f0c6c549c7cf98744038797cdea19ebaa0cbe519ab57bd9ac54bd3d4`. The binary has not been executed; no CUDA runtime API or GPU test is claimed.

The host build command for R2 is:

~~~sh
HERE=/home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/device-tail-bridge-v328-20261018
CUDA_LIB=/usr/local/cuda-12.4/targets/ppc64le-linux/lib
HELPER_DIR=$HERE/build-r1
OUT=$HERE/build-fixture-r2-fixed
mkdir -p "$OUT"
numactl --cpunodebind=0 --membind=0,8 /usr/local/cuda-12.4/bin/nvcc -std=c++17 -arch=sm_70 -cudart shared \
  -I"$HERE" "$HERE/bridge-fixture.cu" "$HELPER_DIR/libac922-tail-bridge.so" \
  -Xlinker -rpath -Xlinker "$CUDA_LIB" \
  -Xlinker -rpath -Xlinker "$HELPER_DIR" \
  -o "$OUT/bridge-fixture"
LD_LIBRARY_PATH="$CUDA_LIB:$HELPER_DIR" ldd "$OUT/bridge-fixture"
~~~

After root explicitly releases a GPU window, run the guarded fixture with:

~~~sh
python3 run-bounded.py
~~~

## Evidence limits

A full PASS proves only this single-GPU fixture's copy ordering, integer graph execution, helper outcome handling, pending invalidation, stream-change decline, cache bound, and context teardown/recreation. It does not prove six-GPU scheduling, full llama.cpp graph eligibility, numerical model behavior, or a performance improvement. The helper and fixture must retain the original graph handles until the helper context is dropped.
