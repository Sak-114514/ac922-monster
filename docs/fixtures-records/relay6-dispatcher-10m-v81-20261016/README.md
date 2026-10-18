# V81 Relay6 10 MiB actual-dispatch fixture

This is a fixture-only V81 copy of
`fixtures/relay6-dispatcher-10m-20261016`. It uses the current frozen V81
runtime and included V81 `allreduce-p2p.cu` source. It does not edit or rebuild
the engine. The fixture exercises the actual `p2p_ar_prepare_launch()`
dispatcher for both segmented kind 2 and relay kind 3, with two independent
communicators, CUDA graph capture/replay, exact rank-ordered F32 outputs,
zero-slice semantics, and input/output canaries.

The source has the required `-DGGML_CUDA_USE_GRAPHS` compile define. V81 build
ID: `20261016T073107.274191Z`; the frozen
`runtime-perf-v81/lib/libggml-cuda.so.0` SHA-256 is
`2f22091738127c31d1bdef499d9b55ec2e21e4cedac911fbce9e61923bdc15e6`.
The V81 P2P source SHA-256 at preparation time is
`50390d1f90c7e001150d661fb4bbfa245dd4eba71a533390de8e4389bdb17101`.

## Host build

On AC922, from the V81 source tree, the fixture-only builder validates the
build note, V81 runtime directory, runtime dependencies, and source paths before
compiling the small fixture executable. NVCC is used to compile this `.cu`
fixture; it does not launch a kernel or consume GPU memory.

```sh
./fixtures/relay6-dispatcher-10m-v81-20261016/build.sh \
  /home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v81 \
  v81-relay10m-host1
```

The build output and binary hashes are retained in the new fixture directory.
Do not reuse the V77 executable from the older fixture.

## Bounded actual-dispatch command

GPU execution remains gated by an explicit root grant. After the V35 and QSA
windows finish and root grants this fixture's window, run:

```sh
RELAY6_DISPATCHER_GPU_WINDOW_AUTHORIZED=1 \
./fixtures/relay6-dispatcher-10m-v81-20261016/run-bounded.sh \
  /home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v81 \
  v81-relay10m-host1
```

The runner retains the existing 58 C ceiling, per-GPU 512 MiB free-memory
floor, no-compute-PID requirement, per-mode 25 second timeout, and 60 second
total bound. Its fixture-owned used-memory delta cap is 1 GiB per GPU, matching
the C++ fixture's per-rank allocation check. The 512 MiB floor is checked on
all six GPUs before, during, and after execution; it is independent of the
used-memory delta ceiling. Every GPU must select kind 2 for segmented mode
and kind 3 for relay mode. A fallback is a failed relay qualification.

The runner reads only GPU status with `nvidia-smi`; it does not change fan or
BMC state and does not start or modify any watcher. Preserve the machine's
intentional fan/BMC masking state and existing watchdog/bootguard state.

The V81 actual-dispatch run passed; see `RESULTS.md` for the exactness, memory,
timing, and log-hash record. It is not an end-to-end model result.

## V86 localgate

V86 changes only `allreduce-p2p.cu` logging relative to its frozen parent.
The fixture source includes `allreduce-p2p.cu` directly. The V81 binary
therefore embeds V81's dispatcher implementation; changing only its
`LD_LIBRARY_PATH` does not qualify it for V86. Build a new small fixture binary
with the V86 source and libraries, then run the host-only closure check:

```sh
./fixtures/relay6-dispatcher-10m-v81-20261016/build.sh \
  /home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v86 \
  v86-relay10m-host1 v86
./fixtures/relay6-dispatcher-10m-v81-20261016/verify-v86-localgate.sh
```

The build and closure check do not execute the binary or access a GPU. They
verify the V86 source hash, embedded logger/dispatcher symbols, runtime build
ID, and exact resolved DSO closure. After a separate root GPU grant, run the
new V86 binary with profile `v86`. The marker validator requires
`p2p-ar-path entry=dispatch` relay6 records for ranks 0 through 5, exact
payload bytes, rank-order markers, and matching relay6 confirmations. This
fixture captures graphs by wrapping `ggml_cuda_op_allreduce()` in
`cudaStreamBeginCapture()` / `cudaStreamEndCapture()`; it does not call
`ggml_cuda_p2p_ar_graph_append()`. The validator reports that path as
`SCOPE_LIMITATION` and does not claim graph-append coverage. Use a distinct log
tag and point `RELAY6_DISPATCHER_BINARY_TAG` at the V86 binary:

```sh
RELAY6_DISPATCHER_GPU_WINDOW_AUTHORIZED=1 \
RELAY6_DISPATCHER_BINARY_TAG=v86-relay10m-host1 \
./fixtures/relay6-dispatcher-10m-v81-20261016/run-bounded.sh \
  /home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v86 \
  v86-relay10m-localgate v86
```

The static V86 localgate is not runtime dispatch proof. No V86 GPU result is
recorded until that separately authorized command completes.
