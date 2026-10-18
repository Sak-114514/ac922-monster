# FlatMeta6 cold graph overlap fixture

This minimal six-GPU F32 fixture checks the first cold merged graph and an
immediate graph topology update. It uses one `MUL_MAT` with W and X split over
K axis 0, a 24-element F32 partial (96 bytes), and a small ADD tail. A and B
share the exact first tensor and merged CUDA child key; B adds one final ADD in
the slot occupied by A's view placeholder, keeping merged graph allocations the same size. The measured
pair submits A asynchronously and then B asynchronously without a host sync;
the control arm synchronizes between them. Each process has independent graph
and CUDA state.

Both shapes are prepared once and synchronized before the measured pair so the
measured calls enter Meta's one-chunk merged graph path. `--scale-nodes 4096`
adds a chain of identity F32 `SCALE` nodes after the common tail. The fixture
reports realized child graph sizes, per-rank `ALLREDUCE` nodes and byte count,
CUDA capture/update logs, output oracle, host read canary, and pthread stack
sizes for the main and Meta worker threads.

## Build on AC922

```sh
./build-fixture.sh /home/debian/ac922-monster/experiments/standard-opt-20261014 \
  /home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v89 v89
```

The build uses `numactl --cpunodebind=0 --membind=0,8` and links the selected
runtime closure. It does not initialize CUDA or load a model.

## Run

The default run executes independent `force=1/sync=0`, `force=1/sync=1`,
`force=0/sync=0`, and `force=0/sync=1` processes. Each process is capped at
20 seconds and all four share a 120-second wall envelope. The runner checks all
six UUIDs for compute PIDs, temperature below 58 C, and at most 512 MiB before
each arm; it stops if runtime memory exceeds 1 GiB or the post-run GPUs do not
return idle.

```sh
./run-bounded.sh /home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v89 v89
```

A single arm and the optional large graph use. SCALE node counts are accepted
from 0 through 10,000; `--node-trace` adds per-node dispatch logging and can
add host overhead for deep graphs.

```sh
./run-bounded.sh RUNTIME TAG --force 1 --sync-between 0 --scale-nodes 4096 --large-stack
```

The V90 cold-drain control can reuse the same V89-linked fixture binary while
placing the V90 closure first in `LD_LIBRARY_PATH`; add
`--single-chunk-cold-drain` to set the V90 opt-in for that process.

The runner sets `GGML_META_CHUNK_NODES=0`,
`GGML_CUDA_GRAPH_STRUCTURAL_MAX_NODES=16384`, and
`GGML_CUDA_P2P_AR_DEBUG=1`. It sets `GGML_CUDA_GRAPH_STRUCTURAL=0` to select the
raw first-node key path associated with the V38 fallback. This bypasses the
structural signature builder, so it tests behavior after fallback to the raw
key and does **not** prove that `external_cap` itself occurred.
`--node-trace` enables per-node CUDA dispatch logging; leave it off for deep
graphs when process-time bounds matter. Child graph logs still report the
number of `SCALE` nodes.

## Evidence gates

Report separately whether each of the following was observed:

- `CHILD_GRAPH` contains exactly one `GGML_OP_ALLREDUCE` with 96 bytes on all
  six ranks for measured A and measured B.
- `CUDA graph path active ... mode=capture` and
  `optimization path ... path=cuda_graph_force_capture ... selected=1` prove
  force capture dispatch; they do not alone prove completed device replay.
- `CUDA graph fallback ... update_result=2` is the runtime's explicit evidence
  for that update failure code. Absence of this line means it was not observed,
  not that the update was successful or necessarily called.
- Both outputs compare exactly with the CPU oracle and retain the host read
  canary.
- For the 4096-node arm, `node-trace` and child graph node counts must confirm
  the SCALE work reached CUDA graph evaluation; a large cgraph count by itself
  is not evidence of kernel capture or launch.

No inference model is loaded. A small fixture result is not proof of a full
model failure cause.
