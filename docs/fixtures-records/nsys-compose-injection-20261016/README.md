# V77 compose probe under Nsight Systems injection

This bounded comparison reuses the already-passed pure-F32 direct AR/compose
probe. It does not modify or rebuild core source. The one comparison run wraps
the same probe process with Nsight Systems 2023.4.4 and
`--capture-range=cudaProfilerApi`. The probe contains no `cudaProfilerStart()`
or `cudaProfilerStop()` calls, so collection remains inactive and no
`.nsys-rep` is expected. This tests the Nsight-wrapped process with the
profiler-API capture range configured; it does not provide an Nsight timeline.

## Fixed baseline identity

- Probe source: `/home/debian/ac922-monster/experiments/standard-opt-20261014/notes/direct-ar-compose-probe.cpp`
- Source SHA-256: `7737f46cb6688c119c6674f0a4e67fdfadd88ed9055390b435b2e59a163166a3`
- Existing probe binary: `/home/debian/ac922-monster/experiments/standard-opt-20261014/notes/direct-ar-compose-shape-probe`
- Binary SHA-256: `2f2ad8d36f49003123a36320e4cd64ccaed9ea4bb1921c78165d818726ca2d42`
- V77 runtime manifest SHA-256: `5d5570afb0bd16cc765232020387b615d7466c3cca20cd23c7b2d1b1c518a9bd`
- Nsight binary: `/usr/local/cuda-12.4/nsight-systems-2023.4.4/target-linux-ppc64le/nsys`
- Baseline run log: `notes/direct-ar-compose-shape-v77-20261016T051738Z.log`

The saved baseline loaded all four GGML libraries from `runtime-perf-v77/lib`,
reported `blocks=64`, and passed both the ordinary segmented control and the
49-child/150-node/48-AR6 composed shape with exact outputs and canaries. The
run wrapper rechecks these hashes and the V77 library resolution before
launching the instrumented comparison. It sets only the probe's required
composition/grid/topology environment: `GGML_META_NESTED_COMPOSE=1`,
`GGML_CUDA_P2P_AR_BLOCKS=64`, PCI bus order, the six existing UUIDs, and the
V77 library path.

The run is limited to 60 seconds, requires an explicit root-issued fixture
token, refuses any existing GPU compute process, and enforces the established
512 MiB free/58°C guards on all six cards. It does not reset devices or modify
BMC settings. It records the current Xid lines before and after, preserving the
V33 Xid 31 seen before this run. The observed record is:

```text
Fri Oct 16 14:18:31 2026: Xid 31, PCI 0004:04:00, pid/name unknown, Ch 0x12;
MMU Fault: ENGINE GRAPHICS HUBCLIENT_FE, address 0x2_437b8000,
FAULT_PDE ACCESS_TYPE_VIRT_WRITE
```

`nvidia-smi` maps BDF `00000004:04:00.0` to logical GPU0 / UUID
`GPU-80452dac-e8c1-99fd-225c-ea846276ae6f`; no later Xid was present at
preflight. This record is preserved verbatim in the run's before/after Xid
artifacts.

## Result

Run `20261016T062636Z-172531` used the exact V77 probe binary and libraries
listed above. The Nsight-wrapped process returned the same exact/canary passes:
ordinary segmented AR, composed first replay, and changed-input replay, with
49 children, 150 scalar nodes per child, and 48 AR6 nodes. No `.nsys-rep` was
produced, consistent with the absent `cudaProfilerStart/Stop` calls. There was
no new Xid after the run; GPU postflight had no compute processes, 16,120–16,125
MiB free, and 38–39°C. The existing Xid 31 remained unchanged in the dmesg
before/after snapshots. This does not establish behavior under active Nsight
collection or under the larger V33 graph.

Saved artifacts are under
`results/nsys-v77-20261016T062636Z-172531/`: run log, V77 `ldd`, manifest/source/
binary hashes, uninstrumented V77 baseline log, and dmesg Xid snapshots.

## Interpretation

- `DIRECT_SEGMENTED_AR_AND_COMPOSE_PASS` under Nsight means injection with an
  inactive CUDA-profiler capture range did not reproduce the failure for this
  exact small workload.
- A failure under Nsight is evidence of an instrumentation/process interaction
  only if the preflight confirms the same probe binary, V77 shared libraries,
  fixed-grid configuration, and six-rank order. Preserve any new Xid/launch
  diagnostics as-is; do not reset the GPUs.
- This is not a model run and makes no throughput claim. The saved uninstrumented
  V77 log is the original comparison arm; no second uninstrumented run is
  needed.
