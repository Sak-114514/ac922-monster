# CUDA graph opaque `tensor->extra` regression

This bounded backend fixture checks the V25/V63 crash condition without loading
an inference model. A first CUDA graph materializes an F32 ADD result into a
normal CUDA buffer. A second graph treats that producer as an external input
leaf. The sentinel mode then sets only that tensor's opaque `extra` field to
`0x57800` and exercises the CUDA graph structural/snapshot lookup. The output
must remain exact for A/B/A frames and retain canaries whenever the structural
path safely declines the opaque descriptor.

The parent fixture process forks the CUDA child. V63's expected SIGSEGV is
contained in the child and accepted only when the runner explicitly expects it;
no core dump is written. V64 external-subgraph probes must exit normally and show the
`unsupported_external` structural rejection when the sentinel is present. The
normal mode leaves `extra == nullptr` and checks exact output as a no-sentinel
control.

Build each binary against the exact runtime under test:

```sh
./build-fixture.sh /home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v63 v63
./build-fixture.sh /home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v64 v64
```

Bounded GPU2 probes (each child has a 60-second timeout, 58 C temperature
limit, 512 MiB preexisting-memory limit, and 1 GiB in-run limit):

```sh
./run-probe.sh /home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v63 v63 sentinel 1 1
./run-probe.sh /home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v64 v64 sentinel 1 0
./run-probe.sh /home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v64 v64 sentinel 0 0
./run-probe.sh /home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v64 v64 normal 1 0
```

This validates the CUDA backend's safe decline and subsequent ordinary graph
execution. It does not load a model or claim an inference-level result.

See [RESULTS.md](RESULTS.md) for the V63 crash reproduction, V64 safe-decline
results, exact-output evidence, artifact hashes, and the snapshot-hit limitation.
