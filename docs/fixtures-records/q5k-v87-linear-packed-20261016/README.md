# Q5_K V87 packed linear MMVQ fixture

This fixture reuses the real 2,048-row Q5_K `output.weight` slice and fixed
16x2560 F32 input from
`fixtures/q5k-v85-chunk4-hc-20261016/`. The slice is 3,604,480 bytes and is
checked against SHA-256
`37f8db597eb0a511b2a968bf7fa6da30cc2c2c6b6a31a3305d4658770645ebc1`. Input A
is the saved V85 activation file with SHA-256
`ce07fd6f0bb2648072014718d70f977746328e81688e0d8642a946d035caf631`. No GGUF
or model is loaded.

The V87 host fixture tests `[K,N]` Q5_K weights with `ne2=1` against these
activation shapes: T4x4, T8x2, T16, T12, and T6. T4x4 supplies the exact
same-process prefix oracle for each other shape. Every shape runs with input
A, a deterministic changed input B, then restored A. The program checks exact
input readback, finite output, canaries, A/B/A restoration, bitwise output
equality to the T4 prefix, sampled FP64 dequantized-row results, and unchanged
weight bytes. It also keeps the existing HC mode-2 T4x4 versus T16 exactness
and canary check.

OFF leaves `GGML_SM70_DECODE_LINEAR_PACKED` unset. ON sets it to exactly `1`.
Both arms set `GGML_SM70_DECODE_LINEAR_CHUNKS=4` and run in separate processes
so the library's cached environment policy cannot leak between arms. The V87
dispatch condition applies to Q5_K because it is quantized, the input and
output are contiguous, the existing MMVQ4 gate accepts the type, and the
weight is a broadcast-channel tensor with `ne2=1`. The packed branch is
eligible for T8, T12, and T16. T4 remains on the existing short path; T6
exercises the non-divisible 4+2 fallback.

After A/B/A gates, the fixture records five warm graph-compute plus
synchronization wall times per shape. Input upload is outside the timed region;
the timer does no tensor or weight readbacks. It reads the result after timing
to confirm output parity. These are small operator-fixture timings only.

## Build and gated run

V87 build ID: `20261016T095614.374883Z`; runtime:
`/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v87/lib`.
Host-link only; it does not rebuild engine code or execute the binary:

```sh
./fixtures/q5k-v87-linear-packed-20261016/build.sh \
  /home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v87/lib \
  host2
```

GPU execution requires a separate root grant. The runner uses GPU3 by default,
retains a 512 MiB free floor, 58 C ceiling, and 512 MiB used-memory delta cap.
It does not change fan or BMC state and does not start a watcher.

```sh
Q5K_V87_PACKED_GPU_WINDOW_AUTHORIZED=1 \
./fixtures/q5k-v87-linear-packed-20261016/run-bounded.sh "$(date -u +%Y%m%dT%H%M%SZ)"
```

The log validator requires a successful `dense_decode_packed_chunks` summary
with `requested=selected=6`, the direct T16 selection marker, and the T6
chunk4 marker. V87's optimization logger prints only its first two detailed
records; the six-entry summary covers the three eligible graph shapes across
their warmup and capture dispatches. If the observed count differs, the gate
fails and needs review; it is not silently relaxed.
