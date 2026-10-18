# Real Meta3 scoped async-input fixture

This is a small real-backend fixture for the private Meta graph-input scope.
It creates two independent Meta3+CPU schedulers over the same three visible
CUDA devices. Each scheduled graph has three explicitly Meta-assigned,
contiguous F32 `GGML_BACKEND_BUFFER_USAGE_COMPUTE` input tensors, all marked
with `ggml_set_input()`: two 8-element vectors and one 12-element vector.
Shared matrices/bias use the established dedicated Meta `WEIGHTS` buffer
strategy from `owner-real-backend-20261016`.

The fixture opens
`ggml_meta_async_input_private::scope(scheduler, graph)`, writes from temporary
vectors, overwrites and frees each source before `finish()`, then reads inputs
back while the scope is open. A partial input write exercises the synchronous
fallback and must read back exact values. In on mode, runs are ordered A/B/A,
B is destroyed after its scope finishes, and A runs again afterward. Off and
on are separate processes because the runtime option is cached on first use.

The root-authorized real-backend qualification uses `--copy-only`: it exercises
input routing, snapshots, fallback, readback, and scheduler-owner lifetime, but
skips graph compute. The original full graph arm is still retained for a later
fix; it is not currently a passing gate.

## CPU helper regression plan

The CPU-only production-helper and ordering tests already live beside the
scope implementation in `fixtures/meta-async-input-20261016/`. Run them after
the input-agent patch is finalized:

```sh
cd /home/debian/ac922-monster/experiments/standard-opt-20261014
python3 ./fixtures/meta-async-input-20261016/run_cpu_tests.py
```

That suite exercises captured-byte snapshots, temporary source mutation,
per-owner enqueue/synchronize counts, capacity decline, upload-before-graph
ordering, and slot lifetime with fake child streams. It does not replace the
real Meta3 fixture below.

Staging-time CPU result on the local mirror:

- `run_cpu_tests.py`: production `write_batch` helper test passed.
- `python3 -m unittest -v test_async_upload_contract.py`: all four contract
  tests passed.
- The real-backend fixture source passed local Clang C++17 syntax-only
  compilation against the current shared GGML headers. This is not a V57 link
  or a runtime/dispatch result.

## V57 host link

Root froze V57 build `20261015T230252.607167Z`; the fixture host-linked against
that runtime with:

```sh
cd /home/debian/ac922-monster/experiments/standard-opt-20261014
RUNTIME_DIR=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v57 \
  ./fixtures/meta-async-input-gpu-20261016/build-real-meta3.sh
```

The build linked only; it did not initialize CUDA or read model payloads. The
binary resolved `libggml-base`, `libggml-cpu`, and `libggml-cuda` from
`runtime-perf-v57/lib`. Corrected copy-only build output is retained at
`results/host-build-copy-only-20261015T231107Z-70878/host-build.log` (SHA-256
`697d258ec6469c4ae569049057425ad73a77c468cbd394c2c3987507fdce3a9f`). Binary
SHA-256: `6a4fc12c2293cd3bafd29857503f7db94e1661d1b7e74df98ac0b8f96f304441`.
The private header is included directly from `src/ggml/src`; this fixture
does not change a public GGML ABI.

## GPU execution only under root ownership

The root-token-guarded runner tests off then on using `--copy-only` on the
existing GPU3–5 island. It checks for occupied devices, enforces 512 MiB free
and 58°C limits, and uses unique per-run result directories. The on route
requires an INFO line with `eligible_inputs=3` for each scheduler plus
nonzero captured writes/bytes, flush batches, and synchronized owners.

```sh
cd /home/debian/ac922-monster/experiments/standard-opt-20261014
META_ASYNC_INPUT_GPU_AUTH=ROOT_META_ASYNC_INPUT \
RUNTIME_DIR=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v57 \
  ./fixtures/meta-async-input-gpu-20261016/run-real-meta3.sh
```

## Full graph attempt and copy-only qualification

The first root-authorized run tag was `20261015T230518Z-70198`. Its OFF process
reached input setup, ephemeral source mutation/free, full input readback, and
partial-write readback with exact values. It then aborted in Meta graph
compute at `ggml-backend-meta.cpp:1166`:

```text
GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer)) failed
```

The failing call is in Meta graph child-node reconstruction. The log did not
identify the specific scheduled split node. The fixture did not provide graph
result or async-route parity evidence from this attempt. The runner stopped
after OFF failed; ON was not started and there was no retry.

Retained artifacts:

- `results/20261015T230518Z-70198/off.log`, SHA-256
  `8c8d2cb658a01fff55cced71262e244f964ce117897683b410b8846ac254cd34`
- `results/20261015T230518Z-70198/summary.log`, SHA-256
  `b36e1012c5ff4302b05d7995f15e846a8bafcc63ba9a3ea6790e7a56afa82a81`

After the abort, GPU3–5 were idle at 16,125 MiB free each and 38/38/39°C.
No GPU reset or core edit was made.

The single corrected V57 copy-only OFF/ON run passed under tag
`20261015T231122Z-70921`. The on log reported `eligible_inputs=3` for both
schedulers. Per-scheduler captured summaries were:

- B: 3 writes, 112 bytes, 1 flush batch, 1 synchronized owner.
- A: 9 writes, 336 bytes, 3 flush batches, 3 synchronized owners.

The synchronous fallback marker appeared for both owners. All four A/B/A plus
post-B-destruction cases passed, with six exact readback barriers covering
full inputs and partial-write results. Each caller buffer was mutated and
freed before scope finish. Both off and on ended with
`PASS META_ASYNC_INPUT_FIXTURE`.

The copy-only run explicitly skipped graph compute. Its
`GRAPH_NODE_AUDIT` lines report original user-graph descriptors, not the
scheduler-owned split graph. Therefore this validates real Meta input staging,
source lifetime, readback, and owner teardown; it does not validate graph
execution, graph parity, full-model behavior, or performance.

Corrected run artifacts:

- `results/20261015T231122Z-70921/off.log`, SHA-256
  `a030b41169607f20a0bf6675614528eeb39c62b60ffab0fca5a74a7274cea791`
- `results/20261015T231122Z-70921/on.log`, SHA-256
  `c516827967ade1344ea19f62484dbae32f7b843dc2ca9889b3f0e55db099ef7c`
- `results/20261015T231122Z-70921/summary.log`, SHA-256
  `52fe97b21b25aa1a74e8410a71d1e9c272ac295b2c531fdd710d09c44730bc3b`

After the corrected run, GPU3–5 were idle at 16,124 MiB free each and
38/38/39°C.
