# EP axis-2 native MTP D2 head-only qualification

This private fixture is based on the prior real-weight D2 head fixture. It
loads exactly the MTP block-48 and shared embedding/output head tensors, while
leaving the trunk and PLE payloads unread. It targets a single Meta3 bank across
the established three-GPU set and checks D2 head behavior with lane counts
4 → 2 → 4. It does not load the full model.

The older `fixtures/mtp-real-head-20261015` run used
`LLAMA_EXPERT_PARALLEL=native=1`; the current parser accepts only the exact
value `1`. Its old D2 results are not evidence for the production EP tensor
layout. This fixture checks that the environment seen by the fixture and the
production split callback both select expert parallelism before any selected
payload bytes are read.

For the selected routed gate/up/down expert tensors, the fixture requires the
model callback to report split axis 2, a single segment, 512 experts, local
counts `170/171/171`, and prefix offsets `0/170/341`. The selected tensor bytes
are loaded using the returned Meta tensor descriptors; the fixture does not
manually TP-slice or rewrite quantized payloads. Dense/shared tensors retain
the ordinary model split policy.

The contexts use the established unified KV layout (`kv_unified=true`),
`n_seq_max=4`, and one Meta3 bank. The three comparisons use one stepwise
context and one unrolled context in this order: initial sequence IDs
`{0,1,2,3}`, same-context reuse `{1,3}`, then reuse `{0,1,2,3}`. Each checks
both full logits, exact token IDs, final hidden state, KV positions, and future
prefix-row rollback. This is head-only D2 parity evidence. It does not prove
target-model GDN semantics, full-model quality, serving dispatch, or a
performance gain.

## Source and runtime

The runner links to the frozen V56 runtime by default:

`/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v56`

The source used to prepare the fixture is the AC922 source tree at
`/home/debian/ac922-monster/experiments/standard-opt-20261014/src`. At
preparation time, relevant file hashes were:

- `src/llama-model.cpp`:
  `f880232505082386e60c1909e8a2d301a5b88904d6202d4d4ec2ed65e8b66ac2`
- `src/llama-context.cpp`:
  `a402784401300f4e333205e0ab592804e8073f1d2d718017f04391d2a1e05f57`
- `ggml/src/ggml-backend-meta.cpp`:
  `cea773c4632c54bfc00cee63a1fe02cdc46d2fabc7d5437189489bed185a7131`
- `ggml/src/ggml-ep-private.h`:
  `477d832b545398de8cf765cfda0deabafefef3ebfec393fd983f5df46391384a`

`llama-model.cpp` routes Qwen4exp `*_exps.weight` tensors with `ne[2] > 1`
through the production axis-2 split only when `ggml_ep_private::requested()`
is true. That parser compares `LLAMA_EXPERT_PARALLEL` with the exact string
`1`. The fixture adds no source/core changes.

## Host-only build

On AC922, the link uses V56 libraries and does not initialize CUDA or read model
payloads:

```sh
cd /home/debian/ac922-monster/experiments/standard-opt-20261014
RUNTIME_DIR=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v56 \
  ./fixtures/mtp-ep-d2-head-20261016/build-head-fixture.sh
```

Host link completed on AC922 with the command above. The binary resolves
`libllama`, `libggml-cpu`, and `libggml-cuda` from `runtime-perf-v56/lib` under
the runner's `LD_LIBRARY_PATH`. The build emitted only the two existing
deprecation warnings for `llama_new_context_with_model`; it did not initialize
CUDA or load a model. Retained build output:
`results/host-build-20261015T224757Z-68242.log` (SHA-256
`918c1dd2ac9379979f824d205542a36be3364836ff3ac5dc4af63a97c3ae8b56`).

Built artifact SHA-256:
`48627309a67a171e4416881b186aa2f3b711131997372483b3caa85c77678722`.
The linked frozen libraries were:

- `libllama.so.0`: `84669dc3a1bec3f297c4305563605d700c9b32c9c99b4bbbbfaa3aac4d0fa322`
- `libggml-cuda.so.0`: `f85aa368136840f01d6b6a63b0ae4c143433b42acee9e726955112d3d9819135`
- `libggml-cpu.so.0`: `c1fd8d19e54df03e97f1c71c8c95565626ec73f44f7d4f3e334bfe3e74f464fc`

## GPU qualification after root grants a window

The runner targets only the established GPU3–5 UUID set. It refuses to run
without the explicit root token, rejects pre-existing compute processes,
checks the 512 MiB free-memory and 58°C limits, gives the process 180 seconds,
and creates a unique UTC-tagged log without truncating earlier results.

```sh
cd /home/debian/ac922-monster/experiments/standard-opt-20261014
MTP_EP_D2_GPU_AUTH=ROOT_MTP_EP_D2 \
RUNTIME_DIR=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v56 \
  ./fixtures/mtp-ep-d2-head-20261016/run-head-matrix.sh \
  /home/debian/models/qwen3.8-flash-next-gsq-rco/integrated/IQ3_XXS-MTP-Q8.gguf
```

The runner sets `LLAMA_EXPERT_PARALLEL=1`, `LLAMA_ISLAND_SPLIT=3`, and fixed
tensor splits `1,1,1`; it also checks all three exact EP layout markers and the
three-case 4→2→4 parity result before it can print success.

## Bounded GPU result

The root-authorized V56 head-only run passed under tag
`20261015T224904Z-68381`. It checked the three routed weights with the real
model split callback before payload reads: gate, up, and down all reported
axis 2, 512 experts, counts `170/171/171`, offsets `0/170/341`. Exactly 34
selected tensors were loaded (`3,485,816,832` payload bytes); trunk and PLE
payload reads were both zero.

All three unified-KV D2 comparisons passed in the requested same-context
sequence order: four lanes `{0,1,2,3}`, two lanes `{1,3}`, then four lanes
`{0,1,2,3}`. Full logits, exact token IDs, final hidden, KV positions, and
future-pad rollback matched the ordinary stepwise path. Every case reported
`max_logit_abs=0` and `max_final_hidden_abs=0`. The one-window log is
`results/mtp-ep-d2-context-matrix-20261015T224904Z-68381.log` (SHA-256
`6039741885c7dd44005bf3e191bfe53d8c43627e83105694a7180dc4c5ecf8e8`).

This is a head-only D2 EP split and graph parity result. It does not establish
full-model target/GDN correctness, server quality, or a performance gain.
After the run, the three assigned GPUs returned to 16,126 MiB free each at
38/38/39°C with no compute processes present.
