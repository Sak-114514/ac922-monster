# Qwen4exp base token lookup fixture

This fixture is owned by the V123 input-placement gate. It uses only public GGML graph/backend APIs and the existing Meta3 construction pattern from `fixtures/qsa-native-mask-20261017/qsa-native-mask-fixture.cpp`. It does not create a CUDA backend wrapper, read model metadata itself, or emit model-loader markers.

## What it will prove

- `cpu` runs CPU `GET_ROWS` on selected raw rows from the real `token_embd.weight` payload and checks A/B/A repeatability, IDs, source-row bytes, and sentinels.
- `meta3` runs the same CPU oracle and Meta3 `GET_ROWS` with the same selected rows and type, requiring bit-exact F32 output for A/B/A. It also requires Meta3 `GET_ROWS` support for that exact type before allocating the test graph.
- `layout` constructs an actual GGML scheduler graph with a 512-ID CPU PLE `GET_ROWS` prefix, MetaA token `GET_ROWS` and computation, then a MetaB consumer. It checks the scheduler's three backends, three splits, one copy and node assignments. Its bounded graph compute triggers the frozen runtime's own `GGML_TURBOPREFILL_TRACE` classifier, whose emitted map must say `topology_eligible=1` and CPU→MetaA→MetaB.

The token operator test compacts the actual first/last, middle, and repeated vocabulary rows into a tiny table, then remaps the selected global token IDs to compact row IDs. This tests real quantized row payloads and dequantization without allocating or reading the entire 273,152,000-byte table on three GPUs. The log explicitly marks `full_vocab_index_addressing=not_tested`; this is an operator-value gate, not proof of full-table index arithmetic, full-table placement, memory fit, Turbo capture, model semantics, or performance.

## Source metadata contract

The existing V122 manifest at `fixtures/layer6-two-bank-v57-20261017/v122-layer-bytes.json` reports:

- `token_embd.weight`: GGML type id 21 (`IQ3_S`), shape `[2560, 248320]`, 273,152,000 bytes.
- `output.weight`: independent tensor, type id 13 (`Q5_K`), 437,043,200 bytes.

The copied pure-header record `original-input-output-offsets-v123.json` (SHA256 `e4f02f72565e5babfdb977231d19c8eeb6ec4fa28e42861cc2a4b37461d40f05`) gives `token_embd.weight` absolute payload offset 461,217,728 (GGUF data section start 11,026,368 plus tensor-relative offset 450,191,360). With `n_embd=2560`, its row size is 1,100 bytes. Pass the absolute offset with `--tensor-data-offset`; this fixture never parses GGUF or Monpack metadata. It reads only the selected row bytes with `pread()`, then copies them unchanged into the CPU and Meta3 tables. Do not replace this compact sample with a full 273 MB table read unless a later memory gate explicitly requires it.

Required operator-mode arguments:

```text
--tensor-file PATH
--tensor-data-offset 461217728
--tensor-type iq3_s
--n-embd 2560
--n-vocab 248320
```

Other GGML types/dimensions can be passed to diagnose a future model artifact, but the V123 target case must use the values above and the precise payload offset supplied by the metadata handoff.

## V123 model markers remain separate

The frozen candidate source contains these model integration markers, which must be checked against a later real model log rather than inferred from this fixture:

- Loader eligibility and destination: `Qwen4exp input placement requested=1 eligible=1 destination=...`.
- Post-load selection and actual buffers: `Qwen4exp input loaded requested=1 selected=1 type=... bytes=... buffer=... output_buffer=... PLE_buffer=...`.
- Graph build order: `independent_PLE_prefix_before_GPU_embedding=1`.

The fixture's `PASS_*` lines only certify the GGML operator or synthetic scheduler graph. They do not stand in for these model markers or full-model semantic gates.

## Preparation and verification state

`build-fixture.sh` links this one fixture translation unit against a caller-supplied frozen runtime; it does not rebuild the engine. `run-bounded.sh` requires a separate explicit authorization token for `meta3` and `layout`, checks GPU process/free-memory/temperature bounds, imposes a 20-second timeout, and preserves each log under `results/` without retrying.

Host linking against frozen `runtime-perf-v123` passed for the single fixture translation unit (`build_id=20261016T204206.497138Z`, V123 report SHA256 `7e321c96c870a53a7f8de72115ec77250c5bc31e1a31b1adaf12784f544fdf0d`). The initial binary `qwen4exp-input-gpu-v123-host2` (SHA256 `1839ea109da7e343b4a5c0279b2d1f2fca364d069cee4bb22ceb3c593f03148e`) passed the bounded Meta3 actual-row A/B/A gate using eight selected original IQ3_S rows; output was bit-exact F32, with repeated IDs, source/ID immutability, and canaries intact (`results/v123-meta3-authorized-20261017T1215Z-meta3-20261016T205626Z.log`, SHA256 `3b915a41c6c88950bdbd3b6eda4710ef250e5040fbe71be31f0600e551fbb091`). It remaps global IDs into the compact eight-row table, so full-vocabulary index arithmetic remains untested.

The first scheduler-layout attempt exposed a fixture construction error before graph allocation: `ggml_backend_sched_new()` requires its last backend to be CPU, but the initial array was `{CPU, MetaA, MetaB}`. That failed log is preserved at `results/v123-layout-authorized-20261017T1215Z-layout-20261016T205634Z.log` (SHA256 `76437e6bd6e37333db5242d44dfc582d9757454b05bd235951722db072a3d924`). The fixture-only correction reordered the array to `{MetaA, MetaB, CPU}` while retaining explicit tensor-to-backend assignments. The new host binary `qwen4exp-input-gpu-v123-host3` (SHA256 `bdeeb193ef1b09c7ebe9a06cfc50a3b0f5927b1da3ff74e65b1281b9fbb3c128`) then passed: the frozen runtime emitted `topology_eligible=1`, `splits=3`, and the actual CPU→MetaA→MetaB map with node ranges `0..1`, `1..3`, `3..4`; 512 PLE IDs and a 512-byte MetaA→MetaB bridge were observed, with exact synthetic outputs (`results/v123-layout-host3-authorized-20261017T1220Z-layout-20261016T205932Z.log`, SHA256 `04cc30b23099f43d8dba16c281377e2c30066f1a1f7a251fae595567f16e3939`). The Meta3 pass and corrected layout pass finished in four seconds each; the initial constructor failure also exited in four seconds. Runs stayed below 40 C, retained at least 16,114 MiB free, and left no compute process. The frozen report and SHA256 values for all runtime GGML DSOs are recorded in `results/host-gates.json`; the copied build report is `results/V123-build.json`.

These are fixture/operator and synthetic scheduler-topology results only. No full model was loaded, and loader destination markers, full-vocabulary embedding placement/indexing, full-model semantics, and performance remain untested. Machine-readable evidence is in `results/host-gates.json` (SHA256 `7f4f2d5c82896f7562b2f66ad82f80af99048743e1daecaf7266b63c56e54286`).
