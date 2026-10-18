# Single-lane D3 real-head fixture for V312

Status: **prepared only**. No build, model load, GPU run, SSH call, or model download has been performed. All fixture changes remain under this directory.

## Reused inputs and physical MTP layout

- The C++ harness starts from `fixtures/mtp-d3-prep-20261016/head-only/mtp-d3-context-fixture.cpp`. It keeps the original GGUF file path, 34 selected tensor names, byte-size checks, raw tensor payload reads, and field rebinds. It replaces the earlier Meta3 tensor-shard loader with a plain physical-CUDA owner loader so the callback-free context can satisfy the dedicated-scheduler device gate.
- Default model argument is the already-used `/home/debian/models/qwen3.8-flash-next-gsq-rco/integrated/IQ3_XXS-MTP-Q8.gguf`; the runner requires that file and never downloads it.
- The metadata model uses three fixed CUDA devices in `split_mode=layer`. `LLAMA_MTP_BASE_DEVICE=0` is set before model loading, so model layer 48 is assigned to visible CUDA0. The selected block 48 plus output head (33 tensors, about 2.99 GiB from the prior fixture's payload receipt) are loaded byte-for-byte into a physical CUDA0 buffer; `token_embd.weight` stays in CPU memory. Trunk and PLE payloads remain unloaded. Expert tensors stay whole on CUDA0 (`LLAMA_EXPERT_PARALLEL` is unset), matching the production plain-LAYER execution shape relevant to the scheduler admission check.
- The fixture asserts both `model.dev_layer(48)==CUDA0` and that the rebound output tensor is buffered on CUDA0. The MTP context constructor filters unused layer devices; with the input embedding on CPU and body/output head colocated on CUDA0, the expected active backends are CUDA0 plus CPU. Under that layout `mtp_body_device()` is ordinary CUDA and `mtp_vocab_backend_allowed()` should accept every registered MTP backend. The run script still requires runtime proof of `admitted=1` and `selected=1`; source reasoning is not treated as runtime evidence.
- A selected-weight preflight caps CUDA0 head payload at 4 GiB and requires at least 512 MiB free after that payload. The run wrapper additionally requires 512 MiB free on all three visible cards, temperature below 58 C, no existing compute processes, and a 600-second timeout.

## D3 checks prepared

One MTP sequence lane only (`n_seq_max=1`). Catch-up positions 0–3 are submitted first, then a seed-only row at position 4. `first_pos=seed_pos=4` and `LLAMA_MTP_FIXED_PREFIX_ROWS=0` preserve the existing no-prefix lane contract. The planned target verify batch is `3 draft + 1 seed = 4`; target verification is not run here.

The callback-enabled semantic context compares stepwise D3 against the single-graph D3 for 100 deterministic input patterns. Pattern order starts A/B/A so the same context sees the same A input on both sides of a different B input. Each case compares all three complete logits, exact three-token history, final hidden, and final KV position: 300 full-logit depth comparisons total. The first case removes KV from the seed position, checks catch-up positions through 3 remain, then runs the generic stepwise fallback from that retained prefix and compares it to the golden.

A separate callback-free context is required because the dedicated-scheduler policy rejects `cb_eval`. It runs four A/B/A/random cases through the dedicated scheduler and compares final full logits, final hidden, exact token history, and final KV position against the stepwise baseline. It also checks rollback from the seed anchor. The runner requires actual runtime logs showing the one-lane D3 policy was admitted, the dedicated scheduler selected, and a later graph reports `rebuilt=0`.

The fixture calls context APIs directly. It does not exercise `common/speculative.cpp` chain selection, target-model verification, accepted-length decisions, full-model behavior, or throughput. Those remain a separate integration gate after this bounded real-head fixture passes.

## Build and launch

Both scripts require `RUNTIME_DIR` to point at the root-frozen V312 runtime. `build-head-fixture.sh` only links the fixture. `run-head-fixture.sh` refuses to launch without `SINGLELANE_D3_GPU_WINDOW_AUTHORIZED=1`; it checks the exact three GPU UUIDs and resource guards above. Each run gets a unique log. The runner verifies parity and dedicated-scheduler admission/reuse markers.

The root orchestrator must release the GPU window and provide the frozen runtime before either script is used. `launch_eligible` in `v312-head-profile.json` remains false until then.
