# P4 / D2-D3 / full-logical-prefill runner preparation

This fixture prepares a V47 P4 candidate run. It has not loaded the model, initialized CUDA, or contacted the AC922. Root owns the runtime build, GPU window, and actual run.

V11 fixes a result-recording defect found in V10: absence of a token event inside the shared decode window previously continued the loop before serializing that lane or applying its semantic check. V11 always records and scores each lane; it computes the shared-throughput window separately and marks that window empty if any lane has no post-start event. The included CPU fixture exercises both cases.

## Frozen candidate shape

- One `llama_model_load_from_file()` call. Two bank target contexts and their two MTP draft contexts share the same model object, so model weights are loaded once. KV, scheduler, graph cache, workspace, and context state are still per-context.
- Two banks × two sequence lanes per bank = four concurrent lanes. Each target/draft context uses the same six-GPU 3+3 island topology; the banks do not own different GPU subsets.
- `n_ctx=132,096` per bank, for 264,192 aggregate context capacity across the two banks. Each individual lane can use the full per-bank pool when its peer is idle; the aggregate across banks is not a single shared KV pool.
- Q8_0 target and draft K/V; native MTP depth D2 by default. D3 is an explicit `ROOT_MTP_DEPTH=3` candidate on this two-lane runner. Full two-lane D2 verify has six rows (`2 × (1 target + 2 draft)`); D3 has eight. The hard verify ceiling stays eight, and no draft-row padding is added.
- `ROOT_SERIAL_CPU_SUBMIT=0` is required. The old runner-wide submission mutex is off. V47's Meta context views and per-physical-GPU stage-tail ordering are the expected safety mechanisms; the actual runtime build identity must be recorded before the run. This candidate does not use the older lock-on measurements as its baseline.
- `LLAMA_EXPERT_PARALLEL=1` is required (exact value `1`; `native=1` disables this route). The runner explicitly opts into two-lane MTP multi-chain/unrolled mode, sets the unrolled depth to `ROOT_MTP_DEPTH`, uses six fixed-prefix rows per lane, and requires the variable-lane gate. The checker requires runtime evidence for both banks and a routed expert log with `total=512`.
- The bank's single long-lived `dual_context_owner` is reused for its prefill calls and target-frame submissions. The two banks submit prefill work concurrently through those existing owner threads; no `std::async` prefill threads are created.

## TurboPrefill eligibility

Every lane is prefetched with one full logical `llama_decode()` call. Its batch contains exactly one sequence with contiguous positions; only the final prompt token requests logits. The 8K corpus requests therefore give the context 16 full 512-token UBs or eight full 1,024-token UBs, with earlier no-logit UBs eligible for the V43/Turbo path when requested. `n_batch=8,192` is sufficient for one full 8K logical prompt and avoids an unnecessary larger host hidden buffer.

The default is ordinary prefill at UBatch 1,024, because V43's warm single-stream PP sample favored 1,024 over 512, while TurboPrefill at 512 showed no steady-state improvement. This is only a candidate choice; P4 must be measured on the actual runtime. Set `ROOT_TURBOPREFILL=1 ROOT_UBATCH=512` for a separate Turbo request; selection/decline is logged and is not a pass requirement or a performance claim.

The launcher records the explicit `LLAMA_TURBOPREFILL` state, requests `GGML_META_CONTEXT_VIEWS=1` and GPU event spans at period 1, and uses `LLAMA_ISLAND_SPLIT=3,3` plus the known-clean `LLAMA_QSA_INDEXER_IMPL=standard`. When Turbo is requested, it also sets the existing frame/FIFO/event prerequisites. It leaves `LLAMA_QSA_POOLED_MULTI` unset and never calls a `get_blk` setter. The normal cache configuration must satisfy the context's existing `get_idx()!=nullptr && get_blk()==nullptr` gate if Turbo is enabled.

The checker requires the actual run log to show Meta context-view and GPU-span diagnostics, their stream-gap caveat, and the Turbo requested/selected/declined state. A Turbo decline is recorded faithfully and does not invalidate P4 decode results. GPU event spans measure first-to-last event envelopes, include stream gaps, and are not kernel-busy time. Do not claim inter-island overlap without timing evidence that establishes it.

## MTP KV and server comparison

`ROOT_SERVER_MTP_KV_POLICY=1` is mandatory. After generating an MTP proposal, the runner removes draft-context KV at and after the prompt position before target verification. It retains the draft context's prompt prefix and allows target nextn rows to refresh the proposal state. `ROOT_SERVER_MTP_KV_POLICY=0` retains proposal KV and is a different runner policy; it is not used here or treated as a comparable server result.

`LLAMA_MTP_KV_ONLY_PREFILL=1` is set to the current V42/Qwen4Exp path and captured in the log/result. It is separate from the draft KV rewind policy above. The server itself has one target and one MTP context serving four lanes; this runner has two target and two draft contexts over shared weights, so its context/workspace/graph memory and scheduling differ. A successful runner rate is a P4 scheduling result, not automatically a server API capacity or latency result.

## Scenario order and quality gates

`scenarios.json` contains 15 ordered scenarios:

1. Four pairs of serial semantic references, then four matching 2+2 P4 semantic scenarios for the frozen 16-case native semantic suite.
2. Two serial 2-lane references for four 8K prompts, followed by one 2+2 P4 shared-generation run.

The runner aborts before proceeding from any semantic scenario that fails its expected output. The checker compares all 16 semantic outputs and all four 8K outputs against same-process serial references by exact generated token ID and output length. The 8K requests each produce exactly 256 tokens with EOS ignored; the checker rejects early/incomplete lanes.

When a lane has only its final target token left in the 256-token budget, `dp.n_max=0` is a planned terminal no-draft step, not an MTP fallback. The runner counts those separately as `planned_terminal_no_draft_lanes`; only an empty draft while rows were requested increments `fallback_lanes`, which remains a zero-fallback gate.

The 8K input token IDs come from the saved `llama-v100-v3-repository-mixed-prose-code-v1` corpus recording and its original tokenizer metadata. The source recording only includes two independent 8K arrays, so the generator derives four unique sliding windows across their adjacent source-backed bodies and appends the recorded 37-token task suffix. Those windows overlap; `input-provenance.json` records that limitation. They are real pretokenized repository prose/code, not synthetic repeated tokens.

The shared decode rate is computed from the common active interval across all four lanes. The wave begins after each bank has completed one serial warmup frame. For each lane, take its first emitted-token time at/after the wave start; the common interval starts at the latest of those four times and ends at the earliest lane's final emitted-token time. Count all four lanes' token events in `(start, end]` and divide by one wall-clock duration. This excludes serial warmup and does not sum per-lane rates. Prefill input count, wall time, and aggregate PP are reported separately.

## Generate, CPU-check, and run

Regenerate input scenarios from the saved local corpus and semantic artifacts:

```bash
python3 generate_v11_scenarios.py \
  --semantic-source /Users/lyx/Desktop/ssh/monster-engine/standard-opt-20261014/results/20261014T172531Z/native-semantics/case-input.json \
  --real-corpus-token-ids /Users/lyx/Desktop/ssh/flashnext-llama-experiment/final-results/ROUND19/real-corpus-token-ids.json \
  --out-dir .
```

Run CPU-only checks (Python checker tests, persistent-owner regression, and scenario shape/hashes):

```bash
./run-cpu-tests.sh
```

After the frozen V47 dynamic library closure and `notes/V47-build.json` are present, build the runner without executing it:

```bash
./compile-v47.sh
```

The build directory contains the executable, `ldd` resolution, and hashes for the frozen V47 build report and every linked llama/ggml runtime library, including `libllama-common.so.0`. The runner links only the frozen `runtime-perf-v47/lib` closure; V47's `common/speculative.cpp.o` is already inside that common shared library.

After root's V47 runtime and tiny GPU fixture are qualified, run on AC922 with the already prepared tuning environment preserved:

```bash
ROOT_RUNTIME_REVISION=20261015T191202.860108Z ROOT_UBATCH=1024 ROOT_MTP_DEPTH=3 \
  ./run-v11.sh /path/to/dual-context-v11 /path/to/IQ3_XXS-MTP-Q8.gguf /path/to/results/p4-v11
```

D3 is selected with `ROOT_MTP_DEPTH=3`; the runner rejects values other than 2 or 3, while keeping four lanes and the eight-row cap. It sets `LLAMA_MTP_GPU_CHAIN_UNROLLED_DEPTH` to the same value and checks that runtime logs arm and activate the requested unrolled path in both banks. The launcher writes `results.json`, `runtime.log`, and `run-manifest.json`; it records source/binary/scenario SHA-256 and model stat metadata, without re-reading/hash-scanning the multi-GB model. Supply `ROOT_MODEL_SHA256` if a trusted installed-model integrity record already provides it. The build report hash and server hash are cross-checked against the frozen V47 manifest.

## CPU-only evidence boundary

The included checker tests validate rejection/pass logic and the persistent-owner helper test. The runner can be syntax-checked and built on the host without model/GPU execution. CPU-only evidence does not establish full-model output parity, inter-island overlap, or any P4 performance result. Those remain root's GPU-run gates.
