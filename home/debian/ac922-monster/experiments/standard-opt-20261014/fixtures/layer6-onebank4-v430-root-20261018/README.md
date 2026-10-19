# V430 onebank D3 single-wave preparation

V430 keeps one model load on the plain-CUDA six-GPU LAYER path and tests one target context plus one generic MTP D3 draft context. Each context must expose four non-unified streams of 17,152 cells (68,608 total). Sequence IDs 0–3 live in bank 0.

The runner first executes the target-state canary on the same model weights. It destroys the serial and four-sequence target-only canary contexts, saves `canary.json` and F32 snapshots, and waits up to ten minutes for a fresh `root-control.json` containing exactly `{"continue":true}`. The control must be a regular file owned by the same Unix account as the runner. Only after that signal does it create the target and MTP contexts, run the four frozen native cases, and start the single 4×16K/512 wave.

The D3 target verification budget is **16 rows total**: four sequences times four rows per active lane. `LLAMA_MTP_VERIFY_ROW_BUDGET=16` is required; the per-lane pad remains four. Prefill uses one owner and synchronized 2,048-token cohorts, followed by a ten-second cooldown after each cohort. The recorded prefill interval includes those cooldowns and does not claim overlap.

`GGML_SM70_DECODE_LINEAR_CHUNKS=4` is the inherited selector request. It does not split a `MUL_MAT_ID` T16 input into four T4 kernels; the Volta MMVQ branch uses `min(8, MMVQ_MAX)`, so T16 is expected to use two T8 chunks. `GGML_CUDA_MOE_FUSED_T16=1` is also only a request: with expert parallel disabled, the fused gate/up/GLU selector remains unselected. Runtime path logs must support any selected-kernel claim.

The V146 gate uses an adapter because its frozen checker requires four bank IDs. V430 first checks the actual result has bank 0, sequence IDs 0–3, four target/draft KV streams, and a total verify budget of 16. It maps a private copy of those verified sequence IDs into the old checker’s bank predicate, then persists the actual onebank topology unchanged.

This directory is preparation only. It has not been built or run; no native pass or performance result is claimed. The profile remains launch-disabled until the root reviews the V426 target-state canary and V430 artifacts. No separate launcher is included; use the root-owned in-band guard.

## Offline checks

```bash
python3 verify-v430-static.py
python3 test-v430-topology.py
bash -n root-host-link-v430.sh
```

These checks do not compile the runner, load a model, or touch a GPU.
