# Prefill MMF trial results

Date: 2026-10-15
Status: fixture compile/CPU prep passed; bounded IQ3_XXS GPU2 case passed; IQ4_NL down MMF core exceeded the 15 s diagnostic watchdog.

## Prep

- No engine/core source files changed.
- No model runtime loaded; no GPU kernel executed in prep.
- Fixture scope is limited to layer 35 local TP shards: IQ3_XXS gate/up [2560,224,512] and IQ4_NL down [224,2560,512].
- Source-backed conversion choice: use existing contiguous `ggml_get_to_fp16_cuda(type)`; the non-contiguous selector does not cover IQ types. Flattened element counts are divisible by 256, and IQ4_NL blocks are independent 32-value blocks, so no K padding is added.
- Build command `fixtures/prefill-mmf-trial-20261015/build-fixture.sh`: PASS. The compiled SM70 PTX contains `mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32`, confirming the existing MMF kernel lowers to Volta Tensor Core MMA.
- CPU ID fixture `prefill-mmf-trial --self-test`: PASS. Balanced route covers all 512 experts exactly 10 times at 512 tokens and 20 times at 1024. The skewed route covers all experts with hot-set counts 64/128 and tail counts 2–3/4–5.

## GPU gate

The compiled fixture has a quick mode for IQ3_XXS gate, 512 tokens, balanced routing. It compares grouped MMF to CPU FP32 dequantized dot reference and grouped DP4A MMQ before running full conversion/MMF timing. Full mode follows only if the quick accuracy screen passes. GPU2 only; no full-model load.

GPU timings and accuracy are recorded below. The full 512/1024 by balanced/skewed matrix was not run.

## Bounded GPU fixture results (2026-10-15)

Scope: one standalone fixture on physical GPU2 only; no model/server loaded and no engine source changed. New phase markers flush before each graph and _Exit(124) on any single graph exceeding 15 s. GPU2 was released after each case.

### IQ3_XXS gate, T=512, balanced top-10 route

- dequant_f16: 4.155 ms
- mmf_id_compact: 0.055 ms
- DtoH metadata check: PASS; exact expert bounds and forward/destination mappings for all 5,120 assignments across 512 experts.
- grouped MMF core: 0.969 ms
- MMQ Q8_1 prep (ID maps, offsets, scatter): 0.097 ms
- grouped MMQ core: 1.013 ms
- Combined fixture case wall time including allocation, graph capture and CPU accuracy screen: 195.529 ms.
- Accuracy vs CPU reference that dequantizes stored IQ3_XXS to FP32 and computes the dot product: MMF rel-L2 2.908666e-4, max abs 1.621246e-4; F16 weight rel-L2 2.040336e-4, max abs 2.849102e-5; PASS_SCREEN. MMQ rel-L2 5.019697e-3, max abs 2.787247e-3 (expected additional Q8_1 activation quantization).
- Decision: this prototype's one-shot conversion + MMF path costs about 5.179 ms per sampled layer versus 1.110 ms for MMQ prep + MMQ core, roughly 4.7x slower end-to-end. The TC core alone is marginally faster than MMQ core, but conversion dominates. This does not establish a model-level speed gain; only a persistent cached-F16 representation could amortize conversion, with substantial memory cost (560 MiB for this one TP-local gate shard).

### IQ4_NL down, bounded diagnostic only

- Initial T=2 run exceeded the 15 s MMF graph watchdog, but is excluded as a production-path test: production switches to compact mul_mat_f_ids only for ncols_dst > 16; T=2 follows the noncompact branch in mmf.cuh.
- Correct-branch T=32 smoke: dequant_f16 2.521 ms, ID compaction 0.018 ms; DtoH map verification passed for all 320 assignments and the 512-entry bound array (bounds_tail=320, 320 active experts).
- During the IQ4_NL grouped MMF core stage at T=32, repeated cudaEventQuery calls returned cudaErrorNotReady for the full 15 s watchdog interval; any other query result would have raised an error instead of timing out. Completion was therefore not observed within the bound, but this does not diagnose why. The process self-terminated. GPU2 returned to 42C/1 MiB/0%. No accuracy or MMQ comparison completed for this shape.
- This result is inconclusive for the down-projection candidate and is not evidence of an engine or MMF defect. Do not crop expert-grid dimensions; production dispatch is fixed to all 512 experts. No larger down-projection run was attempted.

### Final GPU state

After the completed/aborted fixture processes, GPU2 was observed idle at 39C, 1 MiB, 0% utilization. No fixture process remained.

