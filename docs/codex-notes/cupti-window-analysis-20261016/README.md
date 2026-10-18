# V14/R3-R4 CUPTI prefill1/decode4 postprocessing

`analyze_cupti_decode4.py` is a CPU-only analyzer for V14 `prefill1` and `decode4` output from the frozen R3 or R4 collector. It uses CUPTI concurrent-kernel intervals for GPU busy time, not the Meta CUDA-event envelope parser. It reports per-device busy unions, idle gaps, kernel-name-classified MoE/QSA/GDN/AllReduce device time, runtime launch/sync API interval unions, copy records, and correlation coverage.

The clock mapping uses V14 `start_host_before_us` / `start_host_after_us` and stop anchors around the collector calls. It compares the resulting offset brackets and reports their width. For `decode4`, CUPTI intervals are mapped to the results record's `shared_window_start_us/end_us`. For `prefill1`, the analysis window is the mapped CUPTI trace start/stop interval itself: the results shared-generation window describes later decode emissions and is deliberately ignored. This mapping is an explicit host-clock alignment assumption; missing/misaligned brackets produce a warning or invalid-window result, not a guessed conversion.

## Use after the run has written the final trace and `results.json`

```sh
python3 -m unittest -v test_analyze_cupti_decode4.py
python3 analyze_cupti_decode4.py \
  --trace /home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/dual-context-v14-prep-20261016/results/p4-d3-20261015T203652Z/cupti-decode4.jsonl \
  --results /home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/dual-context-v14-prep-20261016/results/p4-d3-20261015T203652Z/results.json \
  --manifest /home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/dual-context-v14-prep-20261016/results/p4-d3-20261015T203652Z/run-manifest.json \
  --label real8k-p4-2x2 \
  --json-out /home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/dual-context-v14-prep-20261016/results/p4-d3-20261015T203652Z/cupti-decode4-analysis.json
```

For `ROOT_CUPTI_PHASE=prefill1`, use the selected label `single8k-pp-probe-32`; the analyzer uses the collector window and reports `prefill_probe_context` for the runner's separate prefill wall/token metadata:

```sh
python3 analyze_cupti_decode4.py \
  --trace /path/to/cupti-prefill1.jsonl \
  --results /path/to/results.json \
  --manifest /path/to/run-manifest.json \
  --label single8k-pp-probe-32 \
  --json-out /path/to/cupti-prefill1-analysis.json
```

If results contain exactly one `prefill1` or `decode4` record, `--label` may be omitted.

## Frozen collector SHA/ABI allowlist

The analyzer accepts only these frozen collector artifacts and maps each exact digest to the marker ABI verified on AC922 with `nm -D --defined-only`:

| Revision | SHA-256 | ABI |
|---|---|---|
| R3 | `2abfa4f07a034353dcc8ac18eee7f9af1aa6c571f36c630d98308b17bed873f2` | `ac922-cupti-window-markers-v1` |
| R4 | `2fd1c3400b7aa90583ff96ce2350d0f7f33682b85e4bcd2bf644c1d8f871309a` | `ac922-cupti-window-markers-v1` |

Both builds export `ac922_trace_prepare`, `ac922_trace_start`, `ac922_trace_stop`, and `ac922_trace_last_error`; R4 preserves the R3 JSONL schema and marker signatures. Legacy R3 manifests without an explicit `abi` field remain accepted by inferring the ABI from the exact digest. If a manifest supplies ABI, revision, or marker exports, those must match the digest mapping. Unknown or mismatched hashes/ABI are rejected before any timing metrics are emitted. A known digest proves artifact identity and the mapped marker contract; it does not prove that a particular trace is valid.

Only pass `--islands '0,1,2|3,4,5'` after confirming the CUPTI-visible IDs follow that physical order for this run. The trace's device events map CUPTI IDs to UUIDs; the runtime manifest may not preserve `CUDA_VISIBLE_DEVICES`. Do not infer island membership from list order if the run reordered devices. By default the aggregate capacity percentage divides by the number of devices with kernel activity and labels that denominator. Pass `--visible-device-count 6` only after verifying the run had six visible GPUs; then the denominator is explicitly all six.

Target-versus-MTP context roles are not present in the R3/R4 event records or V14 `bank-config` lines. The analyzer emits per-context busy results as unmapped by default. An optional `--context-map` JSON file can assign roles only after an independent explicit mapping is available, for example `{"12":{"role":"target","bank":0},"13":{"role":"mtp","bank":0}}`. It does not infer roles from creation order.

## Interpretation boundaries

- `kernel_busy.per_device.kernel_busy_ms` is the union of actual CUPTI kernel execution intervals clipped to the selected analysis window (results shared-generation window for decode4, CUPTI trace window for prefill1). `gpu_kernel_gaps` is the complement within that window. Aggregate device-ms may exceed wall time across GPUs; `aggregate_gpu_capacity_percent` states whether it is normalized by kernel-active devices or an explicitly verified visible-device count.
- MoE, QSA/indexer, GDN, and AllReduce labels are kernel-name heuristics. Unrecognized names remain `other`; they are not silently reclassified.
- Runtime API unions cover CUDA Runtime API intervals only. They do not include arbitrary host work or CPU work outside runtime calls. Correlation IDs link APIs to launches/copies when CUPTI exposes them.
- CUPTI instrumentation changes execution overhead. The report always labels itself diagnostic-only and never marks the run as a clean performance result.
- A trace with nonzero selected-window drops, buffer starvation, overflow, parse errors, no kernel records, missing anchors, or an empty analysis window is rejected with the exact reason and without performance metrics.
- The separate `analyze_p4_gpu_spans.py` reports Meta CUDA-event envelopes. Those include stream gaps and must not be called GPU kernel busy.

The synthetic tests cover interval unions, overlap, categories, runtime API union inputs, explicit context-role mapping, device islands, invalid drop counts, diagnostic-only labeling, R3 backward compatibility, R4 ABI acceptance, and rejection of unknown or mismatched collector provenance. No GPU is used by the analyzer or its tests.
