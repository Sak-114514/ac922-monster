# Results: CUDA graph opaque `tensor->extra` regression

Date: 2026-10-16 UTC. Tiny CUDA backend fixtures only; no model was loaded.

## Main external-subgraph result

The fixture materializes a CUDA F32 ADD producer, then uses `ggml_graph_view(full, 1, n_nodes)` to execute only the consumer suffix. The producer is absent from the current graph node list and is reached through `sum->src[0]` as an external dependency. The suffix gets two stable, distinct nonzero graph UIDs in an A/B/A sequence.

| Runtime | Probe | Result |
|---|---|---|
| V63 | opaque external producer `extra=0x57800`, structural=1, snapshot=1 | Expected SIGSEGV reproduced in the forked child while structural inspection visits the external binding; parent caught it and exited 0. |
| V64 | same external producer, structural=1, snapshot=1 | Structural lookup safely rejected the external descriptor (`reason=unsupported_external`); ordinary CUDA graph execution returned exact A/B/A outputs and intact canaries. |

Exact V64 log markers include `external_producer=1 producer_in_view=0 ancestry_expanded=0`, `reason=unsupported_external`, three exact/canary frame passes, and `OPAQUE_EXTRA_CHILD_PASS`.

## Supplemental probes

The earlier current-node version also reproduced the V63 SIGSEGV and verified V64's `opaque_node_extra` decline with both snapshot=1 and snapshot=0. Its normal no-extra control returned exact A/B/A output with snapshot=1. That normal control had `flat_snapshot_hits=0` and therefore does not establish a snapshot-cache hit. The final external-subgraph revision was not rerun with `extra=null` or snapshot=0 because the GPU was handed back for V26; those are optional follow-ups, not needed for the external opaque-extra regression.

All runs were on GPU 2 only. The runner enforced `<58 C`, `<=512 MiB` before launch, `<=1 GiB` during a fixture, and a 60-second timeout. After the last probe GPU 2 reported 39 C / 5 MiB and no fixture/model compute process remained.

## Artifacts and provenance

- Final fixture source SHA-256: `ecefc7f0c5a0084b0cb3428c9482f7c0b414455852a3f9c2e9a0950bb726f4d6`
- V63 external-subgraph binary SHA-256: `5c09f8957754ecd394f94c8889a810d7c9ec8ff0e7e636757e030eb3a218cb93`
- V64 external-subgraph binary SHA-256: `8cff19757574d29b999e257f1c9f30dfc6851e6a28989ccec50f22ac119bb7e1`
- V63 `libggml-cuda.so.0` SHA-256: `e5a91189f18dce2eed73369a36088696542ee09ebdf94cde195ed3c6c226f983`
- V64 `libggml-cuda.so.0` SHA-256: `91dda4e38adf0de40124467246f521f9007832c347d77a001bc401f52fb5b047`

Main external-subgraph raw logs:

- `results/v63-external-sentinel-snapshot1-20261016T020425Z.log`
- `results/v64-external-sentinel-snapshot1-20261016T020429Z.log`

Supplemental current-node runs are also preserved in `results/`.

This confirms the specific V63 external-binding crash condition and V64's fail-closed behavior for a one-device F32 CUDA graph. It does not prove model-level graph coverage, snapshot hits, or inference performance.
