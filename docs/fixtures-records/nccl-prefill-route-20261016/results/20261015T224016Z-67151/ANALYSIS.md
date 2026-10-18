# Paired NCCL F32 / P2P64 result

Run tag: `20261015T224016Z-67151`  
Runtime: `runtime-perf-v56`  
Devices: GPU0–2, same UUID set for both route processes  
CUDA Graph capture: disabled  
Fixture: synthetic Q8_0 MMID → F32 AllReduce → RMSNorm Meta3 graph

## Correctness and route evidence

The retained summary says `OUTPUT_HASHES_BITWISE_EQUAL=1` and
`ROUTE_COMPARE_PASS`. Each arm reports 7 warm checks, 21/21 CPU-oracle output
passes, 14 rank-consistency checks, and 21 output hashes. The maximum logged
CPU-oracle absolute difference is `1.1920929e-7` in both arms. The two route
hash files are byte-identical (SHA-256
`ef308db1ba0bb94ac00d91611ec67e47f37c358b161f33d73cabfaf991cb4724`).

Dispatch evidence is distinct between the arms:

- P2P64: grid 64, `p2p_launches=114`, `segmented_launches=54`,
  `zero_launches=2`.
- NCCL: `nccl_wire_f32` reports
  `reason=explicit_f32_no_bf16_scratch`; route result requires F32 and reports
  zero P2P launches.

These gates cover this synthetic operator fixture only. They do not establish
full-model correctness, capture safety, or a serving speedup.

## Whole-graph timing

Each timing is the full `graph_compute + meta_sync` wall completion, not an
isolated collective measurement. The 200 KiB repeated shape contributes eight
samples; each other payload has four. Timed order and inputs are the same for
both route arms.

| Actual AllReduce bytes | P2P64 median [range] µs | NCCL F32 median [range] µs | NCCL vs P2P64 |
| ---: | ---: | ---: | ---: |
| 213,044 | 229.5 [198, 242] (n=8) | 277 [273, 303] (n=8) | +20.7% slower |
| 409,700 | 239 [201, 248] | 279 [273, 293] | +16.7% slower |
| 4,211,716 | 478 [471, 483] | 405 [402, 415] | −15.3% |
| 16,797,700 | 1,302.5 [1,299, 1,325] | 858 [857, 865] | −34.1% |
| 33,579,012 | 2,767.5 [2,712, 2,838] | 1,565.5 [1,561, 1,568] | −43.4% |

The 4 MiB NCCL round 13 record was interleaved with another logger's path
summary in the raw file: the prefix ends at line 197 and the remainder is at
line 201 (`... round=13 ... wall_complete_us=405 ...`). That recovered sample
is included above. The other three 4 MiB values are 415, 405, and 402 µs.
The NCCL log therefore has 23 whole-line timing records plus this one
recoverable split record, for the 24 rounds stated by the fixture.

## Recommendation and limits

The measurements place the observed route crossover somewhere between
409,700 B and 4,211,716 B; they do not identify an exact crossover. A
default-off `GGML_META_PREFILL_NCCL_MIN_BYTES=4194304` candidate is conservative
for a source experiment: it keeps the tested 409,700 B case on P2P64 and sends
the tested 4,211,716 B and larger cases to NCCL. This is a threshold candidate,
not a claim that every payload above 4 MiB wins or that model-level latency
improves.

Any future implementation must decide the route consistently at both the
pre-merge/capture decision and the later Meta communication boundary. For
eligible large contiguous F32 reductions on flat Meta3 with three CUDA
children, it should decline P2P merge/callback and use the existing NCCL F32
communicator. Smaller reductions keep the fixed P2P64 route. Parse the
threshold once, keep P2P grid/epoch state unchanged, and never switch routes
during communicator replay. CUDA Graph capture remains untested here because
both arms ran with graphs disabled.

## Provenance

| Artifact | SHA-256 |
| --- | --- |
| `p2p64.log` | `b0fe428cf6e9b718b9dc4214496a586d2ba87bc1869b91f68775de9fb94aa33e` |
| `nccl-f32.log` | `0457cb4851d78f1ea899477483c8a73d809573e7633d03169c94f1c86782d9ff` |
| `summary.log` | `c5af0c35eee1165a30149a7c9654276071939df25520a588e3e2ccbc1bbd64d3` |
| Both `.hashes` files | `ef308db1ba0bb94ac00d91611ec67e47f37c358b161f33d73cabfaf991cb4724` |

Fixture executable SHA-256: `b177fb71c67be9151bda91a3a7686559984a3b3ca9309992ebb63d3f2def6ad0`  
Fixture source SHA-256: `fd22a8942d93cdf2aaa68ed39e98aff3b34578deff769d11050d17a971eeb0cb`

At fixture preparation, the source inspected on AC922 had these hashes:

- `ggml/src/ggml-cuda/ggml-cuda.cu`:
  `0a591c48a351e86ab3b1a6e2a42e1add4e0e48fe260d6f1ee61de6004abb5883`
- `ggml/src/ggml-cuda/allreduce-p2p.cu`:
  `25d3619a3fbd518c06949698e2f886cd3066ea9690d39cd2dc54e907d5ddf0af`
- `ggml/src/ggml-backend-meta.cpp`:
  `cea773c4632c54bfc00cee63a1fe02cdc46d2fabc7d5437189489bed185a7131`
