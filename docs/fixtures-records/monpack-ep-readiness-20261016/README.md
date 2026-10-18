# Monpack EP full-K readiness fixture

Date: 2026-10-16
Scope: read-only audit of the frozen AC922 source plus a bounded CPU-only proof
of Monpack's contiguous expert-axis byte spans. No core source, model, GGUF,
sidecar, runtime, or GPU state is changed.

## Verdict

The old TP K/N shard obstacle does not apply to routed experts placed on EP
axis 2. In the current Qwen4exp split callback, `*_exps.weight` routed tensors
are assigned split axis 2 with rotation 0. The current model contract keeps
each expert's complete K and N dimensions on one rank and partitions E into
contiguous ranges. Monpack layout revision 1 stores E outermost, then N tiles,
then K tiles. Therefore the local EP range is already one contiguous byte span
per tensor. No quantization, K-tile gather, or N-tile repack is needed for this
storage slice.

That storage result is **not** a runtime-readiness result. The current runtime
has several independent, source-confirmed blockers:

1. `llama_monpack::tensor_meta_override()` accepts only split axis 0 or 1 and
   falls back for axis 2. Plain `tensor_src_override()` also requires the full
   global E and full tensor byte count, so it cannot bind a rank-local EP member.
2. `ggml_cuda_mul_mat_id()` deliberately aborts when a registered Monpack
   tensor reaches EP. Removing only that guard is unsafe: the current Monpack
   GEMV reads each route ID directly as local `e`, and the grouped-TC work-list
   indexes `cnt[ids[...]]` while allocating only `meta.ne_e` entries. EP route
   IDs are global, while each rank has only 170 or 171 local experts.
3. The Monpack contract excludes fused gate/up + GLU. The current EP fused
   MMVQ wrapper sends its weights through the stock quantized-vector path, which
   cannot consume registered packed bytes.
4. The archived P3 report contains a full-model grid-family arm whose generated
   text was only about 378 characters, but it reports no logit max-abs,
   per-type numerical error, or isolated CUDA operator comparison. Its version
   claim also conflicts with retained logs: the report says its figures used
   clean `b3-1afa565`, while the retained sidecar-active `p3-ab.log` sessions
   identify `build 7 (1e542c5)`. A separate historical memory summary records
   CUDA GEMV numerical tests passing for all eight packed types, but the
   corresponding fixture source/output is no longer in the current tree. Treat
   the old full-model garbage as an unresolved serving result, not proof that
   IQ2/IQ3 packed GEMV arithmetic is wrong. The direct operator parity fixture
   below has since run against V49: 19/21 format/token cases are bitwise equal
   to stock CUDA; IQ2_S T=4/8 each differ by at most `1.1920929e-7`. This small
   operator delta does not explain the old short garbage output.

The full-K axis-2 storage path is thus feasible to prototype, but the current
seven-format Monpack EP runtime must be rejected as not ready because the
loader, EP route-ID mapping, EP abort/fail-closed handling, and fused consumer
are not implemented. The direct operator test removes evidence for a broad
grid-family GEMV arithmetic failure, but its strict IQ2_S bitwise result remains
unresolved and the old serving corruption still needs a matching-runtime
semantic reproduction. Do not enable `MONSTER_MONPACK` for EP or remove the
abort based on this fixture.

## CPU-only fixture

The executable checks two bounded properties for all seven routed formats:

- With the actual routed tensor dimensions (`gate/up=[2560,640,512]`,
  `down=[640,2560,512]`), checked arithmetic derives the expert stride and
  proves that the contiguous rank spans for E counts `[170,171,171]` cover the
  whole packed tensor exactly, at prefix offsets `[0,170,341]`.
- It materializes only four experts at full K and one 64-row N tile, fills
  uniquely tagged raw bytes in `(expert,N-tile,K-tile)` order, and proves that
  direct `[1,2,1]` E spans concatenate byte-for-byte to the complete fixture.
  This uses less than 4 MiB per format and does not generate a model-sized
  sidecar. The reduced N tile count preserves the layout rule while bounding
  memory; the actual N geometry is separately checked arithmetically.

The fixture also rejects non-axis-2 splits, non-contiguous segment metadata,
incorrect expert-share sums, changed K/N dimensions, non-tileable shapes, and
integer overflow. Its synthetic payload proves only byte addressing; it is not
a quantization/dequantization or CUDA-kernel quality test.

Run from this directory:

```sh
./run-cpu.sh
```

## Required runtime patch seams if the separate quality blocker is fixed

Keep the patch confined to the loader, Monpack metadata/consumer, and the fused
route selection. Do not change the sidecar format or `MONSTER_MONPACK` switch.

1. In `src/llama-monpack-loader.cpp::tensor_meta_override()`, add a dedicated
   axis-2 branch. Validate the original full tensor against the sidecar index,
   require one contiguous axis-2 segment, unchanged full K/N on every member,
   positive local E shares summing exactly to global E, and prefix offsets
   derived from the Meta split state. For each local member, upload the direct
   sidecar interval `[blob_off + e0*expert_stride,
   blob_off + (e0+Elocal)*expert_stride)` and register that member's actual local
   shape. Keep the existing axis-0/axis-1 gather path unchanged.
2. Extend the Monpack registration/dispatch metadata with the global owner
   offset and global expert count, or pass the original EP metadata directly to
   the MMID consumer. In GEMV, map an ID to local E only when it is in the
   owner's range; write exact `+0.0f` for foreign IDs. In the TC prelude, filter
   foreign IDs before any `cnt[]` access, translate owned IDs to local E, and
   clear all output slots before sparse writes. Both paths must remain CUDA-graph
   safe.
3. At `src/ggml/src/ggml-cuda/ggml-cuda.cu::ggml_cuda_mul_mat_id()`, replace
   the existing abort only after both route consumers and fail-closed behavior
   are in place. A registered tensor must never fall through to stock MMVQ/MMQ.
4. At the EP fused MMVQ route (`ggml_cuda_mul_mat_vec_q_fused_route()`), route
   registered Monpack weights through a supported Monpack consumer and a safe
   GLU, or explicitly keep those weights out of fusion. Do not pass packed
   bytes to the stock fused vector kernel.
5. Before GPU use, add per-format consumer parity: all seven actual GGML types,
   owner ranges `[0,170)`, `[170,341)`, `[341,512)`, foreign IDs at either
   boundary, exact foreign zero, full logits/history/rollback oracle, and an
   A/B/A reuse cycle. Use the independent operator fixture below to decide
   whether the current grid-family arithmetic is actually incorrect.

## Storage and capacity cost

Monpack is a byte reordering: each packed tensor keeps its source byte count.
The 2026-10-15 audit recorded a 45,586,996,800-byte sidecar, which would add
that much disk storage beside the original GGUF if retained; it does not
reduce model storage. The earlier audit's read-only search found no `*.monpack*` under
the experiment directory, so the current deployed sidecar path/state was not
confirmed here. The device-resident expert weights remain the same quantized
byte count as their original local EP slice: the documented 39.966 GiB total
routed weights imply about 13.27 GiB on an E=170 rank and 13.35 GiB on an
E=171 rank, before other model weights. Direct axis-2 slicing adds no per-rank
expert-weight capacity and avoids the old K/N repack staging buffer for this
EP case. Existing Monpack grouped-TC work-list allocations (described as a few
MiB per CUDA device in the registry implementation) are additional, and CPU
page-cache usage while all tensor spans are read is not bounded by this
fixture. No runtime capacity or performance claim is made.

## Frozen-source evidence used

Audited read-only under
`/home/debian/ac922-monster/experiments/standard-opt-20261014/src`:

- `src/llama-model.cpp:134-135,332-340`: `ep_weight` detection and axis-2
  expert placement.
- `src/llama-monpack-loader.cpp`: full-E checks in plain override; axis-0/1
  only validation and tile gather in Meta override (`695-957`); plain override
  consistency checks are at `526-591`.
- `src/llama-model-loader.cpp:1775-1837`: plain source override is attempted
  before Meta split override; Meta members are registered after their final
  buffer contents are set.
- `ggml/src/ggml-cuda/ggml-cuda.cu:2847-2858`: registered Monpack + EP abort at
  `ggml_cuda_mul_mat_id()`; the fused EP MMVQ helper near `5900` preserves
  routing metadata but dispatches to the stock quantized-vector kernel.
- `ggml/src/ggml-cuda/monpack/monpack_format.h`: expert-major address formula,
  byte-count invariant, and type/block-size table.
- `ggml/src/ggml-cuda/monpack/monpack-kernels.cuh:712-716,1318-1363`: GEMV's
  `e=channel_x` and TC work-list indexing by route ID.
- `ggml/src/ggml-cuda/monpack/monpack-kernels.cu:185-275,798-848`: registry
  stores local tensor geometry only; `try_mul_mat_id()` has no EP owner range.
- `notes/ep-monpack-readiness-20261015.md`: prior source audit and historical
  P3 result summary. That note is evidence from archived runs, not a new runtime
  test in this task.

No core source was changed, no model was loaded, no CUDA code was built or
called, and no GPU benchmark was run. This follow-up did read seven complete
expert slices (4,134,400 raw bytes) from the original GGUF and matching
same-size spans from the archived sidecar to prepare an independent operator
fixture.

## Historical grid-family quality evidence audit

The retained primary report is
`/home/debian/ac922-monster/docs/P3-WARREPORT-20261014.md` (file mtime
2026-10-14 16:39 +08:00; report text describes the P3 work as 2026-10-13/14).
Its table records `gen_sha16=e39035627dbc3f7a` for the baseline, IQ4_NL, and
IQ4_NL+Q8_0 arms, and reports the all-grid arm only as `garbage (len≈378)`.
The probe `/home/debian/monpack/p3-probe.py` hashes generated text and reports
text length with temperature 0 and up to 96 output tokens. It does not compare
logits, per-element max-abs, or per-format operator outputs. The all-grid
result is a bad full-model output for that run, not an isolated numerical
failure of the CUDA GEMV kernel.

The report's version annotation claims the measurements used a clean
`b3-1afa565` base. Retained logs do not close that provenance:

- Sidecar-active starts in `docs/p3-ab.log` identify `build 7 (1e542c5)`.
  `ENGINE-MASTER.md` §0.5 later identifies that tree as missing the MoE
  weighted-reduction rounding fix.
- `docs/p3-arm.log` contains `build 3 (1afa565)` sessions, but no
  `monpack: sidecar ... active` marker for those sessions.
- `launch-p3.sh` enables `LLAMA_QSA_INDEXER_IMPL=fused` and
  `AC922_TEST_HC_FUSED=1`. The later §0.5 finding concerns the MoE
  weighted-reduction fusion, a distinct path; do not infer that the QSA fused
  indexer is itself the weighted-reduction bug.
- The report does not carry a historical binary SHA-256 or resolved dynamic
  library closure. The current V49 hashes are a different build and cannot
  substitute for P3.

Retained artifact hashes and dates:

| Artifact | Size / mtime (AC922, +08:00) | SHA-256 |
|---|---|---|
| `docs/P3-WARREPORT-20261014.md` | 5,361 B / 2026-10-14 16:39 | `df19671287a4dc722f578a2a1f345b612b965716499c34bcd5acd967e5b2bb93` |
| `docs/p3-ab.log` | 1,803,282 B / 2026-10-14 03:09 | `b98e5a53efaf31ff0496de7865eaa22f35808117f748ee834e3d558c533b6275` |
| `docs/p3-arm.log` | 1,528,678 B / 2026-10-14 22:55 | `a41d053ed7b513611fe24496a6d07b0b5a295920f17d94cbfac3ee68d120e14f` |
| `monpack/p3-probe.py` | 1,867 B / 2026-10-13 22:11 | `80d6a67544c25e1bfff0b65d0f0f5cd5d7357526f9f5b41acf774656249703b2` |
| `monpack/mrt-repack.py` | 27,516 B / 2026-10-13 21:38 | `49ff9c93977550bce8a850fcd04adb60f8719df0a1d03d2430a3dafc84380f43` |
| `monpack/mrt-t1.py` | 24,041 B / 2026-10-13 21:38 | `ac3cd008db1e74fe2272d566796ba9c469cf67b33e57d066986dd1d9c3dc7fab` |
| `monpack/ref.c` | 63,018 B / 2026-10-13 20:29 | `8e190e1115f205d6bc10b5f039230c33b83aa20704d91154f7f6e36d7bfffb40` |
| `monpack/repack-full.log` | 18,408 B / 2026-10-13 21:29 | `2fad5d6c1ea99341715bedb29b4cdb60ccb7e240760fa065295e463c6fea7599` |
| `monpack/repack-full2.log` | 18,409 B / 2026-10-13 21:59 | `da71d8bfa652c4f8507dd8776477f6b5aadf2d38f7576ffceec00fb967e478e0` |
| model `.monpack.manifest.json` | 356,349 B / 2026-10-13 21:59 | `949aeacbc676f9be1bfd08073de0db5b26cedf466998696a6c11102b1831ce86` |

`p3-ab.log` is the retained log with active sidecar sessions; it prints
`build 7 (1e542c5)`. `p3-arm.log` prints `build 3 (1afa565)`, but those
sessions have no active sidecar marker. No archived sidecar-active b3 binary
hash or matching dynamic-library closure was found. The current binary under
`build-monster` SHA-256 is not a historical P3 binary and must not be used as
one.

The retrievable `repack-full.log` records T0 source→sidecar byte round-trips
for 147 tensors. `mrt-t1.py` defines a Python-vs-C dequant reference gate and a
layout-independence gate over real source blocks; those are CPU decoder/layout
tests, not CUDA GEMV tests. The P3 report says its eight-type GEMV unit tests
and sanitizer passed. The
2026-10-13 memory summary also records CUDA GEMV numerical tests passing for
Q8_0, IQ4_NL, Q2_0, and the five IQ2/IQ3 grid types. However, the standalone
CUDA fixture source, run log, output vectors, and source/binary hashes are not
present in the retained remote tree; `/tmp/p3stage` is absent. The retrievable
`mrt-repack.py`, `mrt-t1.py`, `ref.c`, and `repack-full.log` are CPU
byte-repack/dequant-reference gates, not CUDA GEMV operator results. The
historical pass claim is not reproducible here, but it also means the old
full-model arm cannot be promoted to an independent CUDA failure.

The retained P3 server logs have registration debug for some IQ2 types and
`reg=0`/`reg=1` MMID hook hits for sampled types. The debug counter is capped
per process and does not identify whether each IQ2/IQ3 operator reached the
Monpack kernel. No type-specific GPU output/error record is retained. Thus the
old text result cannot distinguish a kernel math defect from registered
pointer/fallback behavior or an uncorroborated runtime build combination.

Current Monpack kernel files match the saved `source-before` snapshot, so no
later source fix was found in those files:

| Source | SHA-256 |
|---|---|
| `ggml-cuda/monpack/monpack-kernels.cu` | `7864d2c4d2501d3bd12aebc8b5ebe868ef98037b59b2045e6654e566d0b83c5e` |
| `ggml-cuda/monpack/monpack-kernels.cuh` | `10155b7b9c5a2f81585766e03a9a57b5fa8abfce816496fb26bece2b3d83b075` |
| V49 `libggml-cuda.so.0` used by the staged host harness | `614cc8685ba9d8c2f5a998a34af330609d7a372d7e764d07052c4ef3d808e7d1` |
| V49 `libggml-cpu.so.0` | `c1fd8d19e54df03e97f1c71c8c95565626ec73f44f7d4f3e334bfe3e74f464fc` |
| V49 `libggml.so.0` | `73ddff0764ae9545cc0cc561baffee5909c820e13e07060cc1bc3ed7314814e2` |
| V49 `libggml-base.so.0` | `e6578aca8f533f14feb05dfc18588330b1dfc8a38dec2003899b5e6592f16c17` |

The V49 shared library exports `monpack::register_tensor`,
`monpack::is_registered`, and context-taking `monpack::try_mul_mat_id`. The
standalone fixture below links this library without rebuilding CUDA sources.

## Independent operator fixture

[`operator-fixture/README.md`](operator-fixture/README.md) documents the direct
operator gate. The extractor preads only expert 170's complete K/N slice for
each type from the original GGUF and matching expert-major sidecar span. It
inverses the d/code-plane permutation and requires byte-for-byte equality with
the original quantized bytes before saving the samples. The 7 raw samples total
4,134,400 bytes; packed samples have the same size.

The C++ harness prepares stock CPU, stock CUDA, and Monpack CUDA `MUL_MAT_ID`
arms for the same raw weights, F32 activations, and local IDs at T=1,4,8. The
host syntax check passed; the CPU-only runner passed all 21 format/token cases
with finite output. The V49 GPU0 operator run is retained at
`operator-fixture/results/cuda-parity-20261015T210022Z-gpu0.log`; 19/21
Monpack-vs-stock-CUDA cases are bitwise equal. IQ2_S T=4 and T=8 each have
`max_abs=1.1920929e-7` and fail the unchanged bitwise gate. CPU-vs-CUDA deltas
are much larger because the CPU IQ2_S dot uses Q8_K while CUDA uses Q8_1; CPU is
not a bitwise CUDA oracle. The source return chain proves `try_mul_mat_id()`
returns true for the completed fixture shapes, though V49 has no per-call
success log beyond the capped hook record.

The new CPU-only owner contract test is `monpack-ep-route-contract.cpp`, run by
`./run-ep-route-cpu.sh`; its local/AC922 logs are under
`operator-fixture/results/ep-route-contract-*-cpu.log`. It tests full-K E
spans, offsets `[0,170,341]`, owner-local ID mapping, foreign `+0.0f`, fused
gate/up ownership, and local-E grouped-TC indices. This contract fixture does
not implement production EP support. See
[`ep-axis2-contract-notes.md`](ep-axis2-contract-notes.md) for the IQ2_S source
comparison, `try_mul_mat_id()` return proof, and minimal loader/consumer patch
seams.

The separate `[K,N,2]` grouped-TC probe uses original expert slices E170/E171
at T=16/32/128. The retained V49 GPU3 run had large IQ2_XXS and IQ3_S Monpack
TC errors. After the root's V52 `monpack-kernels.cuh` repair, the same 21 cases
are finite and all seven formats fall within stock CPU/CUDA baseline
max-absolute error envelopes; V49 logs remain unchanged. V52 host-completion-
wall means are still slower than stock, so this is not a production
enablement. See
[`operator-fixture/tc-probe/README.md`](operator-fixture/tc-probe/README.md)
and its V49/V52 raw and derived logs. The fixture is deliberately denser than
production D3 routing and does not support decode-throughput extrapolation.
