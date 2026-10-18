# Relay6 allreduce fixture results

## V92 ordered Relay6 bounded fixture result

- Frozen runtime build ID: `20261016T112235.532285Z`;
  runtime `runtime-perf-v92`; included `allreduce-p2p.cu` SHA-256
  `b0790248c39350d862e09d0841575a8c7bad1e3c5de19fe50e55c8c9ad311b19`.
- Initial host2 fixture binary SHA-256:
  `7fb6f5ca01999d7edece949e436426aacd319164134081f928de316be91f820a`.
  Its first fallback case selected kind 0, then stopped before numerical
  checks because `cudaEventElapsedTime` was queried before stop events had
  completed. The failed logs remain preserved at
  `results/v92-ordered-v92-ordered-bounded-20261016T114200Z/`.
- Fixed the fixture by recording all six stop events, synchronizing all six,
  then querying elapsed times. The V92 host3 fixture-only build passed with
  no engine rebuild. Binary SHA-256:
  `1b67e1675b7ee2c7a8dbd362311e17ec89dab3c64bdaadcf9d95d249df50bbc4`;
  build record: `build-v92-host3.txt`.
- The authorized bounded run completed 16/16 cases in the 120-second total
  budget. Ordered kind 0 was selected for 160 KiB on both stream-dispatch and
  explicit `ggml_cuda_p2p_ar_graph_append()` paths. Ordered kind 4 was
  selected for 200 KiB and 10 MiB on both paths, all six ranks, and zero
  ranks -1, 0, and 3. The 10 MiB segmented kind 2 and prior Relay6 kind 3
  dispatch controls also passed.
- Every case passed exact rank-order FP32 cancellation oracle checks,
  per-rank changed B inputs, A/B/A replays across two contexts, all-negative-
  zero sign expectations, and output canaries. Two communicators stayed
  below the 1 GiB added-memory bound per GPU. The runner's safety gates and
  final all-GPU idle assertion passed.
- Same-window 10 MiB dispatch timings with zero rank 3 (mean max GPU event
  over three replays per context):

  | Mode | Context 0 | Context 1 |
  |---|---:|---:|
  | Ordered kind 4 | 1.321 ms | 1.330 ms |
  | Relay kind 3 | 1.940 ms | 1.957 ms |
  | Segmented kind 2 | 5.206 ms | 5.626 ms |

  These are operator-fixture timings only; they do not establish model
  throughput or quality.
- Run artifacts:
  `results/v92-ordered-v92-host3-eventwait-20261016T114500Z/`.
  `summary.json` SHA-256:
  `586fed0b3e086dab753059d8de8a60ce02c4b87e4846c6cc1de9f2db0528c605`;
  run log SHA-256:
  `d00796bc82df2c29c89b9e2e5bdaa3629d5d67264bee9c365f129877cf27bdbd`.
- Postflight: GPUs 0–5 were at 37–39 C, 0% utilization, and 8/8/9/9/13/12
  MiB used. The copied V81/V86 logs below remain historical artifacts.

## Historical V81 Relay6 10 MiB actual-dispatch result

## Result

The bounded V81 fixture passed in both modes through the actual dispatcher:

- Payload: 2,621,440 FP32 elements = 10,485,760 bytes.
- Segmented arm selected kind 2 on all six ranks.
- Relay arm selected kind 3 on all six ranks; relay6 selection logs include
  `rank-order-exact=1 cross-payload=4V`.
- Each arm used two independent communicators and three A/B/A graph replays
  per context. All six ranks in all six replays per arm matched the
  rank-ordered FP32 oracle exactly; output canaries passed. Rank 3 was the
  zero-slice rank and still participated in every collective.
- Both modes exited 0. The 1 GiB per-GPU added-used delta ceiling and 512 MiB
  per-GPU global-free floor remained satisfied. The fixture's CUDA free-memory
  ledger after both contexts and graph capture was 116 MiB added per GPU in
  segmented mode and 348 MiB added per GPU in relay mode.
- GPU postflight after the run: 2/2/2/13/5/7 MiB used on GPUs 0–5, 0%
  utilization, 40/40/40/40/40/41 C, no compute PID. The GPUs were released.

## Timing

The fixture reported these means over its three graph replays per context:

| Arm | Context 0 wall / max event | Context 1 wall / max event | Combined wall / max event |
|---|---:|---:|---:|
| Segmented kind 2 | 5.249 / 5.222 ms | 5.231 / 5.205 ms | 5.240 / 5.213 ms |
| Relay kind 3 | 2.024 / 2.014 ms | 1.971 / 1.961 ms | 1.998 / 1.987 ms |

These are bounded operator-fixture timings only. They do not establish an
end-to-end model throughput or quality result.

## Provenance

- Runtime build ID: `20261016T073107.274191Z`.
- Frozen V81 CUDA library SHA-256:
  `2f22091738127c31d1bdef499d9b55ec2e21e4cedac911fbce9e61923bdc15e6`.
- Host fixture binary SHA-256:
  `7196bd1a746344b994c9f37dad2491095e4a0b9052ac06ce84d5ca81bbbb788e`.
- Full run log: `results/v81-relay10m-host1-dispatcher-20261016T075519Z.log`.
- Run log SHA-256:
  `d30377b5d6c265737c6cb7a4b48a6c7d553995c7593e7d9bd56b184334f64d9b`.
- The host fixture was built with `-DGGML_CUDA_USE_GRAPHS`, matching the
  frozen V81 runtime ABI.

No engine source, model, fan, BMC, or watcher was changed for this fixture.

## V86 static localgate prepared

The V86 report at `notes/V86-build.json` is build
`20261016T093817.911598Z`, runtime `runtime-perf-v86`, with no blockers and
`allreduce-p2p.cu` as the only source change since its parent runtime. The V86
CUDA DSO hash is
`cf4bc24d21459ba547c451da3f8cf8267f1fd668a6439f3b0a952e629d537095`; the
source hash is
`a139dbbca28b38d5d433f59ce4ccdbfe2f862ec05c8ac598b04db21470edf93e`.

The fixture source includes `allreduce-p2p.cu` directly. An initial check of
the existing V81 binary showed that its GGML DSO dependencies resolved from
V86, but `nm` then confirmed the binary itself already defined
`ggml_cuda_op_allreduce` and `p2p_ar_prepare_launch`. That V81 binary embeds
the old dispatcher implementation and is not used for V86. Its original
DSO-only check is preserved at `results/v86-localgate-20261016T094414Z.log`
(SHA-256 `e1ed22bf05c595cd0915d259135cb64f35f7a3c2eaa963dcf3232aea93f71490`)
and is superseded.

The fixture-only `build.sh` V86 profile compiled a new executable from the V86
included source, with `-DGGML_CUDA_USE_GRAPHS` and the V86 runtime libraries.
It did not execute the binary or access a GPU. The host build record is
`build-v86-relay10m-host1.txt`; the binary SHA-256 is
`b9229d038d4b8caae7db472a8f9cb97903bd3f66f368bc8786eec8b159172dc8`. The
binary contains the `p2p-ar-path` logger string and defines the dispatcher and
`p2p_ar_prepare_launch`; its RUNPATH and resolved `libggml-{cuda,base,cpu}.so`
and `libggml.so` closure all point to `runtime-perf-v86`.

The V86 build script SHA-256 is
`892f58bfd04c6f35fc0f859d2e34720b9f8dfe930775598a0ce011e5de73d4f5`; the
build record is `build-v86-relay10m-host1.txt` (SHA-256
`c1ec9c09657612b0577a9cdd14e1956ac3116debf22ae3c7ec6bc5a07855307b`). The
new executable SHA-256 is
`b9229d038d4b8caae7db472a8f9cb97903bd3f66f368bc8786eec8b159172dc8`. It
contains the V86 `p2p-ar-path` logger string and defines the dispatcher and
`p2p_ar_prepare_launch`; its RUNPATH and all four GGML DSO resolutions point
to `runtime-perf-v86`.

The corrected static localgate is recorded in
`results/v86-localgate-20261016T095035Z.log` (SHA-256
`867ac2b190e71e014a482e0763fc1718170298fdcda35816fb99b1c7c223199a`). It
verifies the V86 build report, directly included source hash, executable hash
and symbols, and exact resolved DSO closure. It did not query or launch a GPU.
The earlier `094849Z` static log has the same build and closure evidence but
an incomplete next-gate note; it is superseded by `095035Z`.

The V86 runner preserves the existing segmented/relay, two-communicator,
A/B/A, exact-oracle, zero-rank, canary, and timing checks. Its additional
fail-closed marker gate requires `p2p-ar-path entry=dispatch` with algorithm
`relay6`, exact payload size, rank-order marker, and explicit relay6
confirmation for ranks 0 through 5. This fixture builds captured graphs by
calling `ggml_cuda_op_allreduce()` inside a CUDA stream capture; it does not
call `ggml_cuda_p2p_ar_graph_append()`. The validator reports
`SCOPE_LIMITATION entry=graph_append not exercised`; it does not manufacture
or claim graph-append coverage. The stale V81 log was rejected by the marker validator as
expected; its output is
`results/v86-marker-v81-negative.log` (SHA-256
`38b963f0f3a4d0bf7f92d6e769bd06f90f2cfeea6972b6460f94fb4d152e60f6`).

## V86 actual-dispatch GPU result

The bounded V86 actual-dispatch run used the frozen V86 fixture binary and
runtime after the V87 Q5_K operator fixture released GPU3. Both segmented and
relay modes exited 0. Each mode used two communicators and three A/B/A replays
per context. Rank 3 remained the zero-slice participant. All outputs matched
the rank-order FP32 oracle exactly and all canaries passed. The V86 marker gate
confirmed `p2p-ar-path entry=dispatch algorithm=relay6 kind=3` on ranks 0–5
for the 10 MiB payload (`n=2,621,440`), with rank-order exactness.

| Arm | Average wall | Average max GPU event |
|---|---:|---:|
| Segmented kind 2 | 5.233 ms | 5.207 ms |
| Relay kind 3 | 2.272 ms | 2.262 ms |

This is a bounded dispatcher fixture result, not model throughput. The fixture
uses captured-stream dispatch and replay; it does not call
`ggml_cuda_p2p_ar_graph_append()`. The log explicitly records
`SCOPE_LIMITATION entry=graph_append not exercised`.

- V86 build ID: `20261016T093817.911598Z`.
- V86 fixture binary SHA-256:
  `b9229d038d4b8caae7db472a8f9cb97903bd3f66f368bc8786eec8b159172dc8`.
- Run log: `results/v86-relay10m-after-v87-20261016T101607Z-dispatcher-20261016T101609Z.log`.
- Run log SHA-256:
  `46c452f5dde242b159fb67617b78160355ebce3421a3470d966f9e2f5c8b880b`.
- All six GPUs were released after the run. Final state: 38–39 C, 2–6 MiB
  used, at least 16,121 MiB free, 0% utilization, and no compute processes.
