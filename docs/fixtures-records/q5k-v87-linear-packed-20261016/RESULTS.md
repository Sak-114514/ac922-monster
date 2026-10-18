# Q5_K V87 packed linear fixture results

## Provenance and host-link

The fixture host-links against frozen V87 build `20261016T095614.374883Z`
(`runtime-perf-v87`). No engine build or model load was performed.

- V87 `libggml-cuda.so.0`: `a44ce46c44908872adb7a6b093d7df6e78535f6aba6ba9dbae00ea91a7be9ed6`
- V87 `ggml-cuda.cu` backup: `a428f41ba9ea2c10d234669d6aebd0cdb9b0f4f7dbbb87b6b4c256a9616215ca`
- Existing `mmvq.cu`: `a83ed741f20619008d64839dcb2e3f40ddcc9054df396edda5afe5e6b47d0748`
- V87 public/source header hashes are recorded in `build-v87-host2.txt`.
- Corrected fixture source SHA-256: `36a78993b0646195cf8270759595609c97db72bfa49d7d54b3cdf34308ae6678`
- Host2 binary SHA-256: `6e8059e0048f813d620ea3978261dc15851a22cce0ea4d190cf5ad2c5a257da0`
- Host-link record: `build-v87-host2.txt`.
- The fixture includes public GGML headers and links to the frozen V87 DSOs; it
  does not include or compile `ggml-cuda.cu`. The V88 report says the only
  post-V87 source delta is `ggml-cuda.cu`; the V87 backup plus unchanged API
  header hashes establish the host-link boundary.

## GPU run result

The first bounded V87 attempt is preserved at
`results/v87-linear-packed-20261016T101206Z/off.log`. It stopped at the
fixture's T8x2 prefix comparison because the harness compared the 16-row
T8x2 result as 8 rows. This was a fixture shape-count error, before any ON arm;
it is not a kernel parity failure and it is not a passing matrix.

The harness now derives the compared and FP64 sample row count as
`tokens[shape] * segments[shape]`, so T4x4 and T8x2 both use 16 rows. The
corrected source is host-linked as `host2` and ran in separate OFF and ON
processes on GPU3. Both fixture processes exited 0. The wrapper's first
post-run validation attempt exited 1 because CUDA graph diagnostics on stderr
interleaved into two structured stdout timing fields. The parser now recognizes
only those exact split-field forms; it was rerun offline against the preserved
logs. No GPU rerun was made.

Offline verification passed the validator, all 15 OFF/ON output comparisons
and the B input comparison byte-for-byte. For each arm, the log has all five
shapes across A/B/A, exact input readback, canaries, T4-prefix bitwise output
parity, 40 sampled FP64 checks per shape, and HC mode-2 exactness. The ON arm
reported `requested=6 selected=6` for the packed path; the two detailed records
are T16 and T12, and the T6 4+2 chunk fallback marker remained present.

### Warm graph micro-fixture timing

Five repetitions per timing record; input uploads are excluded and an exact
output comparison follows the timed calls. These are per-graph-call operator
fixture measurements, not model throughput:

| Case | OFF median (ms) | ON median (ms) | Note |
|---|---:|---:|---|
| T4x4 | 0.045 | 0.045 | Timing calls one T4 graph; correctness case composes four T4 calls |
| T16 | 0.086 | 0.071 | One T16 graph call |
| T12 | 0.072 | 0.061 | One T12 graph call |
| T8x2 | 0.059 | 0.053 | Timing calls one T8 graph; correctness case composes two T8 calls |
| T6 | 0.055 | 0.055 | Existing 4+2 tail fallback |

The candidate is faster on the measured T16/T12 graph calls and the single-T8
call, with no measured difference for T4/T6. The V87 source/dispatcher evidence
shows the packed path selected for the eligible shapes; no CUDA profiler was
used to count actual quantization or MMVQ launches. These micro-fixture results
do not qualify model-level performance or enabling the candidate in a model
run.

The frozen V87 source does show the structural call-count change: the packed
branch calls `ggml_cuda_mul_mat_vec_q()` once for the broadcast `[K,4,T/4]`
view, while the previous chunk loop calls `ggml_cuda_mul_mat()` separately for
each 4-row chunk (2/3/4 calls at T8/T12/T16). The source comment says the packed
branch shares input quantization and kernel launch across the channel groups.
This establishes the dispatch structure, not a profiler-measured count of
actual CUDA launches.

- Corrected run directory:
  `results/v87-linear-packed-v87-corrected-20261016T101424Z/`
- OFF log SHA-256:
  `1ff4a039c88ee5c70e2ce1bef50878b75bade83c77a460cffee0a588fe3b52d4`
- ON log SHA-256:
  `1861559c44eb504e7465b0bd48505954056f1de8126d48c2ad549d127e634633`
- Wrapper log SHA-256:
  `70672eb00d63cc5d7d1e548a63145bb1f98ff907074f41c4ded1fcb80d98348c`
- GPU3 was released after the run; final state was 38 C, 1 MiB used,
  16,126 MiB free, 0% utilization, and no compute PID.
