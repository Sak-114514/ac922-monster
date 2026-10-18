# EP-MMVQ compact-route fixture results

## V84 operator fixture

Build: `20261016T082325.972432Z`, runtime `runtime-perf-v84`.

- CUDA runtime library SHA-256: `dbd6e23d0063b8871f1bc4e8b3a36c5512134c96f1a55d9a1efc067a355bb2ce`
- Core `mmvq.cu` SHA-256: `cfadc4181bf9762c5f6038cedeb91f9f830f4c03f5b509c40dedccdfaaed6e39`
- Fixture binary with timing: `20b7ddbb68bd8ad09c265f1417fbfb74642ae4a41a20d5b3304b7dc133738e97`
- Fixture source used for that binary: `d3d7e6f0e125b9f94585ec2e6263dbffe5717e683c80a505682c39e468b345ae`

The first bounded run, `results/20261016T083459Z`, ran only IQ2_XXS. Its raw OFF/ON logs have T4 and T8 `PASS case` records and the compact-route summary `requested=4 selected=4`. All six output files compare byte-for-byte. The initial runner stopped because it expected detailed marker lines for both T4 and T8; the logger only prints its first two detailed events. This was a fixture-validator error, not a kernel or output failure. Those original files were retained and reused without rerunning IQ2_XXS.

Run `results/20261016T083947Z` ran the remaining six formats on GPU3. It copied the IQ2_XXS logs and outputs from the first run, then ran IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, and Q2_0. All seven formats have OFF/ON T4 and T8 A/B/A output files; the runner compared 42 pairs byte-for-byte. Each case also checked finite output, nonzero/distinct owned routes, zero foreign-route output, and exact activation, route-ID, and quantized-weight source readback.

For each supported candidate format other than IQ2_S, the candidate log has `ep_mmvq_compact_routes` summary fields with `requested=selected=4`; fixture case records prove both T4 and T8 ran. IQ2_S compact dispatch declined with `selected=0` and the logged native-IQ2S fallback reason, then passed exact OFF/ON output parity.

The new fixture binary reports five synchronized warm `ggml_backend_graph_compute` wall-time samples after the A/B/A correctness checks, with no tensor or weight readbacks in the timed section. Medians in milliseconds were:

| Format | OFF T4 | ON T4 | OFF T8 | ON T8 |
|---|---:|---:|---:|---:|
| IQ2_XS | 0.043 | 0.046 | 0.061 | 0.065 |
| IQ2_S | 0.044 | 0.045 | 0.062 | 0.062 |
| IQ3_XXS | 0.045 | 0.048 | 0.062 | 0.065 |
| IQ3_S | 0.048 | 0.050 | 0.068 | 0.070 |
| IQ4_NL | 0.037 | 0.041 | 0.046 | 0.048 |
| Q2_0 | 0.051 | 0.055 | 0.055 | 0.059 |

IQ2_XXS has no timing sample because its earlier result was retained without rerunning it. These are tiny synthetic-fixture graph wall times, not model performance measurements. The ON times are slightly higher in these samples, so this fixture provides no speedup claim.

The V84 host link passed with `-Wall -Wextra -Werror -pedantic`. GPU3 postflight passed the 512 MiB free-memory and 58 C limits; it ended at 9 MiB used, with no compute PID. The fixture uses synthetic E85 dispatch tensors made by permuting intact raw rows from original expert-170 samples. It does not load a GGUF or claim model semantics.
