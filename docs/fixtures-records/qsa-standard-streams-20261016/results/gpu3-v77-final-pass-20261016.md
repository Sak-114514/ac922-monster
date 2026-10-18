# QSA standard sparse-stream fixture: bounded V77 GPU pass

- Runtime: `/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v77`; physical GPU3 only; no model loaded; one run under a 60-second timeout.
- Source SHA256: `ad240929c62b713d5aad805fa22923e2fcb55ab773c29be11ff2a1d394f486c0` (local and AC922 copies match).
- Executable SHA256: `e49d211e37f813a004bc743c55b3622d2740afd88959fcba3d2d5148c67035eb`.
- Raw log: `qsa-standard-sparse-gpu3-v77-20261016T063643Z.log`, SHA256 `3cb72ca6943d431265bb1cfefe050ea04e3070c72d9c7c84e59bc71724b95eda` (local and AC922 copies match).

All eight combinations passed helper-vs-previous explicit-cell CUDA output bitwise equality, fixed elementwise Q8_0 FP64 bound (`atol=2e-6`, `rtol=2e-5`), finite output, input-variation checks, and KV tail-canary readback. Helper-vs-FP64 maximum absolute error ranged from `5.51e-7` to `8.40e-7`; CPU selected-order vs dense FP64 maximum absolute error ranged from `7.55e-15` to `1.02e-14`.

The actual dense `ggml_flash_attn_ext` arm passed and was finite for the supported `streams=1, ntps=4, heads=4` shape (max absolute difference from the FP64 dense reference `9.21e-4`). It was explicitly skipped for the other seven shapes because the first run showed V77 has no compatible `flash_attn_ext_f16` device code for those sm_70 shape specializations. This is not reported as a pass for those seven dense-FA cases.

Pre-GPU checks passed: host syntax with `-Wall -Wextra -Werror`; CPU backend allocation/upload without graph compute for all eight shapes; host graph plan for all eight; CPU oracle for all eight; and the Python stream-stride test. The first unsupported-dense run and the following fixture upload-bug run remain preserved as separate logs/notes; their corresponding 79 KiB fixture binaries are archived under `results/qsa_standard_sparse_cuda_fixture-before-dense-gate-v77` and `results/qsa_standard_sparse_cuda_fixture-empty-mask-bug-v77` on AC922.

Post-run GPU status: GPU3 was 39 C and 3 MiB used; all six cards were 3–8 MiB used, and no fixture process remained. No model, core source, GPU reset, or system setting was touched.
