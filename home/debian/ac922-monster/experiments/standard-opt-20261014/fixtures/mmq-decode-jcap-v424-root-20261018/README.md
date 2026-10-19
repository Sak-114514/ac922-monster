# V424 decode J-cap microprobe preparation

This is a guarded V291-derived `MUL_MAT_ID` microprobe for all-hot top-10 routing at T4/T6. It covers IQ3_XXS, IQ2_XS, and Q2_0 in `auto`, `cap8`, and `cap16` modes: 18 sequential cases. The route plan accepts positive T; the six representative CPU-oracle indices are clamped to valid tokens and may repeat at these small sizes.

All three arms keep `GGML_CUDA_MOE_ADAPTIVE_J=1` and `GGML_CUDA_MOE_SMALL_J=1`. Auto leaves both J override variables unset. Capped modes set `GGML_CUDA_MOE_GROUPED_J` to 8 or 16; this is a maximum, and the final config may choose a smaller J. The runner requires the grouped `compact_grouped_launch` path in every arm, plus adaptive/small-J decline markers; cap modes also require the source's explicit-override decline reason. The histogram's `cold_j=16`/`hot_j=32` labels describe input counts only and are not treated as final kernel J.

The optional full F32 output dump occurs after timing, uses exclusive creation, and is capped at 600 KiB per case (maximum Q2_0 T6 output is 614,400 bytes). Dumps stay on the server. After GPU postflights prove idle, the comparator performs all three pairwise mode comparisons for each format/T pair, using chunked NumPy memmaps to record bitwise equality, relative L2, max absolute error, and non-finite counts. Numerical and timing differences are report-only.

The case runner retains 2 GiB allocation, 512 MiB free-memory, 58 C temperature, no competing process, and 120 s per-case guards. It records the V236 raw sample/manifest hashes, V411 binary/runtime library hashes, platform/GPU identity, and server-only output hashes. Use only a fresh host-linked V424 C++ harness against frozen R411 shared libraries; CUDA objects are not rebuilt.

```sh
V424_HOSTLINK_AUTHORIZED=1 GGML_SOURCE_ROOT="$BASE/src" RUNTIME_ROOT="$BASE/runtime-perf-v411" \
  bash fixtures/mmq-decode-jcap-v424-root-20261018/host/build-hostlink-v424.sh

V424_GPU_WINDOW_GRANTED=1 RUNTIME_ROOT="$BASE/runtime-perf-v411" \
  BIN="$FIXTURE/host/build-v424/grouped-mmq-fullshape-v424" \
  python3 fixtures/mmq-decode-jcap-v424-root-20261018/host/run-v424-matrix.py \
  --run-tag unique-v424-decode-jcap --device 3 --timeout-seconds 120

python3 fixtures/mmq-decode-jcap-v424-root-20261018/host/compare-v424-outputs.py \
  --matrix "$FIXTURE/host/results/unique-v424-decode-jcap-matrix.json" \
  --report "$FIXTURE/host/results/unique-v424-decode-jcap-output-compare.json"
```

Preparation only: no engine source edit, host-link, GPU run, or output copy was performed here. See `source-hashes.txt` and `sample-input-sha256.txt` for frozen inputs and candidate hashes.
