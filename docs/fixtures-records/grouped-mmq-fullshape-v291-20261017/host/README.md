# V291 grouped-MMQ full-shape baseline fixture

This fixture compares the existing GGML `MUL_MAT_ID` grouped-MMQ path at the model's full expert shapes. It does not call any custom CUDA kernel, load the model, change quantization, or modify engine sources. Inputs are F32. The synthetic E=512 weight bank alternates the exact first 64 rows of raw experts 170 and 171 and repeats those rows to N=640 or N=2560.

## Scope

- Main route: IQ3_XXS gate/up K2560×N640, IQ2_XS gate/up K2560×N640, Q2_0 down K640×N2560; E=512, top-k=10; T=512, 1024, 2048 uniform routing.
- Optional route-shape cases: the same formats at T=512 and 1024 with all tokens routed to the same 10 experts.
- Runtime flags: `GGML_CUDA_MOE_GROUPED=1`, `GGML_CUDA_MOE_ADAPTIVE_J=1`, `GGML_CUDA_MOE_SMALL_J=1`, `GGML_CUDA_MOE_SMALL_J_MAX_TOKENS=512`, and `GGML_CUDA_MOE_FUSED_T16=1`.
- The standalone `MUL_MAT_ID` graph cannot exercise gate/up GLU fusion, so `FUSED_T16` is set but reported as not exercised.
- Quality is report-only: six representative tokens, all 10 routed outputs, and all N rows are compared with a CPU F32 dot product after dequantizing the two exact source experts. It does not check every token's full output.
- Dispatch logs prove the selected grouped-MMQ family. The exact cold/hot J bucket is inferred from the emitted route histogram and the source's expert-count thresholds; the current logger does not print a per-expert J counter.

## Safety and outputs

The case runner requires `V291_GPU_WINDOW_GRANTED=1`, an idle selected GPU, temperature below 58 C, at least 512 MiB free, no competing compute PID, and at most 120 seconds per case. It then polls the same guards once per second, stops the process group on a violation, and verifies the GPU is idle after the case. Both estimated and actual per-case allocations are capped at 2 GiB. Matrix execution is sequential and stops on the first failed case. Logs, command manifests, and guard samples go under `host/results/<run-tag>/`.

## CPU-only preparation check

```sh
python3 fixtures/grouped-mmq-fullshape-v291-20261017/host/verify-v291-static.py
```

This compiles/runs only the route-plan CPU test, checks the six local raw-sample hashes, validates J-bucket calculations, and confirms the release gate. It does not query or use a GPU.

## Root-controlled build and execution

Only after root pins the V290 runtime and grants the host-link window:

```sh
V291_BUILD_AUTHORIZED=1 RUNTIME_ROOT=/path/to/frozen-v290-runtime \
  fixtures/grouped-mmq-fullshape-v291-20261017/host/build-hostlink-v291.sh
```

After root separately grants a bounded GPU window, the primary matrix is:

```sh
V291_GPU_WINDOW_GRANTED=1 RUNTIME_ROOT=/path/to/frozen-v290-runtime DEVICE=3 \
  python3 fixtures/grouped-mmq-fullshape-v291-20261017/host/run-v291-matrix.py \
  --matrix main --run-tag unique-v291-main
```

Optional all-hot routing:

```sh
V291_GPU_WINDOW_GRANTED=1 RUNTIME_ROOT=/path/to/frozen-v290-runtime DEVICE=3 \
  python3 fixtures/grouped-mmq-fullshape-v291-20261017/host/run-v291-matrix.py \
  --matrix allhot --run-tag unique-v291-allhot
```

No build or GPU run is authorized by the local preparation step. These measurements are operator-level baselines, not LLM throughput results.
