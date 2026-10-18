# Independent Monpack operator parity fixture

This fixture is independent of the full model, text generation, QSA/fused
indexer, MTP, and EP runtime integration. It uses seven original, unmodified
quantized expert matrices with full K/N and one complete E slice (global
expert 170). It does not prune or requantize model weights. A fixture-only
mapping `global_id=170 - owner_offset=170 -> local_id=0` tests the local
Monpack math contract; it does **not** establish that production EP offsets are
passed to Monpack.

The sample extractor reads metadata through the installed `gguf-py` reader,
then preads only the chosen complete expert from the original GGUF and the
corresponding expert-major sidecar span. It reverses the sidecar's `(E,Ntile,
Ktile,d-plane,code-plane)` arrangement and byte-compares that result with the
source GGUF expert. This is a byte permutation check, not a re-quantization.
The seven files together are about 3.95 MiB raw plus 3.95 MiB packed.

## Prepare original bits on AC922

Run from the registered `operator-fixture` path on AC922. This opens the GGUF
as metadata/mmap, reads only seven expert slices (global E=170, about 4 MiB),
and reads the matching same-size packed sidecar slices. It does not construct a
`llama_model`, scan tensor payloads, or start CUDA.

```sh
./extract-original-samples.py --out ./samples
```

The default source is
`/home/debian/models/qwen3.8-flash-next-gsq-rco/integrated/IQ3_XXS-MTP-Q8.gguf`;
the default sidecar is its archived `.monpack.disabled-by-p3` file. The script
requires the matching sidecar manifest and the repository `gguf-py` package.
It saves exact raw/packed binary samples and `sample-manifest.json` locally in
this fixture directory.

| Type | Source tensor | Shape `[K,N,E]` | Raw expert bytes |
|---|---|---:|---:|
| IQ2_XXS | `blk.1.ffn_gate_exps.weight` | `[2560,640,512]` | 422,400 |
| IQ2_XS | `blk.0.ffn_gate_exps.weight` | `[2560,640,512]` | 473,600 |
| IQ2_S | `blk.12.ffn_gate_exps.weight` | `[2560,640,512]` | 524,800 |
| IQ3_XXS | `blk.35.ffn_gate_exps.weight` | `[2560,640,512]` | 627,200 |
| IQ3_S | `blk.8.ffn_gate_exps.weight` | `[2560,640,512]` | 704,000 |
| IQ4_NL | `blk.0.ffn_down_exps.weight` | `[640,2560,512]` | 921,600 |
| Q2_0 | `blk.1.ffn_down_exps.weight` | `[640,2560,512]` | 460,800 |

## CPU reference arm

The C++ host harness creates original GGUF-layout `MUL_MAT_ID` operations with
the GGML CPU backend for T=1,4,8, using the same deterministic F32 activations
as the optional CUDA arms. It verifies that all seven original-byte samples
are accepted by the stock CPU operation and produce finite outputs. Run on
AC922 after extraction:

```sh
./run-host-cpu.sh ./samples
```

`build-host.sh` uses ordinary `g++` and the frozen V49 shared libraries; it does
not invoke NVCC. It leaves the executable under `/tmp` and removes it after
the CPU run.

The host run is prepared and passed: remote `g++ -fsyntax-only -Wall -Wextra
-Werror -pedantic` succeeded, and `run-host-cpu.sh` passed all 21 format/token
cases. The mapper also checks rank ranges `[0,170)`, `[170,341)`, and
`[341,512)`, mapping fixture global expert 170 to local ID 0. The actual
payloads and per-slice SHA-256 values are retained in `samples/`; command
outputs are retained in `results/host-syntax.log` and `results/host-cpu.log`.

## CUDA parity arm and V49 GPU0 result

The same host C++ harness creates, for each type and T=1,4,8:

- the stock-CUDA reference with the original raw GGML bytes;
- the Monpack CUDA result with the exact sidecar expert bytes registered
  directly through the exported Monpack library API;
- the stock CPU reference as a separate cross-check.

The harness requires exact bitwise Monpack-vs-stock-CUDA equality and reports
CPU-vs-stock-CUDA equality plus max-absolute/max-relative errors. A mismatch
prints the metrics for every case and exits nonzero. It was run in the
root-assigned GPU0 window with the explicit guard enabled:

```sh
MONPACK_OPERATOR_GPU_WINDOW=1 CUDA_VISIBLE_DEVICES=0 ./run-cuda-parity.sh ./samples
```

The harness registers only an E=1 local tensor and sends local route ID 0.
It cannot validate global-to-local EP mapping, foreign-zero behavior, fused
gate/up + GLU, a full model, token history, or performance. Those remain
separate production gates. The V49 `libggml-cuda.so.0` exports the existing
Monpack registration hooks, so the fixture links the frozen library and does
not rebuild CUDA sources. Its SHA-256 at preparation time was
`614cc8685ba9d8c2f5a998a34af330609d7a372d7e764d07052c4ef3d808e7d1`;
`nm -D -C` confirmed `register_tensor`, `is_registered`, and the context-taking
`try_mul_mat_id` symbols. No CUDA function was called by the host-only run.

The fixture does not assume the old P3 full-model text result is a numerical
kernel verdict. It preserves the actual original weight bits and creates a
direct operator comparison that can independently confirm or falsify that
claim.

### Run record

- UTC: `2026-10-15T21:00:22Z` (AC922 local time: `2026-10-16 05:00:22 +08:00`).
- Preflight: GPU0 was a Tesla V100-SXM2-16GB at 39 C, 0 MiB used, 0% utilization,
  with no compute processes. It was 39 C / 0 MiB / no compute process after the
  runner exited.
- Runtime: existing `runtime-perf-v49/lib/libggml-cuda.so.0`, SHA-256
  `614cc8685ba9d8c2f5a998a34af330609d7a372d7e764d07052c4ef3d808e7d1`;
  ordinary `g++` host compilation, no NVCC. The fixture completed in about four
  seconds and did not load a model.
- Raw output log:
  `results/cuda-parity-20261015T210022Z-gpu0.log`, SHA-256
  `7391b2f9ab9d26664516e51149c79651dabbf9bbfb20917467996331d5a0d2d4`.
- Strict result: 19 of 21 format/token cases were bitwise equal to stock CUDA.
  IQ2_S at T=4 and T=8 failed the unchanged bitwise gate: respectively 1,445 / 2,560
  and 2,906 / 5,120 values differed, with `max_abs=1.1920929e-7` and
  `max_rel=0.000182248952` for both. The other 19 cases had zero unequal values
  and `max_abs=0`. The runner exited 1 because these two cases failed the strict
  gate; no failure was suppressed or repaired.
- CPU outputs were finite in all 21 cases but were not bitwise equal to stock
  CUDA. Per-case CPU-vs-CUDA absolute and relative differences remain in the raw
  log; the CPU arm is not a bitwise oracle for these kernels.
- Registration logs show all seven types registered. The CUDA callsite emitted
  six `[mp-dbg] mul_mat_id hook` records, all for IQ2_XXS because the source
  counter is capped at six. Those records show the packed tensor pointer with
  `reg=1`, alternating with the raw stock-CUDA pointer at `reg=0`. V49 does not
  log the `try_mul_mat_id()` success return or a per-kernel launch marker, so the
  hook record alone is not proof of successful Monpack dispatch for every type.
  Exact Monpack-vs-stock outputs for 19 cases are corroborating evidence; do not
  overstate the marker coverage.

This direct test does not validate production EP mapping, fused gate/up + GLU,
full-model logits/history/rollback, or performance. The IQ2_S one-ULP mismatch
is retained as a numerical failure against the fixture's strict bitwise
criterion; no tolerance was added.
