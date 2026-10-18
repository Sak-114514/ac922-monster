# Onecat variable-lane XQA fixture — work log

Status: **partially investigated; V34 stock/Onecat matrix is not complete** because AC922 became unreachable before the follow-up build/run. Do not treat this note as a Onecat correctness result.

## Scope and V34 provenance checked

- Target runtime: `/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v34`.
- V34 build record reported `run_completed=true`, with only `ggml/src/ggml-backend.cpp` changed for that build snapshot.
- `strings` and `nm -D` on V34's `libggml-cuda.so.0` showed the `using opt-in Onecat Flash-V100 XQA decode path` marker and exported `ggml_onecat_v100::{env_enabled,supported,get_alloc_size,launch}` symbols.
- The first fixture build linked directly to V34, without `fattn-volta-onecat-fixed.o`; its undefined Onecat references are intended to resolve from the V34 CUDA library.
- GPU2 UUID was confirmed as `GPU-e6529eb8-aa1f-1d02-5d63-dfb7995bf1eb`. The initial runner restricted CUDA visibility to that UUID and found no other GPU2 compute PID before launch.

## First bounded stock attempt

The first stock process at `GGML_CUDA_FATTN_ONECAT=0` completed T=1 and T=3 at streams 1/2/3 with `NKV=2303`, Q8_0 K/V, gathered stride, causal triangular masks, canary, CPU oracle, masked-tail mutation, and q0/q1 triangle mutation. Recorded CPU oracle maximum absolute errors were 0.0018904, 0.0019006, 0.0014331, and 0.0015864 respectively. All printed `PASS` for those checks.

The same process aborted at T=10 with `fattn-mma-f16.cuh:1858: NO_DEVICE_CODE` (`ncols1*ncols2 < 32` on Volta). The attempted `NKV=2303` disabled the stock GQA grouped path, exposing a stock fallback template hole. This is not an Onecat failure, and no Onecat-enabled process had started. The abort released its CUDA process; a subsequent `nvidia-smi` query showed no compute PID on GPU2.

To keep verifying the Onecat admission/fallback contract without changing core code, the local fixture now keeps partial `NKV=2303` for T=1/T=3 and uses aligned `NKV=2304` for T=6/9/10/12. This retains the odd partial partition on the actually supported Onecat T=1 path while avoiding the observed stock fallback hole for larger verify shapes. It includes T=3/S=2 and T=3/S=3 for six/nine total query rows, a T=1 max-bias case, CPU softmax oracle checks, exact masked-tail invariance, q0/q1 triangular-mask mutation, and T6→T9→T6 A/B/A.

## Current interruption and continuation

After editing the follow-up fixture and runner locally, `ssh ac922` began timing out to Tailscale IP `100.66.42.47`; `ssh ac9226` reported no IPv6 route. `tailscale status` showed the peer `ac922` as `Online=false`, and `tailscale ping 100.66.42.47` timed out. Therefore the revised fixture has **not** been re-synced, rebuilt, or run. No follow-up GPU process was started.

When SSH is restored, sync `fixtures/onecat-variable-lanes-20261015/` and this note to the AC922 mirror, build against V34, then run `run-matrix.sh`. The runner reuses the completed initial stock T1/T3 files when present and runs only the stock remainder; if those files are absent, it runs the full stock suite. It then runs the Onecat-enabled full suite and strict comparison. The final gate requires the real V34 Onecat marker for T1/no-bias, explicit stock-fallback admission for T>1/key-bias, all nine stock/Onecat output files, strict abs error 0.004, CPU oracle, causal-mask mutation, bias effect, canary, and A/B/A replay. GPU2 occupancy is checked before and after.
