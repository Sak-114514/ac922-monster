# Cold P2P AllReduce validation

This is a bounded fixture plan for the existing `GGML_META_P2P_COLD=1` route. It does not load the model or edit engine/backend source. `run-cold-p2p-matrix.py` derives a temporary six-rank variant from the existing `fixtures/p4-v3-local-repro/two-meta-concurrent.cpp`; generated C++ lives in a temporary directory and is removed on exit.

## Reused code and current gap

- `two-meta-concurrent.cpp` already has two independent Meta contexts, four dependent partial reductions, a concurrent graph phase, and replay checks. The checked-in fixture is intentionally 3 ranks and 4096 elements, and its first shape is primed serially with cold P2P disabled. It therefore does not cover six ranks or concurrent *first cold* calls.
- The derived fixture sets `RANKS=6`, keeps `BANKS=2`, reads one `COLD_P2P_N` shape per fresh process, and makes the first graph compute concurrent across the two Meta contexts. This tests two independent six-device P2P state objects sharing the six physical GPUs.
- It changes only the temporary fixture’s FP32 inputs to an order-sensitive cancellation pattern and corrects the old three-rank `square_ab` / `square_c` oracle to sum over six ranks. It dumps every stage/rank output before reporting any oracle mismatch, so NCCL output can still be compared with the cold-P2P output if the NCCL tree uses different addition order.

## Matrix and gates

For each `N = 1, 4, 12, 3072`, run a fresh NCCL-control process and a fresh `--cold-p2p` process. Each process performs:

1. One first-use graph compute, launched concurrently on both Meta contexts. The control must report zero P2P kernel host launches at this point; the cold arm must report `4 reductions × 6 ranks × 2 contexts = 48` launches.
2. Up to six same-shape compute rounds. Keep every round's output dump and host P2P launch delta. The runner marks the actual capture/replay round from CUDA graph path mode lines in that round's log segment; it does not infer replay from the call number. The small graph may need several warmups before replay appears.

The runner checks callback registration, the exact first-call route, four graph-boundary reductions, no cold declines, no in-graph rejection, no P2P stuck diagnostics, and an actual replay marker within six rounds. It also requires `nccl_wire_f32` host-path logging in the NCCL control and no such fallback-path entry in the cold-P2P arm. Those logger entries prove host collective-path selection only; they are not GPU completion or replay-kernel counts. It dumps output words for every phase and requires:

- every cold-P2P output (first call and each warmup/capture/replay round) matches the exact CPU FP32 rank-order oracle at every stage, rank, and element;
- cold-P2P outputs stay bit identical across the first call and all later rounds (the ABA gate);
- NCCL-control differences are reported per phase and are not allowed to weaken the cold oracle or ABA gate. NCCL may use a different valid FP32 addition order.

The CPU order-sensitivity check uses a six-rank cancellation vector. For the first partial reduction, rank-order FP32 gives `0.5` (`0x3f000000`) while one adjacent-pair tree gives `0.0` (`0x00000000`). This proves that the test values can expose changed addition order; it does **not** predict the installed NCCL algorithm’s order. Actual NCCL output is compared directly in the bounded GPU fixture.

All four sizes are at most 3072 FP32 elements (12 KiB), below the P2P relay6 threshold of 204800 bytes and the segmented threshold of 4 MiB. This matrix exercises the one-shot six-rank kernel and its two-half epoch reuse. It does not validate relay6, segmented, compact-staging, or stage-capacity growth after a larger reservation.

## Peer ordering and cross-thread capture

The P2P kernel uses system-scope fences and peer flag words, not CUDA event objects, to publish data and establish peer arrival. The cold helper launches all rank kernels before synchronizing each rank stream. The first-call-concurrent phase tests this route from two host threads; the four dependent collectives and repeated phases exercise epoch increments and double-buffer-half reuse without any shared P2P stage/flag/epoch arrays between Meta contexts.

Static source reading found no cold-route CUDA Graph reentrancy prohibition. The Meta cold callback is called from the unmerged fallback loop after child graph dispatch, and its direct helper launches and synchronizes the P2P kernels. CUDA graph capture uses `cudaStreamCaptureModeRelaxed` in `ggml-cuda.cu`; the global capture counter only makes backend-context destruction wait so a cuBLAS handle is not destroyed during another thread’s capture. The new concurrent first-cold run is the direct fixture check for this conclusion. It is not run while any model/server is using the GPUs.

## Commands and safety

On AC922, with no GPU compute process present and `CUDA_VISIBLE_DEVICES` explicitly set to the verified `[island A × 3, island B × 3]` UUID order:

```sh
python3 fixtures/cold-p2p-validation-20261015/run-cold-p2p-matrix.py --cpu-oracle
python3 fixtures/cold-p2p-validation-20261015/run-cold-p2p-matrix.py --compile-only
python3 fixtures/cold-p2p-validation-20261015/run-cold-p2p-matrix.py --run
```

`--run` refuses to start if `nvidia-smi` reports any compute PID or if six explicit visible-device IDs are not supplied. It builds against the existing private include tree and `runtime-perf-v21`; it does not build the engine. Use `--sizes 1` to isolate a shape, or omit `--sizes` to run the full matrix. It stores logs, FP32 output snapshots, actual capture/replay rounds, and `comparison.json` in a timestamped directory beneath this fixture.

## First bounded run evidence

The root's 20261015T114701Z NCCL-control n=1 log is preserved under `results/20261015T114701Z/`. It showed the external NCCL path (`nccl_wire_f32 requested=8 selected=8`) and an FP32 order difference: the NCCL result for stage 0 was `0.0`, while the serial rank-order oracle was `0.5`. This difference is reported as an NCCL comparison result; it does not relax the cold-P2P rank-order oracle.

That run stopped at the replay assertion because the third graph compute still logged capture. The runner now permits six same-shape rounds, stores each round output, and requires an actual `mode=replay` graph-path marker before the matrix qualifies.

The failed first attempt is preserved both remotely and locally at `results/20261015T114701Z/`. The corrected bounded matrix completed at `results/20261015T115339Z/` and is backed up locally with all logs, comparison JSON, and FP32 snapshots. For every size and both arms, the actual CUDA graph log showed capture in warmup round 2 and replay in rounds 3 and 4. Each cold-P2P first call logged exactly 48 P2P rank launches, zero NCCL fallback entries, zero cold declines, and no stuck kernel reports. All cold outputs matched the rank-order oracle bit for bit and remained unchanged through the first call, capture, and replay rounds. The NCCL control produced the same outputs after merge/capture/replay, but its first unmerged pass differed: stage 0 was 0.0 on bank 0 and 0.5 on bank 1, opposite the cold rank-order results; this is recorded per shape in comparison.json and is not counted as a cold-P2P failure. After the matrix, nvidia-smi showed no compute PID, 0% GPU utilization, and 2–3 MiB per card. The fixture did not load a model.
