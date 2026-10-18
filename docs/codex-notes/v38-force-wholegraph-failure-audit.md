# V38 force-wholegraph failure audit

Date: 2026-10-16 UTC  
Status: the bounded fixture completed; it did not reproduce the V38 SIGSEGV/Xid signature.

## Original V38 evidence and limits

The handoff identifies the failing unit as `root-p4-v38-flatmeta6-ep6-d3-16k-20261016t102725z`, with SIGSEGV 139 and GPU 4 Xid 43. The retained log interval was `3.12.858.177` to `3.21.196` (about eight seconds). No core backtrace was available. A per-bucket line lacking rank 0 is not evidence that rank 0 failed to submit.

The V89 hypothesis came from the frozen source evidence supplied with the task: with `GGML_META_CHUNK_NODES=0`, a merged graph has one chunk, so the `n_chunks > 1 && !ready_at_entry` readiness scan and lockstep path are skipped. In CUDA graph dispatch, `GGML_CUDA_GRAPHS_FORCE=1` can capture and launch a compatible cold graph immediately. This fixture checks that cold graph/update ordering on six V100s; it does not model the full 49-layer unit or load the IQ3XXS/Q8MTP model.

The fixture sets `GGML_CUDA_GRAPH_STRUCTURAL=0` to exercise raw first-node key behavior. It bypasses structural signature building, so this does **not** show that `external_cap` occurred or reproduce the signature builder's rejection decision.

## Fixture and provenance

See [the fixture README](../fixtures/flatmeta6-cold-flat-ar-20261016/README.md). It reuses the existing Meta fixture allocation pattern and creates one F32 `MUL_MAT`, with W and X split on K axis 0 across six ranks. The partial is 24 F32 values (96 bytes), followed by one AllReduce and a small tail.

A and B share the original first tensor. A ends with a view placeholder and B uses an ADD in that slot, keeping both graph node counts equal. This also keeps the remapped merged child graph's first-node address equal across A/B. The measured pair calls `ggml_backend_graph_compute_async(A)` then `ggml_backend_graph_compute_async(B)`; the `sync_between=0` arm does not synchronize between them. The final sync, CPU oracle, and host read canary run afterward. The `sync_between=1` arm inserts exactly one synchronization between A and B.

`--scale-nodes 4096` adds 4,096 identity F32 SCALE nodes before the A view/B ADD tail. The fixture records each child graph key, AllReduce node size, CUDA graph path, update result, SCALE node trace, output comparison, and pthread stack size.

The V89 host build used the requested CPU placement, `numactl --cpunodebind=0 --membind=0,8`; it did not use a model or start CUDA. The same PPC64LE fixture binary was used with the V90 closure by putting V90 first in `LD_LIBRARY_PATH`.

| Artifact | SHA-256 |
|---|---|
| Fixture source | `6c3e06c0ea424b97f614cb200cb423985af964325c670c681f5d2efc37a04aec` |
| Fixture binary | `814aba2eba1e4a155972514323609d8be589e308be02eb91eebd73ad0f8068f7` |
| V89 `libggml-base.so.0` | `7a95bd46168f701861d006a6cf133ea2032b3cfa7d82276624245e6f78129293` |
| V90 `libggml-base.so.0` | `14cfbe0a90af969394a4dc0d58bbc1f7097e40862c2fc29770a14af6d290e118` |
| V89 and V90 `libggml-cuda.so.0` | `1efd377371ba8ac31b4f8a84ab32a37a45f17632f8dddcc860f459a1e0404477` |

The V89/V90 `libggml.so.0` and `libggml-cpu.so.0` hashes also matched. `ldd` confirmed all `libggml*` libraries resolved from the selected runtime directory on each run.

## V89 results

All four valid small arms exited 0 in three seconds each. Every arm reported one 96-byte AllReduce on each of the six ranks for measured A and B, the same measured merged child key for A/B on all six ranks, exact outputs (`max_abs=0`), and an intact host read canary.

| Arm | `GGML_CUDA_GRAPHS_FORCE` | Sync between A/B | `update_result=2` records | Result log |
|---|---:|---:|---:|---|
| A | 1 | 0 | 12 | `results/v89-force1-sync0-20261016T110436Z.log` |
| B | 1 | 1 | 12 | `results/v89-force1-sync1-20261016T110441Z.log` |
| C | 0 | 0 | 0 | `results/v89-force0-sync0-20261016T110446Z.log` |
| D | 0 | 1 | 0 | `results/v89-force0-sync1-20261016T110450Z.log` |

In both force-on small arms, the 12 `update_result=2` records cover six ranks during the B shape preparation and six ranks during measured B. The measured B update calls `cudaGraphExecUpdate`, receives result 2, and takes the runtime's `action=reinstantiate` fallback; both outputs still pass. The force path summary reports six selected captures per device. `sync_between=1` also passes and still reports the same update result, so result 2 alone does not reproduce the V38 failure.

The small graph did not reproduce SIGSEGV, timeout, or incorrect output. A larger force-on/no-sync arm was therefore run within the same 120-second envelope. The 4,096-node arm exited 0 in six seconds. Its A and B child graphs each had 4,100 nodes and the same first-node key on every rank. Each measured graph had one 96-byte AllReduce on every rank. It recorded 98,304 `begin SCALE` and 98,304 `end SCALE` events across the six devices (4,096 nodes × four graph evaluations per device), so the large cgraph did reach CUDA graph evaluation. Measured A/B host submission times were 607,635 and 555,344 microseconds. Both outputs were exact with intact canaries. Log: `results/v89-force1-sync0-20261016T110524Z.log`.

The process stack soft limit was 8,192 KiB. `pthread_getattr_np` reported 8,323,072 bytes for the main thread and 8,388,608 bytes for each of the five Meta worker threads. The 4,096-node run passed, so no 64 MiB stack retry was made.

## V90 cold-drain control

The V90 run reused the same fixture binary and enabled `GGML_META_SINGLE_CHUNK_COLD_DRAIN=1`, with force on, no A/B sync, and 4,096 SCALE nodes. It exited 0 in six seconds. The log contains two `meta-cold-graph-drain` markers with `ranks=6 chunks=1 ready_at_entry=0 action=drain_before_capture_and_lockstep`, for measured A and B. The measured child keys match across A/B on all ranks, each graph has one 96-byte AllReduce per rank, `update_result=2` is observed on all six ranks during measured B, and both CPU oracle/canary checks pass. Log: `results/v89-force1-sync0-20261016T110835Z.log`.

## Safety and conclusion

No model was loaded. Each runner arm enforced a 20-second process timeout, temperatures below 58 C, at most 512 MiB before launch and 1 GiB during execution. After V89 and V90, all six GPU compute PID lists were empty; the V90 postflight read was 38–39 C and 2/2/2/8/7/6 MiB used. The final kernel log query showed no Xid after the most recent pre-test Xid entry at 18:31:53 local time (10:31:53 UTC).

The fixture confirms that a same-raw-key topology change can produce `cudaGraphExecUpdate` result 2 under force capture, and that the fallback reinstantiate completes correctly in these small and 4,096-node F32 cases. The V90 cold-drain marker confirms the intended one-chunk guard ran. Neither result establishes the cause of the original V38 SIGSEGV/Xid 43; full-model graph complexity, external binding pressure, stream timing, and worker/driver stack conditions remain outside this fixture. Preserve the original V38 artifacts and treat any causal claim as unproven.

The first host draft's four logs from `11:00:50Z` to `11:01:04Z` are retained but excluded: that draft used unequal A/B graph sizes, so the measured merged child keys differed. The corrected source keeps A/B node counts equal and asserts the six measured keys are identical.

## V91/V90 graph-depth follow-up

After the V39 gate, two additional single-process arms used the public-API fixture linked to the V91 closure with an 8 MiB thread stack, force on, `GGML_META_SINGLE_CHUNK_COLD_DRAIN=1`, raw-key mode, and no A/B sync. Both were capped at a 9-second TERM timeout plus 1-second KILL grace, under a shared 60-second envelope.

- V91 at 8,192 SCALE nodes exited 0 in 4 seconds. Child graphs had 8,196 nodes (8,192 SCALE nodes) per rank, both cold-drain markers appeared, the six A/B child keys matched, every measured graph had a 96-byte AllReduce on all six ranks, and both oracle/canary checks passed. `GGML_CUDA_GRAPHS_FORCE=1` was **not selected**: V91 logged `selected=0 reason=structural_binding_signature_unavailable` for the raw structural-disabled graph and ran the deep graph in eager mode. Log: `results/v91-depth-force1-sync0-20261016T113629Z.log`.
- V91 at 10,000 SCALE nodes exited 0 in 6 seconds with the same raw-mode behavior: 10,004 child nodes per rank, cold-drain markers, same keys, 96-byte reductions, exact oracle, and intact canaries. Log: `results/v91-depth-force1-sync0-20261016T113717Z.log`.
- The V90 whole closure was then used with the same 10,000-node binary to test the earlier raw-key force behavior. It reached the cold merged A graph and timed out at the 10-second wall cap with status 124. The log contains 10,002-node CUDA capture paths during preparation and 10,004-node measured A child graphs on all six ranks. Measured A did not return from submit; B was not dispatched, so no oracle/canary result exists. Under `GGML_CUDA_P2P_AR_DEBUG=1`, the log records 512 `p2p allreduce stuck` lines for the 96-byte one-shot AllReduce: ranks 0 and 3 waited at call 3 while peer flags remained at epoch 2. The cold-drain marker appeared once. No SIGSEGV or post-run Xid was observed; postflight returned all GPUs to idle. Log: `results/v91-depth-force1-sync0-20261016T113821Z.log`.

The V91 arms test deep eager graph submission and do not test deep CUDA graph capture. The V90 timeout shows that a cold-drain marker and per-rank host submission still do not guarantee all six peers have advanced to the same collective before a 10,000-node cold path reaches the AllReduce. This is a bounded synchronization observation, not proof of the V38 crash cause or a stack-overflow diagnosis.
