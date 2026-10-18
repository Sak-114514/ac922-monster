# P4 cold graph / external-view lock order

Actual failed run: `fixtures/dual-context-v10-prep-20261016/results/p4-d3-20261015T192743Z`.
No throughput result was produced. Two early semantic reference records existed, but a runner serialization bug omitted lanes finishing before the shared window; those records do not establish the semantic gate.

Before debugger attachment, root read `/proc/33697/task/*/syscall` and read-only `/proc/33697/mem`:

| Waiting thread | Futex address | Mutex owner read from pthread state |
| --- | --- | --- |
| 34527 | `0x7fff81471550` | 34588 (recursive count 2) |
| 34588 | `0x7ffd77486b10` | 34527 (recursive count 1) |

The first address maps to frozen V47 `libggml-base.so.0`, ELF offset `0x151550`; `nm -anC` identifies `(anonymous namespace)::ggml_meta_replay_guard_mutex()::mutex` at exactly that address. This establishes a CPU mutex wait cycle. The second mutex is in an anonymous allocation; no successful native backtrace identified its exact C++ instance. Static inspection finds four external-view owner-state lock entry points that can enter the replay/registry guard recursively, which are now covered by a single lock-order scope.

The subsequent CUDA-GDB attach enumerated threads, then timed out at 30 seconds. Its log says `cuda-gdb has received a SIGSEGV` while quitting. The target also exited 139 during that window. This does not identify an inference kernel fault, nor prove the exact mechanism of the target's exit. No further debugger attach is required to establish the earlier CPU wait cycle.

Fix: acquire the global replay guard before each owner-state recursive mutex; release the owner mutex first. Context ownership, graph leases and GPU stage-tail FIFO remain intact; no parent-wide execution lock or GPU synchronization was added.

Regression: `run.py` builds an old-order variant that deterministically deadlocks under a two-second subprocess timeout, then the actual new private helper, which completes the same two-thread cold-graph/recursive-registry scenario. It also checks recursive depth, balanced release, and inactive scope. Mac and POWER9 runs pass. Real Meta/CUDA owner fixtures and full-model validation are separate gates.
