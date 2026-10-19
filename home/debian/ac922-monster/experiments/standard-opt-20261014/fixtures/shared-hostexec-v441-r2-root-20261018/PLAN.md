# V441 R2 shared-helper readiness fix

This folder preserves the R1 helper and direct fixture snapshots as the exact
R1 build inputs. The R2 candidate is a separate source copy; it changes only
the shared V2 readiness predicate and its regression fixture.

## Readiness contract

- Shared V2 readiness is a CPU metadata probe. It validates non-null owner,
  key, graph, and the owner/device context identity without calling CUDA or
  checking the caller thread's current device/last-error state.
- A prepared per-key binding is ready when its graph matches, its recorded
  stream has a live generation-matched shared slot, that slot has an exec, and
  neither binding nor slot has an unfenced launch. The slot may currently be
  bound to another key because every real V2 launch rebinds from its supplied
  captured graph.
- The separate per-key topology fallback retains R1's readiness check.
- Actual launches and V3 preparation retain R1's current-device and prior
  asynchronous-error validation.

## Regression coverage

The eight-key same-stream ring queries key zero after key seven has occupied
the slot. The fixture sets key zero's output to a sentinel, checks readiness
while GPU 5 is current for the GPU 0 owner, confirms the current device remains
GPU 5, then switches back to GPU 0 and launches key zero. The exact output and
final slot summary must show that the launch rebound key zero's graph.

This fixture is staged only. The author did not compile it, run its CPU test, or
use a GPU. Root owns build and runtime verification.
