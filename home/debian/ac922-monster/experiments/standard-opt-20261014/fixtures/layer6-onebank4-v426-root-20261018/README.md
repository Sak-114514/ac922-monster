# V426 one-bank four-stream preparation

This candidate is intended to test one plain-CUDA six-GPU LAYER model with one target context and one MTP draft context, each holding four independent non-unified 17,152-cell sequences (68,608 cells total per context). It keeps generic MTP D3, PAD6, target UBatch 256, draft UBatch request 512, the V306 18-GiB hybrid PLE placement, and the existing V409 target graph-tail/runtime path. CUDA linear decode chunking remains at the inherited value 4 for the initial state canary; the environment request is not proof that a fused T16 kernel was selected.

## First stage: target-state canary only

The runner loads the model once, builds four 1,024-token prefixes from the pinned real corpus, records serial target-context F32 logits and unmasked `h_nextn`, then compares a four-sequence target context over prefix, teacher-forced rows, and rollback accept vectors `[0,1,2,3]` and `[3,2,1,0]`. It destroys those contexts, writes `canary.json` and the referenced F32 snapshots, then waits at most ten minutes for a fresh `root-control.json` containing exactly `{"continue":true}`. The control must be a regular file owned by the runner account; symlinks and stale files are rejected. `ROOT_V426_CANARY_ONLY=1` exits after this gate without allocating benchmark target or draft contexts.

The root controller should inspect the JSON and F32 snapshot index while the process waits, then explicitly release the canary-only process with:

```bash
printf '{"continue":true}\n' > /path/to/this-run/root-control.json
```

Use the exact output directory printed by the guarded launcher. The shell writing the control must run as the same Unix account as the runner.

The canary is not yet built or run. The saved V410 four-bank native output and the older synthetic native-only adapter are not V426 onebank evidence. This fixture does not claim that four-lane MTP draft state, performance, or the full benchmark path passed. Root must review and host-link before any model run; launch remains disabled in `v426-root-profile.json`.

## Remaining integration work

- Reconcile the copied V418/V410 performance runner with one bank of four lanes. Several inherited record fields and checkers still describe four one-lane banks.
- Complete the scenario/profile/host-link source pins after the runner is frozen.
- Run the canary first. Only after its saved output is reviewed should the root controller write the continue file and proceed to native/performance gates.

## Offline check

```bash
python3 verify-v426-canary-static.py
```

This checks source ordering and gate contracts only. It does not build or run llama.cpp, load model weights, or touch a GPU.
