# V38 status

## Frozen candidate

- Runtime: V89, build ID `20261016T101245.741848Z`, report SHA-256 `4f1d5799decf78e8139dc10a6d7296ee6af5b8f81dddbdf1deba36f10b0afc47`.
- Topology and shape: flat Meta6/EP6; one bank × four lanes; D3; 16K per sequence; target ubatch 1024; MTP ubatch 512; Q8_0 KV; standard indexer.
- Candidate flags: chunkless Meta merge; structural bound 16384; packed dense decode; force graph capture; CPU async-copy limit 32 MiB with the unchanged 64 MiB pinned pool; PLE prefix 12 GiB over six GPU NUMA nodes.
- Reserves remain 4 GiB HBM per GPU and 3 GiB DDR. The PLE loader's actual placement/capacity result is authoritative; this candidate never relaxes reserves.
- Quality gates are unchanged from V37, with a new exact per-flow V37 token-array comparison. The V27 comparison remains visible.

## Micro gates

Root authorized the one-load V38 model run after confirming the bounded MTP, CPU-copy, packed-path, graph-force, and PLE12 policy gates. Their evidence is recorded in `v38-micro-gates.json`. The full model run still must prove actual dispatch and preserve all semantic/F32/token gates.

## Launch history

The first supervisor attempt used tag `root-p4-v38-flatmeta6-ep6-d3-16k-20261016t102240z` and exited before model initialization. The journal showed `write_run_manifest_v38.py` importing a wrapper that did not expose `ENV_KEYS`; GPU memory stayed idle and no model PID was created. The wrapper is being corrected to reuse the V35 manifest helper directly. Preserve this failed harness attempt; use a new tag for the authorized model run.

## Pending result

No V38 model correctness or performance result is recorded yet. Do not infer a performance win from successful micro fixtures, a linked harness, or requested environment variables. The run must finish its native semantic, F32, generated-token, clean single-8K/P4, V37 token-array, V27 token-array, runtime-activation, capacity, and safety checks.
