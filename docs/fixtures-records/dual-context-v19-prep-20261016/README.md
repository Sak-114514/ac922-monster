# V19 P4 limited-context KV-memory candidate

V19 is an independent harness derived from V18. It addresses the V18 free-memory-floor stop by reducing only the non-unified KV backing capacity; it keeps the externally selected MTP ubatch unchanged (the first comparison is expected to use 512). No core or V18 files are changed.

## Accepted layouts

`run-v19.sh` allows exactly these two pairs and the runner rechecks them before backend initialization:

| Variant | `ROOT_KV_UNIFIED` | `ROOT_BANK_CTX` | `n_seq_max` | Actual context per sequence | KV streams per bank | Capacity interpretation |
|---|---:|---:|---:|---:|---:|---|
| `unified_132096` | 1 | 132096 | 2 | 132096 | 1 | One shared 132096-cell pool per bank |
| `per_seq_64k` | 0 | 131072 | 2 | 65536 | 2 | Two independent 65536-cell streams per bank |

The four per-sequence streams across two banks total 262144 cells. That is aggregate capacity, not a per-request context guarantee. The prior `per_seq_128k` layout is intentionally not admitted by this limited-memory candidate harness.

Select the smaller per-sequence candidate with:

```bash
ROOT_KV_UNIFIED=0 ROOT_BANK_CTX=131072 ./run-v19.sh RUNNER MODEL OUTPUT
```

The target and MTP draft context are both checked with public `llama_n_ctx()`, `llama_n_ctx_seq()`, and `llama_n_seq_max()` queries; stream count is derived from actual total/per-sequence context. The semantic suite, same-prompt exact-token gates, and four 8K request test remain unchanged. No longer-context workload was added.

V19 retains separate target-decode-plus-sync, MTP-prefill, diagnostic dump/scan, per-lane total, and wall-time measurements, along with cold and same-process warm-graph uncached 8K probes. The warm probe checks an exact 32-token prefix against the serial 8K reference.

## Build and run boundary

The host-only compile script targets the frozen V56 runtime. It only links the runner; it must not start a model. Root supervises any model run with the existing GPU free-memory floor. This preparation has not run a model or used the GPU.
