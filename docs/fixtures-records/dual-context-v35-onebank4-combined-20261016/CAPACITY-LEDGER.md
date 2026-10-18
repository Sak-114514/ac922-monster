# V35 short-context capacity estimate

This is a preflight estimate only. V35 has not loaded model weights or created
target/draft contexts. The estimate reuses the V28 16K capacity review and saved
V23 buffer observations; it does not inspect model payloads or include any new
V35 GPU allocation evidence.

## Requested capacity

- One target bank and one MTP draft context.
- `kv_unified=0`, `n_ctx=65536` cells per context, `n_seq_max=4`.
- Four non-unified streams at 16,384 cells each. Aggregate capacity is 65,536
  cells; the per-request limit remains 16,384.
- `n_batch=8192`, target ubatch 1,024, draft ubatch 512.
- D3 with four lanes permits 16 target verification rows. The fixed-prefix MTP
  input has 24 rows (six per lane).

## Tracked HBM estimate per card

The older V28 ledger estimated two 2-lane target banks plus two draft contexts
at 3,802.86 MiB/card before omitted graph and allocator costs. V35 uses the
same four-stream KV capacity but removes one target context bank. The estimate
below keeps conservative aggregate recurrent and draft costs and uses saved
V23 values as proxies:

| Buffer class | V35 proxy per card | Basis |
| --- | ---: | --- |
| Target QSA + indexer KV | 374 MiB | Same four 16K-cell streams as the previous two-bank layout |
| Target recurrent state | 762 MiB | Existing conservative two-bank/four-sequence subtotal |
| Target compute workspace | 908.15 MiB | One saved V23 bank workspace; four-lane graph growth is unmeasured |
| MTP draft compute workspace | 850.56 MiB | Retained two-bank D3 workspace proxy for one D3x4 graph |
| MTP draft KV + indexer KV | 153 MiB | Saved V23 buffers at the same 65,536-cell aggregate |
| **Tracked estimate** | **3,047.71 MiB (2.98 GiB)** | Excludes graph, allocator, and other untracked costs |

The V28 static weight-plus-PLE estimate left 4.80–4.92 GiB/card before runtime
buffers. Subtracting the V35 tracked estimate leaves about 1.82–1.94 GiB/card
for untracked costs. This is not a fit guarantee: the one-bank/four-lane target
graph and D3x4 draft graph have not been allocated. Preserve the existing
per-card reserve and context-init guards; do not infer safe headroom from this
static estimate.

## Evidence boundary

The V28 capacity review and V23 runtime-buffer observations are read-only
inputs. These totals are derived estimates, not V35 measurements. No model
payload was opened, no full-model process was started, and no GPU was used for
this ledger.
