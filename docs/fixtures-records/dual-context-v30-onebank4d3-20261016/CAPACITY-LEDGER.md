# V30 static capacity ledger

This is a preflight estimate only. V30 has not loaded the model or allocated
target/draft contexts. The estimate reuses the existing V28 16K capacity review
and saved V23 buffer observations; it does not scan or download the GGUF.

## Exact requested capacity

- One target bank and one MTP draft context.
- `kv_unified=0`, `n_ctx=65536` cells per context, `n_seq_max=4`.
- Four non-unified streams at 16,384 cells each. Aggregate capacity is 65,536
  cells; the per-request limit remains 16,384.
- `n_batch=8192`, target ubatch 1024, draft ubatch 512.
- D3 with four lanes plans 16 target verification rows. The fixed-prefix MTP
  input has 24 rows (six per lane).

## Tracked HBM estimate per card

V28's existing 16K ledger estimated two 2-lane target banks plus two draft
contexts at 3,802.86 MiB/card before omitted graph/allocator costs. For V30,
the total four-stream KV capacity is unchanged, while one target context is
removed. The ledger below uses the saved per-bank values as a proxy and keeps
the aggregate four-sequence recurrent and draft costs:

| Buffer class | V30 proxy per card | Basis |
| --- | ---: | --- |
| Target QSA + indexer KV | 374 MiB | Same 4 x 16K cells as the two V28 banks combined |
| Target recurrent state | 762 MiB | Existing conservative two-bank/four-sequence subtotal |
| Target compute workspace | 908.15 MiB | One saved V23 bank workspace; four-lane graph growth is unmeasured |
| MTP draft compute workspace | 850.56 MiB | Retain the combined two-bank D3 workspace as a conservative proxy for one D3x4 graph |
| MTP draft KV + indexer KV | 153 MiB | Saved V23 136 MiB + 17 MiB buffer sizes at the same 65,536-cell aggregate |
| **Tracked estimate** | **3,047.71 MiB (2.98 GiB)** | Before graph, allocator, and other untracked allocations |

The V28 static weight-plus-PLE estimate left 4.80–4.92 GiB/card before runtime
buffers. Subtracting the V30 tracked estimate leaves about 1.82–1.94 GiB/card
for untracked costs. This is not a fit guarantee: the single-bank four-lane
target graph and D3x4 draft graph have not been allocated. Preserve the existing
V69 per-card reserve and context-init guards; do not infer usable headroom from
the static estimate.

## Evidence boundary

V28 `CAPACITY-REVIEW.md` and `V23-runtime-buffer-evidence.txt` are the read-only
inputs. The totals above are derived estimates for capacity planning; they are
not V30 runtime measurements. No model payload was opened, no full-model
process was started, and no GPU was used for this ledger.
