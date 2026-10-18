# V47 QSA pooled-score audit

## Decision

For complete, dense blocks, the per-sequence pooled scorer is not a different algorithm from
the V47 ordinary QSA selector. The ordinary `build_qsa_top_k()` already gathers raw per-cell
indexer K rows by `blk_cells`, averages them in member order, applies RMSNorm and RoPE, computes
per-head ReLU dot products, and broadcasts each block score back to its cells before cell top-k.
`build_qsa_mblk_update()` performs the same raw-K mean followed by RMSNorm/RoPE for the same
`(sequence, position/ratio)` members. Under the fast-path dense/member guards, these are the same
full-block score formula, modulo floating-point operation ordering.

This corrects the broader concern in the earlier `v47-phase-ledger-rebase-report.md`: comparing
`RMS(mean(raw K))` to a hypothetical mean of independently normalized/scored cells is the wrong
comparison, because V47's ordinary route does the former too.

## CPU minimum example

For `K0=(1,0)`, `K1=(0,3)`, query `(1,1)`, ratio 2, RMS epsilon `1e-6`, and identity RoPE at
position 0:

- ordinary block formula: `dot(RMS(q), RMS((K0+K1)/2)) = 1.788852772`;
- multi full-block formula: `1.788852772`;
- a different per-cell-normalize-first interpretation: mean of the two per-cell scores is
  `1.414212070`.

The CPU fixture asserts equality of the first two and rejects the third. It also leaves RoPE
identity only for this minimal arithmetic example; the source call order places RoPE after the
same pooled RMSNorm on both complete-block routes.

## Remaining tail distinction

V47 standard `set_input_qsa()` leaves an incomplete group in the spare/dead block: its member IDs
are zero-filled and block position remains zero; `blk_bias` gives that spare block `+1e9`. The
multi-row update instead pools the present tail members plus dummy cell 0 and uses the actual first
tail member's position. Thus the tail score itself can differ. Both paths give the tail the same
`+1e9` selection priority, and the standard selector budgets up to `ratio-1` tail cells; this is a
tail-specific edge, not a full-block pooling-algorithm difference. The fixture does not claim
unbounded-score mathematical equivalence or full-model logits parity for this tail behavior.

The device-bias candidate in this directory is independent: it reproduces the V47 host `blk_bias`
values on the ordinary QSA route and does not require the pooled-cache path.
