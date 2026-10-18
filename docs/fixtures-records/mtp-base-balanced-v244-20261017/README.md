# V244 V241 base-balanced two-family candidate

This is a local-only combined harness candidate. It reuses the V238 D4 runner
and V233 sequential context-family lifecycle. One model is loaded once, then
the target and MTP contexts are created, drained, and destroyed for each
family while the model remains resident:

1. `single64k-d4-ub512`: one lane, 66,048-token context, first the 249-token
   Paris semantic canary, then 64K input plus 8 output tokens, target
   `n_batch=32768` / `n_ubatch=512`, two 32K prefill chunks.
2. `fourbank16k-d4-ub256`: four independent one-lane banks, each with a
   17,152-token context; four disjoint 16K inputs and 512 output tokens, target
   `n_batch=4096` / `n_ubatch=256`, with the frozen four-case semantic gate
   before the long case.

Both families use D4 on the generic, non-unrolled MTP GPU chain and PAD6. The
private context policy checks the actual per-sequence capacity and the five
target verify rows per lane. The model split is layer mode with the exact
`[8, 8, 8, 8, 8, 10]` ratio. The prepared environment selects
`LLAMA_MTP_BASE_DEVICE=0`, a 12 GiB PLE prefix, small-J, the cached device
properties path, a 16 MiB pinned-input floor under the 1 GiB cap, binding keys,
and forced CUDA Graph capture off. The draft UBatch request and the actual
draft context must both be 256 for both families. Target UBatch remains 512
for the single64K family and 256 for the four-bank family.

The native gate dispatches only the two exact V244 scenario identities. It
uses the existing V50 semantic implementation for the Paris canary and the
frozen V146 implementation for the four-bank canary. It accepts the exact
native-only prefix before performance, while any complete two-record result
must satisfy the full family shape. Unknown/mutated scenario labels fail
closed.

## Status and limits

V241 is frozen as build `20261017T102503.118440Z` with report SHA-256
`3d84a8da88f4a2dee04940c6d083842280aa49a98397c4e31e7d13e8a1966129`.
Root reported the V241 source-extraction, ASAN/UBSAN, and source-hash gates
passing. The V243 tiny UVA host-feedback prerequisite is still pending.

## R1 failure and R2 correction

R1 loaded the model and created contexts, then exited with code 1 at the
startup assertion because `LLAMA_MTP_N_UBATCH=512` requested a draft UBatch
that the actual context clamps to 256. It stopped before any inference, native
canary, or performance case. The R1 source, profile, host-link script, launch
manifest, runtime log, and guard status are preserved with an R1 receipt.

R2 changes only the draft UBatch request to 256 for both families, updates the
assertions/profile metadata, and links into the unique
`build-root-v244-r2/runner` path. The assertion that the actual draft UBatch is
256 remains in place. The R2 source/profile/host-link are ready for root review;
R2 has not been built or run. V243 is still a launch prerequisite.

The future summary reports single64K prefill from actual input-token count and
timing, plus four-flow shared and inclusive decode rates. No V244 performance
result exists yet.
