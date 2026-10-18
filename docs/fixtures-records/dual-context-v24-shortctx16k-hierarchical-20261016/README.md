# Hierarchical EP6 16K-per-request fallback preparation

Status: **PPC64LE host link complete; runner not executed**. No hierarchical full model or GPU was launched. V61 exposed a nested Meta source-binding bug; the runner is now linked against frozen V62 after its source-binding fix. Root's nested-context/dense qualification and a separate GPU grant are still required before model launch. The older V61 closure note is historical evidence only and must not be used to launch.

## Capacity rationale

See `CAPACITY-REVIEW.md` and `V23-runtime-buffer-evidence.txt`. Using the V23 observed per-island buffers, a hierarchical 2-bank × 64K-per-sequence layout accounts for about 4.81 GiB/card in known target and MTP compute/KV/RS categories before omitted graph and draft-state buffers. Nominal room after the prior V61 metadata-only weight estimate and 8 GiB six-node PLE prefix is only about 4.80–4.92 GiB/card. That is not a safe fit assumption; V62's actual per-card loader and graph gates remain authoritative.

The fallback caps every sequence at **16,384 tokens**. Each scenario uses an 8,192-token real prompt plus 256 generated tokens. The context capacity is explicit: bank context 32,768 for two non-unified streams, four streams aggregate to 65,536 across both banks, and maximum single-request context is 16,384. The measured buffer subtotal is still only a lower bound; loader free-memory/reserve checks, graph allocation, semantic gates, full-F32 comparisons, and exact-token gates must all pass.

At 16K, the conservative tracked subtotal is about 3.71 GiB/card: target KV/indexer shrinks linearly to one quarter, while recurrent state, target compute workspace, and MTP compute are left at their full V23 measured values. Unmapped draft KV/indexer storage and graph/allocator usage are additional. **This is a safer candidate, not a fit claim.**

## Preserved runtime and quality policy

The fixture preserves all experts, IQ3_XXS weights, Q8_0 K/V, standard indexer, owner-local decode loop, and V23's original native semantic, full-F32 preclean, exact token, and diagnostic/clean-family gates. PLE uses the existing six-node hybrid path with an 8 GiB GPU prefix; CPU lookup precision and HBM/DDR reserves stay unchanged.

The short-context KV policy accepts only `ROOT_KV_UNIFIED=0`, `ROOT_KV_VARIANT=per_seq_16k`, `ROOT_BANK_CTX=32768`, two lanes, and two KV streams. The comparator checks actual target and draft capacities from the initialized contexts and rejects any 64K-shaped metadata. It does not label aggregate context as per-request capacity.

## Runtime and launch gate

`compile-runtime-v24-shortctx.sh` is a runner-only host link against `runtime-perf-v62`; it does not rebuild engine libraries or initialize CUDA. The V62 build ID and actual library hashes must be recorded after the V62 freeze.

The V62 host-link artifacts are recorded in `v62-hostlink.file.txt`, `v62-hostlink.ldd.txt`, and `v62-hostlink.sha256.txt` (build ID `20261016T010951.975659Z`; runner SHA `8f25242b2c8bb8fe09eba2002386d5d98f5b8de08886e9e491a6499b76a5c44`). The full-model launcher must remain gated on the corrected V62 tiny nested-context/graph qualification and a separate root GPU grant. It must use the V62 whole dynamic-library closure, the corrected BMC/GPU supervisor (DDR node0+8 and RSS record-only), and a fresh output directory. Do not use the historical V61 closure or run V24 while V62's loader/graph contract is unresolved.
