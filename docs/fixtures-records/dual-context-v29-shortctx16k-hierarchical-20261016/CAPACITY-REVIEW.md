# Hierarchical EP6 per-card capacity review (short-context fallback)

Status: **static budget only**. The V25 full-model attempt reached its first prefill and aborted in CUDA Graph structural-signature construction. V65 is the new candidate runtime; its opaque-extra tiny gate is still pending. The estimates below combine the V23 real-runtime buffer lines preserved in `V23-runtime-buffer-evidence.txt` with the prior V61 metadata estimate supplied by root. They are not a fit guarantee; V65's real per-card free-memory/reserve gate and graph construction remain authoritative.

## Static HBM room after weights and the six-node PLE prefix

Using 15.47 GiB usable HBM/card, 9.22–9.34 GiB/card hierarchical weight estimate, and an 8 GiB PLE prefix divided over six GPU NUMA nodes (1.333 GiB/card), the nominal room is:

- low-weight end: 15.47 − 9.22 − 1.333 = **4.92 GiB/card**
- high-weight end: 15.47 − 9.34 − 1.333 = **4.80 GiB/card**

The existing PLE policy retains a 4 GiB/card runtime reserve and a 2 GiB/card hierarchical-weight check. V65 includes the fail-closed opaque-extra graph-signature fix and preserves raw graph UID; this fixture does not change either reserve, and the real V65 loader must confirm them before any model run.

## Known V23 runtime allocations, mapped to hierarchical two-island replication

V23 used two three-GPU Meta islands. The log reports per-island Meta buffers for target QSA KV/indexer KV, recurrent state, and scheduler compute. Hierarchical EP6 mirrors dense/GDN/QSA paths across both islands; the known per-card tracked amount therefore sums both island entries for each bank:

| Known buffer class | V23 measured amount per island/card | Hierarchical two-island amount per bank/card |
|---|---:|---:|
| Target QSA KV + indexer KV | (272 + 102) MiB | **748 MiB** |
| Target recurrent state (RS) | 202 + 179 MiB across the two island reports | **381 MiB** |
| Target compute workspace | 453.17 + 454.98 MiB | **908.15 MiB** |
| **Target subtotal per bank** |  | **2037.15 MiB** |
| Two target banks |  | **4074.30 MiB** |
| Two MTP draft compute workspaces | 2 × 425.28 MiB | **850.56 MiB** |
| **Tracked runtime subtotal at 64K/sequence** |  | **4924.86 MiB = 4.81 GiB** |

The 64K/sequence tracked subtotal is already approximately the full 4.80–4.92 GiB left after static weights and PLE. Its margin ranges from about **−10 MiB to +113 MiB** before additional graph, allocator, and draft-state overhead. V23 also logs draft KV/indexer cache buffers (136 MiB and 17 MiB); their per-card nested mapping is not included in the table, so this subtotal is a known-buffer lower bound. **Do not launch 64K/sequence as a presumed-fit configuration.**

## 16K-per-sequence fallback budget for the real 8K prompts

The independent fallback fixes each request at 16,384 tokens, enough for an 8,192-token prompt plus the existing 256-token output. It keeps two lanes per bank, so each bank context is 32,768 cells, four streams total 65,536 across both banks, and the maximum request capacity is explicitly **16,384**.

For the conservative tracked subtotal below, QSA KV and indexer KV scale linearly to one quarter at 16K; recurrent state and target compute workspace are deliberately kept at the full V23 measured size instead of assuming they shrink. Draft compute remains unchanged:

| Known buffer class | 16K/sequence tracked amount per bank/card |
|---|---:|
| Target QSA KV + indexer KV (¼ of 748 MiB) | 187 MiB |
| Target recurrent state (unchanged) | 381 MiB |
| Target compute workspace (unchanged) | 908.15 MiB |
| Two target banks | **2952.30 MiB** |
| Two MTP draft compute workspaces | **850.56 MiB** |
| **Tracked 16K subtotal** | **3802.86 MiB = 3.71 GiB** |

This leaves roughly 1.09–1.21 GiB against the nominal 4.80–4.92 GiB room, before the unallocated graph, allocator, and draft KV/indexer costs. This is a safer capacity candidate than 32K or 64K, but **still requires the actual V65 loader/reserve check, graph capture, full-F32 gate, native semantic gate, and exact-token gate**. No memory reserve is reduced.

For comparison, at 32K/sequence the same conservative model tracks about 4.08 GiB/card, leaving only about 0.72–0.84 GiB before omitted allocations; 16K is the preferred fallback for the requested 8K workload.

## Quality and safety policy

The short-context runner keeps the V23 test prompts, 256-token real-prompt output, diagnostic native semantic gate, full-F32 preclean gate, exact serial/P4 token comparison, and clean-family measurement. It reports actual target/draft context sizes and KV-stream counts from the initialized contexts. It does not describe the 65,536-token aggregate pool as a per-request limit.

The PLE byte format and lookup precision are unchanged. `LLAMA_PLE_NUMA_GPU_NODES=6`, `LLAMA_PLE_NUMA_GPU_GIB=8`, `LLAMA_PLE_NUMA_HYBRID=1`, `LLAMA_PLE_SHARD=0`, and `LLAMA_PLE_SHARD_ROWS=0` preserve the existing six-node PLE placement and CPU lookup path. A full-model launch remains gated on root's V65 tiny qualification and a separate GPU grant.
