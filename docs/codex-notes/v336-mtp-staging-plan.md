# V336 candidate: single staging of ordinary MTP prefill

Status: source audit only; no code/build/GPU change.

In common/speculative.cpp1923–50 and1970–85 the nonshared catch-up path builds token/position/sequence rows and shifted hidden embeddings twice. First buffer is only consumed by the fold/draft_pending verification branch. Under last_only_prefill && def_tok.empty(), draft_pending is false by definition and flush_deferred returns immediately; therefore the first build is dead. Preserve the second build and all carry/history updates.

Proposal: default-off LLAMA_MTP_PREFILL_SINGLE_BUILD; when requested and last_only_prefill && def_tok.empty(), skip only the first build. Keep set_h lambda visible to second build. Every other branch retains original implementation. Log selected rows and avoided host memcpy bytes, without claiming target D2H or GPU compute savings. At32K×10240F32, duplicate shift avoided1,342,136,320bytes/chunk. This is a low-cost execution cleanup, not the primary throughput bet.

Proof requirements: confirm no first-buffer consumer under predicate; native+32K-crossing correctness and final carry; whole-model speed separately from host byte count. Integrate with persistent-owner overlap only after ownership boundary is explicit; never let next target decode overwrite borrowed hidden rows.
