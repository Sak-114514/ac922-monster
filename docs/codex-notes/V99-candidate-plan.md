# V99 singleton Turbo admission

Source review found all n_seq_max=1 contexts allocate an optional pooled-QSA block cache even with unified=false and pooled-multi=0. The installed frame scheduler correctly refuses this mutable cache, so planned single32K/64K would silently benchmark the normal route.

Add default-on LLAMA_QSA_POOLED_SINGLE; exact0 disables only singleton optional pooled cache. Preserve raw indexer/attention/recurrent caches, original Q8 KV and all model weights; use existing per-cell QSA and frozen metadata path already used by multi-stream contexts. Frame admission/ownership/events/budgets remain strict, unchanged. Log the allocation policy. Default and four-stream modes remain unchanged. New combined candidate uses exact0, original standard indexer, and verifies actual selected frames plus small semantic canaries before any claim.

Correct stale topology diagnostic claiming adapter_not_installed/execution_enabled0; distinguish adapter installed/topology observation from actual submit outcome. Two CPU translation units, no CUDA rebuild.
