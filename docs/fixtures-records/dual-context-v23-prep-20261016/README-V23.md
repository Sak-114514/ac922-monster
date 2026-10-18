# V23 owner-local speculative loop

This fixture is V22-derived and uses the V23 candidate runner plus the V23 owner-loop helper. It preserves the V22 diagnostic and clean context-family quality/performance gates and uses the frozen V59 runtime closure. The owner-loop candidate is sourced from `fixtures/autonomous-owner-loop-20261016/candidate-v22/dual-context-v23.cpp` (SHA-256 3f1e6d5c49b8ff70eeeb16f306ae6f28a67139f2506743817eb75a4ea296de15).

The candidate keeps V22 CLI and quality-checker arguments. This fixture links only the standalone runner; no runtime engine library is rebuilt here. The authorized run is two banks x two lanes, D3, per-sequence 64K, target ubatch 1024, MTP ubatch 512, CPU copy mode 1 capped at 1 MiB, NCCL threshold 0, standard QSA indexer, with CUPTI and LD_PRELOAD disabled.
