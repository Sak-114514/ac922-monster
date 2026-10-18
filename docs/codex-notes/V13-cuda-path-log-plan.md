# V13 bounded CUDA dispatch logs

No arithmetic or dispatch policy change. Replace each one-time thread-local chunk log with a per-type/per-row-count 64-bit seen mask, so a prefill tail cannot hide later12/16-row decode activation. Only the already admitted Volta <=64-row chunk path is observed. Log actual selected F32 wire when GGML_CUDA_NCCL_F32 is explicitly enabled, using the existing bounded host-dispatch registry. No public header/dependency changes; one CUDA translation unit.
