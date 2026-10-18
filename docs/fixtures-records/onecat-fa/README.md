# Onecat Flash-V100 bounded fixture

This fixture exercises a standalone `GGML_OP_FLASH_ATTN_EXT` without loading a model. It uses D=256, QH=24/KVH=2, one query token per stream, F16 per-stream masks, and both Q8_0 and F16 K/V at NKV=2304. The matrix covers streams 1/4/8 at NKV=2304 for both K/V types, plus Q8_0 at NKV=512/8192. The S4→S8→S4 passes reuse stable graph/tensor addresses and check exact A replay. Additional S4 cases use contiguous K/V and strided K/V with an all-zero mask. A canary tensor follows the FA op allocation to detect writes beyond the advertised extra-data size.

The executable links the frozen `runtime-perf-v5` ggml CUDA backend plus a separately compiled `fattn-volta-onecat-fixed.o` from the private source. The object supplies the patched Onecat admission/allocation/launch symbols, so this does not rebuild or replace the server runtime. Its Onecat symbols are exported from the executable and ELF binding was confirmed from `libggml-cuda.so.0` to the executable.

Set `GGML_CUDA_FATTN_ONECAT=0` and `=1` in separate fresh processes, each writing to its own output directory. The Onecat run should log `using opt-in Onecat Flash-V100 XQA decode path`; the marker is once per process. Compare CPU-copied F32 outputs with:

```sh
python3 fixtures/onecat-fa/compare.py <stock-output-dir> <onecat-output-dir> --atol 0.02 --rtol 0.02
```

The strict recorded comparison is `results/onecat-fa-fixture-20261015-v6/compare-strict.txt`; it uses `--atol 0.004 --rtol 0`, stricter than the script's default `0.02 + 0.02*abs(ref)`. All 14 stock/Onecat outputs passed that absolute-error gate. The current bounded run passed canary and A/B/A checks. It does not prove model-level accuracy or throughput. `compute-sanitizer` was not installed, and the fixture has no model weights or MoE path.
