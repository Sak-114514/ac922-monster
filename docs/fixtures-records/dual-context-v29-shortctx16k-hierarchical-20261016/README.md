# V29 nested asynchronous Meta dispatch candidate

V29 reuses the frozen V28 runner binary and V27 quality/comparison gates. It keeps the V28 topology, context capacity (16K per sequence), two banks × two lanes, D3, UB1024/MTP512, fragment cache CAP64/MIB128, CPU-copy mode 1, standard indexer, and no-CUPTI/no-preload policy. It selects the frozen V68 closure and adds only `GGML_META_NESTED_ASYNC=1`.

The supervisor requires the V68 fifth-frame hot-path proof, the inherited cache/merge gates, and a root GPU grant. No model is launched during preparation. After a run, `analyze_v29_run.py` reports direct `LLAMA_STEP_TIMING` compute milliseconds alongside the clean P4 record; it does not infer step time from token rate or report a cold probe as a speed result. The V27 auxiliary prefix failures remain in force.
