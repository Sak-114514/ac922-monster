# V326: Layer-Weighted Hybrid PLE Quota Estimate

## Decision

For a first 9/9/8/8/8/6 main-layer layout, use this candidate 18 GiB HBM prefix distribution:

| Logical GPU | Main layers | HBM PLE quota |
|---:|---:|---:|
| 0 | 0–8 | 3456 MiB |
| 1 | 9–17 | 4160 MiB |
| 2 | 18–25 | 3776 MiB |
| 3 | 26–33 | 2880 MiB |
| 4 | 34–41 | 2752 MiB |
| 5 | 42–47 | 1408 MiB |
| **Total** | **48 main layers** | **18432 MiB = 18 GiB** |

All quota values are multiples of 64 MiB and sum to the existing 18 GiB prefix. Based on the saved V324 CUDA-free measurements and exact raw per-layer GGUF sizes, this distribution leaves approximately 0.73–0.76 GiB on every card above the unchanged 4 GiB runtime reserve. This is a preselection estimate only. The target-layout model load must pass the existing per-GPU CUDA-free and Linux NUMA-available admission checks before any PLE mapping/copy proceeds.

This is the Hybrid18 path. It does not place the full PLE table on GPU. The PLE remains one byte-exact contiguous virtual table; only the per-range NUMA placement policy changes. With an 18 GiB prefix, its unchanged 8.822 GiB suffix remains on CPU DDR.

## Metadata source and reproduction

The model was read from:

    /home/debian/models/qwen3.8-flash-next-gsq-rco/integrated/IQ3_XXS-MTP-Q8.gguf

The GGUF parser source was:

    /home/debian/ac922-monster/experiments/standard-opt-20261014/src/gguf-py/gguf/gguf_reader.py

GGUFReader maps the file and reads the GGUF header/tensor descriptors. The extraction below uses each descriptor's name and type-derived n_bytes; it never accesses tensor.data or reads weight payload bytes. The file is 78,615,622,080 bytes, has 1256 tensors, and the GGUF tensor-data area starts at byte 11,026,368.

No standalone extraction script was saved; the following exact read-only script body was run inline, with the model path above passed as argv[1]. The parser implementation path above and this embedded body are the reproduction references:

    import collections
    import re
    import sys
    from gguf.gguf_reader import GGUFReader

    reader = GGUFReader(sys.argv[1], mode="r")
    sizes = collections.defaultdict(int)
    for tensor in reader.tensors:
        match = re.match(r"^blk\.(\d+)\.", tensor.name)
        if match:
            sizes[int(match.group(1))] += int(tensor.n_bytes)
    for layer in range(50):
        print(layer, sizes[layer])

The installed package implementation creates NumPy views for tensor payloads while building ReaderTensor objects, but this script only reads n_bytes and never dereferences those views. V324's current CUDA model-buffer totals independently match the descriptor raw-byte sums to within the log's rounding, providing a cross-check of the layer accounting.

## Raw bytes by main layer

Values are exact descriptor n_bytes, in ascending layer order within each row.

| Layer indices | Raw bytes in order |
|---|---|
| 0–7 | 1,018,105,216; 800,034,176; 1,019,070,336; 734,523,392; 1,025,222,016; 789,753,216; 733,981,056; 732,667,392 |
| 8–15 | 1,019,536,256; 791,401,856; 798,354,816; 735,670,272; 839,342,976; 736,082,816; 787,554,176; 839,624,192 |
| 16–23 | 740,038,016; 841,406,336; 1,024,902,016; 787,505,152; 1,023,839,616; 833,992,576; 1,026,814,336; 1,023,619,072 |
| 24–31 | 1,026,007,936; 800,610,176; 838,055,296; 744,187,392; 1,264,558,976; 1,081,265,536; 1,268,465,536; 980,116,992 |
| 32–39 | 1,261,540,736; 1,022,869,376; 1,270,431,616; 1,182,362,112; 953,017,216; 1,256,502,656; 1,086,546,816; 1,032,712,192 |
| 40–47 | 846,728,576; 940,004,736; 1,084,255,616; 1,268,134,912; 1,189,164,416; 948,537,216; 1,083,231,616; 1,173,143,552 |

The main layers that GPU5 gains in the 9/9/8/8/8/6 layout are:

| Layer | Raw bytes | MiB | GiB |
|---:|---:|---:|---:|
| 42 | 1,084,255,616 | 1034.027 | 1.009792 |
| 43 | 1,268,134,912 | 1209.388 | 1.181043 |
| **42 + 43** | **2,352,390,528** | **2243.414** | **2.190834** |

Other fixed/root tensors relevant to GPU5:

| Tensor | Raw bytes | Placement/evidence |
|---|---:|---|
| blk.48 MTP | 2,775,621,632 | 2.584999 GiB; assigned to CUDA5 in V324 |
| blk.49 | 0 | No payload bytes |
| output.weight | 437,043,200 | 416.797 MiB; output layer is offloaded to CUDA5 |
| output_hc_down.weight | 6,553,600 | 6.250 MiB |
| output_hc_norm.weight | 40,960 | 0.039 MiB |
| output_hc_up.weight | 6,553,600 | 6.250 MiB |
| token_embd.weight | 273,152,000 | 260.498 MiB; V324 logged this as CPU_Mapped |
| per_layer_token_embd.weight | 28,800,138,240 | 27,465.952 MiB = 26.822219 GiB total PLE |

MTP plus output/output_hc fixed bytes total 3,225,812,992 bytes (3.004272 GiB). The target GPU5 main-layer range 42–47 is 6.283137 GiB, so target GPU5 raw model weight total is 9.287410 GiB, about 2.190834 GiB above its current 44–47 plus fixed-weight allocation.

## V324 baseline and target admission estimate

V324's current main-layer ranges were 0–9, 10–19, 20–28, 29–35, 36–43, and 44–47; MTP and output/output_hc stayed on GPU5. The target main-layer ranges are 0–8, 9–17, 18–25, 26–33, 34–41, and 42–47. The target moves weights only; output, MTP, token embedding and PLE placements remain as described above.

The table's baseline CUDA_free values are copied from V324's PLE admission log and are rounded there to 0.01 GiB. Target free is estimated by subtracting the exact descriptor raw-weight delta from that rounded baseline. A 4 GiB per-GPU runtime reserve is subtracted to estimate the PLE quota cap.

| GPU | Current layer raw GiB | Target layer raw GiB | Target minus current weights GiB | V324 CUDA_free before PLE GiB | Estimated target CUDA_free before PLE GiB | Estimated PLE cap after 4 GiB reserve GiB | Candidate quota MiB | Estimated margin above reserve GiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 8.069253 | 7.332203 | -0.737050 | 7.37 | 8.107050 | 4.107050 | 3456 | 0.732050 |
| 1 | 7.572100 | 6.621215 | -0.950885 | 7.87 | 8.820885 | 4.820885 | 4160 | 0.758385 |
| 2 | 7.992317 | 7.028962 | -0.963355 | 7.45 | 8.413355 | 4.413355 | 3776 | 0.725855 |
| 3 | 7.513028 | 7.879976 | +0.366948 | 7.93 | 7.563052 | 3.563052 | 2880 | 0.750552 |
| 4 | 7.886349 | 7.979857 | +0.093508 | 7.54 | 7.446492 | 3.446492 | 2752 | 0.758992 |
| 5 | 7.096575 | 9.287410 | +2.190834 | 8.33 | 6.139166 | 2.139166 | 1408 | 0.764166 |

The V324 log reports Linux GPU-NUMA available slightly above CUDA_free on all six devices (logical GPUs 0–5: 7.39, 7.89, 7.46, 7.95, 7.57, 8.36 GiB). CUDA_free is therefore the lower of the two admission constraints in the baseline. New layout startup must remeasure both constraints; the table is not a substitute for that check.

The V324 log also reports the current CUDA model buffers as 8262.92, 7753.83, 8184.14, 7693.34, 8075.63, and 7266.90 MiB. They match the current raw per-GPU GGUF tensor sums to the displayed precision.

With an 18 GiB prefix, CPU DDR bytes remain:

    28,800,138,240 - 18 * 1,073,741,824 = 9,472,785,408 bytes
    = 8.822218895 GiB total, or 4.411109447 GiB per node when balanced.

The unchanged CPU runtime reserve is 3 GiB total (1.5 GiB per node). V324 reported node0/node8 available capacity of 9.38/12.79 GiB and 22.17 GiB aggregate before PLE placement. The target layer assignment does not change the prefix/suffix arithmetic; the loader still rechecks actual CPU node capacity before copy.

## Why not 8/8/8/8/8/8 first

Equal eight-main-layer ranges would move layers 40–43 to GPU5. Those layers total 3.854859 GiB raw, leaving estimated GPU5 free memory of 4.475 GiB before PLE and only about 0.475 GiB for a PLE stripe after the 4 GiB reserve. The aggregate may be redistributed over the other GPUs, but this creates a substantially tighter GPU5 margin. The 9/9/8/8/8/6 layout has a much more balanced projected margin and is the preferred first candidate.

## V326 parser and downstream reporting caveat

V326's opt-in LLAMA_PLE_NUMA_GPU_STRIPES_MIB parser accepts six page-aligned MiB quotas whose sum equals the configured prefix. It preserves the existing uniform default when unset, contiguous virtual addresses, original raw bytes, per-GPU admission against both CUDA free and Linux NUMA available, the 4 GiB GPU reserve, and the CPU suffix. A zero quota skips mbind for that device while still applying the reserve check; the recommended candidate has no zero entries.

The existing V324 runner still derives ple_gpu_gib_per_node as total LLAMA_PLE_NUMA_GPU_GIB divided by LLAMA_PLE_NUMA_GPU_NODES. That field would falsely report 3 GiB/device with custom quotas. For any custom-quota run, record LLAMA_PLE_NUMA_GPU_STRIPES_MIB in the run metadata or use the actual per-device stripe/offset lines in the engine log. Do not interpret the legacy per-node field as the physical distribution.

Relevant artifacts:

- V324 admission and tensor-buffer evidence: fixtures/single64k-golden-fresh-v307-20261018/results/root-v324-inband-20261017t184856625772z-runtime.log
- V324 runner with equal-per-node metadata assumption: fixtures/single64k-golden-fresh-v307-20261018/dual-context-v324-direct-cupti64k.cpp
- GGUFReader implementation: src/gguf-py/gguf/gguf_reader.py in the AC922 checkout
- V326 CPU policy fixture: fixtures/ple-stripe-quotas-v326-20261018/policy.cpp
- Active placement policy: src/src/llama-ple-numa.h

No GPU fit test for the new target layout is claimed here. Retain fail-closed admission and the 4 GiB reserve unchanged.
