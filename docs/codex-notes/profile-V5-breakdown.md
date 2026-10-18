# V5 Nsight SQLite profile breakdown

Read-only analysis of the existing prefill8k.sqlite and decode4.sqlite. No model, GPU, build, or source operations were performed.

## prefill8k

- Capture duration: 8967.692 ms; completed kernel rows: 139416; CUDA runtime API rows: 108911.
- CUDA context device IDs: 0, 1, 2, 3, 4, 5.
- Kernel activity device IDs: 0, 1, 2, 3, 4, 5.
- Runtime API summed duration: 5062.557 ms across calls; this is not wall time.

| Device | Kernel rows | Sum of captured kernel durations (ms) | First-to-last event span (ms) |
|---:|---:|---:|---:|
| 0 | 21400 | 2933.133 | 7799.891 |
| 1 | 21384 | 2855.095 | 7792.669 |
| 2 | 21396 | 2545.229 | 7787.251 |
| 3 | 23204 | 2691.913 | 7933.011 |
| 4 | 26020 | 2609.994 | 7922.828 |
| 5 | 26012 | 2341.019 | 7912.923 |

Category percentages are within that device's captured kernel-duration sum only; intervals can overlap.

| Device | Category | Kernel rows | Sum ms | Share of device kernel sum |
|---:|---|---:|---:|---:|
| 0 | groupedMoE | 600 | 860.717 | 29.3% |
| 0 | P2P_AR | 250 | 459.435 | 15.7% |
| 0 | NCCL_AllReduce | 150 | 416.912 | 14.2% |
| 0 | denseGEMM | 4103 | 301.669 | 10.3% |
| 0 | other | 5628 | 232.467 | 7.9% |
| 0 | convert_dequant | 6758 | 160.636 | 5.5% |
| 0 | GDN | 152 | 127.912 | 4.4% |
| 0 | hyperconnection | 799 | 110.429 | 3.8% |
| 0 | attention_Onecat_QSA | 64 | 95.099 | 3.2% |
| 0 | normalization | 1048 | 77.425 | 2.6% |
| 0 | groupedMoE_prep | 1800 | 76.711 | 2.6% |
| 0 | QSA_indexer | 48 | 13.721 | 0.5% |
| 1 | groupedMoE | 596 | 854.120 | 29.9% |
| 1 | P2P_AR | 250 | 473.184 | 16.6% |
| 1 | NCCL_AllReduce | 150 | 330.186 | 11.6% |
| 1 | denseGEMM | 4096 | 303.530 | 10.6% |
| 1 | other | 5630 | 232.783 | 8.2% |
| 1 | convert_dequant | 6754 | 160.359 | 5.6% |
| 1 | GDN | 152 | 128.051 | 4.5% |
| 1 | hyperconnection | 800 | 110.022 | 3.9% |
| 1 | attention_Onecat_QSA | 64 | 95.610 | 3.3% |
| 1 | normalization | 1048 | 77.628 | 2.7% |
| 1 | groupedMoE_prep | 1796 | 75.909 | 2.7% |
| 1 | QSA_indexer | 48 | 13.713 | 0.5% |
| 2 | groupedMoE | 600 | 859.906 | 33.8% |
| 2 | P2P_AR | 250 | 456.285 | 17.9% |
| 2 | denseGEMM | 4100 | 303.113 | 11.9% |
| 2 | other | 5630 | 232.775 | 9.1% |
| 2 | convert_dequant | 6758 | 159.942 | 6.3% |
| 2 | GDN | 152 | 128.191 | 5.0% |
| 2 | hyperconnection | 796 | 109.495 | 4.3% |
| 2 | attention_Onecat_QSA | 64 | 95.121 | 3.7% |
| 2 | normalization | 1048 | 77.812 | 3.1% |
| 2 | groupedMoE_prep | 1800 | 76.330 | 3.0% |
| 2 | NCCL_AllReduce | 150 | 32.551 | 1.3% |
| 2 | QSA_indexer | 48 | 13.708 | 0.5% |
| 3 | groupedMoE | 552 | 720.526 | 26.8% |
| 3 | P2P_AR | 230 | 430.481 | 16.0% |
| 3 | NCCL_AllReduce | 139 | 401.289 | 14.9% |
| 3 | denseGEMM | 4008 | 300.274 | 11.2% |
| 3 | other | 5698 | 193.465 | 7.2% |
| 3 | convert_dequant | 6702 | 159.646 | 5.9% |
| 3 | GDN | 136 | 114.124 | 4.2% |
| 3 | hyperconnection | 992 | 102.297 | 3.8% |
| 3 | attention_Onecat_QSA | 64 | 94.952 | 3.5% |
| 3 | groupedMoE_prep | 2168 | 71.465 | 2.7% |
| 3 | normalization | 1697 | 71.313 | 2.6% |
| 3 | quantMMQ | 768 | 18.201 | 0.7% |
| 3 | QSA_indexer | 48 | 13.661 | 0.5% |
| 3 | quantmatvec | 1 | 0.207 | 0.0% |
| 3 | BF16_GEMV | 1 | 0.012 | 0.0% |
| 4 | groupedMoE | 552 | 725.514 | 27.8% |
| 4 | P2P_AR | 230 | 421.074 | 16.1% |
| 4 | denseGEMM | 4008 | 300.035 | 11.5% |
| 4 | NCCL_AllReduce | 139 | 298.823 | 11.4% |
| 4 | other | 6722 | 202.355 | 7.8% |
| 4 | convert_dequant | 6702 | 158.533 | 6.1% |
| 4 | GDN | 136 | 114.998 | 4.4% |
| 4 | hyperconnection | 992 | 102.774 | 3.9% |
| 4 | attention_Onecat_QSA | 64 | 95.826 | 3.7% |
| 4 | groupedMoE_prep | 2680 | 73.017 | 2.8% |
| 4 | normalization | 1953 | 71.447 | 2.7% |
| 4 | quantMMQ | 1792 | 31.638 | 1.2% |
| 4 | QSA_indexer | 48 | 13.740 | 0.5% |
| 4 | quantmatvec | 1 | 0.207 | 0.0% |
| 4 | BF16_GEMV | 1 | 0.013 | 0.0% |
| 5 | groupedMoE | 552 | 724.347 | 30.9% |
| 5 | P2P_AR | 230 | 421.442 | 18.0% |
| 5 | denseGEMM | 4008 | 300.132 | 12.8% |
| 5 | other | 6718 | 202.120 | 8.6% |
| 5 | convert_dequant | 6698 | 157.090 | 6.7% |
| 5 | GDN | 136 | 114.949 | 4.9% |
| 5 | hyperconnection | 992 | 102.756 | 4.4% |
| 5 | attention_Onecat_QSA | 64 | 95.464 | 4.1% |
| 5 | groupedMoE_prep | 2680 | 72.936 | 3.1% |
| 5 | normalization | 1953 | 71.358 | 3.0% |
| 5 | NCCL_AllReduce | 139 | 32.825 | 1.4% |
| 5 | quantMMQ | 1792 | 31.656 | 1.4% |
| 5 | QSA_indexer | 48 | 13.725 | 0.6% |
| 5 | quantmatvec | 1 | 0.207 | 0.0% |
| 5 | BF16_GEMV | 1 | 0.013 | 0.0% |

### Runtime API totals

| Category | Calls | Sum ms |
|---|---:|---:|
| sync | 7078 | 2909.835 |
| kernel_launch | 77133 | 599.443 |
| graph_launch | 1107 | 541.394 |
| memcpy | 6859 | 421.671 |
| graph_instantiate | 222 | 322.121 |
| other | 10557 | 215.901 |
| memset | 4422 | 42.417 |
| graph_update | 222 | 5.791 |
| graph_capture | 1311 | 3.984 |

### Top kernels

| Device | Kernel | Calls | Sum ms | Avg us | Max us |
|---:|---|---:|---:|---:|---:|
| 0 | k_allreduce_p2p_seg | 250 | 459.435 | 1837.74 | 10661.69 |
| 0 | ncclDevKernel_AllReduce_Sum_bf16_RING_LL | 150 | 416.912 | 2779.41 | 9048.13 |
| 0 | mul_mat_q_moe_grouped | 176 | 277.228 | 1575.16 | 1861.14 |
| 0 | void cutlass::Kernel2<cutlass_70_tensorop_s884gemm_f16_128x128_tn_align8>(T1::Params) | 880 | 147.773 | 167.92 | 624.38 |
| 0 | gated_delta_net_cuda_128x4_volta | 152 | 127.912 | 841.53 | 951.93 |
| 0 | mul_mat_q_moe_grouped | 80 | 126.631 | 1582.89 | 1882.74 |
| 0 | mul_mat_q_moe_grouped | 96 | 117.835 | 1227.44 | 1343.16 |
| 0 | convert_unary_cont_sat_vec4 | 3200 | 101.472 | 31.71 | 81.03 |
| 0 | flash_attn_ext_f16 | 32 | 94.349 | 2948.41 | 5175.07 |
| 0 | mul_mat_q_moe_grouped | 48 | 76.648 | 1596.84 | 1763.92 |
| 0 | mul_mat_q_moe_grouped | 64 | 74.816 | 1169.00 | 1284.02 |
| 0 | dsv4_hc_post_f32 | 400 | 61.599 | 154.00 | 705.08 |
| 0 | volta_s884gemm_fp16_64x64_ldg8_tn | 487 | 60.403 | 124.03 | 161.53 |
| 0 | rms_norm_f32 | 424 | 59.700 | 140.80 | 152.60 |
| 0 | mul_mat_q_moe_grouped | 48 | 58.729 | 1223.53 | 1483.54 |
| 1 | k_allreduce_p2p_seg | 250 | 473.184 | 1892.74 | 13366.94 |
| 1 | ncclDevKernel_AllReduce_Sum_bf16_RING_LL | 150 | 330.186 | 2201.24 | 9068.88 |
| 1 | mul_mat_q_moe_grouped | 172 | 270.396 | 1572.07 | 1849.97 |
| 1 | mul_mat_q_moe_grouped | 96 | 154.154 | 1605.77 | 2372.53 |
| 1 | void cutlass::Kernel2<cutlass_70_tensorop_s884gemm_f16_128x128_tn_align8>(T1::Params) | 884 | 150.264 | 169.98 | 624.95 |
| 1 | gated_delta_net_cuda_128x4_volta | 152 | 128.051 | 842.44 | 959.80 |
| 1 | convert_unary_cont_sat_vec4 | 3196 | 101.470 | 31.75 | 81.09 |
| 1 | mul_mat_q_moe_grouped | 64 | 97.240 | 1519.37 | 1648.50 |
| 1 | flash_attn_ext_f16 | 32 | 94.847 | 2963.97 | 5238.65 |
| 1 | mul_mat_q_moe_grouped | 80 | 94.846 | 1185.58 | 1471.41 |
| 1 | mul_mat_q_moe_grouped | 48 | 61.529 | 1281.85 | 1364.95 |
| 1 | volta_s884gemm_fp16_64x64_ldg8_tn | 496 | 61.402 | 123.79 | 162.20 |
| 1 | dsv4_hc_post_f32 | 400 | 61.132 | 152.83 | 174.68 |
| 1 | rms_norm_f32 | 424 | 59.741 | 140.90 | 151.07 |
| 1 | mul_mat_q_moe_grouped | 48 | 57.145 | 1190.53 | 1333.02 |
| 2 | k_allreduce_p2p_seg | 250 | 456.285 | 1825.14 | 7419.41 |
| 2 | mul_mat_q_moe_grouped | 176 | 276.883 | 1573.20 | 1849.59 |
| 2 | mul_mat_q_moe_grouped | 128 | 202.442 | 1581.58 | 1882.90 |
| 2 | void cutlass::Kernel2<cutlass_70_tensorop_s884gemm_f16_128x128_tn_align8>(T1::Params) | 888 | 150.287 | 169.24 | 628.00 |
| 2 | gated_delta_net_cuda_128x4_volta | 152 | 128.191 | 843.36 | 955.71 |
| 2 | convert_unary_cont_sat_vec4 | 3200 | 100.771 | 31.49 | 81.69 |
| 2 | mul_mat_q_moe_grouped | 80 | 96.739 | 1209.24 | 1471.93 |
| 2 | flash_attn_ext_f16 | 32 | 94.369 | 2949.03 | 5173.87 |
| 2 | mul_mat_q_moe_grouped | 48 | 60.648 | 1263.50 | 1342.05 |
| 2 | volta_s884gemm_fp16_64x64_ldg8_tn | 480 | 60.570 | 126.19 | 161.85 |
| 2 | dsv4_hc_post_f32 | 396 | 60.492 | 152.76 | 175.35 |
| 2 | rms_norm_f32 | 424 | 59.898 | 141.27 | 152.06 |
| 2 | mul_mat_q_moe_grouped | 48 | 57.835 | 1204.90 | 1349.24 |
| 2 | mul_mat_q_moe_grouped | 32 | 50.215 | 1569.22 | 1651.13 |
| 2 | dsv4_hc_pre_f32 | 400 | 49.003 | 122.51 | 130.20 |
| 3 | k_allreduce_p2p_seg | 230 | 430.481 | 1871.66 | 7407.71 |
| 3 | ncclDevKernel_AllReduce_Sum_bf16_RING_LL | 138 | 401.069 | 2906.30 | 9093.76 |
| 3 | mul_mat_q_moe_grouped | 96 | 152.860 | 1592.29 | 2229.94 |
| 3 | mul_mat_q_moe_grouped | 120 | 144.138 | 1201.15 | 1425.75 |
| 3 | void cutlass::Kernel2<cutlass_70_tensorop_s884gemm_f16_128x128_tn_align8>(T1::Params) | 800 | 133.729 | 167.16 | 462.65 |
| 3 | mul_mat_q_moe_grouped | 96 | 115.689 | 1205.10 | 1469.08 |
| 3 | gated_delta_net_cuda_128x4_volta | 136 | 114.124 | 839.15 | 960.02 |
| 3 | mul_mat_q_moe_grouped | 64 | 96.661 | 1510.32 | 1705.37 |
| 3 | convert_unary_cont_sat_vec4 | 3184 | 95.527 | 30.00 | 813.78 |
| 3 | flash_attn_ext_f16 | 32 | 94.184 | 2943.25 | 5213.58 |
| 3 | mul_mat_q_moe_grouped | 64 | 74.433 | 1163.01 | 1300.38 |
| 3 | dsv4_hc_post_f32 | 368 | 55.958 | 152.06 | 174.97 |
| 3 | rms_norm_f32 | 1129 | 55.837 | 49.46 | 1507.99 |
| 3 | volta_s884gemm_fp16_64x64_ldg8_tn | 424 | 55.055 | 129.85 | 162.62 |
| 3 | dsv4_hc_pre_f32 | 624 | 46.339 | 74.26 | 124.99 |
| 4 | k_allreduce_p2p_seg | 230 | 421.074 | 1830.76 | 7329.43 |
| 4 | ncclDevKernel_AllReduce_Sum_bf16_RING_LL | 138 | 298.592 | 2163.71 | 9120.68 |
| 4 | mul_mat_q_moe_grouped | 120 | 145.319 | 1210.99 | 1520.92 |
| 4 | mul_mat_q_moe_grouped | 112 | 135.647 | 1211.13 | 1557.65 |
| 4 | void cutlass::Kernel2<cutlass_70_tensorop_s884gemm_f16_128x128_tn_align8>(T1::Params) | 792 | 131.555 | 166.10 | 461.53 |
| 4 | mul_mat_q_moe_grouped | 80 | 128.574 | 1607.17 | 1927.63 |
| 4 | gated_delta_net_cuda_128x4_volta | 136 | 114.998 | 845.57 | 955.64 |
| 4 | mul_mat_q_moe_grouped | 96 | 111.799 | 1164.57 | 1310.48 |
| 4 | mul_mat_q_moe_grouped | 64 | 96.943 | 1514.73 | 1710.48 |
| 4 | flash_attn_ext_f16 | 32 | 95.057 | 2970.53 | 5257.82 |
| 4 | convert_unary_cont_sat_vec4 | 3184 | 94.546 | 29.69 | 80.89 |
| 4 | dsv4_hc_post_f32 | 368 | 56.404 | 153.27 | 175.35 |
| 4 | volta_s884gemm_fp16_64x64_ldg8_tn | 432 | 55.395 | 128.23 | 163.93 |
| 4 | rms_norm_f32 | 1129 | 54.777 | 48.52 | 151.96 |
| 4 | mul_mat_q_moe_grouped | 32 | 47.058 | 1470.56 | 1738.87 |
| 5 | k_allreduce_p2p_seg | 230 | 421.442 | 1832.36 | 7495.72 |
| 5 | mul_mat_q_moe_grouped | 120 | 145.001 | 1208.34 | 1511.77 |
| 5 | mul_mat_q_moe_grouped | 112 | 135.577 | 1210.51 | 1479.01 |
| 5 | void cutlass::Kernel2<cutlass_70_tensorop_s884gemm_f16_128x128_tn_align8>(T1::Params) | 792 | 131.760 | 166.36 | 459.61 |
| 5 | mul_mat_q_moe_grouped | 80 | 128.158 | 1601.98 | 1910.08 |
| 5 | gated_delta_net_cuda_128x4_volta | 136 | 114.949 | 845.21 | 961.15 |
| 5 | mul_mat_q_moe_grouped | 64 | 96.985 | 1515.39 | 1700.54 |
| 5 | flash_attn_ext_f16 | 32 | 94.698 | 2959.31 | 5221.41 |
| 5 | convert_unary_cont_sat_vec4 | 3184 | 94.222 | 29.59 | 81.82 |
| 5 | mul_mat_q_moe_grouped | 80 | 93.941 | 1174.26 | 1298.68 |
| 5 | dsv4_hc_post_f32 | 368 | 56.284 | 152.95 | 174.75 |
| 5 | volta_s884gemm_fp16_64x64_ldg8_tn | 432 | 55.806 | 129.18 | 164.09 |
| 5 | rms_norm_f32 | 1129 | 54.726 | 48.47 | 152.00 |
| 5 | mul_mat_q_moe_grouped | 32 | 46.741 | 1460.67 | 1583.96 |
| 5 | dsv4_hc_pre_f32 | 624 | 46.472 | 74.47 | 128.32 |

### Data completeness

- Diagnostics: Not all NVTX events might have been collected.; Not all CUDA events might have been collected.; Not all OS runtime libraries events might have been collected..
- Nonzero CUPTI runtime API return codes: 0.

## decode4

- Capture duration: 7079.196 ms; completed kernel rows: 83736; CUDA runtime API rows: 3212.
- CUDA context device IDs: 0, 1, 2, 3, 4, 5.
- Kernel activity device IDs: 5.
- Runtime API summed duration: 653.599 ms across calls; this is not wall time.

| Device | Kernel rows | Sum of captured kernel durations (ms) | First-to-last event span (ms) |
|---:|---:|---:|---:|
| 5 | 83736 | 1366.460 | 4378.953 |

Category percentages are within that device's captured kernel-duration sum only; intervals can overlap.

| Device | Category | Kernel rows | Sum ms | Share of device kernel sum |
|---:|---|---:|---:|---:|
| 5 | other | 40670 | 330.029 | 24.2% |
| 5 | BF16_GEMV | 8618 | 270.538 | 19.8% |
| 5 | quantmatvec | 9008 | 254.270 | 18.6% |
| 5 | P2P_AR | 3103 | 225.680 | 16.5% |
| 5 | quantMMQ | 6326 | 117.102 | 8.6% |
| 5 | attention_Onecat_QSA | 1118 | 50.289 | 3.7% |
| 5 | normalization | 6641 | 45.281 | 3.3% |
| 5 | convert_dequant | 685 | 25.327 | 1.9% |
| 5 | GDN | 1054 | 21.645 | 1.6% |
| 5 | hyperconnection | 2978 | 11.307 | 0.8% |
| 5 | groupedMoE_prep | 3163 | 9.078 | 0.7% |
| 5 | QSA_indexer | 372 | 5.914 | 0.4% |

### Runtime API totals

| Category | Calls | Sum ms |
|---|---:|---:|
| graph_launch | 428 | 619.670 |
| kernel_launch | 2698 | 21.933 |
| graph_instantiate | 7 | 10.976 |
| memcpy | 58 | 0.660 |
| graph_update | 7 | 0.251 |
| graph_capture | 14 | 0.108 |

### Top kernels

| Device | Kernel | Calls | Sum ms | Avg us | Max us |
|---:|---|---:|---:|---:|---:|
| 5 | gemv_sm70_bf16_kernel | 8618 | 270.538 | 31.39 | 51.65 |
| 5 | k_allreduce_p2p | 3103 | 225.680 | 72.73 | 29495.79 |
| 5 | mul_mat_vec_f | 5580 | 68.491 | 12.27 | 21.12 |
| 5 | cpy_scalar_contiguous | 868 | 53.610 | 61.76 | 106.88 |
| 5 | mul_mat_q | 1612 | 44.059 | 27.33 | 46.30 |
| 5 | mul_mat_vec_q_moe | 930 | 41.811 | 44.96 | 46.15 |
| 5 | mul_mat_vec_q_moe | 434 | 38.150 | 87.90 | 96.10 |
| 5 | onecat_xqa_partition_kernel | 496 | 37.244 | 75.09 | 79.94 |
| 5 | pad_f32 | 496 | 26.931 | 54.30 | 55.55 |
| 5 | mul_mat_vec_q_moe | 372 | 25.167 | 67.65 | 82.88 |
| 5 | mul_mat_vec_q | 806 | 25.123 | 31.17 | 35.97 |
| 5 | mul_mat_q | 62 | 22.810 | 367.90 | 370.11 |
| 5 | mul_mat_vec_q | 1488 | 22.449 | 15.09 | 28.80 |
| 5 | gated_delta_net_cuda_128x4_volta | 1054 | 21.645 | 20.54 | 25.28 |
| 5 | k_get_rows | 496 | 20.982 | 42.30 | 45.89 |

### Data completeness

- Diagnostics: Not all NVTX events might have been collected.; Not all CUDA events might have been collected.; Not all OS runtime libraries events might have been collected..
- Nonzero CUPTI runtime API return codes: 0.

## Decode failure / partial capture

The SQLite export records six CUDA contexts but only device 5 has completed kernel activity rows; device 2 has no kernel row, so its last valid kernel cannot be identified from this export. The server log separately records the illegal-memory error and the failing cudaStreamSynchronize on device 2. The CUPTI runtime returnValue column contains no nonzero codes.
decode4 stop CLI output: Collection stop is not allowed in this state.

Do not interpret device-duration sums as wall-time contributions or add GPU totals as if they were critical-path time. API duration sums also overlap across host threads.
