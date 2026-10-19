# R396 direct host launch of the identical captured child

CUDA12.4 Programming Guide 3.2.8.7.7 explicitly permits host launching DeviceLaunch-flag graphs. Separate helper changes only controller launch to the already-owned child; no host/device mixed launch and no arithmetic or tensor changes. Keep controller allocation/ready invariants, all retirement/fallback/cap32 protection. First run actual GGML ABA plus CUPTI to determine whether child kernels become visible; no whole-model speed claim from this probe. OriginalR5 retained. New full-model candidate only after helper gates pass; no driver/kernel/toolkit upgrades.

Source: https://docs.nvidia.com/cuda/archive/12.4.1/cuda-c-programming-guide/index.html#device-graph-launch
