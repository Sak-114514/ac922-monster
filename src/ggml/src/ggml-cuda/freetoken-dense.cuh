#pragma once
#include "ggml-cuda-ft.h"
#include "common.cuh"
const ggml_cuda_ft_dense_api * ggml_cuda_ft_dense_get_api();
bool ggml_cuda_ft_dense_supports(const ggml_tensor * op, int device);
void ggml_cuda_ft_dense_compute(ggml_backend_cuda_context & ctx, ggml_tensor * op);
