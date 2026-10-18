#pragma once
#include "ggml.h"
#include <cuda_runtime.h>
#include <cstddef>

// Internal experimental interface, initially used only by the numerical fixture.
struct monomoe_sm70_params {
    const float * input;
    const void * router;
    const void * gate;
    const void * up;
    const void * down;
    float * output;
    ggml_type router_type;
    ggml_type gate_type;
    ggml_type down_type;
    int hidden, ffn, experts, selected, tokens;
    size_t router_row, gate_row, gate_expert, up_row, up_expert, down_row, down_expert;
    float routing_scale;
    bool normalize_selected;
};
size_t monomoe_sm70_workspace_size(const monomoe_sm70_params & p);
cudaError_t monomoe_sm70_launch(const monomoe_sm70_params & p, void * workspace,
                               size_t workspace_bytes, cudaStream_t stream);
