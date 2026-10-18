#pragma once
#include <cstdint>

// Host-side dispatch/capture telemetry only. It must never change a kernel,
// synchronization, or fallback decision; graph replay does not re-run this hook.
void ggml_cuda_optimization_path_log(int device, const char * path, bool requested, bool selected,
                                    const char * reason, int64_t tokens, int64_t width, int64_t auxiliary);
