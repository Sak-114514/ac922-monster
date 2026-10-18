// llama-monpack-loader.h — P3 monpack v2 sidecar loader (engine integration)
//
// Loads `<primary-gguf>.monpack` next to the model, validates it against the
// format constitution (ggml/src/ggml-cuda/monpack/monpack_format.h) and, for
// tensors the sidecar repacked (index family_id > 0), substitutes the weight
// bytes uploaded into the GPU buffer with the packed blob (same byte count,
// invariant I1) and registers the final device pointer with the CUDA-side
// monpack kernel registry (monpack::register_tensor).
//
// Failure policy: any validation problem logs one WARN ("monpack ignored") and
// leaves the engine on the 100% original GGUF path. The GGUF itself is always
// loaded as usual, so rollback == deleting the sidecar file (or MONSTER_MONPACK=0).
//
// Env:
//   MONSTER_MONPACK=0|1           sidecar enabled (default 1)
//   MONSTER_MONPACK_TYPES="20,21" fork-type whitelist (empty/unset = all packed types)
//   MONSTER_MONPACK_VERIFY=1      additionally hash every redirected blob against
//                                 the index blob_sha128 (slow, debug only)
//
// The sidecar mapping is kept for the lifetime of the process (never munmap'd
// mid-run); the OS releases it at exit.
#pragma once

#include <cstddef>

struct ggml_tensor;

namespace llama_monpack {

// Idempotent; called at the top of llama_model_loader::load_all_data with the
// primary GGUF path. No-op when the sidecar is missing, disabled, or invalid,
// and when the build has no monpack CUDA kernels (MONPACK_AVAILABLE unset).
void init(const char * gguf_path);

// When non-null: this tensor's weight bytes must be taken from the sidecar
// blob instead of the GGUF (same byte count), and register_tensor() must be
// called once the tensor's buffer contents are final. Only ever non-null for
// tensors living in a GPU device buffer; everything else keeps the GGUF path.
const void * tensor_src_override(ggml_tensor * t);

// Registers the tensor (whose t->data now holds the packed blob) with the
// monpack kernel registry. No-op for tensors that were not redirected.
void register_tensor(ggml_tensor * t);

// v2 slice-aware path for island/meta buffers: when this tensor lives in a meta
// (island) buffer and the sidecar packed it, every member device's K-slice is
// repacked on the host into its own packed staging buffer (same constitution
// formula, slice geometry, equal byte count), uploaded straight into the member's
// device buffer and registered with the CUDA monpack registry. Returns true when
// the tensor was fully handled — the caller must then skip its own upload of
// this tensor. Returns false (original GGUF path, engine unaffected) when the
// sidecar is off, the tensor is not packed, or any geometry assumption fails.
bool tensor_meta_override(ggml_tensor * t);

} // namespace llama_monpack
