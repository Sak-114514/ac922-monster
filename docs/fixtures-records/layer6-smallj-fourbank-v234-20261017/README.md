# V234 four-flow small-J comparison

V234 reuses the exact frozen V229 four-bank runner executable and scenario
file. It loads the model once using the frozen V232 libraries, enables
`GGML_CUDA_MOE_SMALL_J=1`, retains device-property caching, raises the
pinned-input minimum to 16 MiB under a 1024 MiB cap, and keeps
`GGML_CUDA_GRAPHS_FORCE=0`. All other V229 fast-path settings stay inherited.

The run remains a four-canary native semantic gate followed by the same
four-flow 16K prompt/512-output case. The long outputs remain unscored. The
small-J path is confirmed only if the runtime log contains
`path=mmq_moe_small_j requested=1 selected=1` with the expected
`gpu_prefix_J16_cold_J32_hot_original_dot_kernels` reason.

`summarize-v234-v229.py` compares every output token ID across the four flows,
per-flow and aggregate MTP proposed/accepted counts, full prefill wall PP,
target-prefill service rate, shared-window TG, inclusive emission-span TG, and
whole-scenario output rate. It labels the inclusive emission span as including
serialized warmup and tail; it does not turn per-flow averages into aggregate
throughput.

The launcher creates `results/` before writing its local manifest and uses the
required `root-v234-inband-<UTC stamp>` model unit prefix. The V231 short
operator fixture must release the GPUs before root invokes the launcher. This
package was prepared locally only; it did not build, contact AC922, load a
model, or start the guard/service.
