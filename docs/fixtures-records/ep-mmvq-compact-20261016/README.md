# EP-MMVQ compact-route operator fixture

This fixture compares the stock route path with V84's default-off
`GGML_CUDA_EP_MMVQ_COMPACT=1` path using the seven raw expert-170 samples in
`fixtures/monpack-ep-readiness-20261016/operator-fixture/samples/`. The sample
verifier checks manifest SHA-256 values without reading a GGUF.

Each sample is reused as an intact quantized row and cyclically permuted to
make a synthetic local E=85 tensor. No block is decoded, edited, or
requantized. These are not the model's actual 85 expert weights. Each test
uses `[K,N,E=85]` weights, EP metadata `(offset=170, global_experts=512)`, and
top-10 global IDs. Each token routes one owned expert plus nine foreign
experts. A/B/A runs change activation and ID contents on the same graph
buffers.

IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, and IQ3_S use fused gate/up plus SwiGLU at
K=2560/N=640. IQ4_NL and Q2_0 use plain down projection at K=640/N=2560. Both
T=4 and T=8 are tested. Candidate marker validation parses summary fields
(`requested == selected > 0`) and checks the fixture's T4/T8 `PASS case`
records. The logger limits detailed marker lines, so detailed lines are not
used to decide shape coverage. IQ2_S must report compact `selected=0` with
the native-IQ2S fallback reason; its outputs are still checked for exact
OFF/ON parity.

After each dtype's A/B/A correctness gates, the binary reports five warm
`ggml_backend_graph_compute` wall timings for T4 and T8. The timed section
does no tensor or weight readbacks. Timings are tiny fixture measurements and
do not establish model performance.

## Build and run

Use only a frozen V84 runtime. The host build uses the frozen V84 CUDA
library; it does not rebuild engine code:

```sh
LLAMA_SRC_ROOT=/home/debian/ac922-monster/experiments/standard-opt-20261014/src \
LLAMA_RUNTIME_LIB=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v84/lib \
LLAMA_BUILD_NOTE=/home/debian/ac922-monster/experiments/standard-opt-20261014/notes/V84-build.json \
LLAMA_BUILD_ID=20261016T082325.972432Z \
./fixtures/ep-mmvq-compact-20261016/build-host.sh \
  /home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/ep-mmvq-compact-20261016/ep-mmvq-compact-v84-timing
```

The bounded runner requires an explicit root grant and one assigned GPU. It
reuses the preserved IQ2_XXS result in `results/20261016T083459Z` by default,
so it will not rerun that dtype:

```sh
ROOT_EP_MMVQ_COMPACT_GPU_WINDOW_AUTHORIZED=1 \
LLAMA_SRC_ROOT=/home/debian/ac922-monster/experiments/standard-opt-20261014/src \
LLAMA_RUNTIME_LIB=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v84/lib \
LLAMA_BUILD_NOTE=/home/debian/ac922-monster/experiments/standard-opt-20261014/notes/V84-build.json \
LLAMA_BUILD_ID=20261016T082325.972432Z \
CUDA_VISIBLE_DEVICES=GPU_ID \
./fixtures/ep-mmvq-compact-20261016/run-cuda-parity.sh \
  /home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/ep-mmvq-compact-20261016/ep-mmvq-compact-v84-timing
```

The runner preserves a 512 MiB free-memory floor and 58 C ceiling on the
selected GPU. It does not change fan or BMC state and does not start a
watcher. Full results and scope limits are in [RESULTS.md](RESULTS.md).
