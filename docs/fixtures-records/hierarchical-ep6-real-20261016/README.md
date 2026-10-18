# Hierarchical EP6 real-GGUF fixture

This fixture checks the V60/V61 loader's nested Meta2{Meta3A,Meta3B} layout
against a flat Meta6 reference. It is fixture-only: it does not load a model
runtime, change model precision, or edit core source. The only GGUF payloads
read are the seven selected routed-expert matrices plus block-0 routed and
shared-expert matrices.

The intended scope is one layer, not a serving benchmark:

- Each of the seven target formats (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S,
  IQ4_NL, Q2_0) is exercised with the real packed bytes and global IDs.
- A 64-token route set covers all 512 global expert IDs in every format. The
  8-token arm covers each physical owner and uses its actual MUL_MAT_ID
  consumer. The 128-token arms cover larger matrix shapes.
- The complete block-0 routed gate/up/down banks are read without pruning.
  The fixture executes gate MUL_MAT_ID + up MUL_MAT_ID + SwiGLU + down
  MUL_MAT_ID, route weighting, slot aggregation, and a final RMSNorm consumer.
  It checks exact foreign +0.0, unique ownership, flat6/nested parity, and
  A/B/A replay.
- The block-0 shared-expert gate/up matrices use flat TP6 as the reference and
  nested TP3 with an outer 2-way island mirror. Physical GGUF bytes are checked
  against the 6-way flat slices and both 3-way nested island slices. Output
  comparison uses one logical flat6 tensor read against each island's logical
  inner-Meta3 tensor read in A/B/A.

The expert partition expected on physical ranks 0..5 is:

    0: [0,85)   1: [85,170)   2: [170,256)
    3: [256,341) 4: [341,426) 5: [426,512)

The nested route IDs stay global. V60/V61 Meta contexts compose the offsets
for each nested MUL_MAT_ID; the fixture does not rewrite route IDs to local
indices. Static weights use ggml_backend_meta_alloc_ctx_tensors_from_buft,
matching the model loader's static-weight allocation path.

## Host-only checks

    ./run-cpu-tests.sh
    ./build-fixture-v61.sh /home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v61
    ./run-gpu-bounded-v61.sh dense-only

The host-link script only compiles and links the fixture. --inspect MODEL.gguf
reads GGUF metadata without initializing CUDA and prints the exact block-0
formats, shapes, and offsets.
The dense-only GPU mode skips all routed format and block-0 tests and runs only
the shared gate/up T=64 flat6-vs-two-island TP3 A/B/A comparison.

## GPU run gate

Do not run the GPU fixture without a separate root-assigned six-GPU window.
The runner requires HIERARCHICAL_EP6_GPU_WINDOW_AUTHORIZED=1, exactly six
unique ordered CUDA UUIDs, no preexisting compute process, less than 512 MiB
preexisting memory per GPU, and temperature below 58 C. During the run it
aborts at 58 C or above 2 GiB used per card, with a 10-minute wall timeout.
It explicitly sets GGML_META_CONTEXT_VIEWS=1,
GGML_META_REPLAY_GUARD=1, and GGML_META_FIFO_PIPELINE=1. The runner validates
these three prerequisites before checking GPU state; nested Meta event
creation returns null when FIFO_PIPELINE is disabled. The log gate requires both 2-member outer and
3-member inner meta-context-submit FIFO markers, the GPU_tail_wait=1 path,
and a logged P2P6/NCCL6 reduction route. A fixture PASS therefore does not
depend on V59 defaults and exercises repeated A/B/A graph submissions on the
same V60/V61 Meta contexts.

## V61 GPU qualification status (2026-10-16)

The authorized bounded runs did not pass. Raw logs are retained under results/
and are not overwritten:

- run-v61-20261016T004701Z.log: the first attempt reached CUDA and created the
  flat6/nested contexts, then aborted in fixture setup because one scheduler
  was asked to allocate two graphs without a reset. This was fixture misuse of
  ggml_backend_sched_alloc_graph, not evidence about EP or V60.
- run-v61-20261016T004914Z.log: the second attempt exposed a second fixture
  omission: final_graph had not been allocated. The source fix is recorded in
  the current fixture.
- run-v61-20261016T005212Z.log: with both graph roots allocated, the flat6
  IQ2_XXS, T=64, A local-output check passed exact foreign-zero for all 640
  routes and six owners. The following nested run aborted at
  ggml-backend-meta.cpp:3328 because the fixture runner omitted
  GGML_META_FIFO_PIPELINE=1. Source line 385 shows Meta event creation returns
  null when that prerequisite is off. This was a runner-environment omission,
  not evidence of a V61 core defect; no nested consumer result was emitted.
- run-v61-20261016T005420Z.log: after explicitly enabling FIFO_PIPELINE, the
  expected outer/inner FIFO events and P2P6 rank-order reduction appeared.
  All seven real target formats passed T=64 flat6/nested A/B/A ownership and
  exact foreign-zero checks; the IQ2_XS T=8/T=128 arms and full block-0
  gate/up/down T=8/T=64 consumer, parity, and ABA checks also passed. Its
  dense comparison incorrectly expected each flat6 leaf to have a TP3 slice;
  the later dense-only attempt corrected this comparator contract.
- run-v61-20261016T005840Z.log: dense-only mode verified actual flat6 TP6
  packed-weight slices and both nested TP3 island slices against GGUF bytes.
  It then called ggml_backend_tensor_get on the flat6 logical dense output,
  which aborted at ggml-backend-meta.cpp:3030 because offset_j did not equal
  chunk_size_full. Nested island logical output comparisons did not run. This
  run used the intended N/6 flat versus N/3 per-island layout; source-slice
  checks passed before the output getter assertion.

After each exit, nvidia-smi showed no remaining compute process, about 1–2 MiB
used per card, and 37–39 C. No full model was loaded. The overall runner gate
did not pass because flat6 dense logical output retrieval stopped at the Meta
offset assertion. The routed-expert and full-block gates listed above passed
individually. Flat6-versus-two-island dense output parity and dense A/B/A remain
unverified. Do not treat the complete fixture as PASS.
