# BF16 panel-width fixture

This fixture reuses the existing private `kernel_sm70_bf16.cuh` implementation.
It adds no kernel or engine changes. `build-host.sh` checks the header SHA-256
before compiling the standalone fixture for SM70 and links only against CUDA
Runtime. It refuses to build while `CUDA_VISIBLE_DEVICES` is set.

The reference uses the original `<4,4,256,false>` kernel in a loop of four
columns. Candidates call the same kernel once over the full activation panel:
`<1,T,256,false>`, `<2,T,256,false>`, or `<4,T,256,false>` for `T` in 12, 16,
24, and 32. Thus the candidate matrix contains 12 configurations, reducing
baseline launch counts 3/4/6/8 to one launch.

Correctness covers K=2560,N=8192; K=10240,N=2560; and the non-four-row tail
K=2560,N=8195. Each candidate uses A/B/A activation panels and compares all
F32 outputs bit for bit with the looped reference. BF16 weights are generated
as finite, nonoverflowing 16-bit values; activations remain F32 with full
mantissas. The fixture checks finite outputs, weight/activation immutability,
and device canaries.

Five CUDA-event timing repetitions cover all 12 configurations on the first
two shapes. Each weight panel exceeds the V100's 6 MiB L2, and the fixture
checks per-case allocations remain below 256 MiB. It reports the reference
launch group and the one-launch candidate separately. These measurements are
operator-fixture results; they do not establish a model-level throughput win.

The source header SHA pinned by the host build is
`cc8edc374923244b47313b675d2c9870378fc70fb29b14725841668bd02185c5`.
No raw weight or activation files are included; deterministic inputs are
generated at runtime.
