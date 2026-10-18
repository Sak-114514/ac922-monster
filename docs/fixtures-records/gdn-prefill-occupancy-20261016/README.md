# GDN prefill occupancy fixture

This isolated public-GGML fixture measures the existing scalar-gate GDN operator
at the shapes where the SM70 X4 kernel changes the third grid dimension from
ceil(S_v / 4) = 32 to 8 for S_v = 128. It does not change GDN or model source.

The K=1 cases use H = 6, 12; n_seqs = 1, 4; one-sequence T = 1024, 4096;
four-sequence T = 1024; and one T = 1025 tail case. A K=5 snapshot case uses
H=12, n_seqs=4, T=1024, matching the four-lane rollback snapshot count used by
the target recurrent cache. Every case uses F32 state, scalar log-gate, and
FP32 Q/K vectors pre-normalized to unit L2 length with the Qwen
RMSNorm(eps / S) times 1/sqrt(S) contract. The F64 oracle replays the scalar
recurrence from those exact FP32 inputs, includes the GDN output factor
1/sqrt(S), and checks attention rows plus every requested state snapshot with
atol = 5e-4 and rtol = 5e-4.

Each case runs an A/B/A input sequence on one graph. It checks exact A replay,
verifies the read-only initial-state tensor and its guard sequence remain
bitwise unchanged, and surrounds the copied output with device-side canaries.
The canonical binary file contains A, B, and replayed A values in a fixed case
order, with no timing or backend-mode fields, so CUDA OFF and ON files can be
compared with cmp.

The program reports the backend allocation for each case and refuses any
single allocation at or above 512 MiB. Cases run serially and each CUDA run
uses one process at a time. This bound covers the fixture's GGML backend tensor
buffer; it does not count the CUDA context or other loaded runtime components.

## Host build

On AC922, the CPU-only build links against the V95 runtime and does not load the
CUDA backend:

    RUNTIME_DIR=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v95 \
      ./build-gdn-prefill-occupancy.sh cpu

The host-only compile check is also safe before any GPU window:

    SRC_ROOT=/home/debian/ac922-monster/experiments/standard-opt-20261014/src
    c++ -std=c++17 -Wall -Wextra -Werror -fsyntax-only \
      -I"$SRC_ROOT/ggml/include" gdn-prefill-occupancy.cpp

The CPU executable can run the same F64, A/B/A, and canary checks on the host.
Its timings describe CPU execution and do not predict CUDA occupancy.

## Full CUDA OFF/ON comparison

The default binary mode performs the full F64 oracle for every case and every
A/B/A replay. This is the reference path. It takes longer than the bounded
GPU window used for the follow-up check.

Run only inside an explicitly opened root GPU window. The full runner has a
grant token guard, uses one GPU process at a time, asks the runtime to log the
actual volta_gdn_x4 dispatch, and compares the complete output files bitwise:

    GDN_OCCUPANCY_GPU_GRANT=ROOT_GRANTED \
    RUNTIME_DIR=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v95 \
      ./run-gdn-prefill-occupancy.sh cuda0

The full output file contains all eight cases. The reported latency is the
median host wall time around
ggml_backend_graph_compute() plus ggml_backend_synchronize() for seven
repetitions after two warmups. It excludes input upload and output readback.
The per-case grid CTA counts are estimates from the launch shapes. The runtime
optimization-path record proves whether the X4 path was selected; neither
record measures achieved SM occupancy. This is an operator check, not
model-quality or end-to-end prefill evidence. No H=12, n_seqs=4, T=4096
allocation is attempted.

## Bounded X4 rerun against the validated OFF reference

The saved GPU3 OFF run completed all eight cases with the full F64 oracle,
A/B/A, and canary checks. Its canonical binary is retained on AC922 at
/home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/gdn-prefill-occupancy-20261016/results/root-gpu3-20261016T130844Z/off.bin
with SHA256
1f64287292fdb9b21ca65d3b161e968aa6edbc02f226a2cb68ad168cf03e76eb. The
462 MiB binary is not copied into this fixture's workspace archive; the small
OFF log and hash record are retained under results/root-gpu3-20261016T130844Z.

The quick mode keeps the same eight-case canonical output layout, A/B/A
replay, canaries, and timing. It skips the duplicated long-shape F64 loops,
then runs a separate H=6, n_seqs=4, T=128, K=5 A/B F64 gate into a separate
file. The quick output is compared byte-for-byte with the validated OFF binary,
so long-shape X4 output inherits the completed OFF F64 result. The separate
short F64 file is not included in the OFF comparison.

Use a fresh external output directory and the retained OFF reference:

    GDN_OCCUPANCY_GPU_GRANT=ROOT_GRANTED \
    RUNTIME_DIR=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v95 \
    OUT_DIR=/tmp/gdn-prefill-occupancy-v96-gpu3 \
      ./run-gdn-prefill-occupancy-quick.sh \
      /home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/gdn-prefill-occupancy-20261016/results/root-gpu3-20261016T130844Z/off.bin

The quick runner validates the pinned source/runtime closure, requires the
X4-selected path log, checks the short F64 gate, and compares the canonical
eight-case output file with the retained OFF reference. The output directory
must be new for each run so logs and result tags are not reused.
