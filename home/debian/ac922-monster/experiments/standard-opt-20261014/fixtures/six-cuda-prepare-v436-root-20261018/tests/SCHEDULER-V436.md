# V436 six-CUDA scheduler fixture

This fixture exercises the staged V436 pre-capture hook on an actual GGML
scheduler graph. It builds a CPU-resident `GET_ROWS` root followed by six
explicitly assigned CUDA `ADD` stages, with one resident bias tensor per GPU.
The scheduler backend list keeps CPU last as required by GGML; explicit node
assignments make the expected split order CPU, CUDA 0 through CUDA 5.

`scheduler6-fixture-v436.cpp` first calls the private CUDA prepare proc on six
small, separately allocated CUDA `ADD` graphs. Each graph has a `-777` output
sentinel checked after initial and repeated preparation, then runs normally and
must match an exact CPU oracle. The fixture sets the CUDA capture hint on the
parent thread through the backend registry proc and verifies it remains set
across the direct calls and scheduler submit before restoring its previous
value.

The scheduler case then computes exact CPU-oracle outputs for one baseline
frame with scheduler preparation disabled and 24 frames with preparation
enabled. Each scheduler frame has new graph metadata and a distinct CPU row-ID
payload. The existing R411 copy-depth choice is preserved; the fixture accepts
four or eight copies and reports which policy was active.

## Host link and run

After the root releases a frozen R436 runtime, host-link without building or
changing engine sources:

```sh
V436_SCHED_HOSTLINK_AUTHORIZED=1 \
GGML_SRC_ROOT=/path/to/pinned/source \
RUNTIME_ROOT=/path/to/frozen/runtime \
V436_SCHED_BUILD_TAG=unique-tag \
bash tests/build-scheduler6-host.sh
```

After the separately authorized six-GPU run window, set the binary, backend
plugin directory, frozen runtime, and helper v3 library paths, then run:

```sh
V436_SCHED_FIXTURE_RUN_AUTHORIZED=1 \
V436_SCHED_BINARY=/path/to/scheduler6-fixture-v436 \
V436_SCHED_PLUGIN_DIR=/path/to/isolated/backend-plugins \
V436_SCHED_RUNTIME_ROOT=/path/to/frozen/runtime \
V436_SCHED_HELPER_SO=/path/to/libac922-tail-bridge-v3.so \
V436_SCHED_RUN_TAG=unique-tag \
bash tests/run-scheduler6-matrix.sh
```

The runner does not set a scheduler copy-depth selector. Its checker requires
six direct sentinel/oracle markers, one exact scheduler frame in baseline,
24 exact scheduler frames in parallel, and 24 actual
`layer-parallel-prepare requested=1 selected=1 prepared=6 workers=6` markers.
The baseline must have no scheduler prepare marker. These checks do not treat
the environment variable alone as proof of scheduler selection.

## Evidence boundary

This is an isolated CPU-host-link and six-GPU fixture proposal, not full-model
or end-to-end performance evidence. The CUDA prepare-only calls assert that
they do not execute graph math; ordinary graph execution separately checks
exact outputs. CPU-only scheduler tests cannot establish CUDA event or
cross-device concurrency safety. No build or GPU run is performed by the
fixture author; root must link and run it against the pinned R436 runtime,
preserving runtime provenance and the requested memory/device guards.
