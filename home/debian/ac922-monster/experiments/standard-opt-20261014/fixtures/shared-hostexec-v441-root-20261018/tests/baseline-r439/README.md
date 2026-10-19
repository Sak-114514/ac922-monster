# V439 six-device fixtures

The direct CUDA fixture calls V3 prepare/can-prepare and the unchanged V2
host-child launch API on six distinct devices. It checks prepare-only output
sentinels, exact repeat preparation, same-key in-flight decline, safe rebind
after completion, the 32-entry cache bound, unsupported-node decline, and the
exact graph output oracle. Its runner stores stdout/stderr separately and
requires the host-child single-exec marker.

The scheduler fixture reuses the R436 staged runtime interface without editing
engine sources: CPU-resident `GET_ROWS`, then six explicitly assigned CUDA
`ADD` stages, one frame in default-off baseline and 24 frames with the
parallel-prepare scheduler gate. The checker requires six flags=0 prepares,
the R398 host-child marker, exact CPU oracle outputs, and 24 actual
`selected=1 prepared=6 workers=6` markers on the parallel case. It accepts the
two streams as unordered evidence but retains all 24 complete frame-line
assertions.

After root releases the helper/runtime build and run windows, use:

```sh
V439_HELPER_BUILD_AUTHORIZED=1 bash helper/build-helper.sh
V439_FIXTURE_BUILD_AUTHORIZED=1 bash tests/build-fixture.sh
V439_FIXTURE_RUN_AUTHORIZED=1 V439_FIXTURE_RUN_TAG=unique-direct bash tests/run-fixture.sh
```

For the GGML scheduler fixture, host-link the same pinned R436 source/runtime
closure used by the V436 fixture, then run its matrix script with the candidate
helper at `V439_SCHED_HELPER_SO`. Do not set a scheduler copy-depth selector;
the fixture accepts the active four- or eight-copy policy.

`AC922_TAIL_STAGE_TIMING=1` can be set before either process starts to collect
child-instantiation wall/thread-CPU diagnostic lines. It is off by default.
The measurement does not establish full-model correctness or a performance
gain. No build or GPU execution is performed by this fixture author.
