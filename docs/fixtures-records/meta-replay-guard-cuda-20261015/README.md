# CUDA 3+3 shared-weight replay-guard fixture

Bounded GPU-only correctness fixture for runtime-perf-v13. It uses the verified six-V100 order: PCI 04:04-06 in island 0 and PCI 35:03-05 in island 1. The fixture constructs two CUDA Meta devices of three members each, then two independent API contexts, each with its own Meta backend pair and scheduler. Two small static weight buffers are shared by both contexts, one per island.

The graph crosses the island boundary with a small vector add, then reduces to one F32 output. It is not a model or performance benchmark. The scheduler-bound gate is checked against actual split graphs; readiness requires every active Meta split to be merged and CUDA-graph ready.

The sequence is: cold A/B allocation; serialized first deferred dispatch and subsequent capture passes for both contexts; hot A replay; B graph switch to an 8-row view while A replays during B's guarded second capture; scheduler reset invalidation; B return to its original 16-row graph; final hot A replay. Every output is compared to the exact CPU scalar expectation and guarded by host canaries.

Build and run on the AC922 host with native CUDA 12.4/driver 550 and runtime-perf-v13:
- build-fixture.sh compiles only this small harness against the frozen V13 libraries;
- run-fixture.sh uses explicit UUID order and writes run-001.log.

No full model, broad load, system setting, driver change, or engine rebuild is part of this fixture. The root runner guard must remain in place until this gate passes and root reviews the result.
