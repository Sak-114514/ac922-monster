# Coverage and limits

## Prepared assertions

- Seven-backend layout: six distinct CUDA backend instances plus one CPU backend, with exactly four scheduler copies.
- Fixture admission predicate rejects a two-CUDA topology and any copy count other than four.
- The root graph is captured, initially allocated with a full split, then restored once before each of 24 CPU input rewrites.
- Every frame must take the plan API success path, seal bindings, and receive a fresh root UID; the runner checks logged root and split UID uniqueness.
- The frame pattern is A/B/A repeated eight times. Each frame submits asynchronously and poisons the CPU `GET_ROWS` result after the scheduler has staged it for CUDA0.
- The final pinned output readback has 24 independent canary-bounded slots and is compared exactly against a host-side deterministic CPU oracle.
- Each of the six split inputs must cycle through four distinct physical destination tensors. After the final drain, each slot's latest data is checked against the expected pre-ADD value for that edge.
- A changed output layout must return the private API's pre-mutation decline result, preserve root edges and UID, and allow a normal full-split fallback.
- Runner checks the R389 independent pinned upload, eight-slot source-wait summary, peer-fence activation, and plan helper-field logs.

## Not covered

- No compilation, host link, or runtime fixture result exists yet.
- The fixture does not measure SM occupancy, throughput, or model-quality behavior.
- It is only a primitive gate for later R5 device-tail work; it does not exercise R398 capture/replay scope or any model tail.
- The two-CUDA decline is a fixture-side predicate check; it does not invoke the private API or prove the production `llama-context.cpp` gate.
- Rate-limited peer-fence logs show activation but do not report a complete wait count. The 24-frame asynchronous result and per-slot oracle checks provide the data-integrity assertion.
- CPU and exact primitive arithmetic do not prove safety for arbitrary model graphs, Meta/MTP paths, eight-copy rings, or CUDA graph capture.
- The source headers and libraries must be pinned after the root agent reloads R402; current active source did not yet contain the private API header.
