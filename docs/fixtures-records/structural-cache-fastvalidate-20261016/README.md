# CUDA graph snapshot fast-validation probe

Status: CPU-only candidate, **not integrated**. This fixture tests the only safe
shortcut found for repeated snapshot hits: keep checking every live tensor
descriptor and every source/view edge, but avoid constructing temporary
`tensor_input` vectors and allocating graph/external hash sets on each hit.
It does not use a generation counter, trust UID as semantic identity, change
canonical comparison, or authorize replay.

## Source findings

Current V56 files were inspected on AC922. `ggml_cuda_graph_tensor_input`
copies type, buffer, dimensions/strides, op parameters, flags, view offset,
data pointer, name, opaque extra identity, padding, all source/view pointers,
and CUDA data-device plus stream-event identities
(`src/ggml/src/ggml-cuda/ggml-cuda.cu:4033-4076`). The current validator still
must compare those live values to preserve binding, mutation, and ABA safety
(`src/ggml/src/ggml-cuda/graph-structural-snapshot-cache.h:185-258`). Without a
reliable version counter incremented by every writer to these fields, a
generation-only shortcut is unsafe.

The current hit path allocates a graph-ID vector and creates graph and external
hash sets plus a pending-closure vector (`ggml-cuda.cu:4274-4290`,
`graph-structural-snapshot-cache.h:212-225`). It invokes the descriptor builder
by value for every current node and external descendant. It then deep-copies
the cached signature even when its `signature` output argument is null
(`ggml-cuda.cu:4292-4295`, `4311-4313`). The latter occurs in
`ggml_backend_cuda_graph_optimize` at `ggml-cuda.cu:7616`. Since the LRU lookup
returns a `shared_ptr`, a read-only pointer into that snapshot remains alive
through the comparisons in `get_key`; on a null-output call the signature copy
can be skipped without changing validation or graph selection.

Full misses still walk the graph and its external closure, construct temporary
descriptors, and build a canonical signature (`ggml-cuda.cu:4144-4202`). This
fixture does not accelerate that miss path.

## Candidate and correctness boundary

`fast-validation-candidate.h` builds sorted graph/external identity indexes at
snapshot-capture time. On every hit it still checks all graph identities,
descriptor bytes, source/view edges, extra-state validity, and CUDA
data/event identities. A fixed 4096-byte bitmap replaces per-hit graph/external
sets and pending vector. It reads an external tensor only after a current,
already-validated parent reaches it. Cached identities are integer keys only;
no old tensor pointer is dereferenced. Index storage is included in the
reported per-snapshot size, but any production integration would also need to
charge it against the current 8 MiB context budget.

The synthetic input includes 18 CUDA extra-state addresses per tensor
(six device slots, one data pointer and two event slots per device), and 256
external descriptors. This approximates the shape of the live descriptor and
event checks; it does not model every GGML topology or CPU-side allocator cost.

## AC922 POWER9 results

Host-only O2 binary used `g++ -std=c++17 -O2 -mcpu=power9 -mtune=power9` and
`nice +15`; CPU max frequency was 2,300,000 kHz before and after. No CUDA,
NVCC, model, or GPU calls were made. The current validator in this comparison
reads a `tensor_input` by value; the candidate uses a stable reference and
prebuilt identity index. Timings are median of five separate runs; each row's
per-run median itself aggregates seven timed batches.

| graph nodes | externals | current validation | candidate validation | validator ratio | signature deep-copy cost | index bytes | snapshot bytes |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 633 | 256 | 335.745 us | 286.996 us | 1.170× | 176.289 us | 9,208 | 915,536 |
| 2,048 | 256 | 831.254 us | 772.581 us | 1.076× | 454.212 us | 20,528 | 2,375,816 |
| 4,000 | 256 | 1,543.308 us | 1,427.561 us | 1.081× | 838.386 us | 36,144 | 4,390,280 |

The 633-node case is the largest graph observed in the V17 real runtime log:
72 structural graph-path records, 38–633 nodes, mean 413.7. 2,048 is the V56
default structural node cap. 4,000 is synthetic stress only, beyond the
default cap; it is not an observed workload size.

Correctness: 31/31 cases passed on POWER9 and under ASan+UBSan. Cases include
all descriptor fields, source and view rewiring, external descendants, opaque
extra validity, event/data state, UID/raw-key mismatch, graph identity changes,
same-address ABA, and rejection before reading an unexpected new descendant.

Artifacts:

- `results/ppc-host-only-20261015T223644Z.log` — POWER9 O2/ASan correctness run
- `results/ppc-bench-copy-5runs-20261016.txt` — five-run POWER9 medians source
- `results/ppc-microbench-full-20261016.txt` — independent full-case smoke

## V17 runtime evidence and decision

The real V17 runtime log is
`/home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/dual-context-v17-prep-20261016/results/p4-d3-20261015T214844Z/runtime.log`.
For the six main graph devices it reports 1,743–1,860 full signature builds
per device and 0–52 flat snapshot hits (49/6/6 on devices 0/1/2; 52/0/0 on
devices 3/4/5). For device 0, 5,391,775 us of `signature_build_us` over 1,860
builds is about 2.90 ms/build. The existing `lookup_us` timing begins only
after snapshot validation and the signature deep copy, so it does **not** time
the work measured in this fixture.

At the observed 633-node maximum, the candidate saves about 49 us for one
successful validation hit; the recorded V17 hit count would imply only a few
milliseconds total across that run. Even an all-hit counterfactual saves about
90 ms over roughly 1,900 calls, while leaving the ~5.4 s of full signature
build time on misses untouched. Current V56 LRU could change the hit rate, but
V18 did not run long enough to establish that.

**Decision:** do not integrate the sorted-index validator. Its only measured
benefit is small, applies only to confirmed cache hits, and the current evidence
shows low hit counts. Preserve the safety boundary. The narrow null-output
signature-copy elimination looks more promising (176 us at the observed
633-node size) and does not weaken validation; evaluate it as a separate small
core change only if V56 live traces show the null-output call is frequent.

The POWER9 validation microbenchmark is a CPU fixture result, not an inference
throughput claim and not a full-model performance result.
