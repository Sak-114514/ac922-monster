# CUDA structural signature flat-snapshot cache probe

This is a CPU-only preparation fixture. It does not modify `src/`, the shared
tree, CUDA code, a build directory, or a running model.

## Source audited

Remote private source at `/home/debian/ac922-monster/experiments/standard-opt-20261014`:

- `ggml-cuda.cu` SHA256: `8590a6fa301ec3fc3749fd44418ace079b245dd30340900224753064b6d02cbc`
- `graph-structural-signature.h` SHA256: `763337e076cf0bea53eb9db570af5a1a5f53ccd3f688f6265a03d4ab6cba98d3`
- The exact signature header is copied under `reference/` for this fixture.

Relevant seams in the audited `ggml-cuda.cu`:

- `ggml_cuda_graph_tensor_input()` around line 4022 snapshots descriptor,
  source/view identities, `extra`, CUDA per-device data pointers, and event
  pointers. Non-CUDA opaque extras reject the structural path.
- `ggml_cuda_graph_make_structural_signature()` around line 4088 walks every
  graph node and external source/view closure, then canonicalizes and checks
  cycles with `build_signature()`.
- `ggml_cuda_graph_get_key()` around line 4234 keeps the raw-key fast path only
  for a nonzero matching UID already present in `cuda_graphs`. Otherwise it
  rebuilds the full signature before checking active graphs and FT variants.
- Callers include graph readiness (~1775), graph compute (~7362), and graph
  optimization (~7568). The optimize call requests no output signature, but
  still pays key selection and structural lookup.
- `ggml_cuda_graph_structural_drop_context()` (~1132), called by the CUDA
  context destructor (~1164), is the lifecycle seam for any per-context slot.

## Proposed cache boundary

Cache one immutable snapshot per live CUDA context, keyed by `(raw_key, uid)`;
only nonzero UIDs are eligible. Before a hit:

1. Compare current `cgraph->nodes` count and ordered pointer identities with the
   snapshot. A mismatch returns to the existing full builder without reading a
   cached pointer.
2. Read each *current* graph node through the existing tensor-input extractor
   and compare every captured descriptor field, all source/view edges, and
   `extra_state_valid` exactly.
3. Walk the external closure in current first-use order. A descendant pointer
   is read only after it was obtained from an already validated current parent
   edge and matched to the expected cached identity. Its descriptor, edges,
   CUDA device bindings, and event bindings must also match exactly.
4. On any miss, unsupported opaque extra, cycle/cap issue, or mismatch, use the
   existing `make_structural_signature()` path. A hit supplies only the exact
   canonical signature; existing active/variant comparison, readiness,
   compatibility, warmup, and replay checks remain authoritative.

Do not return `raw_key` as an alias or use UID alone to choose a graph. The
cache stores addresses as integers for equality only; it never dereferences a
descendant from the old snapshot. Replace the previous snapshot on successful
capture, retain at most one per context, and clear it in
`ggml_cuda_graph_structural_drop_context()`. The fixture uses an 8 MiB retained
snapshot ceiling; a larger candidate simply skips caching. A production
implementation should hold an immutable snapshot through `shared_ptr` after a
short registry lock, then validate outside that lock.

`tensor_input` does not hash or snapshot tensor payload bytes. That matches the
existing structural contract: input values may change in-place while the exact
graph and bindings stay the same. EP-dependent device/event bindings are
covered only to the extent the current extractor exposes them; this proposal
does not add a new interpretation of opaque EP state.

## CPU verification

```sh
c++ -std=c++17 -O2 -Wall -Wextra -Werror \
  -I monster-engine/standard-opt-20261014/fixtures/structural-cache-probe-20261016 \
  -I monster-engine/standard-opt-20261014/fixtures/structural-cache-probe-20261016/reference \
  -I monster-engine/standard-opt-20261014/src/ggml/include \
  monster-engine/standard-opt-20261014/fixtures/structural-cache-probe-20261016/test.cpp \
  -o /tmp/ac922-structural-cache-probe
/tmp/ac922-structural-cache-probe --bench
```

ASan+UBSan build and run:

```sh
c++ -std=c++17 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
  -I monster-engine/standard-opt-20261014/fixtures/structural-cache-probe-20261016 \
  -I monster-engine/standard-opt-20261014/fixtures/structural-cache-probe-20261016/reference \
  -I monster-engine/standard-opt-20261014/src/ggml/include \
  monster-engine/standard-opt-20261014/fixtures/structural-cache-probe-20261016/test.cpp \
  -o /tmp/ac922-structural-cache-probe-asan
/tmp/ac922-structural-cache-probe-asan
```

Latest local run: 44/44 correctness checks passed under both normal and
ASan+UBSan builds. Checks include every tensor descriptor field, source/view
rewiring, opaque extras, CUDA data/event state, changed raw key/UID/node order,
same-address ABA, parent-edge-before-descendant-read ordering, external closure
cap, retained-byte cap, and replacement/context-drop lifetime.

Synthetic microbench (1024 graph nodes + 256 external descriptors, 200
iterations; includes the copy required by the current `get_key` output API;
median of five runs):

```text
rebuild_us=55,897 validate_us=31,142 ratio=1.796
```

This is only the standalone CPU helper cost; it is not a full production
signature trace, CUDA graph, or model speed result. Before integration, verify
with a trace that stable `(context, raw_key, uid, ordered node identities)`
repeats account for a material share of the measured signature-build time.

## Integration status

Review-only integration patch:
`patches/graph-structural-flat-snapshot-candidate.patch`. It adds the helper at
the CUDA source include path, captures validated input snapshots after the
existing successful canonical build, tries the one-slot snapshot before the
full build, logs hit/miss counts, and drops the slot with the existing CUDA
context registry. `patch --dry-run -p1` passed against the audited source copy.
The patch is not applied to `src/`, has not been CUDA-compiled, and is not
production-ready. Keep it deferred if actual CUPTI/CPU timing does not show
this path is material.
