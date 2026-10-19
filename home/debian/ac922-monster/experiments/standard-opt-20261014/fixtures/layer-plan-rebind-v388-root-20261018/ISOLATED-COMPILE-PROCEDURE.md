# Root-only isolated backend compile procedure

This is a recipe for later use, after V395 has finished and the root agent confirms the shared build tree is idle. It is based on saved R393 build metadata and pinned source/header hashes. No compile, link, or fixture run was performed here.

## R393 provenance

R393 is `runtime-perf-v393`, build `20261018T025135.968927Z`. The R393 report is `notes/V393-build.json`, SHA-256 `c3166e9d2e522b9e81ed010873452369d46d3647163dc962abd6f4527fcb14d0`; `notes/v393-gates.json` records that report hash.

The R393 frozen libraries hash to:

```text
libggml-base.so.0  3a1ca698839224d0128481f09e15e66ee06aae698cff524ea693b9c3d5eeed8f
libggml-cpu.so.0   63525deba1db08199b25852ba71c9115961fdd4db8632f7defe88172613bd31c
```

The exact `ggml-backend.cpp` source hash is `649e7ee75a110fb87de087476b6fbdc9f1654dba546688f90645c40dbc6d0138`; the pinned include closure is [`source-current/ggml-headers.sha256`](source-current/ggml-headers.sha256). The full patched source is staged in `candidate/`, and the private rebind header sits next to it.

The final R393 report says the backend object was reused from successful compile run `20261018T024850.572378Z`, then linked into the frozen library by final run `20261018T025135.968927Z`. The exact object compile argv is in `source_closure_compile_plan` and the reused compile result; the final link line is in the R393 `ggml/src:ggml-base` link result.

## Exact R393 commands

The R393 backend compile command was:

```sh
/usr/bin/c++ \
  -DGGML_BUILD -DGGML_SCHED_MAX_COPIES=4 -DGGML_SHARED -DGGML_USE_OPENMP \
  -D_GNU_SOURCE -D_XOPEN_SOURCE=600 -Dggml_base_EXPORTS \
  -I/home/debian/ac922-monster/experiments/standard-opt-20261014/src/ggml/src/. \
  -I/home/debian/ac922-monster/experiments/standard-opt-20261014/build-reuse/ggml/src \
  -I/home/debian/ac922-monster/experiments/standard-opt-20261014/src/ggml/src/../include \
  -O3 -DNDEBUG -std=gnu++17 -fPIC \
  -Wmissing-declarations -Wmissing-noreturn -Wall -Wextra -Wpedantic -Wcast-qual \
  -Wno-unused-function -Wno-array-bounds -Wextra-semi -fopenmp \
  -o CMakeFiles/ggml-base.dir/ggml-backend.cpp.o \
  -c /home/debian/ac922-monster/experiments/standard-opt-20261014/src/ggml/src/ggml-backend.cpp
```

The R393 base link command was:

```sh
/usr/bin/c++ -fPIC -O3 -DNDEBUG -shared -Wl,-soname,libggml-base.so.0 \
  -o ../../bin/libggml-base.so.0.25.1 \
  CMakeFiles/ggml-base.dir/ggml.c.o \
  CMakeFiles/ggml-base.dir/ggml.cpp.o \
  CMakeFiles/ggml-base.dir/ggml-alloc.c.o \
  CMakeFiles/ggml-base.dir/ggml-backend.cpp.o \
  CMakeFiles/ggml-base.dir/ggml-backend-meta.cpp.o \
  CMakeFiles/ggml-base.dir/ggml-opt.cpp.o \
  CMakeFiles/ggml-base.dir/ggml-threading.cpp.o \
  CMakeFiles/ggml-base.dir/ggml-quants.c.o \
  CMakeFiles/ggml-base.dir/gguf.cpp.o \
  -lm /usr/lib/gcc/powerpc64le-linux-gnu/12/libgomp.so \
  /usr/lib/powerpc64le-linux-gnu/libpthread.a
```

## Temporary object replacement

After V395 ends, copy the R393 `ggml-base` object closure into a new temporary tree. Keep the copied backend object under a second name. First relink that unchanged object set and require the resulting base library SHA-256 to equal the frozen R393 hash above. This check detects build-reuse objects that changed after R393; if the hash differs, stop and recover a matching R393 object closure.

```sh
BASE=/home/debian/ac922-monster/experiments/standard-opt-20261014
FIXTURE=$BASE/fixtures/layer-plan-rebind-v388-root-20261018
RUNTIME=$BASE/runtime-perf-v393
OBJDIR=$BASE/build-reuse/ggml/src/CMakeFiles/ggml-base.dir
TMP=$(mktemp -d /tmp/layer-plan-rebind-v388.XXXXXX)
mkdir -p "$TMP/ggml/src/CMakeFiles/ggml-base.dir" "$TMP/bin" "$TMP/lib"
cp "$OBJDIR"/*.o "$TMP/ggml/src/CMakeFiles/ggml-base.dir/"
cp "$TMP/ggml/src/CMakeFiles/ggml-base.dir/ggml-backend.cpp.o" \
   "$TMP/ggml/src/CMakeFiles/ggml-base.dir/ggml-backend.cpp.r393.o"
```

From `$TMP/ggml/src`, run the recorded link line with the original temporary `ggml-backend.cpp.o`. Then compile the staged candidate source into only that temporary object path, preserving the R393 flags and replacing source/include roots with the pinned fixture files:

```sh
cd "$TMP/ggml/src"
/usr/bin/c++ \
  -DGGML_BUILD -DGGML_SCHED_MAX_COPIES=4 -DGGML_SHARED -DGGML_USE_OPENMP \
  -D_GNU_SOURCE -D_XOPEN_SOURCE=600 -Dggml_base_EXPORTS \
  -I"$FIXTURE/source-current/src/ggml/src/." \
  -I"$BASE/build-reuse/ggml/src" \
  -I"$FIXTURE/source-current/src/ggml/include" \
  -O3 -DNDEBUG -std=gnu++17 -fPIC \
  -Wmissing-declarations -Wmissing-noreturn -Wall -Wextra -Wpedantic -Wcast-qual \
  -Wno-unused-function -Wno-array-bounds -Wextra-semi -fopenmp \
  -o CMakeFiles/ggml-base.dir/ggml-backend.cpp.o \
  -c "$FIXTURE/candidate/ggml-backend.cpp"
```

Rerun the same recorded link line. Only the temporary `ggml-backend.cpp.o` has changed; the other base objects remain the verified R393 objects. Copy the resulting library into an isolated directory and link the fixture against it plus the frozen R393 CPU library:

```sh
cp "$TMP/bin/libggml-base.so.0.25.1" "$TMP/lib/libggml-base.so.0.25.1"
ln -s libggml-base.so.0.25.1 "$TMP/lib/libggml-base.so.0"
ln -s libggml-base.so.0 "$TMP/lib/libggml-base.so"
ln -s "$RUNTIME/lib/libggml-cpu.so.0" "$TMP/lib/libggml-cpu.so.0"
GGML_SOURCE_ROOT="$FIXTURE/source-current/src" \
GGML_LIBRARY_DIR="$TMP/lib" \
GGML_FIXTURE_BUILD_DIR="$TMP/fixture-build" \
  "$FIXTURE/build-cpu-fixture.sh"
LD_LIBRARY_PATH="$TMP/lib:$RUNTIME/lib" \
GGML_FIXTURE_BUILD_DIR="$TMP/fixture-build" \
  "$FIXTURE/run-cpu-fixture.sh"
```

The isolated fixture uses mock CPU devices and the real scheduler/gallocr/CPU graph executor. It needs no model and should not initialize or use CUDA. Keep the procedure idle until V395 is complete.
