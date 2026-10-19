#!/usr/bin/env bash
set -euo pipefail

if [[ ${AC922_PRODUCTION_BUILD:-} != 1 ]]; then
  echo "refusing V306 host-link: root must set V306_BUILD_AUTHORIZED=1 after the CPU profile/KV gate" >&2
  exit 2
fi

B=${AC922_EXPERIMENT_ROOT:-/home/debian/ac922-monster/experiments/standard-opt-20261014}
F="$B/production/combo-20261018"
S="$B/src"
R="$B/runtime-perf-v444/lib"
REPORT="$B/notes/V444-build.json"
SOURCE="$F/bank-worker-combo.cpp"
EXPECTED_SOURCE_SHA=8c740eaeb24aac21918e2717705b66e6da04799e3e56bfa4ce3de29ba0990a69
EXPECTED_REPORT_SHA=b993f9a857c73586a9ff39b5b655ec5bc1984ae48924b33ef033abcef5daf4f4
O=${AC922_PRODUCTION_BUILD_DIR:-$F/build-combo-c12}

[[ -f "$SOURCE" ]]
[[ "$(sha256sum "$SOURCE" | awk '{print $1}')" == "$EXPECTED_SOURCE_SHA" ]]
[[ "$(sha256sum "$REPORT" | awk '{print $1}')" == "$EXPECTED_REPORT_SHA" ]]
[[ ! -e "$O" ]] || { echo "refusing to overwrite existing V306 build directory: $O" >&2; exit 2; }
python3 - "$REPORT" <<'PY'
import json, sys
report=json.load(open(sys.argv[1]))
assert report.get("run_completed") is True
assert report.get("build_run_id") == "20261018T092503.472988Z"
PY
mkdir -p "$O"

numactl --cpunodebind=0 --membind=0,8 /usr/bin/nice -n 10 /usr/bin/g++ -std=c++17 -O3 -DNDEBUG -pthread \
  -DGGML_BACKEND_SHARED -DGGML_SHARED -DGGML_USE_CPU -DGGML_USE_CUDA -DLLAMA_SHARED \
  -I"/usr/local/cuda-12.4/targets/ppc64le-linux/include" -I"$S/common" -I"$S/src" -I"$S/include" -I"$S/ggml/include" -I"$S/ggml/src" \
  -I"$S/vendor" -I"$S/tools/server" -I"$S/tools/mtmd" \
  -I"$B/build-reuse/include" -I"$B/build-reuse/ggml/include" -I"$B/fixtures" \
  -I"$B/fixtures/long-prefill-combo-v45-20261016" \
  -I"$B/fixtures/dual-context-v35-onebank4-combined-20261016" \
  -I"$B/fixtures/layer6-fourbank-v146-20261017" \
  -I"$B/fixtures/mtp-vocab-fourbank-v221-20261017" -I"$F" \
  "$SOURCE" -rdynamic -fno-omit-frame-pointer -o "$O/bank-worker" \
  -Wl,-rpath,"$R" -Wl,--no-as-needed \
  "$R/libllama-common.so.0" "$R/libmtmd.so.0" "$R/libllama.so.0" \
  "$R/libggml.so.0" "$R/libggml-cpu.so.0" "$R/libggml-cuda.so.0" "$R/libggml-base.so.0" \
  -Wl,--as-needed /usr/local/cuda-12.4/targets/ppc64le-linux/lib/stubs/libcuda.so \
  > "$O/compile.log" 2>&1

LD_LIBRARY_PATH="$R" ldd "$O/bank-worker" > "$O/ldd.txt"
! grep -q 'not found' "$O/ldd.txt"
for name in libllama-common.so.0 libmtmd.so.0 libllama.so.0 libggml.so.0 \
            libggml-cpu.so.0 libggml-cuda.so.0 libggml-base.so.0; do
  grep -F "$name => $R/" "$O/ldd.txt" >/dev/null || {
    echo "V299 runtime identity check failed for $name" >&2
    exit 2
  }
done
sha256sum "$SOURCE" "$F/thermal-pacing.h" "$F/production-ipc.h" "$F/production-shape-policy.h" "$O/bank-worker" "$R/"*.so* "$REPORT" > "$O/sha256.txt"
echo "HOSTLINK_PASS build_dir=$O runtime=$R model_loaded=0 gpu_touched=0"
