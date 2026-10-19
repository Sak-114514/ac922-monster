#!/usr/bin/env bash
set -euo pipefail

BASE=${AC922_EXPERIMENT_ROOT:-/home/debian/ac922-monster/experiments/standard-opt-20261014}
FIXTURE=$BASE/production/diagnostic-graphlevel-20261019
HELPERS=$BASE/fixtures/layer6-cupti32k-v223-20261017
BUILD=$FIXTURE/build-graphlevel
CUDA=${CUDA_HOME:-/usr/local/cuda-12.4}
CUPTI=$CUDA/extras/CUPTI

mkdir -p "$BUILD"
${CXX:-g++} -std=c++11 -O2 -Wall -Wextra -fPIC -fvisibility=hidden -shared -Wl,-z,defs \
    -I"$CUDA/include" -I"$CUPTI/include" -I"$HELPERS" \
    "$FIXTURE/graphlevel-collector.cpp" \
    -L"$CUPTI/lib64" -Wl,-rpath,"$CUPTI/lib64" -lcupti -pthread \
    -o "$FIXTURE/libac922-cupti-graphlevel.so" \
    >"$BUILD/collector-compile.log" 2>&1

ldd "$FIXTURE/libac922-cupti-graphlevel.so" >"$BUILD/collector-ldd.txt"
sha256sum "$FIXTURE/graphlevel-collector.cpp" "$FIXTURE/libac922-cupti-graphlevel.so" \
    >"$BUILD/collector-sha256.txt"
nm -S -C "$FIXTURE/libac922-cupti-graphlevel.so" >"$BUILD/collector-symbol-sizes.txt"
printf 'built %s\n' "$FIXTURE/libac922-cupti-graphlevel.so"
