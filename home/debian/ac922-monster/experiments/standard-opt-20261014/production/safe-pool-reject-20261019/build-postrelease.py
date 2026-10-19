"""Compile the worker against the unchanged, qualified L4 runtime closure."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
SOURCE = BASE / "src"
BUILD = BASE / "build-reuse"
BASELINE = Path("/home/debian/ac922-monster/production/releases/20261019-r444-banked-l4")
OUT = HERE / "build-lastwave"
LIB = OUT / "lib"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if os.environ.get("AC922_POSTRELEASE_BUILD") != "1":
        raise SystemExit("Set AC922_POSTRELEASE_BUILD=1 for the isolated host build")
    started = time.monotonic()
    baseline = json.loads((BASELINE / "build-lastwave/build-report.json").read_text())
    names = list(baseline["libraries"])
    for name in names + ["libac922-tail-bridge.so"]:
        assert sha(LIB / name) == sha(BASELINE / "build-lastwave/lib" / name), name
    for relative, expected in baseline["host_source_sha256"].items():
        assert sha(HERE / "host-src" / relative) == expected, relative
    includes = [SOURCE / "common", SOURCE / "src", SOURCE / "include", SOURCE / "ggml/include",
                SOURCE / "ggml/src", SOURCE / "vendor", SOURCE / "tools/server", SOURCE / "tools/mtmd",
                BUILD / "include", BUILD / "ggml/include", BASE / "fixtures", HERE]
    for child in ["long-prefill-combo-v45-20261016", "dual-context-v35-onebank4-combined-20261016",
                  "layer6-fourbank-v146-20261017", "mtp-vocab-fourbank-v221-20261017"]:
        includes.append(BASE / "fixtures" / child)
    argv = ["/usr/bin/nice", "-n", "5", "/usr/bin/g++", "-std=c++17", "-O3", "-DNDEBUG", "-pthread",
            *["-D" + name for name in ["GGML_BACKEND_SHARED", "GGML_SHARED", "GGML_USE_CPU", "GGML_USE_CUDA", "LLAMA_SHARED"]],
            "-I/usr/local/cuda-12.4/targets/ppc64le-linux/include", *["-I" + str(p) for p in includes],
            str(HERE / "bank-worker-combo.cpp"), "-rdynamic", "-fno-omit-frame-pointer", "-o", str(OUT / "bank-worker"),
            "-Wl,-rpath,$ORIGIN/lib", "-Wl,--no-as-needed", *[str(LIB / name) for name in names],
            "-Wl,--as-needed", "/usr/local/cuda-12.4/targets/ppc64le-linux/lib/stubs/libcuda.so"]
    with (OUT / "worker.compile.log").open("w") as log:
        subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True)
    closure = subprocess.check_output(["ldd", str(OUT / "bank-worker")],
                                     env=dict(os.environ, LD_LIBRARY_PATH=str(LIB)), text=True)
    assert "not found" not in closure
    for name in names:
        assert name + " => " + str(LIB) + "/" in closure, name
    (OUT / "ldd.txt").write_text(closure)
    report = {"status": "BUILD_PASS", "model_loaded": False, "cuda_tus": 0, "host_tus": [],
              "relinked": [], "reused_engine": True, "base_release": str(BASELINE),
              "elapsed_seconds": time.monotonic() - started,
              "worker_sha256": sha(OUT / "bank-worker"), "source_sha256": sha(HERE / "bank-worker-combo.cpp"),
              "headers": {p.name: sha(p) for p in sorted(HERE.glob("*.h"))},
              "host_source_sha256": baseline["host_source_sha256"],
              "libraries": {name: sha(LIB / name) for name in names}}
    (OUT / "build-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
