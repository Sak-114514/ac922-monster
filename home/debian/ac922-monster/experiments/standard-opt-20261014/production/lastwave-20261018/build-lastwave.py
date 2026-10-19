"""Incremental host-only build against the exact frozen C12 object closure."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time

HERE = Path(__file__).resolve().parent
BASE = Path(os.environ.get("AC922_EXPERIMENT_ROOT", "/home/debian/ac922-monster/experiments/standard-opt-20261014"))
SOURCE = BASE / "src"
OLD = Path("/home/debian/ac922-monster/src")
BUILD = BASE / "build-reuse"
FALLBACK = Path("/home/debian/ac922-monster/production/releases/20261018-r444-banked-combo-c12-r2")
OUT = HERE / "build-lastwave"
LIB = OUT / "lib"
HOST = HERE / "host-src"
NAMES = ["libllama-common.so.0", "libmtmd.so.0", "libllama.so.0", "libggml.so.0",
         "libggml-cpu.so.0", "libggml-cuda.so.0", "libggml-base.so.0"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(argv: list[str], log: Path, cwd: Path | None = None) -> None:
    with log.open("w") as stream:
        subprocess.run(argv, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, check=True)


def build() -> None:
    if os.environ.get("AC922_LASTWAVE_BUILD") != "1":
        raise SystemExit("Set AC922_LASTWAVE_BUILD=1 for the isolated host build")
    OUT.mkdir(exist_ok=True)
    LIB.mkdir(exist_ok=True)
    manifest = json.loads((FALLBACK / "release-manifest.json").read_text())
    for name in NAMES:
        expected = manifest["sha256"]["lib/" + name]
        actual = digest(FALLBACK / "lib" / name)
        if actual != expected:
            raise RuntimeError(f"frozen runtime changed: {name}")
        if not (LIB / name).exists():
            shutil.copy2(FALLBACK / "lib" / name, LIB / name)
    shutil.copy2(FALLBACK / "lib" / "libac922-tail-bridge.so", LIB / "libac922-tail-bridge.so")
    for name in NAMES:
        alias = LIB / name.removesuffix(".0")
        if not alias.exists():
            alias.symlink_to(name)

    cc = json.loads((BUILD / "compile_commands.json").read_text())
    entries = {Path(entry["file"]).name: entry for entry in cc}
    changed = []
    for relative in ["ggml/src/ggml-backend.cpp", "src/llama-context.cpp", "common/speculative.cpp"]:
        private = HOST / relative
        if private.exists():
            changed.append((relative, private))
    if not changed:
        raise RuntimeError("host source patches must be prepared before building")

    def compile_one(item: tuple[str, Path]):
        relative, private = item
        entry = entries[private.name]
        args = shlex.split(entry["command"])
        mapped = []
        for arg in args:
            arg = arg.replace(str(OLD / "build-monster"), str(BUILD))
            arg = arg.replace(str(OLD), str(SOURCE))
            mapped.append(arg)
        obj = OUT / (private.stem + ".o")
        mapped[mapped.index("-o") + 1] = str(obj)
        mapped[mapped.index("-c") + 1] = str(private)
        log = OUT / (private.stem + ".compile.log")
        identity = hashlib.sha256(private.read_bytes() + json.dumps(mapped).encode()).hexdigest()
        receipt = OUT / (private.stem + ".compile-identity")
        if not (obj.exists() and receipt.exists() and receipt.read_text().strip() == identity):
            run(["/usr/bin/nice", "-n", "5", *mapped], log, BUILD)
            receipt.write_text(identity + "\n")
        return relative, obj, mapped

    started = time.time()
    # These three independent C++ TUs are the complete engine delta; CUDA
    # compilation remains zero. Do not manufacture extra jobs or rebuild headers.
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        objects = list(executor.map(compile_one, changed))
    replacements = {Path(relative).name + ".o": obj for relative, obj, _ in objects}
    links = [("ggml/src", "ggml-base", "libggml-base.so.0"),
             ("src", "llama", "libllama.so.0"),
             ("common", "llama-common", "libllama-common.so.0")]
    linked = []
    for directory, target, name in links:
        if target == "llama-common" and "speculative.cpp.o" not in replacements:
            continue
        args = shlex.split((BUILD / directory / "CMakeFiles" / (target + ".dir") / "link.txt").read_text())
        args[args.index("-o") + 1] = str(LIB / name)
        for i, arg in enumerate(args):
            if Path(arg).name in replacements:
                args[i] = str(replacements[Path(arg).name])
            elif "-Wl,-rpath," in arg:
                args[i] = "-Wl,-rpath,$ORIGIN"
            elif arg.startswith("../bin/lib"):
                library = Path(arg).name.split(".so.")[0] + ".so.0"
                args[i] = str(LIB / library)
            elif arg.endswith((".o", ".a")) and not Path(arg).is_absolute():
                frozen = OUT / "base-objects" / target / arg
                frozen.parent.mkdir(parents=True, exist_ok=True)
                if not frozen.exists():
                    shutil.copy2(BUILD / directory / arg, frozen)
                args[i] = str(frozen)
        run(args, OUT / (target + ".link.log"), BUILD / directory)
        linked.append(name)

    if os.environ.get("AC922_LASTWAVE_ENGINE_ONLY") == "1":
        print(json.dumps({"status": "HOST_LIBS_PASS", "cuda_tus": 0,
                          "host_tus": [relative for relative, _ in changed],
                          "relinked": linked, "elapsed_seconds": time.time()-started}), flush=True)
        return

    includes = [SOURCE / "common", SOURCE / "src", SOURCE / "include", SOURCE / "ggml/include",
                SOURCE / "ggml/src", SOURCE / "vendor", SOURCE / "tools/server", SOURCE / "tools/mtmd",
                BUILD / "include", BUILD / "ggml/include", BASE / "fixtures", HERE]
    for child in ["long-prefill-combo-v45-20261016", "dual-context-v35-onebank4-combined-20261016",
                  "layer6-fourbank-v146-20261017", "mtp-vocab-fourbank-v221-20261017"]:
        includes.append(BASE / "fixtures" / child)
    flags = ["-D" + name for name in ["GGML_BACKEND_SHARED", "GGML_SHARED", "GGML_USE_CPU", "GGML_USE_CUDA", "LLAMA_SHARED"]]
    argv = ["/usr/bin/nice", "-n", "5", "/usr/bin/g++", "-std=c++17", "-O3", "-DNDEBUG", "-pthread",
            *flags, "-I/usr/local/cuda-12.4/targets/ppc64le-linux/include",
            *("-I" + str(path) for path in includes), str(HERE / "bank-worker-combo.cpp"),
            "-rdynamic", "-fno-omit-frame-pointer", "-o", str(OUT / "bank-worker"),
            "-Wl,-rpath," + str(LIB), "-Wl,--no-as-needed", *(str(LIB / name) for name in NAMES),
            "-Wl,--as-needed", "/usr/local/cuda-12.4/targets/ppc64le-linux/lib/stubs/libcuda.so"]
    run(argv, OUT / "worker.compile.log")
    env = dict(os.environ, LD_LIBRARY_PATH=str(LIB))
    closure = subprocess.check_output(["ldd", str(OUT / "bank-worker")], env=env, text=True)
    (OUT / "ldd.txt").write_text(closure)
    if "not found" in closure:
        raise RuntimeError("unresolved private library")
    for name in NAMES:
        if name + " => " + str(LIB) + "/" not in closure:
            raise RuntimeError("library mixed outside private closure: " + name)
    report = {"status": "BUILD_PASS", "model_loaded": False, "cuda_tus": 0,
              "base_release": str(FALLBACK), "elapsed_seconds": time.time()-started,
              "host_tus": [relative for relative, _ in changed], "relinked": linked,
              "worker_sha256": digest(OUT / "bank-worker"),
              "source_sha256": digest(HERE / "bank-worker-combo.cpp"),
              "headers": {path.name: digest(path) for path in sorted(HERE.glob("*.h"))},
              "host_source_sha256": {relative: digest(private) for relative, private in changed},
              "libraries": {name: digest(LIB / name) for name in NAMES}}
    (OUT / "build-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    build()
