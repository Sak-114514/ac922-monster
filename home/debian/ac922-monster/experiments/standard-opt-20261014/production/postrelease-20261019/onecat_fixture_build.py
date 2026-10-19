#!/usr/bin/env python3
"""Build the V372 Onecat first-capture fixture against the private R5/R444 closure.

This is a host-only compile/link step. It does not run the fixture, initialize
CUDA, load a model, query GPU state, or change any runtime/service state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone


DEFAULT_REPO = Path("/home/debian/ac922-monster/experiments/standard-opt-20261014")
DEFAULT_POSTRELEASE = DEFAULT_REPO / "production/postrelease-20261019"
R444_RELEASE = Path(
    "/home/debian/ac922-monster/production/releases/20261018-r444-banked-combo-c12-r2"
)
FIXTURE_REL = Path("fixtures/onecat-init-concurrency-v372-20261018")
EXPECTED_HOST_TUS = {
    "ggml/src/ggml-backend.cpp",
    "src/llama-context.cpp",
    "common/speculative.cpp",
}
EXPECTED_HOST_RELINKS = {
    "libggml-base.so.0",
    "libllama.so.0",
    "libllama-common.so.0",
}
CORE_DSOS = (
    "libggml-cuda.so.0",
    "libggml-cpu.so.0",
    "libggml.so.0",
    "libggml-base.so.0",
)
ALL_RUNTIME_DSOS = (
    "libggml-base.so.0",
    "libggml-cpu.so.0",
    "libggml-cuda.so.0",
    "libggml.so.0",
    "libllama-common.so.0",
    "libllama.so.0",
    "libmtmd.so.0",
)
R5_RELINKED_DSOS = (
    "libggml-base.so.0",
    "libllama.so.0",
    "libllama-common.so.0",
)
R5_HELPER = "libac922-tail-bridge.so"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"BUILD GATE FAILED: {message}")


def run(argv: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        argv,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode:
        sys.stdout.write(result.stdout)
        raise SystemExit(f"BUILD GATE FAILED: {shlex.join(argv)} returned {result.returncode}")
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="unique artifact tag; never reuses an existing build")
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--postrelease-dir", type=Path, default=DEFAULT_POSTRELEASE)
    args = parser.parse_args()
    require(bool(re.fullmatch(r"[A-Za-z0-9._-]{1,80}", args.tag)), "unsafe tag")
    require(platform.machine() == "ppc64le", "run this host-only build on the AC922 ppc64le host")

    repo = args.repo_root.resolve()
    postrelease = args.postrelease_dir.resolve()
    fixture_dir = repo / FIXTURE_REL
    fixture_cpp = fixture_dir / "onecat-init-concurrency-v372.cpp"
    oracle_cpp = repo / "fixtures/onecat-variable-lanes-20261015/onecat-variable-lanes.cpp"
    verify_py = fixture_dir / "verify-result.py"
    build_root = postrelease / "build-lastwave"
    private_lib = build_root / "lib"
    build_report = build_root / "build-report.json"
    artifact_dir = postrelease / "onecat-fixture" / "builds" / args.tag
    binary = artifact_dir / "onecat-init-concurrency-v372"
    backend_dir = artifact_dir / "backend"
    manifest_path = artifact_dir / "manifest.json"
    build_log_path = artifact_dir / "host-build.log"

    require(fixture_cpp.is_file() and oracle_cpp.is_file() and verify_py.is_file(), "V372 fixture sources are missing")
    require(build_report.is_file(), "private L5 build report is missing")
    require(private_lib.is_dir(), "private build-lastwave/lib is missing")
    require(not artifact_dir.exists(), f"artifact tag already exists: {artifact_dir}")
    require(not os.getenv("AC922_POSTRELEASE_GPU_WINDOW"), "unset AC922_POSTRELEASE_GPU_WINDOW for the host-only build")

    report = json.loads(build_report.read_text())
    require(report.get("status") == "BUILD_PASS", "private L5 build report is not BUILD_PASS")
    require(report.get("model_loaded") is False, "private build report unexpectedly says a model was loaded")
    require(report.get("cuda_tus") == 0, "L5 must not rebuild CUDA TUs for this fixture closure")
    require(Path(report.get("base_release", "")).resolve() == R444_RELEASE.resolve(), "L5 base is not frozen R444")
    require(set(report.get("host_tus", [])) == EXPECTED_HOST_TUS, "private build report host TU set changed")
    require(set(report.get("relinked", [])) == EXPECTED_HOST_RELINKS, "private host DSO set changed")
    report_libraries = report.get("libraries", {})
    require(set(report_libraries) == set(ALL_RUNTIME_DSOS), "private build report runtime DSO inventory changed")

    dso_hashes: dict[str, str] = {}
    for name in ALL_RUNTIME_DSOS:
        path = private_lib / name
        require(path.is_file(), f"missing private closure DSO {path}")
        actual = sha256(path)
        require(report_libraries.get(name) == actual, f"{name} does not match the L5 build report")
        dso_hashes[name] = actual

    frozen_cuda = R444_RELEASE / "lib/libggml-cuda.so.0"
    require(frozen_cuda.is_file(), f"frozen R444 CUDA DSO missing: {frozen_cuda}")
    frozen_cuda_sha = sha256(frozen_cuda)
    require(dso_hashes["libggml-cuda.so.0"] == frozen_cuda_sha, "private CUDA DSO differs from frozen R444")

    for name in EXPECTED_HOST_RELINKS:
        path = private_lib / name
        require(path.is_file(), f"missing private R5 host DSO {path}")
        actual = sha256(path)
        require(report_libraries.get(name) == actual, f"{name} does not match the L5 build report")
        dso_hashes[name] = actual

    # Record the helper, but do not imply the fixture exercises it.
    helper_path = private_lib / R5_HELPER
    helper_record = {
        "name": R5_HELPER,
        "path": str(helper_path),
        "present": helper_path.is_file(),
        "sha256": sha256(helper_path) if helper_path.is_file() else None,
        "linked_or_loaded_by_fixture": False,
        "note": "Recorded for provenance; V372 is a direct GGML fixture and does not claim to exercise the R5 runtime helper.",
    }

    artifact_dir.mkdir(parents=True)
    (backend_dir).mkdir()
    for soname, backend_name in (("libggml-cuda.so.0", "libggml-cuda.so"),
                                 ("libggml-cpu.so.0", "libggml-cpu.so")):
        (backend_dir / backend_name).symlink_to(private_lib / soname)

    src_root = repo / "src"
    build_headers = repo / "build-reuse/ggml/src"
    for path in (src_root / "ggml/include/ggml.h", src_root / "ggml/src/ggml-impl.h", build_headers):
        require(path.exists(), f"fixture compile include prerequisite missing: {path}")

    argv = [
        "/usr/bin/g++", "-std=c++17", "-O2", "-g", "-pthread", "-fopenmp",
        "-DGGML_SHARED", "-D_GNU_SOURCE", "-D_XOPEN_SOURCE=600",
        f"-I{src_root / 'ggml/include'}", f"-I{src_root / 'ggml/src'}", f"-I{build_headers}",
        str(fixture_cpp), "-o", str(binary),
        f"-L{private_lib}", f"-Wl,-rpath,{private_lib}",
        "-L/usr/local/cuda-12.4/targets/ppc64le-linux/lib",
        "-Wl,-rpath,/usr/local/cuda-12.4/targets/ppc64le-linux/lib",
        "-Wl,-rpath,/usr/local/cuda/lib",
        "-l:libggml-cuda.so.0", "-l:libggml-cpu.so.0", "-l:libggml.so.0", "-l:libggml-base.so.0",
        # Force the two llama-side R5 host DSOs into DT_NEEDED even though this
        # direct-GGML fixture does not call llama APIs. This proves the exact
        # private host closure can coexist with the R444 CUDA DSO; it does not
        # claim to exercise llama-context or the tail-bridge helper.
        "-Wl,--no-as-needed", "-l:libllama.so.0", "-l:libllama-common.so.0", "-Wl,--as-needed",
        "-lcudart", "-lcublas", "-lcublasLt", "-L/usr/local/cuda/lib", "-l:libnccl.so.2", "-ldl", "-pthread",
    ]
    compile_output = run(argv)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(private_lib)
    ldd_output = run(["ldd", "-r", str(binary)], env=env)
    require("not found" not in ldd_output and "undefined symbol:" not in ldd_output,
            "ldd -r reported missing or unresolved runtime dependencies")
    required_loaded_dsos = tuple(dict.fromkeys((*CORE_DSOS, *R5_RELINKED_DSOS)))
    for name in required_loaded_dsos:
        expected = (private_lib / name).resolve()
        pattern = re.compile(r"^\s*" + re.escape(name) + r" => (\S+)", re.MULTILINE)
        match = pattern.search(ldd_output)
        require(match is not None, f"ldd -r did not resolve {name}")
        require(Path(match.group(1)).resolve() == expected, f"ldd -r resolved {name} outside the private closure")

    loaded_helper = R5_HELPER in ldd_output
    helper_record["linked_or_loaded_by_fixture"] = loaded_helper
    (artifact_dir / "host-build.log").write_text(
        "HOST_ONLY_BUILD=1\nGPU_PROCESS_STARTED=0\nMODEL_LOADED=0\n"
        + "COMPILE: " + shlex.join(argv) + "\n"
        + compile_output
        + "\nLDD_R:\n" + ldd_output
    )

    manifest = {
        "status": "HOST_BUILD_PASS",
        "tag": args.tag,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host_only": True,
        "gpu_process_started": False,
        "model_loaded": False,
        "runtime_path": str(private_lib),
        "private_build_report": str(build_report),
        "private_build_report_sha256": sha256(build_report),
        "private_build_report_status": report.get("status"),
        "private_build_report_base_release": report.get("base_release"),
        "private_host_tus": report.get("host_tus"),
        "private_relinked_host_dsos": report.get("relinked"),
        "fixture_source": str(fixture_cpp),
        "fixture_source_sha256": sha256(fixture_cpp),
        "oracle_source": str(oracle_cpp),
        "oracle_source_sha256": sha256(oracle_cpp),
        "verifier": str(verify_py),
        "verifier_sha256": sha256(verify_py),
        "binary": str(binary),
        "binary_sha256": sha256(binary),
        "backend_dir": str(backend_dir),
        "dso_closure": {name: {"path": str(private_lib / name), "sha256": digest} for name, digest in dso_hashes.items()},
        "runtime_libraries_sha256": dict(dso_hashes),
        "required_loaded_dsos": list(required_loaded_dsos),
        "frozen_r444_cuda": {"path": str(frozen_cuda), "sha256": frozen_cuda_sha,
                              "matches_private_closure": True},
        "r5_helper": helper_record,
        "ldd_r_sha256": hashlib.sha256(ldd_output.encode()).hexdigest(),
        "ldd_r_text": ldd_output,
        "build_log": str(build_log_path),
        "run_gate": str(postrelease / "onecat_fixture_gate.py"),
        "gpu_test": "not run; requires AC922_POSTRELEASE_GPU_WINDOW=1 and the separate gate script",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"HOST_BUILD_PASS manifest={manifest_path}")
    print(f"BINARY={binary}")
    print(f"BACKEND_DIR={backend_dir}")
    print(f"RUNTIME_LIBDIR={private_lib}")
    print(f"R444_CUDA_SHA256={frozen_cuda_sha}")
    print(f"R5_HELPER_PRESENT={str(helper_record['present']).lower()} LOADED_BY_FIXTURE={str(loaded_helper).lower()}")
    print("GPU_PROCESS_STARTED=0 MODEL_LOADED=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
