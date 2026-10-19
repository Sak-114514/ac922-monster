#!/usr/bin/env python3
"""Guarded, sequential six-card V372 Onecat first-capture gate.

This script refuses to query or launch GPU work unless
AC922_POSTRELEASE_GPU_WINDOW=1 is explicitly set by the caller. It runs only
the bounded V372 direct-GGML fixture, never the inference service or a model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone


POSTRELEASE = Path("/home/debian/ac922-monster/experiments/standard-opt-20261014/production/postrelease-20261019")
FIXTURE_DIR = Path("/home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/onecat-init-concurrency-v372-20261018")
GPU_UUIDS = (
    (0, "GPU-80452dac-e8c1-99fd-225c-ea846276ae6f"),
    (1, "GPU-ac5b33c1-276d-c1ae-15a6-46f7894daefb"),
    (2, "GPU-e6529eb8-aa1f-1d02-5d63-dfb7995bf1eb"),
    (3, "GPU-fadb0459-4d30-907b-289a-862b3014f165"),
    (4, "GPU-760ce501-32f6-69b6-215b-1020da68bbfc"),
    (5, "GPU-d3558623-deca-c702-3cc0-01b4aa9c667a"),
)
CORE_DSOS = (
    "libggml-cuda.so.0",
    "libggml-cpu.so.0",
    "libggml.so.0",
    "libggml-base.so.0",
)
R5_RELINKED_DSOS = (
    "libggml-base.so.0",
    "libllama.so.0",
    "libllama-common.so.0",
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
AUTH_ENV = "AC922_POSTRELEASE_GPU_WINDOW"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checked_run(argv: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def assert_under(path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"GATE REFUSED: path is outside postrelease directory: {path}") from exc


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def gpu_snapshot(uuid: str, phase: str) -> dict:
    query = checked_run([
        "nvidia-smi", "-i", uuid,
        "--query-gpu=memory.used,memory.free,temperature.gpu,utilization.gpu",
        "--format=csv,noheader,nounits",
    ])
    if query.returncode != 0:
        raise RuntimeError(f"nvidia-smi telemetry failed for {uuid} {phase}: {query.stdout.strip()}")
    try:
        row = next(csv.reader([query.stdout.strip()]))
        used, free, temp, util = [int(value.strip()) for value in row]
    except (StopIteration, ValueError) as exc:
        raise RuntimeError(f"invalid nvidia-smi telemetry for {uuid} {phase}: {query.stdout!r}") from exc

    pids_result = checked_run([
        "nvidia-smi", "-i", uuid, "--query-compute-apps=pid", "--format=csv,noheader,nounits",
    ])
    if pids_result.returncode != 0:
        raise RuntimeError(f"nvidia-smi PID query failed for {uuid} {phase}: {pids_result.stdout.strip()}")
    pids = [line.strip() for line in pids_result.stdout.splitlines() if line.strip() and line.strip() != "N/A"]
    return {
        "phase": phase,
        "uuid": uuid,
        "memory_used_mib": used,
        "memory_free_mib": free,
        "temperature_c": temp,
        "utilization_percent": util,
        "compute_pids": pids,
        "pass": used <= 512 and free >= 512 and temp < 58 and util < 5 and not pids,
    }


def all_card_preflight(phase: str) -> list[dict]:
    snapshots = [gpu_snapshot(uuid, f"{phase}-gpu{index}") for index, uuid in GPU_UUIDS]
    failed = [s for s in snapshots if not s["pass"]]
    if failed:
        raise RuntimeError("strict all-card idle/thermal preflight failed: " + json.dumps(failed, sort_keys=True))
    return snapshots


def verify_manifest(path: Path) -> tuple[dict, Path, Path, Path]:
    postrelease = POSTRELEASE.resolve()
    manifest_path = path.resolve()
    assert_under(manifest_path, postrelease)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "HOST_BUILD_PASS" or manifest.get("host_only") is not True:
        raise RuntimeError("fixture manifest is not a successful host-only build receipt")
    if manifest.get("gpu_process_started") is not False or manifest.get("model_loaded") is not False:
        raise RuntimeError("host build receipt claims GPU or model activity")

    binary = Path(manifest["binary"]).resolve()
    backend_dir = Path(manifest["backend_dir"]).resolve()
    runtime_libdir = Path(manifest["runtime_path"]).resolve()
    for item in (binary, backend_dir, runtime_libdir):
        assert_under(item, postrelease)
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError(f"fixture binary is missing or not executable: {binary}")
    if sha256(binary) != manifest.get("binary_sha256"):
        raise RuntimeError("fixture binary changed after host build")

    for name, record in manifest.get("dso_closure", {}).items():
        path = Path(record["path"]).resolve()
        assert_under(path, postrelease)
        if not path.is_file() or sha256(path) != record.get("sha256"):
            raise RuntimeError(f"private runtime DSO changed after build: {path}")
    for name in CORE_DSOS:
        if name not in manifest.get("dso_closure", {}):
            raise RuntimeError(f"manifest lacks required private DSO: {name}")
    runtime_hashes = manifest.get("runtime_libraries_sha256", {})
    if tuple(runtime_hashes.keys()) != ALL_RUNTIME_DSOS:
        raise RuntimeError("manifest does not contain the ordered seven-library R5/R444 inventory")
    if any(manifest["dso_closure"].get(name, {}).get("sha256") != runtime_hashes[name]
           for name in ALL_RUNTIME_DSOS):
        raise RuntimeError("runtime library hash map disagrees with DSO closure manifest")
    required_loaded = tuple(dict.fromkeys((*CORE_DSOS, *R5_RELINKED_DSOS)))
    if tuple(manifest.get("required_loaded_dsos", [])) != required_loaded:
        raise RuntimeError("manifest does not require the exact loaded R444+R5 DSO closure")

    verify_py = FIXTURE_DIR / "verify-result.py"
    if not verify_py.is_file() or sha256(verify_py) != manifest.get("verifier_sha256"):
        raise RuntimeError("V372 result verifier is missing or changed after build")
    return manifest, binary, backend_dir, runtime_libdir


def ldd_private_closure(binary: Path, runtime_libdir: Path, manifest: dict) -> str:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(runtime_libdir)
    result = checked_run(["ldd", "-r", str(binary)], env=env)
    if result.returncode != 0 or "not found" in result.stdout or "undefined symbol:" in result.stdout:
        raise RuntimeError("private runtime ldd -r validation failed:\n" + result.stdout)
    required_loaded_dsos = tuple(manifest["required_loaded_dsos"])
    for name in required_loaded_dsos:
        match = re.search(r"^\s*" + re.escape(name) + r" => (\S+)", result.stdout, re.MULTILINE)
        if not match or Path(match.group(1)).resolve() != (runtime_libdir / name).resolve():
            raise RuntimeError(f"{name} did not resolve from the postrelease private closure:\n{result.stdout}")
    return result.stdout


def run_one(index: int, uuid: str, tag: str, binary: Path, backend_dir: Path,
            runtime_libdir: Path, log_path: Path, verify_py: Path,
            manifest: dict, mark_gpu_process_started) -> dict:
    before_all = all_card_preflight(f"before-{tag}")
    ldd_output = ldd_private_closure(binary, runtime_libdir, manifest)
    header = (
        "ALL_CARD_GPU_PREFLIGHT_BEFORE=PASS\n"
        + "\n".join("GPU_PREFLIGHT " + json.dumps(snapshot, sort_keys=True) for snapshot in before_all)
        + "\n"
        f"runtime_libdir={runtime_libdir}\n"
        f"binary={binary}\n"
        f"V372_RUNTIME_CLOSURE_PASS libdir={runtime_libdir}\n"
        "LDD_R:\n" + ldd_output + "\n"
    )
    env = os.environ.copy()
    env.pop("GGML_CUDA_DISABLE_GRAPHS", None)
    env.pop("GGML_CUDA_GRAPH_STRUCTURAL_TEST_POLICY", None)
    env.update({
        "GGML_CUDA_FATTN_ONECAT": "1",
        "GGML_CUDA_ONECAT_INIT_READY": "1",
        "GGML_CUDA_FATTN_ONECAT_PART": "256",
        "GGML_CUDA_GRAPH_OPT": "1",
        "GGML_CUDA_GRAPHS_FORCE": "1",
        "GGML_CUDA_GRAPH_STRUCTURAL": "1",
        "GGML_CUDA_GRAPH_PATH_LOG": "1",
        "GGML_OPTIMIZATION_PATH_LOG": "1",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": uuid,
        "LD_LIBRARY_PATH": str(runtime_libdir),
        "OMP_NUM_THREADS": "2",
    })
    argv = [
        "timeout", "--signal=INT", "--kill-after=3s", "60s",
        str(binary), str(backend_dir), "0",
    ]
    started = datetime.now(timezone.utc).isoformat()
    mark_gpu_process_started()
    result = checked_run(argv, env=env)
    stdout = result.stdout
    try:
        after_all = all_card_preflight(f"after-{tag}")
    except Exception as exc:
        log_path.write_text(header + stdout + f"\nALL_CARD_GPU_PREFLIGHT_AFTER=FAIL\n{exc}\n")
        raise RuntimeError(f"post-run GPU safety preflight failed for GPU{index}: {exc}") from exc
    try:
        log_path.write_text(
            header + stdout + "\nALL_CARD_GPU_PREFLIGHT_AFTER=PASS\n"
            + "\n".join("GPU_PREFLIGHT " + json.dumps(snapshot, sort_keys=True) for snapshot in after_all)
            + "\n"
        )
    except OSError as exc:
        raise RuntimeError(f"cannot save V372 output log {log_path}: {exc}") from exc

    if result.returncode != 0:
        log_path.write_text(log_path.read_text() + f"\nV372 fixture process failed status={result.returncode}; no automatic retry\n")
        raise RuntimeError(f"V372 GPU{index} returned {result.returncode}; see {log_path}")

    verifier = checked_run([sys.executable, str(verify_py), str(log_path), str(runtime_libdir)])
    with log_path.open("a") as f:
        f.write("\n" + verifier.stdout)
    if verifier.returncode != 0:
        raise RuntimeError(f"V372 verifier failed for GPU{index}; see {log_path}")
    with log_path.open("a") as f:
        f.write(f"\nexit_status={result.returncode}\n")
    return {
        "gpu_index": index,
        "gpu_uuid": uuid,
        "tag": tag,
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "exit_status": result.returncode,
        "verified": True,
        "log": str(log_path),
        "log_sha256": sha256(log_path),
        "fixture_source_sha256": manifest["fixture_source_sha256"],
        "oracle_source_sha256": manifest["oracle_source_sha256"],
        "simultaneous_first_capture": True,
        "actual_onecat_init_ready": True,
        "graph_capture_and_replay": True,
        "cpu_oracle_pass": True,
        "bit_exact_replay": True,
        "passed": True,
        "gpu_process_started": True,
        "before_all_cards": before_all,
        "after_all_cards": after_all,
        "passed_markers": [
            "V372_PASS first_use_concurrent=1",
            "Onecat init-ready prepared device=0",
            "CUDA graph path active ... mode=capture",
            "CUDA graph path active ... mode=replay",
            "CPU_ORACLE ... PASS",
            "V372 result verification PASS",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="HOST_BUILD_PASS manifest from onecat_fixture_build.py")
    parser.add_argument("--run-tag", help="unique gate tag; defaults to UTC timestamp")
    args = parser.parse_args()

    # This authorization is checked before manifest access or any GPU query.
    if os.environ.get(AUTH_ENV) != "1":
        print(f"GPU GATE REFUSED: set {AUTH_ENV}=1 explicitly after model stop and owner authorization", file=sys.stderr)
        return 2

    if platform_machine() != "ppc64le":
        print("GPU GATE REFUSED: this gate must run on the AC922 ppc64le host", file=sys.stderr)
        return 2
    receipt_path: Path | None = None
    receipt: dict | None = None
    gate_root: Path | None = None
    try:
        run_tag = args.run_tag or datetime.now(timezone.utc).strftime("onecat-r5-r444-%Y%m%dT%H%M%SZ")
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,80}", run_tag):
            raise RuntimeError("unsafe run tag")
        gate_root = POSTRELEASE / "onecat-fixture/gate-runs" / run_tag
        if gate_root.exists():
            raise RuntimeError(f"gate run tag already exists: {gate_root}")
        receipt_path = POSTRELEASE / "onecat-fixture-receipt.json"
        if receipt_path.exists():
            raise RuntimeError(f"refusing to overwrite existing canonical receipt: {receipt_path}")
        gate_root.mkdir(parents=True)
        receipt = {
            "schema": "ac922-onecat-fixture-receipt-v1",
            "status": "IN_PROGRESS",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "run_tag": run_tag,
            "gpu_window_authorized_env": True,
            "gpu_process_started": False,
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256(args.manifest.resolve()) if args.manifest.is_file() else None,
            "cases": [],
            "cards": [],
            "gpu_process_scope": "V372 direct GGML microfixture only; no model or service",
        }
        atomic_json(receipt_path, receipt)
        atomic_json(gate_root / "receipt.json", receipt)

        manifest, binary, backend_dir, runtime_libdir = verify_manifest(args.manifest)
        verify_py = FIXTURE_DIR / "verify-result.py"
        receipt.update({
            "binary": str(binary),
            "binary_sha256": sha256(binary),
            "runtime_libdir": str(runtime_libdir),
            "runtime_libraries_sha256": manifest["runtime_libraries_sha256"],
            "fixture_source": manifest["fixture_source"],
            "fixture_source_sha256": manifest["fixture_source_sha256"],
            "oracle_source": manifest["oracle_source"],
            "oracle_source_sha256": manifest["oracle_source_sha256"],
            "private_build_report_sha256": manifest["private_build_report_sha256"],
            "frozen_r444_cuda_sha256": manifest["frozen_r444_cuda"]["sha256"],
            "r5_helper": manifest.get("r5_helper"),
        })
        atomic_json(receipt_path, receipt)
        atomic_json(gate_root / "receipt.json", receipt)

        def mark_gpu_process_started() -> None:
            receipt["gpu_process_started"] = True
            atomic_json(receipt_path, receipt)
            atomic_json(gate_root / "receipt.json", receipt)

        for index, uuid in GPU_UUIDS:
            tag = f"{run_tag}-gpu{index}"
            log_path = gate_root / f"gpu{index}.log"
            try:
                card = run_one(index, uuid, tag, binary, backend_dir, runtime_libdir, log_path, verify_py,
                               manifest, mark_gpu_process_started)
            except Exception as exc:
                receipt["status"] = "FAIL"
                receipt["failed_gpu_index"] = index
                receipt["failed_gpu_uuid"] = uuid
                receipt["error"] = str(exc)
                receipt["failure_log"] = str(log_path) if log_path.exists() else None
                receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
                atomic_json(receipt_path, receipt)
                atomic_json(gate_root / "receipt.json", receipt)
                raise
            receipt["cards"].append(card)
            receipt["cases"] = receipt["cards"]
            atomic_json(receipt_path, receipt)
            atomic_json(gate_root / "receipt.json", receipt)
            print(f"V372_GPU_PASS index={index} uuid={uuid} log={log_path}", flush=True)
        receipt["status"] = "PASS"
        receipt["cases"] = receipt["cards"]
        receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(receipt_path, receipt)
        atomic_json(gate_root / "receipt.json", receipt)
        print(f"V372_SIX_CARD_PASS receipt={receipt_path}")
        return 0
    except Exception as exc:  # Preserve a clear, fail-closed operator-facing error.
        if receipt_path is not None and receipt is not None and receipt.get("status") == "IN_PROGRESS":
            receipt["status"] = "FAIL"
            receipt["error"] = str(exc)
            receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
            atomic_json(receipt_path, receipt)
            if gate_root is not None and gate_root.exists():
                atomic_json(gate_root / "receipt.json", receipt)
        print(f"GPU GATE FAILED/CLOSED: {exc}", file=sys.stderr)
        return 1


def platform_machine() -> str:
    import platform
    return platform.machine()


if __name__ == "__main__":
    raise SystemExit(main())
