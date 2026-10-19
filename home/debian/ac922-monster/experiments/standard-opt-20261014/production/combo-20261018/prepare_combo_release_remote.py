#!/usr/bin/env python3
"""Stage an immutable combo release; never promote or touch running services.

Run on the AC922 after the combo worker and its real smoke receipt are ready.
This script only creates production/releases/<version>; it never writes the
production/current symlink, systemd units, gateway source, or model data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any
from datetime import datetime, timezone


RUNTIME_DSOS = (
    "libllama-common.so.0",
    "libmtmd.so.0",
    "libllama.so.0",
    "libggml.so.0",
    "libggml-cpu.so.0",
    "libggml-cuda.so.0",
    "libggml-base.so.0",
)
EXPECTED_SMOKE_STATUS = "COMBO_NATIVE_PREFILL32K_D3_FOUR512_CAPACITY_PASS"
EXPECTED_SMOKE_WAVES = (
    "combo-formal32k-cold",
    "combo-formal32k-warm",
    "combo-quality8192-after-cache",
    "combo-native4-after-prefill",
    "combo-native4-8k-prefill",
    "combo-four512",
    "combo-524288-capacity-multi",
    "combo-native4-after-large",
    "combo-max-single-UB512",
    "combo-native4-final",
)
DEFAULT_BASE = Path("/home/debian/ac922-monster/experiments/standard-opt-20261014")
REMOTE_BASE = DEFAULT_BASE
DEFAULT_MODEL = Path("/home/debian/models/qwen3.8-flash-next-gsq-rco/integrated/IQ3_XXS-MTP-Q8.gguf")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def file_info(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "sha256": sha256_file(path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path(os.environ.get("AC922_EXPERIMENT_ROOT", DEFAULT_BASE)))
    parser.add_argument("--production-root", type=Path, default=Path("/home/debian/ac922-monster/production"))
    parser.add_argument("--worker-build", default="build-combo-c12",
                        help="worker build directory under production/combo-20261018 (default: build-combo-c12)")
    parser.add_argument("--worker", type=Path, default=None,
                        help="explicit worker binary; overrides --worker-build and AC922_COMBO_WORKER")
    parser.add_argument("--v35-checker", type=Path, default=None,
                        help="explicit auxiliary verifier path; otherwise use the established fixture path")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--version", default=None, help="release id; defaults to a UTC timestamped combo id")
    parser.add_argument("--smoke-report", type=Path, default=None,
                        help="real C10 combined worker-smoke-result.json; omitting it creates a non-promotable release")
    parser.add_argument("--check-only", action="store_true",
                        help="report source-closure availability without writing release artifacts")
    return parser.parse_args()


def paths(args: argparse.Namespace) -> dict[str, Path]:
    base = args.base.resolve()
    combo = base / "production/combo-20261018"
    candidate = base / "production/candidate-20261018"
    env_file = combo / "combo-environment.json"
    env_data: dict[str, Any] = {}
    if env_file.is_file():
        try:
            env_data = json.loads(env_file.read_text(encoding="utf-8"))
        except Exception:
            env_data = {}
    if args.worker is not None:
        worker = args.worker
    elif os.environ.get("AC922_COMBO_WORKER"):
        worker = Path(os.environ["AC922_COMBO_WORKER"])
    else:
        worker = combo / args.worker_build / "bank-worker"
    def configured_path(name: str, fallback: Path) -> Path:
        value = env_data.get(name)
        if not isinstance(value, str) or not value:
            return fallback
        configured = Path(value).expanduser()
        # Allow the local workspace's check-only scan to resolve host-configured
        # experiment paths when the corresponding local copy exists. On the
        # AC922, the original absolute path exists and is used unchanged.
        if configured.is_absolute() and not configured.exists():
            try:
                local_copy = base / configured.relative_to(REMOTE_BASE)
            except ValueError:
                local_copy = configured
            if local_copy.exists():
                configured = local_copy
        return configured.resolve()

    v35 = args.v35_checker or (base / "fixtures/dual-context-v35-onebank4-combined-20261016/preclean_quality_gate_v22.py")
    return {
        "base": base,
        "combo": combo,
        "candidate": candidate,
        "worker": worker.resolve(),
        "v35_checker": v35.resolve(),
        "model": args.model.resolve(),
        "env": env_file,
        "runtime_lib": configured_path("ROOT_DYNAMIC_LIB_DIR", base / "runtime-perf-v444/lib"),
        "build_report": configured_path("ROOT_BUILD_REPORT", base / "notes/V444-build.json"),
        "tail_bridge": configured_path("GGML_CUDA_GRAPH_DEVICE_TAIL_LIBRARY", Path("/nonexistent/tail-bridge")),
        "native_cases": base / "fixtures/dual-context-v35-onebank4-combined-20261016/native-semantic/case-input.json",
        "native_evaluator": base / "fixtures/dual-context-v35-onebank4-combined-20261016/native-semantic/run_semantic_native.py",
        "corpus": configured_path("ROOT_V131_CORPUS_SOURCE",
                                   base / "fixtures/long-prefill-combo-v45-20261016/v45-real-corpus-source.json"),
        "guard_model": base / "notes/inband-guard-v222-20261017/model_guard.py",
        "guard_stop": base / "notes/inband-guard-v222-20261017/stop_model.py",
        "guard_sdr": base / "notes/inband-guard-v222-20261017/sdr.cache",
    }


def required_sources(p: dict[str, Path]) -> dict[str, Path]:
    result = {
        "api.py": p["candidate"] / "api.py",
        "llm_routes.py": p["candidate"] / "llm_routes.py",
        "test_api.py": p["candidate"] / "test_api.py",
        "candidate_verify_release.py": p["candidate"] / "verify_release.py",
        "candidate_wait_safety.py": p["candidate"] / "wait_safety.py",
        "combo_safety_monitor.py": p["combo"] / "combo_safety_monitor.py",
        "combo_temperature_policy.py": p["combo"] / "combo_temperature_policy.py",
        "combo_environment.json": p["env"],
        "verify_combo_native.py": p["combo"] / "verify_combo_native.py",
        "combo_worker_source.cpp": p["combo"] / "bank-worker-combo.cpp",
        "production_ipc.h": p["combo"] / "production-ipc.h",
        "thermal_pacing.h": p["combo"] / "thermal-pacing.h",
        "combo_build_script.sh": p["combo"] / "build-combo.sh",
        "combo_worker_binary": p["worker"],
        "model": p["model"],
        "runtime_build_report": p["build_report"],
        "tail_bridge_r5": p["tail_bridge"],
        "auxiliary_v35_checker": p["v35_checker"],
        "native_semantic_cases": p["native_cases"],
        "native_semantic_evaluator": p["native_evaluator"],
        "v45_corpus_source": p["corpus"],
        "safety_model_guard": p["guard_model"],
        "safety_stop_model": p["guard_stop"],
        "safety_sdr_cache": p["guard_sdr"],
    }
    result.update({f"runtime_{name}": p["runtime_lib"] / name for name in RUNTIME_DSOS})
    return result


def inventory(p: dict[str, Path]) -> dict[str, Any]:
    required = required_sources(p)
    missing = {name: str(path) for name, path in required.items() if not path.is_file()}
    if p["worker"].is_file() and not (p["worker"].stat().st_mode & 0o111):
        missing["combo_worker_executable_bit"] = str(p["worker"])
    runtime_names = [name for name in RUNTIME_DSOS if (p["runtime_lib"] / name).is_file()]
    return {
        "status": "SOURCE_CLOSURE_READY" if not missing else "SOURCE_CLOSURE_INCOMPLETE",
        "missing": missing,
        "runtime_dso_count": len(runtime_names),
        "runtime_dsos": runtime_names,
        "runtime_lib_dir": str(p["runtime_lib"]),
        "helper_path": str(p["tail_bridge"]),
        "worker_path": str(p["worker"]),
        "worker_hash_policy": "captured_at_prepare_time_not_hardcoded",
    }


def patched_native_checker(source: Path) -> tuple[str, str]:
    text = source.read_text(encoding="utf-8")
    needle = "BASE = HERE.parents[1]"
    if text.count(needle) != 1:
        raise ValueError("verify_combo_native.py BASE anchor changed; update release path patch deliberately")
    patched = text.replace(needle,
        'BASE = Path(os.environ.get("AC922_EXPERIMENT_ROOT", str(HERE.parents[1])))', 1)
    return text, patched


def validate_smoke_report(path: Path, p: dict[str, Path]) -> dict[str, Any]:
    worker = p["worker"]
    report_path = path.resolve()
    # Only accept guarded fixture evidence; don't let an arbitrary hand-written
    # JSON outside the experiment tree authorize a release.
    fixture_root = p["base"]
    try:
        report_path.relative_to(p["base"] / "fixtures")
    except ValueError as exc:
        raise ValueError("combined smoke receipt must live below the experiment fixtures directory") from exc
    # Worker path is under production/combo-20261018/build-combo-c10; derive the
    # experiment root by walking known path components when using the default.
    for parent in report_path.parents:
        if parent.name == "fixtures":
            fixture_root = parent.parent
            break
    if report_path.name != "worker-smoke-result.json" or not report_path.is_file():
        raise ValueError("smoke receipt must be an existing worker-smoke-result.json")
    data = json.loads(report_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if data.get("status") != EXPECTED_SMOKE_STATUS:
        errors.append(f"combined smoke status is {data.get('status')!r}")
    if data.get("worker_exit") != 0:
        errors.append(f"combined smoke worker_exit is {data.get('worker_exit')!r}")
    started, finished = data.get("started"), data.get("finished")
    if not isinstance(started, (int, float)) or not isinstance(finished, (int, float)) or finished < started:
        errors.append("combined smoke timestamps are absent or invalid")
    if not isinstance(data.get("ready"), dict):
        errors.append("combined smoke lacks worker-ready record")
    waves = data.get("waves")
    if not isinstance(waves, list):
        errors.append("combined smoke lacks wave records")
        waves = []
    by_id = {row.get("id"): row for row in waves if isinstance(row, dict)}
    missing_waves = [wave for wave in EXPECTED_SMOKE_WAVES if wave not in by_id]
    if missing_waves:
        errors.append("combined smoke missing required waves: " + ",".join(missing_waves))
    for wave_id in EXPECTED_SMOKE_WAVES:
        row = by_id.get(wave_id)
        if row and row.get("errors"):
            errors.append(f"combined smoke wave {wave_id} contains errors")
    # Bind the smoke to the current C10 worker and all source/environment inputs
    # so a later rebuild or thermal/environment edit invalidates the receipt.
    artifacts = data.get("artifacts")
    if not isinstance(artifacts, dict):
        artifacts = {}
        errors.append("smoke receipt lacks artifact identity hashes")
    actual_worker_sha = sha256_file(worker)
    source_sha = sha256_file(p["combo"] / "bank-worker-combo.cpp")
    env_sha = sha256_file(p["env"])
    thermal_header_sha = sha256_file(p["combo"] / "thermal-pacing.h")
    thermal_policy_sha = sha256_file(p["combo"] / "combo_temperature_policy.py")
    for key, expected in (("worker_sha256", actual_worker_sha), ("source_sha256", source_sha),
                          ("environment_sha256", env_sha), ("thermal_header_sha256", thermal_header_sha),
                          ("thermal_policy_sha256", thermal_policy_sha)):
        if artifacts.get(key) != expected:
            errors.append(f"smoke artifacts.{key} is absent or differs from current release input")
    env_data = json.loads(p["env"].read_text(encoding="utf-8"))
    if data.get("runtime_build_id") != env_data.get("ROOT_RUNTIME_REVISION"):
        errors.append("smoke runtime_build_id differs from combo-environment.json")
    if data.get("thermal_policy") != {"gpu_stop_c": 65, "pause_c": 59, "resume_c": 54}:
        errors.append("smoke thermal policy differs from required 65/59/54 C contract")
    worker_path = data.get("worker_path") or artifacts.get("worker_path")
    if worker_path and Path(worker_path).resolve() != worker.resolve():
        errors.append("smoke receipt worker_path differs from selected worker binary")
    if errors:
        raise ValueError("combined smoke receipt rejected: " + "; ".join(errors))
    return {"status": "PASS", "path": str(report_path), "sha256": sha256_file(report_path),
            "worker_path": str(worker.resolve()), "worker_sha256": actual_worker_sha,
            "source_sha256": source_sha, "environment_sha256": env_sha,
            "thermal_header_sha256": thermal_header_sha, "thermal_policy_sha256": thermal_policy_sha,
            "started": started, "finished": finished, "wave_ids": list(EXPECTED_SMOKE_WAVES),
            "experiment_root": str(fixture_root)}


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def normalize_release_permissions(root: Path) -> None:
    """Make staged immutable assets readable/traversable by the debian unit user."""
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_symlink():
            continue
        if path.is_dir():
            path.chmod(0o755)
        elif path.is_file():
            executable = bool(path.stat().st_mode & 0o111)
            path.chmod(0o755 if executable else 0o644)
    root.chmod(0o755)


def verify_build_identity(build_report: Path, checker_source: Path) -> dict[str, str]:
    report = json.loads(build_report.read_text(encoding="utf-8"))
    checker = checker_source.read_text(encoding="utf-8")
    id_match = re.search(r'EXPECTED_RUNTIME_BUILD_ID\s*=\s*"([^"]+)"', checker)
    sha_match = re.search(r'EXPECTED_RUNTIME_REPORT_SHA256\s*=\s*"([0-9a-f]{64})"', checker)
    if not id_match or not sha_match:
        raise ValueError("native checker runtime pin anchors are missing")
    digest = sha256_file(build_report)
    if report.get("build_run_id") != id_match.group(1) or digest != sha_match.group(1):
        raise ValueError("runtime build report does not match verify_combo_native.py's R444 pin")
    if report.get("run_completed") is not True:
        raise ValueError("V444 build report is not marked run_completed")
    return {"build_run_id": id_match.group(1), "report_sha256": digest}


def make_release(args: argparse.Namespace, p: dict[str, Path], smoke_gate: dict[str, Any]) -> dict[str, Any]:
    root = args.production_root.resolve()
    releases = root / "releases"
    releases.mkdir(parents=True, exist_ok=True)
    version = args.version or ("combo-c10-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{2,95}", version):
        raise ValueError("invalid release version token")
    final = releases / version
    if final.exists() or final.is_symlink():
        raise FileExistsError(f"refusing to overwrite release path {final}")
    if not os.access(releases, os.X_OK):
        raise PermissionError(f"release root is not traversable: {releases}")
    sources = required_sources(p)
    missing = {name: str(path) for name, path in sources.items() if not path.is_file()}
    if p["worker"].is_file() and not (p["worker"].stat().st_mode & 0o111):
        missing["combo_worker_executable_bit"] = str(p["worker"])
    if missing:
        raise FileNotFoundError("production source closure incomplete: " + json.dumps(missing, sort_keys=True))

    temp = Path(tempfile.mkdtemp(prefix=f".{version}.staging-", dir=releases))
    try:
        temp.chmod(0o755)
        lib = temp / "lib"
        final_lib = final / "lib"
        lib.mkdir()
        for name in RUNTIME_DSOS:
            copy_file(p["runtime_lib"] / name, lib / name)
            alias = name.removesuffix(".0")
            alias_path = lib / alias
            if not alias_path.exists():
                alias_path.symlink_to(name)
        helper_name = "libac922-tail-bridge.so"
        copy_file(p["tail_bridge"], lib / helper_name)

        copy_file(p["combo"] / "README.md", temp / "README.md")
        copy_file(p["candidate"] / "api.py", temp / "api.py")
        copy_file(p["candidate"] / "llm_routes.py", temp / "llm_routes.py")
        copy_file(p["candidate"] / "test_api.py", temp / "test_api.py")
        copy_file(p["candidate"] / "verify_release.py", temp / "verify_release_base.py")
        copy_file(p["candidate"] / "wait_safety.py", temp / "wait_safety.py")
        for name in ("ac922-inference.service","ac922-inference-safety.service"):
            copy_file(p["candidate"] / name,temp / name)
        unit=temp / "ac922-inference.service"
        unit_text=unit.read_text()
        assert unit_text.count("[Service]\n")==1
        unit.write_text(unit_text.replace("[Service]\n","[Service]\nLimitMEMLOCK=4G\n",1))
        copy_file(p["combo"] / "20-bounded-pinned-memory.conf",temp / "20-bounded-pinned-memory.conf")
        copy_file(p["combo"] / "promote_combo_gateway.py",temp / "promote_combo_gateway.py")
        copy_file(p["combo"] / "production-shape-policy.h",temp / "production-shape-policy.h")
        copy_file(p["combo"] / "combo_safety_monitor.py", temp / "combo_safety_monitor.py")
        # The installed ac922-inference-safety.service still executes
        # current/safety_monitor.py; ship the combo monitor under that expected
        # entrypoint as well, without editing the installed unit.
        copy_file(p["combo"] / "combo_safety_monitor.py", temp / "safety_monitor.py")
        copy_file(p["combo"] / "combo_temperature_policy.py", temp / "combo_temperature_policy.py")
        copy_file(p["combo"] / "bank-worker-combo.cpp", temp / "bank-worker-combo.cpp")
        copy_file(p["combo"] / "production-ipc.h", temp / "production-ipc.h")
        copy_file(p["combo"] / "thermal-pacing.h", temp / "thermal-pacing.h")
        copy_file(p["combo"] / "build-combo.sh", temp / "build-combo.sh")
        copy_file(p["worker"], temp / "bank-worker")
        copy_file(p["build_report"], temp / "V444-build.json")
        copy_file(p["v35_checker"], temp / "verify-v35-preclean.py")

        original_checker, packaged_checker = patched_native_checker(p["combo"] / "verify_combo_native.py")
        (temp / "verify_combo_native.py").write_text(packaged_checker, encoding="utf-8")
        checker_build_pin = verify_build_identity(p["build_report"], p["combo"] / "verify_combo_native.py")
        environment = json.loads((p["combo"] / "combo-environment.json").read_text(encoding="utf-8"))
        if not isinstance(environment, dict) or not all(isinstance(k, str) for k in environment):
            raise ValueError("combo-environment.json must be a string-keyed JSON object")
        environment = {key: str(value) for key, value in environment.items() if value is not None}
        environment.update({
            "LD_LIBRARY_PATH": str(final_lib),
            "ROOT_DYNAMIC_LIB_DIR": str(final_lib),
            "ROOT_BUILD_REPORT": str(final / "V444-build.json"),
            "GGML_CUDA_GRAPH_DEVICE_TAIL_LIBRARY": str(final_lib / helper_name),
            "AC922_EXPERIMENT_ROOT": str(p["base"]),
            "ROOT_CUPTI_PHASE": "off",
            "ROOT_NSYS_PREFILL_PROFILE": "0",
            "LD_PRELOAD": "",
            "AC922_SYNC_AUDIT": "0",
            "AC922_POOL_TOKENS": "524288",
            "AC922_MIN_GPU_FREE_MIB": "256",
            "AC922_REQUEST_TOKENS": "262140",
        })
        (temp / "combo-environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")

        model = p["model"]
        worker_command = {
            "argv": [
                "/usr/bin/env", "-u", "LLAMA_QSA_INDEXER_TRACE", "-u", "LLAMA_MTP_BASE_DEVICE",
                "-u", "GGML_CUDA_ALLREDUCE", "-u", "LLAMA_ISLAND_SPLIT",
                "-u", "GGML_META_CPU_ASYNC_COPY_BUDGET_MIB", "-u", "GGML_META_CPU_ASYNC_COPY_MAX_BYTES",
                "/usr/bin/numactl", "--cpunodebind=0", "--membind=0,8", str(final / "bank-worker"),
                str(model), str(root / "run/model"), str(final / "verify_combo_native.py"),
                str(final / "verify-v35-preclean.py"),
            ],
            "env": environment,
        }
        (temp / "worker-command.json").write_text(json.dumps(worker_command, indent=2) + "\n", encoding="utf-8")

        wrapper = '''"""Start gate for a staged combo release; never promotes a release."""
import hashlib, json, subprocess, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
manifest=json.loads((HERE/"release-manifest.json").read_text())
gate=manifest.get("smoke_gate", {})
if not manifest.get("promotion_eligible") or gate.get("status") != "PASS":
    raise SystemExit("COMBO_SMOKE_GATE_PENDING: current symlink must not point here")
smoke=Path(gate["path"])
if hashlib.sha256(smoke.read_bytes()).hexdigest() != gate.get("sha256"):
    raise SystemExit("combined smoke receipt changed after preparation")
data=json.loads(smoke.read_text())
if data.get("status") != "COMBO_NATIVE_PREFILL32K_D3_FOUR512_CAPACITY_PASS" or data.get("worker_exit") != 0:
    raise SystemExit("combined smoke receipt is not a real PASS")
if data.get("artifacts", {}).get("worker_sha256") != manifest.get("worker", {}).get("sha256"):
    raise SystemExit("combined smoke worker hash does not match this release")
subprocess.run([sys.executable, str(HERE/"verify_release_base.py")], check=True)
'''
        (temp / "verify_release.py").write_text(wrapper, encoding="utf-8")

        # Dynamic linker checks validate the staged closure without launching a
        # model or initializing CUDA. The GPU smoke remains a separate gate.
        linker_env = dict(os.environ, LD_LIBRARY_PATH=str(lib), LD_PRELOAD="")
        for binary in (temp / "bank-worker", lib / helper_name):
            result = subprocess.run(["ldd", "-r", str(binary)], env=linker_env,
                                    capture_output=True, text=True, timeout=30, check=True)
            if "not found" in result.stdout or "undefined symbol" in result.stdout:
                raise RuntimeError(f"linker closure failed for {binary}: {result.stdout[-3000:]} {result.stderr[-1000:]}")
            if binary.name == "bank-worker":
                for name in RUNTIME_DSOS:
                    if f"{name} => {lib}/{name}" not in result.stdout:
                        raise RuntimeError(f"unexpected runtime resolution for {name}")
                (temp / "ldd-worker.txt").write_text(result.stdout.replace(str(temp), str(final)), encoding="utf-8")
            else:
                (temp / "ldd-tail-bridge.txt").write_text(result.stdout.replace(str(temp), str(final)), encoding="utf-8")

        external_inputs = [p["native_cases"], p["native_evaluator"],
                           p["corpus"], p["guard_model"], p["guard_stop"], p["guard_sdr"]]
        external = {str(path): file_info(path) for path in external_inputs}
        files = {}
        if smoke_gate.get("status") == "PASS":
            source_smoke = Path(smoke_gate["path"])
            copy_file(source_smoke, temp / "smoke-acceptance.json")
            smoke_gate = dict(smoke_gate)
            smoke_gate["source_path"] = str(source_smoke)
            smoke_gate["path"] = str(final / "smoke-acceptance.json")
        for item in sorted(temp.rglob("*")):
            if item.is_file() and item.name != "release-manifest.json":
                files[str(item.relative_to(temp))] = sha256_file(item)
        model_stat = model.stat()
        manifest = {
            "schema": "ac922-combo-release-v1",
            "version": version,
            "status": "FROZEN_NOT_PROMOTED",
            "prepared_unix": time.time(),
            "prepare_only": True,
            "current_symlink_mutated": False,
            "systemd_mutated": False,
            "gateway_mutated": False,
            "worker": file_info(p["worker"]),
            "worker_source": str(p["combo"] / "bank-worker-combo.cpp"),
            "runtime": {"build": checker_build_pin, "runtime_dir": str(p["runtime_lib"]),
                        "private_dsos": list(RUNTIME_DSOS), "tail_bridge": file_info(p["tail_bridge"])},
            "model_identity": [{"path": str(model), "bytes": model_stat.st_size, "mtime_ns": model_stat.st_mtime_ns}],
            "smoke_gate": smoke_gate,
            "promotion_eligible": smoke_gate.get("status") == "PASS",
            "api_sources": {name: sha256_file(p["candidate"] / name) for name in ("api.py", "llm_routes.py")},
            "checker_source_sha256": hashlib.sha256(original_checker.encode("utf-8")).hexdigest(),
            "checker_packaged_sha256": sha256_file(temp / "verify_combo_native.py"),
            "external_inputs": external,
            "sha256": {**files, **{name: record["sha256"] for name, record in external.items()}},
            "promotion_contract": "root-owned integration must verify promotion_eligible and the PASS receipt before atomically replacing production/current; this script has no promotion path",
        }
        (temp / "release-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        normalize_release_permissions(temp)
        temp.rename(final)
        return {"status": manifest["status"], "release": str(final),
                "promotion_eligible": manifest["promotion_eligible"],
                "worker_sha256": manifest["worker"]["sha256"],
                "private_dso_count": len(RUNTIME_DSOS), "tail_bridge": helper_name,
                "current_symlink_mutated": False, "systemd_mutated": False, "gateway_mutated": False}
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def main() -> int:
    args = parse_args()
    p = paths(args)
    report = inventory(p)
    if args.check_only:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["status"] == "SOURCE_CLOSURE_READY" else 1
    if report["missing"]:
        raise SystemExit("Refusing incomplete source closure: " + json.dumps(report["missing"], sort_keys=True))
    # No receipt is acceptable for staging, but it is never promotable. A
    # supplied receipt must be complete and bound to the exact selected worker.
    smoke_gate = {"status": "PENDING", "reason": "no real combined smoke receipt supplied"}
    if args.smoke_report:
        smoke_gate = validate_smoke_report(args.smoke_report, p)
    result = make_release(args, p, smoke_gate)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
