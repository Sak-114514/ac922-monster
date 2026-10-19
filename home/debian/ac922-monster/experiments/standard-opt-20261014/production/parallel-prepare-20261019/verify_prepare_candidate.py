#!/usr/bin/env python3
"""Fail-closed identity and six-device fixture gate for the private prepare candidate."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PRODUCTION = Path("/home/debian/ac922-monster/production")
L5 = PRODUCTION / "releases/20261019-r444-banked-l5"
MODEL = Path("/home/debian/models/qwen3.8-flash-next-gsq-rco/integrated/IQ3_XXS-MTP-Q8.gguf")
HELPER_V3 = Path("/home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/six-cuda-prepare-v436-root-20261018/helper/build-v436/libac922-tail-bridge-v3.so")
CUDA_LIB = "libggml-cuda.so.0"
TAIL_LIB = "libac922-tail-bridge.so"
PARALLEL_FLAGS = {
    "GGML_SCHED_LAYER_PARALLEL_PREPARE": "1",
    "GGML_SCHED_LAYER_GPU_SPANS": "1",
    "GGML_META_GPU_SPANS": "1",
    "GGML_META_GPU_SPANS_PERIOD": "1",
    "AC922_LASTWAVE_RECORD_OUTPUTS": "1",
}
TAIL_LIBRARY_ENV = "GGML_CUDA_GRAPH_DEVICE_TAIL_LIBRARY"
PATH_OVERRIDES = {"ROOT_DYNAMIC_LIB_DIR", "LD_LIBRARY_PATH", TAIL_LIBRARY_ENV}
ALLOWED_ENV_OVERRIDES = PATH_OVERRIDES | set(PARALLEL_FLAGS)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read JSON receipt {path}: {error}") from None
    require(isinstance(value, dict), f"expected JSON object in {path}")
    return value


def worker_command(directory: Path) -> dict[str, Any]:
    for name in ("lastwave-worker-command.json", "worker-command.json"):
        path = directory / name
        if path.is_file():
            return read_json(path)
    raise RuntimeError(f"frozen L5 worker command is missing under {directory}")


def worker_index(argv: list[str]) -> int:
    matches = [index for index, arg in enumerate(argv)
               if isinstance(arg, str) and Path(arg).name == "bank-worker"]
    require(len(matches) == 1, "worker argv must contain exactly one bank-worker")
    return matches[0]


def device_ids(receipt: dict[str, Any]) -> list[int]:
    raw = receipt.get("devices", receipt.get("device_ids"))
    require(isinstance(raw, list), "fixture-result.json needs a six-device list in devices")
    ids: list[int] = []
    for item in raw:
        if isinstance(item, int):
            ids.append(item)
        elif isinstance(item, dict):
            value = item.get("device_id", item.get("id"))
            require(isinstance(value, int), "fixture device entries need integer id/device_id")
            ids.append(value)
        else:
            raise RuntimeError("fixture device entries must be integer IDs or objects")
    return ids


def verify() -> dict[str, Any]:
    report = read_json(HERE / "build/build-report.json")
    require(report.get("status") == "BUILD_PASS", "private build report is not BUILD_PASS")
    require(report.get("cuda_tus") == 1 and report.get("template_tus") == 0,
            "candidate must contain exactly one CUDA TU and zero template TUs")

    old_report = read_json(L5 / "build-lastwave/build-report.json")
    frozen_worker = L5 / "build-lastwave/bank-worker"
    require(frozen_worker.is_file(), "frozen L5 bank-worker is missing")
    frozen_worker_sha = sha(frozen_worker)
    require(report.get("worker_reused_sha256") == frozen_worker_sha,
            "candidate core worker SHA does not match frozen L5")
    require(report.get("worker_reused_sha256") == old_report.get("worker_sha256"),
            "build receipt core worker SHA differs from the L5 build receipt")

    old_libraries = old_report.get("libraries")
    new_libraries = report.get("libraries")
    require(isinstance(old_libraries, dict) and isinstance(new_libraries, dict),
            "L5 and candidate build reports must list library hashes")
    require(CUDA_LIB in old_libraries and CUDA_LIB in new_libraries,
            "L5/candidate build report must include libggml-cuda.so.0")
    require(report.get("control_relink_exact_sha256") == old_libraries[CUDA_LIB],
            "relink control does not reproduce the frozen L5 CUDA DSO")

    build_lib = HERE / "build/lib"
    actual_libraries = {path.name: sha(path) for path in build_lib.iterdir()
                        if path.is_file() and not path.is_symlink()}
    require(actual_libraries == new_libraries,
            "private build/lib contents differ from build-report.json")
    unchanged = [name for name, expected in old_libraries.items()
                 if name not in {CUDA_LIB, TAIL_LIB}]
    require(len(unchanged) == 6,
            f"expected six unchanged non-CUDA/non-helper engine DSOs, found {len(unchanged)}")
    for name in unchanged:
        require(new_libraries.get(name) == old_libraries[name],
                f"engine DSO differs from frozen L5: {name}")
    require(set(old_libraries).issubset(new_libraries), "candidate dropped an L5 engine DSO")
    allowed_changed = {CUDA_LIB, TAIL_LIB}
    changed = {name for name in old_libraries
               if new_libraries.get(name) != old_libraries[name]}
    added = set(new_libraries) - set(old_libraries)
    require(CUDA_LIB in changed and changed <= allowed_changed and added <= {TAIL_LIB},
            f"unexpected engine-library delta: changed={sorted(changed)} added={sorted(added)}")
    require(TAIL_LIB in new_libraries, "V3 helper DSO is absent from the private build report")
    require(HELPER_V3.is_file() and new_libraries[TAIL_LIB] == sha(HELPER_V3),
            "candidate helper is not the frozen V3 bridge DSO")

    fixture = read_json(HERE / "fixture-result.json")
    require(fixture.get("status") == "PASS", "six-device prepare fixture is not PASS")
    ids = device_ids(fixture)
    require(len(ids) == 6 and set(ids) == set(range(6)),
            f"fixture must enumerate exactly devices 0..5, got {ids}")
    require(fixture.get("guard_on") is True and fixture.get("old_guard_negative_reproduced") is True,
            "fixture must prove the old guard fails and candidate guard is on")
    require(fixture.get("native_frames") == 24 and fixture.get("changed_cpu_payloads") == 24,
            "fixture must exercise all 24 distinct CPU payload frames")
    require(fixture.get("oracle") == "exact" and fixture.get("prepare_sentinel") == "unchanged" and
            fixture.get("parent_hint_restored") is True,
            "fixture oracle, prepare sentinel, or TLS restore gate failed")
    cases = fixture.get("cases")
    require(isinstance(cases, list), "fixture must retain negative and positive case receipts")
    case_by_name = {case.get("case"): case for case in cases if isinstance(case, dict)}
    require(case_by_name.get("guard-l5-negative-c1", {}).get("returncode") != 0 and
            case_by_name.get("guard-candidate-positive-c2", {}).get("returncode") == 0,
            "fixture must include the expected failing-L5 and passing-candidate cases")

    fixture_libraries = fixture.get("runtime_libraries_sha256")
    require(isinstance(fixture_libraries, dict),
            "fixture-result.json must include runtime_libraries_sha256")
    require(fixture_libraries == new_libraries,
            "fixture loaded-library SHA set does not match all eight build-reported DSOs")

    for name in ("api.py", "candidate_server.py"):
        private = HERE / name
        frozen = L5 / name
        require(private.is_file() and frozen.is_file(), f"missing API/server source: {name}")
        require(sha(private) == sha(frozen), f"API/server file differs from frozen L5: {name}")

    command = read_json(HERE / "lastwave-worker-command.json")
    baseline = worker_command(L5)
    argv = command.get("argv")
    baseline_argv = baseline.get("argv")
    require(isinstance(argv, list) and isinstance(baseline_argv, list),
            "candidate and L5 worker command must contain argv arrays")
    index = worker_index(argv)
    baseline_index = worker_index(baseline_argv)
    expected_argv = list(baseline_argv)
    require(str(frozen_worker) == baseline_argv[baseline_index],
            "frozen L5 command does not point to the exact build-lastwave worker")
    require(baseline_argv[baseline_index + 1] == str(MODEL),
            "frozen L5 worker command does not use the original model file")
    require(baseline_argv[baseline_index + 3:] == [
        str(L5 / "verify_lastwave_native.py"), str(L5 / "verify-v35-preclean.py")],
        "frozen L5 worker command does not use the two frozen native helpers")
    expected_argv[baseline_index + 2] = str(HERE / "run/model")
    require(argv == expected_argv,
            "candidate argv must match frozen L5 except for HERE/run/model at worker argument +2")
    require(MODEL.is_file(), "original IQ3_XXS-MTP-Q8 model file is missing")
    model_runtime = HERE / "run/model"
    for runtime_dir in (model_runtime, HERE / "run/api"):
        if runtime_dir.exists():
            require(runtime_dir.is_dir() and not runtime_dir.is_symlink() and
                    os.access(runtime_dir, os.W_OK | os.X_OK),
                    f"existing runtime directory is not writable by debian: {runtime_dir}")

    env = command.get("env")
    baseline_env = baseline.get("env")
    require(isinstance(env, dict) and isinstance(baseline_env, dict),
            "candidate and L5 worker command must contain env objects")
    for key in set(env) | set(baseline_env):
        if key not in ALLOWED_ENV_OVERRIDES:
            require(env.get(key) == baseline_env.get(key),
                    f"environment differs from L5 outside approved prepare/path flags: {key}")
    for key, value in PARALLEL_FLAGS.items():
        require(env.get(key) == value, f"candidate prepare flag must be {key}={value}")
    expected_lib = str(build_lib)
    require(env.get("ROOT_DYNAMIC_LIB_DIR") == expected_lib,
            "ROOT_DYNAMIC_LIB_DIR must point at the private candidate DSOs")
    baseline_lib = baseline_env.get("ROOT_DYNAMIC_LIB_DIR")
    baseline_ld = baseline_env.get("LD_LIBRARY_PATH")
    require(isinstance(baseline_lib, str) and isinstance(baseline_ld, str) and
            baseline_lib in baseline_ld,
            "frozen L5 library search path cannot be normalized safely")
    expected_ld = baseline_ld.replace(baseline_lib, expected_lib, 1)
    require(env.get("LD_LIBRARY_PATH") == expected_ld,
            "LD_LIBRARY_PATH must differ only by replacing the frozen L5 library directory")
    require(env.get(TAIL_LIBRARY_ENV) == str(build_lib / TAIL_LIB),
            f"{TAIL_LIBRARY_ENV} must point at the build-reported V3 helper")

    for key, value in {
        "AC922_POOL_TOKENS": "524288",
        "AC922_REQUEST_TOKENS": "262140",
        "ROOT_MTP_DEPTH": "3",
        "LLAMA_MTP_PAD": "6",
        "LLAMA_MTP_GPU_CHAIN": "1",
        "LLAMA_QSA_INDEXER_IMPL": "standard",
        "GGML_CUDA_GRAPH_OPT": "1",
        "ROOT_CUPTI_PHASE": "off",
        "ROOT_NSYS_PREFILL_PROFILE": "0",
    }.items():
        require(env.get(key) == value, f"qualified L5 serving setting changed: {key}")
    require(env.get("GGML_CUDA_DISABLE_GRAPHS") in (None, "0", "off", "false"),
            "candidate must not disable CUDA graphs")
    for key, value in env.items():
        if "CUPTI" in key or "NSYS" in key:
            require(value == baseline_env.get(key), f"CUPTI/NSYS setting changed: {key}")

    return {"status": "VERIFY_PASS", "cuda_tus": 1, "template_tus": 0,
            "l5_worker_sha256": frozen_worker_sha,
            "candidate_cuda_sha256": new_libraries[CUDA_LIB],
            "candidate_helper_sha256": new_libraries[TAIL_LIB],
            "unchanged_l5_engine_dsos": unchanged, "fixture_devices": sorted(ids)}


if __name__ == "__main__":
    print(json.dumps(verify(), sort_keys=True))
