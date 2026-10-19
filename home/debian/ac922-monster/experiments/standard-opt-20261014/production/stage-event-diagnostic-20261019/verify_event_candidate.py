#!/usr/bin/env python3
"""Fail-closed identity gate for the private event worker and observer DSO."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
ROOT = HERE.parents[3]
PRODUCTION = ROOT / "production"
L5 = PRODUCTION / "releases/20261019-r444-banked-l5"
SAFE = PROJECT / "production/safe-pool-reject-20261019"
OBSERVER = PROJECT / "production/captured-argument-observer-20261019/libac922-tail-observer.so"
SCENARIO_SOURCE = PROJECT / "fixtures/fourbank-d3-newruntime-v306-20261017/v221-d3-scenarios.json"
WORKER_SHA256 = "8b7654c41987acca55d168b58520dced2ec52c71c9c251ddbccf14edbdbbfa7b"
WORKER_SOURCE_SHA256 = "25edba5f4dbd55e8cd242b27bbf82838bffa8dc5fe1f8b8b1f808dca220c09f9"
OBSERVER_SHA256 = "80b433e73a93ef095e3bd04ed35dd7f64bb4c8b60103f6a88b14d624780857db"
OBSERVER_NAME = "libac922-tail-observer.so"
R5_TAIL_NAME = "libac922-tail-bridge.so"
API_PORT = "8012"
CONTROL_SOCKET = str(PRODUCTION / "stage-event-control.sock")
OVERRIDE = Path("/etc/systemd/system/ac922-inference.service.d/96-stage-event-diagnostic.conf")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read JSON {path}: {error}") from None
    require(isinstance(value, dict), f"expected JSON object in {path}")
    return value


def baseline_command() -> dict[str, Any]:
    for candidate in (HERE / "l5-frozen-worker-command.json",
                      L5 / "lastwave-worker-command.json", SAFE / "lastwave-worker-command.json"):
        if candidate.is_file():
            return read_json(candidate)
    raise RuntimeError("frozen L5 worker command is missing")


def worker_index(argv: list[str]) -> int:
    indices = [i for i, value in enumerate(argv)
               if isinstance(value, str) and Path(value).name == "bank-worker"]
    require(len(indices) == 1, "worker argv must contain one bank-worker")
    return indices[0]


def verify(runtime_required: bool = False) -> dict[str, Any]:
    report = read_json(HERE / "build/build-report.json")
    require(report.get("status") == "BUILD_PASS", "private worker build receipt is not BUILD_PASS")
    require(report.get("worker_sha256") == WORKER_SHA256,
            "private event worker hash differs from the pinned build receipt")
    worker_path = HERE / "build/bank-worker"
    require(worker_path.is_file() and sha(worker_path) == WORKER_SHA256,
            "private event worker binary is missing or differs from the pinned SHA-256")
    require(report.get("source_sha256") == WORKER_SOURCE_SHA256 and
            sha(HERE / "bank-worker-combo.cpp") == WORKER_SOURCE_SHA256,
            "private worker source differs from the tested event-selector source")

    l5_report = read_json(L5 / "build-lastwave/build-report.json")
    old_libraries = l5_report.get("libraries")
    libraries = report.get("libraries")
    require(isinstance(old_libraries, dict) and isinstance(libraries, dict),
            "frozen and candidate build reports must list engine DSOs")
    require(len(old_libraries) == 7 and set(libraries) == set(old_libraries),
            "private worker must use exactly the seven frozen L5 engine DSOs")
    require(libraries == old_libraries, "one or more candidate engine DSOs differ from L5")
    libdir = HERE / "build/lib"
    actual_engine = {p.name: sha(p) for p in libdir.iterdir()
                     if p.is_file() and not p.is_symlink() and p.name in old_libraries}
    require(actual_engine == old_libraries, "build/lib engine DSO bytes differ from L5")
    require(set(actual_engine) == set(old_libraries), "a frozen L5 engine DSO is missing")
    forbidden_libraries = {"libac922-tail-bridge-v3.so", "libac922-tail-bridge-b96.so"}
    require(not any(p.name in forbidden_libraries for p in libdir.iterdir() if p.is_file()),
            "candidate build directory contains a forbidden V3/B96 bridge library")

    original_tail = L5 / "build-lastwave/lib" / R5_TAIL_NAME
    candidate_tail = libdir / R5_TAIL_NAME
    require(original_tail.is_file() and candidate_tail.is_file(), "frozen R5 tail bridge is missing")
    require(sha(candidate_tail) == sha(original_tail) == report.get("original_r5_sha256"),
            "candidate must retain the exact frozen R5 tail bridge in build/lib")
    require(OBSERVER.is_file() and sha(OBSERVER) == OBSERVER_SHA256,
            "captured-argument observer DSO is missing or has an unexpected SHA-256")

    command = read_json(HERE / "lastwave-worker-command.json")
    baseline = baseline_command()
    argv, base_argv = command.get("argv"), baseline.get("argv")
    env, base_env = command.get("env"), baseline.get("env")
    require(isinstance(argv, list) and isinstance(base_argv, list) and
            isinstance(env, dict) and isinstance(base_env, dict),
            "worker command must contain argv and env objects")
    index, base_index = worker_index(argv), worker_index(base_argv)
    expected = list(base_argv)
    expected[base_index] = str(HERE / "build/bank-worker")
    expected[base_index + 2] = str(HERE / "run/model")
    require(argv == expected, "worker argv differs from frozen L5 beyond worker and private output root")
    require(argv[index + 1] == base_argv[base_index + 1], "diagnostic model path differs from frozen L5")
    require(argv[index + 3:] == base_argv[base_index + 3:],
            "diagnostic native/preclean helpers differ from frozen L5")
    require(Path(argv[index + 1]).is_file(), "frozen IQ3_XXS model is missing")

    allowed = {"ROOT_DYNAMIC_LIB_DIR", "LD_LIBRARY_PATH", "GGML_CUDA_GRAPH_DEVICE_TAIL_LIBRARY",
               "AC922_STAGE_EVENT_DIAGNOSTIC", "AC922_CAPTURED_ARGUMENT_OBSERVER",
               "AC922_LASTWAVE_RECORD_OUTPUTS"}
    for key in set(env) | set(base_env):
        if key not in allowed:
            require(env.get(key) == base_env.get(key), f"worker environment changed from L5: {key}")
    libdir_text = str(libdir)
    old_libdir = base_env.get("ROOT_DYNAMIC_LIB_DIR")
    require(isinstance(old_libdir, str) and old_libdir in base_env.get("LD_LIBRARY_PATH", ""),
            "frozen L5 library path cannot be rewritten safely")
    require(env.get("ROOT_DYNAMIC_LIB_DIR") == libdir_text and
            env.get("LD_LIBRARY_PATH") == base_env["LD_LIBRARY_PATH"].replace(old_libdir, libdir_text, 1),
            "worker runtime must load the private directory containing exactly L5 DSOs")
    require(env.get("GGML_CUDA_GRAPH_DEVICE_TAIL_LIBRARY") == str(OBSERVER),
            "R5 tail helper variable does not point at the captured-argument observer")
    require(env.get("AC922_CAPTURED_ARGUMENT_OBSERVER") == "1" and
            env.get("AC922_STAGE_EVENT_DIAGNOSTIC") == "1",
            "worker command must opt in to both the observer and event diagnostic")
    require(env.get("AC922_LASTWAVE_RECORD_OUTPUTS") == "1",
            "diagnostic must retain full native flow/token-ID records")
    require(env.get("LD_PRELOAD", "") == "", "LD_PRELOAD must remain empty")
    for key, value in {"GGML_CUDA_GRAPH_DEVICE_TAIL": "1",
                       "GGML_CUDA_GRAPH_DEVICE_TAIL_CAPTURE_ONLY": "1",
                       "LLAMA_TARGET_GRAPH_DEVICE_TAIL": "0",
                       "LLAMA_MTP_GRAPH_DEVICE_TAIL": "0",
                       "AC922_POOL_TOKENS": "524288",
                       "ROOT_MTP_DEPTH": "3",
                       "LLAMA_MTP_PAD": "6",
                       "LLAMA_QSA_INDEXER_IMPL": "standard"}.items():
        require(env.get(key) == value, f"frozen L5 requirement changed: {key}")
    forbidden_paths = ("/parallel-prepare-20261019/", "/prepare-pool-20261019/",
                       "/b96/", "libac922-tail-bridge-v3.so")
    require(not any(any(term in str(value).lower() for term in forbidden_paths)
                    for value in argv + list(env.values())),
            "worker command points at a forbidden parallel-prepare/B96/V3 path")

    bootstrap = read_json(HERE / "diagnostic-bootstrap.json")
    require(bootstrap.get("worker_sha256") == WORKER_SHA256 and
            bootstrap.get("observer_sha256") == OBSERVER_SHA256,
            "bootstrap identity does not match the worker and observer")
    require(bootstrap.get("private_api_port") == 8012 and
            bootstrap.get("former_inference_backend_port") == 8007 and
            bootstrap.get("public_gateway_unit") == "ac922-apigateway.service",
            "bootstrap must keep the private API on 8012 and leave the public gateway unit untouched")
    require(bootstrap.get("engine_dso_names") == sorted(old_libraries),
            "bootstrap DSO list differs from the exact L5 set")
    stories = bootstrap.get("four512_requests")
    require(isinstance(stories, list) and len(stories) == 4 and
            all(request.get("max_tokens") == 512 and request.get("cache_prompt") is False
                for request in stories),
            "bootstrap must retain exactly four uncached 512-token story probes")
    require(sha(SCENARIO_SOURCE) == bootstrap.get("standard_scenario_sha256"),
            "standard D3 probe fixture hash changed")
    cpu_case = read_json(HERE / "policy_cpu_test_receipt.json")
    require(cpu_case.get("status") == "PASS" and
            cpu_case.get("policy_sha256") == sha(HERE / "stage-event-diagnostic-policy.h"),
            "CPU admission-policy case is missing or stale")

    scenario_path = HERE / "run/model/copy0/fourbank16k-d3/scenarios.json"
    if runtime_required:
        require(scenario_path.is_file() and sha(scenario_path) == sha(SCENARIO_SOURCE),
                "private run/model D3 scenario bootstrap is missing or changed")
        require((HERE / "run/api").is_dir(), "private API log directory is missing")
    elif scenario_path.exists():
        require(sha(scenario_path) == sha(SCENARIO_SOURCE), "existing private D3 scenario bootstrap changed")

    if os.environ.get("AC922_API_PORT"):
        require(os.environ["AC922_API_PORT"] == API_PORT, "private API must use port 8012")
    if os.environ.get("AC922_ACCEPTANCE_SOCKET"):
        require(os.environ["AC922_ACCEPTANCE_SOCKET"] == CONTROL_SOCKET,
                "private in-band control must use the new stage-event socket")
    print("STAGE_EVENT_IDENTITY_PASS worker=8b7654c4 observer=80b433e7 engine_dsos=7 api=8012 gateway=8007", flush=True)
    return {"status": "VERIFY_PASS", "worker_sha256": WORKER_SHA256,
            "worker_source_sha256": WORKER_SOURCE_SHA256, "observer_sha256": OBSERVER_SHA256,
            "engine_dsos": sorted(old_libraries), "scenario_sha256": bootstrap["standard_scenario_sha256"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-required", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.runtime_required), sort_keys=True))
