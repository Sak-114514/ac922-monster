"""Validate the private host delta and immutable baseline before model startup."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(require_acceptance: bool = False, identity_only: bool = False) -> None:
    manifest = json.loads((HERE / "candidate-manifest.json").read_text())
    for relative, expected in manifest["sha256"].items():
        actual = sha(HERE / relative)
        if actual != expected:
            raise RuntimeError("candidate artifact changed: " + relative)
    report = json.loads((HERE / "build-lastwave/build-report.json").read_text())
    if report["status"] != "BUILD_PASS" or report["cuda_tus"] != 0:
        raise RuntimeError("expected successful host-only incremental build")
    if sha(HERE / "build-lastwave/bank-worker") != report["worker_sha256"]:
        raise RuntimeError("worker hash differs from build receipt")
    if sha(HERE / "bank-worker-combo.cpp") != report["source_sha256"]:
        raise RuntimeError("worker source changed since build")
    for name, expected in report["headers"].items():
        if sha(HERE / name) != expected:
            raise RuntimeError("worker header changed since build: " + name)
    for name, expected in report["libraries"].items():
        if sha(HERE / "build-lastwave/lib" / name) != expected:
            raise RuntimeError("private library changed: " + name)
    for relative, expected in report["host_source_sha256"].items():
        if sha(HERE / "host-src" / relative) != expected:
            raise RuntimeError("private engine source changed: " + relative)
    cmd = json.loads((HERE / "lastwave-worker-command.json").read_text())
    env = cmd["env"]
    for key, value in {"ROOT_MTP_DEPTH": "3", "LLAMA_MTP_PAD": "6", "LLAMA_MTP_GPU_CHAIN": "1",
                       "LLAMA_QSA_INDEXER_IMPL": "standard", "AC922_POOL_TOKENS": "524288",
                       "AC922_REQUEST_TOKENS": "262140", "AC922_MIN_GPU_FREE_MIB": "256"}.items():
        if env.get(key) != value:
            raise RuntimeError("unqualified serving configuration: " + key)
    if not identity_only and env.get("AC922_CONCURRENT_SETUP") == "1":
        if env.get("GGML_CUDA_ONECAT_INIT_READY") != "1":
            raise RuntimeError("parallel setup requires per-device initialization readiness")
        gate = json.loads((HERE / "onecat-fixture-receipt.json").read_text())
        if gate.get("status") != "PASS" or len(gate.get("cases", [])) != 6:
            raise RuntimeError("all-six-device simultaneous first-capture gate is missing")
        for name, expected in gate["runtime_libraries_sha256"].items():
            if sha(HERE / "build-lastwave/lib" / name) != expected:
                raise RuntimeError("concurrent-setup fixture runtime changed: " + name)
    if require_acceptance:
        receipt = json.loads((HERE / "lastwave-acceptance.json").read_text())
        if receipt.get("status") != "PASS" or receipt.get("worker_sha256") != report["worker_sha256"]:
            raise RuntimeError("matching real-model acceptance is missing")
    print("LASTWAVE_CANDIDATE_IDENTITY_PASS cuda_tus=0 fixed_D3=1 pool=524288", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-acceptance", action="store_true")
    parser.add_argument("--identity-only", action="store_true")
    options = parser.parse_args()
    verify(options.require_acceptance, options.identity_only)
