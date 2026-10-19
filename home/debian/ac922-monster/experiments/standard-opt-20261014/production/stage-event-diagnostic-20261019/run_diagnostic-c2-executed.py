#!/usr/bin/env python3
"""Run one bounded stage-event diagnostic window on an already staged private API."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import time
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
PRODUCTION = PROJECT.parents[1] / "production"
CONTROL_SOCKET = PRODUCTION / "stage-event-control.sock"
API_PORT = 8012
RUN = HERE / "run"
WAVES = RUN / "waves"
LOGS = RUN / "logs"
WORKER_LOG = RUN / "api/worker.log"
RECEIPT = RUN / "diagnostic-receipt.json"
LOG = RUN / "diagnostic.log"
SCENARIOS = PROJECT / "fixtures/fourbank-d3-newruntime-v306-20261017/v221-d3-scenarios.json"
FOUR512_EVIDENCE = PROJECT / "production/safe-pool-reject-20261019/evidence/four512-flow-record.json"
EXPECTED_WORKER_SHA = "8b7654c41987acca55d168b58520dced2ec52c71c9c251ddbccf14edbdbbfa7b"
EXPECTED_OBSERVER_SHA = "80b433e73a93ef095e3bd04ed35dd7f64bb4c8b60103f6a88b14d624780857db"

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {message}\n"
    print(line, end="", flush=True)
    with LOG.open("a", encoding="utf-8") as output:
        output.write(line)
        output.flush()


def wait_ready(timeout_s: int = 600) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urlopen(f"http://127.0.0.1:{API_PORT}/ready", timeout=3) as response:
                value = json.load(response)
            if value.get("status") == "ready":
                return value
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError("private API 8012 did not become ready within ten minutes")


def send_wave(label: str, requests: list[dict], index: int, timeout_s: int = 600) -> dict:
    message = {"id": label, "requests": requests, "refill": False}
    started_us = time.monotonic_ns() // 1000
    with socket.socket(socket.AF_UNIX) as connection:
        connection.settimeout(timeout_s)
        connection.connect(str(CONTROL_SOCKET))
        connection.sendall(json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode() + b"\n")
        stream = connection.makefile("rb")
        raw = stream.readline(8 << 20)
        if not raw.endswith(b"\n"):
            raise RuntimeError(f"wave {label} returned no complete IPC JSON line")
        result = json.loads(raw)
    ended_us = time.monotonic_ns() // 1000
    result["host_start_monotonic_us"] = started_us
    result["host_end_monotonic_us"] = ended_us
    result["host_full_wave_us"] = ended_us - started_us
    result["measurement_scope"] = "host send through complete IPC response; includes full wave overhead"
    result["request_sha256"] = hashlib.sha256(
        json.dumps(requests, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    atomic_json(WAVES / f"{index:02d}-{label}.json", {"request": message, "result": result})
    return result


def validate_wave(result: dict, requests: list[dict], expected_worker_pid: int | None) -> int:
    if result.get("status") == "ERROR":
        raise RuntimeError(f"IPC wave error: {result.get('error')}")
    expected_ids = {request["id"] for request in requests}
    if result.get("errors"):
        raise RuntimeError(f"worker emitted request errors: {result['errors']}")
    if set(result.get("done", {})) != expected_ids:
        raise RuntimeError(f"done IDs mismatch: {sorted(result.get('done', {}))} vs {sorted(expected_ids)}")
    pid = result.get("worker_pid")
    if not isinstance(pid, int) or (expected_worker_pid is not None and pid != expected_worker_pid):
        raise RuntimeError(f"worker PID changed across the combined window: {pid} != {expected_worker_pid}")
    if result.get("timings", {}).get("cached_tokens", 0) != 0:
        raise RuntimeError("a supposedly uncached diagnostic wave reported cached input tokens")
    return pid


def source_inputs() -> tuple[dict, list[dict], list[dict]]:
    scenarios = json.loads(SCENARIOS.read_text(encoding="utf-8"))
    if len(scenarios) < 2 or scenarios[1].get("label") != "v221-d3-fourbank-4x16k-512":
        raise RuntimeError("standard fourbank D3 fixture layout changed")
    native = [dict(request) for request in scenarios[0]["requests"]]
    bootstrap = json.loads((HERE / "diagnostic-bootstrap.json").read_text(encoding="utf-8"))
    stories = bootstrap.get("four512_requests")
    if not isinstance(stories, list) or len(stories) != 4:
        raise RuntimeError("bootstrap must retain the exact four previously accepted story prompts")
    if any(request.get("max_tokens") != 512 or request.get("cache_prompt") is not False
           for request in stories):
        raise RuntimeError("four512 requests must remain 512-token and uncached")
    corpus = {"id": "stage-event-corpus32", "corpus_token_start": 0,
              "corpus_token_count": 32768,
              "prompt_source": "v45-curated-real-corpus-runtime-token-slice",
              "max_tokens": 8, "ignore_eos": True, "cache_prompt": False}
    return corpus, native, stories


def assert_native(texts: dict, native: list[dict]) -> dict:
    checked = {}
    for request in native:
        request_id = request["id"]
        text = texts[request_id].strip()
        if request["checker"] == "strict_json":
            actual = json.loads(text)
            ok = actual == request["expected"]
        else:
            ok = text == str(request["expected"])
        if not ok:
            raise RuntimeError(f"native semantic probe failed for {request_id}: {text[:160]!r}")
        checked[request_id] = {"status": "PASS", "checker": request["checker"], "text": text}
    return checked


def assert_four512(result: dict, stories: list[dict]) -> dict:
    evidence = json.loads(FOUR512_EVIDENCE.read_text(encoding="utf-8"))
    expected = {flow["id"]: flow for flow in evidence["flows"]}
    parity = {}
    for request in stories:
        request_id = request["id"]
        flow = expected.get(request_id)
        if not flow:
            raise RuntimeError(f"accepted four512 evidence lacks {request_id}")
        text = result["texts"][request_id]
        done = result["done"][request_id]
        if done.get("output_tokens") != 512 or text != flow["output"]:
            raise RuntimeError(f"real four512 full-text/output-count comparison failed: {request_id}")
        parity[request_id] = {"full_text_exact": True, "output_tokens": 512,
                              "expected_source": "safe-pool-reject evidence/four512-flow-record.json"}
    return parity


def copy_log_evidence(receipt: dict) -> dict:
    if not WORKER_LOG.is_file():
        return {"worker_log": str(WORKER_LOG), "available": False}
    content = WORKER_LOG.read_text(encoding="utf-8", errors="replace")
    (LOGS / "worker.log.snapshot").write_text(content, encoding="utf-8")
    interesting = [line for line in content.splitlines()
                   if any(key in line for key in (
                       "layer6-model-device", "stage-event-diagnostic", "scheduler layer GPU spans",
                       "meta-gpu-span", "captured-argument-observer", "ac922-tail-bridge admission"))]
    event_lines = [line for line in interesting if any(key in line for key in (
        "layer6-model-device", "stage-event-diagnostic", "scheduler layer GPU spans", "meta-gpu-span"))]
    observer_lines = [line for line in interesting if "captured-argument-observer" in line]
    (LOGS / "stage-event-lines.log").write_text("\n".join(event_lines) + ("\n" if event_lines else ""), encoding="utf-8")
    (LOGS / "observer-lines.log").write_text("\n".join(observer_lines) + ("\n" if observer_lines else ""), encoding="utf-8")
    selected = [line for line in event_lines if "scheduler layer GPU spans requested=1 selected=1" in line]
    target_spans = [line for line in event_lines if "meta-gpu-span device=" in line and " approx_begin_us=" in line]
    drops = [line for line in event_lines if "meta-gpu-span device=" in line and " dropped=" in line]
    device_ids = sorted({int(part.split("=", 1)[1]) for line in target_spans
                         for part in line.split() if part.startswith("device=")})
    observer_selected = [line for line in observer_lines if "requested=1 selected=1" in line]
    replay_authorized = any("replay_authorized=1" in line for line in observer_lines)
    summary = {"worker_log": str(WORKER_LOG), "available": True,
               "filtered_event_log": str(LOGS / "stage-event-lines.log"),
               "filtered_observer_log": str(LOGS / "observer-lines.log"),
               "scheduler_init_selected_count": len(selected),
               "target_span_rows": len(target_spans), "span_device_ids": device_ids,
               "drop_records_seen": len(drops),
               "drop_values": [int(line.rsplit("dropped=", 1)[1].split()[0]) for line in drops],
               "observer_records": len(observer_lines), "observer_selected_records": len(observer_selected),
               "observer_replay_authorized_seen": replay_authorized,
               "final_drop_gate": "PENDING_PARENT_STOP_AND_COMPLETE_LOG_ANALYSIS"}
    receipt["log_evidence"] = summary
    return summary


def main() -> None:
    for directory in (WAVES, LOGS):
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if RECEIPT.exists():
        raise RuntimeError(f"diagnostic receipt already exists; refusing to mix runs: {RECEIPT}")
    build = json.loads((HERE / "build/build-report.json").read_text(encoding="utf-8"))
    if build.get("worker_sha256") != EXPECTED_WORKER_SHA or build.get("source_sha256") != "25edba5f4dbd55e8cd242b27bbf82838bffa8dc5fe1f8b8b1f808dca220c09f9":
        raise RuntimeError("worker/source identity changed after preparation")
    observer = PROJECT / "production/captured-argument-observer-20261019/libac922-tail-observer.so"
    if not observer.is_file() or sha(observer) != EXPECTED_OBSERVER_SHA:
        raise RuntimeError("observer DSO identity changed after preparation")

    report = {"schema": "ac922-stage-event-diagnostic-result-v1", "status": "WAITING_FOR_READY",
              "started_unix": time.time(), "worker_sha256": EXPECTED_WORKER_SHA,
              "worker_source_sha256": build["source_sha256"], "observer_sha256": EXPECTED_OBSERVER_SHA,
              "engine_libraries": build.get("libraries"), "api_port": API_PORT,
              "inference_backend_port_during_window": 8007,
              "public_gateway_unit_active_unmodified": "ac922-apigateway.service",
              "control_socket": str(CONTROL_SOCKET),
              "no_throughput_claim": True, "model_load_count_requested": 1,
              "phase_scope": "full in-band wave; host timestamps include IPC and wave overhead",
              "waves": []}
    atomic_json(RECEIPT, report)
    log("waiting for private API 8012 readiness")
    report["ready"] = wait_ready()
    report["status"] = "RUNNING_COMBINED_DIAGNOSTIC"
    atomic_json(RECEIPT, report)
    corpus, native, stories = source_inputs()
    pid = None

    waves = [
        ("stage-event-primer32", [dict(corpus, id="primer32")]),
        ("diag-event-prefill32", [dict(corpus, id="prefill32")]),
        ("diag-event-decode4", stories),
    ]
    for index, (label, requests) in enumerate(waves, 1):
        log(f"wave start id={label} requests={len(requests)} uncached=1")
        result = send_wave(label, requests, index)
        pid = validate_wave(result, requests, pid)
        if label in {"stage-event-primer32", "diag-event-prefill32"}:
            if result.get("timings", {}).get("prefill_input_tokens") != 32768:
                raise RuntimeError(f"{label} did not process the exact 32768-token real-corpus slice")
            request_id = requests[0]["id"]
            expected_output = 8
            if result["done"][request_id].get("output_tokens") != expected_output:
                raise RuntimeError(f"{label} did not produce the requested eight tokens")
        if label == "diag-event-decode4":
            report["four512_full_text_parity"] = assert_four512(result, stories)
            report["four512_evidence_sha256"] = sha(FOUR512_EVIDENCE)
        report["waves"].append({"id": label, "host_start_monotonic_us": result["host_start_monotonic_us"],
                                "host_end_monotonic_us": result["host_end_monotonic_us"],
                                "host_full_wave_us": result["host_full_wave_us"],
                                "request_sha256": result["request_sha256"],
                                "worker_pid": pid, "timings_raw": result.get("timings", {}),
                                "request_ids": [request["id"] for request in requests]})
        report["worker_pid"] = pid
        report["status"] = "RUNNING_COMBINED_DIAGNOSTIC"
        copy_log_evidence(report)
        atomic_json(RECEIPT, report)
        log(f"wave complete id={label} host_full_wave_us={result['host_full_wave_us']}")

    final_native = []
    for source in native:
        request = dict(source)
        request["cache_prompt"] = False
        request["capacity_tokens"] = 17916
        final_native.append(request)
    result = send_wave("diag-native4-cap17916", final_native, 4)
    pid = validate_wave(result, final_native, pid)
    report["native_quality"] = assert_native(result["texts"], native)
    report["waves"].append({"id": "diag-native4-cap17916", "host_start_monotonic_us": result["host_start_monotonic_us"],
                            "host_end_monotonic_us": result["host_end_monotonic_us"],
                            "host_full_wave_us": result["host_full_wave_us"],
                            "request_sha256": result["request_sha256"], "worker_pid": pid,
                            "timings_raw": result.get("timings", {}),
                            "request_ids": [request["id"] for request in final_native],
                            "capacity_tokens_per_request": 17916})
    report["worker_pid"] = pid
    copy_log_evidence(report)
    atomic_json(RECEIPT, report)
    log("native4 capacity=17916 semantic checks passed")

    paris = {"id": "long-paris", "prompt": native[0]["prompt"],
             "semantic_pad_to_tokens": 32768, "max_tokens": 24,
             "cache_prompt": False, "ignore_eos": False}
    result = send_wave("diag-long-paris-quality", [paris], 5)
    pid = validate_wave(result, [paris], pid)
    text = result.get("texts", {}).get("long-paris", "").strip()
    if text != "Paris":
        raise RuntimeError(f"long-paris semantic check failed: {text[:160]!r}")
    report["long_paris_quality"] = {"status": "PASS", "text": text,
                                   "prompt_tokens_expected": 32768}
    report["waves"].append({"id": "diag-long-paris-quality",
                            "host_start_monotonic_us": result["host_start_monotonic_us"],
                            "host_end_monotonic_us": result["host_end_monotonic_us"],
                            "host_full_wave_us": result["host_full_wave_us"],
                            "request_sha256": result["request_sha256"], "worker_pid": pid,
                            "timings_raw": result.get("timings", {}), "request_ids": ["long-paris"]})
    report["worker_pid"] = pid
    log_evidence = copy_log_evidence(report)
    report["log_evidence"] = log_evidence
    report["status"] = "DIAGNOSTIC_WAVES_AND_SEMANTIC_GATES_PASS_PENDING_PARENT_FINAL_LOG_REVIEW"
    report["finished_unix"] = time.time()
    atomic_json(RECEIPT, report)
    log("combined diagnostic waves complete; stage selection, spans, drops, and observer output remain in separate logs")


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        RUN.mkdir(mode=0o700, parents=True, exist_ok=True)
        LOGS.mkdir(mode=0o700, parents=True, exist_ok=True)
        report = json.loads(RECEIPT.read_text(encoding="utf-8")) if RECEIPT.exists() else {}
        report.update(status="FAIL_FAST_PARENT_RESTORE_REQUIRED", error=repr(error), finished_unix=time.time())
        try:
            copy_log_evidence(report)
        finally:
            atomic_json(RECEIPT, report)
        log(f"FAIL_FAST_PARENT_RESTORE_REQUIRED {error!r}")
        raise
