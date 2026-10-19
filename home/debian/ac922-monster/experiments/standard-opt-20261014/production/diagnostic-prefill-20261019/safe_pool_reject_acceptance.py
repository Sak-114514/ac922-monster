#!/usr/bin/env python3
"""Insert a terminal near-boundary refill-mode pool rejection into the affinity campaign."""
from __future__ import annotations
import argparse
import json
import os
import socket
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import affinity_acceptance as aff  # noqa: E402

SOCKET = Path("/home/debian/ac922-monster/production/safe-pool-reject-control.sock")
OUT = HERE / "run/safe-pool-reject-result.json"
CASES = HERE.parents[1] / "fixtures/fourbank-d3-newruntime-v306-20261017/v221-d3-scenarios.json"
BOUNDARY = [169212, 169212, 169212, 1276]
ERROR_CODE = "retention_plan_pool_exceeded"


class GateError(RuntimeError):
    pass


def need(ok: bool, msg: str) -> None:
    if not ok:
        raise GateError(msg)


class RefillControl:
    """Strict-boolean refill wave; the in-band wrapper observes close/ack and returns meta."""
    def __init__(self, path: Path, timeout: float = 600):
        self.path = path
        self.timeout = timeout

    def wave(self, label: str, requests: list[dict]) -> dict:
        public = [{k: v for k, v in r.items() if not k.startswith("_")} for r in requests]
        epoch = f"safe-pool-{label}-{uuid.uuid4().hex[:10]}"
        cmd = {"type": "wave", "id": epoch, "refill": True, "requests": public}
        if cmd["refill"] is not True:
            raise GateError("refill must be the JSON boolean true")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(self.timeout)
            s.connect(str(self.path))
            s.sendall(json.dumps(cmd, ensure_ascii=False).encode() + b"\n")
            line = s.makefile("rb").readline(8 << 20)
        if not line:
            raise GateError("refill control socket closed without result")
        try:
            result = json.loads(line)
        except json.JSONDecodeError:
            raise GateError("invalid refill control JSON") from None
        if result.get("status") == "ERROR":
            raise GateError(f"refill control failed: {result.get('error')}")
        result["label"] = label
        return result


def pool_request(request: dict, capacity: int, request_id: str | None = None) -> dict:
    result = {k: v for k, v in request.items() if not k.startswith("_")}
    if request_id is not None:
        result["id"] = request_id
    result["capacity_tokens"] = capacity
    return result


def native_cases(path: Path) -> list[dict]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    rows = doc[0]["requests"]
    need(len(rows) == 4, "expected original four native canaries")
    cases = [{"id": r["id"], "prompt": r["prompt"], "max_tokens": 24, "_expected": r["expected"],
              "_checker": r["checker"], "_expected_bank": i} for i, r in enumerate(rows)]
    need(all(len(case["prompt"]) < 1000 for case in cases),
         "native canaries must remain short enough for the 1276-token bank")
    return cases


def worker_path(explicit: str | None) -> str:
    if explicit:
        return str(Path(explicit).expanduser())
    for command_file in (HERE / "lastwave-worker-command.json", HERE / "worker-command.json"):
        if not command_file.exists():
            continue
        try:
            argv = json.loads(command_file.read_text(encoding="utf-8")).get("argv", [])
        except (OSError, json.JSONDecodeError, AttributeError):
            continue
        match = next((str(arg) for arg in argv if isinstance(arg, str) and Path(arg).name == "bank-worker"), None)
        if match:
            return match
    raise GateError("cannot identify bank-worker path; pass --worker-path or AC922_WORKER_PATH")

class BoundaryDecorator:
    """Inject short native canaries after cold4 to exercise near-boundary rejection."""

    def __init__(self, base: Any, probe: RefillControl, canaries: list[dict]):
        self.base = base
        self.probe_transport = probe
        self.canaries = canaries
        self.probe: dict | None = None
        self.worker_pid: Any = None
        self.seed_geometry: dict | None = None

    def wave(self, label: str, requests: list[dict]) -> dict:
        result = self.base.wave(label, requests)
        pid = result.get("worker_pid")
        if label == "cold4":
            need(isinstance(pid, int) and pid > 0, "wrapper omitted loaded worker PID on seed wave")
            self.worker_pid = pid
            self.seed_geometry = aff.geometry_signature(result.get("timings"))
            need(self.seed_geometry is not None, "seed wave omitted four-bank geometry")
            # The fourth bank is only 1276 tokens, so use the short native canaries,
            # not the ~10.5K-token cache-seeding prompts, for this refusal probe.
            overflow = [pool_request(req, BOUNDARY[i], f"pool-canary-{req['id']}")
                        for i, req in enumerate(self.canaries)]
            probe = self.probe_transport.wave("near-boundary-overflow", overflow)
            probe["_requested"] = [{"id": r["id"]} for r in overflow]
            self.probe = probe
            self._validate_probe(probe)
            result["safe_pool_reject_probe"] = {k: v for k, v in probe.items() if k != "texts"}
            # The legacy preflight overflow remains separate from the new
            # retention rejection. Neither may disturb the seeded checkpoints.
            over_requests = [pool_request(req, [174588, 174588, 174588, 1276][i],
                                         f"overpool-{req['id']}")
                             for i, req in enumerate(self.canaries)]
            over = self.base.wave("true-overpool-preflight", over_requests)
            errors = over.get("errors", [])
            need(len(errors) == 4 and {e.get("id") for e in errors} ==
                 {r["id"] for r in over_requests} and
                 all(e.get("status") == 400 and e.get("message") ==
                     "shared context pool exceeded" for e in errors),
                 "true over-pool wave did not retain its legacy preflight 400")
            need(not over.get("accepted") and not over.get("done") and
                 over.get("worker_pid") == self.worker_pid and
                 over.get("wave_done_seen") is True,
                 "true over-pool preflight accepted work or lost the worker")
            result["legacy_overpool_probe"] = over
        elif label == "same4":
            need(pid == self.worker_pid, "worker PID changed after near-boundary rejection")
            for req in requests:
                cached = result.get("cache_status", {}).get(req["id"], {}).get("cached_tokens")
                need(cached == 10496, f"{req['id']}: post-rejection cache hit was {cached}, expected 10496")
        return result

    def _validate_probe(self, result: dict) -> None:
        need(result.get("refill_mode_requested") is True, "wrapper did not confirm strict refill=true")
        need(result.get("wave_closing_seen") is True, "worker wave_closing was not observed")
        need(result.get("close_ack_sent") is True, "wrapper did not send close_wave acknowledgement")
        need(result.get("wave_done_seen") is True, "worker wave_done was not observed")
        pre = result.get("worker_pid_before")
        post = result.get("worker_pid_after")
        pid = result.get("worker_pid")
        if pre is None:
            pre = pid
        if post is None:
            post = pid
        need(isinstance(pre, int) and isinstance(post, int),
             "wrapper omitted numeric worker PID across terminal rejection")
        need(pre == self.worker_pid and post == self.worker_pid,
             "worker PID changed across terminal rejection")
        errors = result.get("errors", [])
        expected_ids = {r["id"] for r in result.get("_requested", [])}
        need(len(errors) == 4 and {e.get("id") for e in errors} == expected_ids and
             all(e.get("status") == 400 and e.get("code") == ERROR_CODE for e in errors),
             "near-boundary request wave did not return four correlated 400 retention_plan_pool_exceeded errors")
        need(not result.get("accepted") and not result.get("done"),
             "terminal pool rejection accepted or completed requests")
        timing = result.get("timings", {})
        need(timing.get("context_rebuilds_before_execution") == 0, "rejection rebuilt a context")
        need(timing.get("bank_releases_before_execution") == 0, "rejection released a bank context")
        geometry = aff.geometry_signature(timing)
        need(geometry == self.seed_geometry, "terminal rejection changed one of the four bank layouts")
        input_events = [e.get("input_tokens") for e in errors if isinstance(e.get("input_tokens"), int)]
        self.probe_metadata = {
            "refill_mode_requested": True,
            "wave_closing_seen": True,
            "close_ack_sent": True,
            "wave_done_seen": True,
            "worker_pid": pid or pre,
            "errors": errors,
            "context_rebuilds_before_execution": 0,
            "bank_releases_before_execution": 0,
            "geometry_unchanged": True,
            "capacity_tokens_requested": BOUNDARY,
            "capacity_tokens_requested_sum": sum(BOUNDARY),
            "actual_input_tokens": sum(input_events) if len(input_events) == 4 else None,
            "actual_input_tokens_observed": len(input_events) == 4,
        }

class MockTransport:
    def __init__(self, failure: str | None = None):
        self.failure = failure
        self.pid = 4701
        self.affinity = aff.MockTransport()

    def wave(self, label: str, requests: list[dict]) -> dict:
        if label == "true-overpool-preflight":
            return {"id": label, "errors": [
                {"id": r["id"], "status": 400, "message": "shared context pool exceeded"}
                for r in requests], "accepted": {}, "done": {},
                "worker_pid": self.pid, "wave_done_seen": True}
        if label == "near-boundary-overflow":
            errors = [{"id": r["id"], "status": 400, "code": ERROR_CODE,
                       "message": "pool plan exceeded"} for r in requests]
            if self.failure == "non400":
                errors[0]["status"] = 500
            if self.failure == "wrong_code":
                errors[0]["code"] = "wrong_code"
            geometry = [{"bank": i, "target_context": 17152, "draft_context": 17152,
                         "target_ubatch": 256} for i in range(4)]
            if self.failure == "bad_geometry":
                geometry[0]["target_context"] = 17408
            return {"id": label, "errors": errors, "accepted": {}, "done": {}, "timings": {
                "banks": geometry, "context_rebuilds_before_execution": 0,
                "bank_releases_before_execution": 0}, "refill_mode_requested": True,
                "wave_closing_seen": True, "close_ack_sent": self.failure != "missing_close_ack",
                "wave_done_seen": True, "worker_pid": self.pid, "worker_pid_before": self.pid,
                "worker_pid_after": self.pid + 1 if self.failure == "worker_pid" else self.pid}
        result = self.affinity.wave(label, requests)
        result["worker_pid"] = self.pid
        if label == "same4":
            for event in result.get("cache_status",{}).values():
                event.update(hit=True, cached_tokens=10496, checkpoint_tokens=10496)
        if self.failure == "worker_pid" and label == "same4":
            result["worker_pid"] = self.pid + 1
        return result

def mocked_run(failure: str | None = None, canaries: list[dict] | None = None) -> dict:
    source = aff.load_corpus(aff.DEFAULT_CORPUS)
    transport = MockTransport(failure)
    with tempfile.TemporaryDirectory(prefix="safe-pool-affinity-mock-") as tmp:
        decorated = BoundaryDecorator(transport, transport, canaries or native_cases(CASES))
        report = aff.run_campaign(decorated, source, Path(tmp) / "affinity.json", emit_summary=False)
        report["pool_probe_meta"] = decorated.probe_metadata
        return report

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    parser.add_argument("--socket", default=str(SOCKET))
    parser.add_argument("--cases", default=str(CASES))
    parser.add_argument("--worker-path", default=os.environ.get("AC922_WORKER_PATH"))
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    cases = native_cases(Path(args.cases).expanduser())
    if not args.run:
        # The mock transport drives the exact same decorator and existing affinity campaign.
        good = mocked_run(canaries=cases)
        need(good["status"] == "PASS", "mock safe-pool affinity campaign failed")
        # Separately exercise rejection metadata/geometry gates with deterministic bad wrappers.
        outcomes = {}
        for failure in ("bad_geometry","non400","wrong_code","missing_close_ack","worker_pid"):
            try:
                mocked_run(failure, cases)
                outcomes[failure] = False
            except (GateError, aff.AcceptanceError):
                outcomes[failure] = True
        need(all(outcomes.values()), "a boundary negative mock was accepted")
        print(json.dumps({"self_test": "PASS", "transport": "in-memory only; no socket/API/GPU/proc",
                          "affinity_campaign_waves": len(good["waves"]),
                          "injected_refill_probe": good["waves"][0]["response"].get("safe_pool_reject_probe") is not None,
                          "near_boundary_prompts_chars": [len(case["prompt"]) for case in cases],
                          "negative_modes_rejected": outcomes}, ensure_ascii=False))
        return 0
    try:
        expected = worker_path(args.worker_path)
        path = Path(args.socket)
        base = aff.UnixWaveTransport(path, timeout_s=900)
        probe = RefillControl(path, timeout=900)
        decorated = BoundaryDecorator(base, probe, cases)
        source = aff.load_corpus(aff.DEFAULT_CORPUS)
        out = Path(args.out).expanduser()
        report = aff.run_campaign(decorated, source, out, emit_summary=True)
        need(decorated.probe is not None, "near-boundary probe did not run")
        report["safe_pool_boundary_probe"] = {k: v for k, v in decorated.probe.items() if k != "texts"}
        report["worker_executable_path"] = expected
        aff.write_report(out, report)
        print(json.dumps({"status": report["status"], "result": str(out)}, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(f"safe pool rejection acceptance failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
