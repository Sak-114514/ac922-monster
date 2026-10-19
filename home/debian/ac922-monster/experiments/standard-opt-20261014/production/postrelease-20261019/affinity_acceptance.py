#!/usr/bin/env python3
"""In-band, same-service acceptance for four-bank exact-prefix affinity.

Normal execution only sends waves to an already-running private Unix control
socket. It never starts, stops, or configures the model service. ``--self-test``
uses an in-process fake transport and makes no socket or HTTP requests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import time
import uuid
from typing import Any, Protocol


HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
DEFAULT_CORPUS = BASE / "fixtures/long-prefill-combo-v45-20261016/v45-real-corpus-source.json"
DEFAULT_SOCKET = Path("/home/debian/ac922-monster/production/postrelease-control.sock")
DEFAULT_OUT = HERE / "run/affinity-result.json"
CORPUS_CHARACTERS = 32_768
DEFAULT_BANK_CONTEXT = 17_152
DOCUMENTS = (
    ("doc-alpha", "DOC-AFFINITY-ALPHA-01", "3179"),
    ("doc-bravo", "DOC-AFFINITY-BRAVO-02", "6842"),
    ("doc-charlie", "DOC-AFFINITY-CHARLIE-03", "5301"),
    ("doc-delta", "DOC-AFFINITY-DELTA-04", "9274"),
)


class AcceptanceError(RuntimeError):
    pass


class WaveTransport(Protocol):
    def wave(self, label: str, requests: list[dict[str, Any]]) -> dict[str, Any]: ...


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_corpus(path: Path) -> str:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AcceptanceError(f"cannot read corpus fixture: {type(exc).__name__}") from None
    corpus = document.get("corpus", {}) if isinstance(document, dict) else {}
    segments = corpus.get("segments", []) if isinstance(corpus, dict) else []
    text = "\n\n".join(str(segment.get("text", "")) for segment in segments
                       if isinstance(segment, dict) and segment.get("text"))
    if not text:
        raise AcceptanceError("corpus fixture has no text segments")
    return text


def make_corpus_body(source: str) -> str:
    if not source:
        raise AcceptanceError("empty source corpus")
    body = (source * (CORPUS_CHARACTERS // len(source) + 1))[:CORPUS_CHARACTERS]
    if len(body) != CORPUS_CHARACTERS:
        raise AcceptanceError("failed to construct the fixed 32K-character corpus body")
    return body


def make_prompt(document_id: str, value: str, body: str, revision: str = "baseline") -> str:
    # Stable identity and answer are deliberately at the prompt's very front.
    # The long, identical real-corpus section follows those bank-specific keys.
    return (
        f"DOCUMENT_ID={document_id}\n"
        f"NOVA = {value}\n"
        f"REFERENCE_REVISION={revision}\n"
        "REFERENCE MATERIAL BEGIN\n"
        f"{body}\n"
        "REFERENCE MATERIAL END\n"
        "QUESTION: For the document identified at the beginning, what is NOVA? "
        "Return only its decimal value.\n"
    )


def request_for(document: tuple[str, str, str], body: str,
                *, revision: str = "baseline", answer: str | None = None) -> dict[str, Any]:
    request_id, document_id, original_answer = document
    prompt = make_prompt(document_id, answer or original_answer, body, revision)
    return {
        "id": request_id,
        "prompt": prompt,
        "max_tokens": 16,
        "cache_prompt": True,
        "_expected": answer or original_answer,
        "_document_id": document_id,
        "_prompt_sha256": sha256_text(prompt),
    }


def campaign_waves(body: str) -> list[tuple[str, list[dict[str, Any]], str]]:
    baseline = [request_for(document, body) for document in DOCUMENTS]
    return [
        ("cold4", baseline, "cold"),
        ("same4", [request_for(DOCUMENTS[i], body) for i in range(4)], "hit"),
        ("shuffle_3_1_0_2", [request_for(DOCUMENTS[i], body) for i in (3, 1, 0, 2)], "hit"),
        ("subset2_2_0", [request_for(DOCUMENTS[i], body) for i in (2, 0)], "hit"),
        ("subset3_1_3_2", [request_for(DOCUMENTS[i], body) for i in (1, 3, 2)], "hit"),
        # Change the earliest per-document reference fields and the NOVA answer.
        # At least two requests are sent so the service retains four-bank mode.
        ("edited_early_body2", [
            request_for(DOCUMENTS[0], body, revision="edited-early-v1", answer="7301"),
            request_for(DOCUMENTS[2], body, revision="edited-early-v1", answer="4159"),
        ], "miss"),
    ]


class UnixWaveTransport:
    def __init__(self, path: Path, timeout_s: float = 1500.0):
        self.path = path
        self.timeout_s = timeout_s

    def wave(self, label: str, requests: list[dict[str, Any]]) -> dict[str, Any]:
        public_requests = [{key: value for key, value in request.items() if not key.startswith("_")}
                           for request in requests]
        epoch = f"affinity-{label}-{uuid.uuid4().hex[:10]}"
        began = time.monotonic()
        payload = json.dumps({"id": epoch, "requests": public_requests}, ensure_ascii=False).encode("utf-8") + b"\n"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(self.timeout_s)
            connection.connect(str(self.path))
            connection.sendall(payload)
            wire = connection.makefile("rb").readline(8 << 20)
        if not wire:
            raise AcceptanceError(f"control socket closed without a result for {label}")
        try:
            result = json.loads(wire)
        except json.JSONDecodeError:
            raise AcceptanceError(f"control socket returned invalid JSON for {label}") from None
        if not isinstance(result, dict):
            raise AcceptanceError(f"control socket result for {label} is not an object")
        if result.get("status") == "ERROR":
            raise AcceptanceError(f"control wave {label} failed: {result.get('error', 'unknown error')}")
        result["label"] = label
        result["control_roundtrip_s"] = time.monotonic() - began
        return result


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.chmod(temp, 0o600)
    os.replace(temp, path)


def optional_bank(event: Any) -> int | None:
    if not isinstance(event, dict):
        return None
    value = event.get("bank")
    return value if isinstance(value, int) and 0 <= value < 4 else None


def enrich_wave(label: str, requests: list[dict[str, Any]], result: dict[str, Any],
                *, expected_cache: str, bank_context: int) -> tuple[dict[str, Any], bool, dict[str, int]]:
    errors = result.get("errors", [])
    accepted = result.get("accepted", {})
    done = result.get("done", {})
    cache_status = result.get("cache_status", {})
    texts = result.get("texts", {})
    first_tokens = result.get("first_token_s", {})
    last_tokens = result.get("last_token_s", {})
    good = not errors and len(accepted) == len(requests) and len(done) == len(requests)
    rows: list[dict[str, Any]] = []
    mapping: dict[str, int] = {}

    for position, request in enumerate(requests):
        request_id = str(request["id"])
        expected = str(request["_expected"])
        actual = str(texts.get(request_id, "")).strip()
        accepted_event = accepted.get(request_id)
        done_event = done.get(request_id)
        cache_event = cache_status.get(request_id)
        input_tokens = accepted_event.get("input_tokens") if isinstance(accepted_event, dict) else None
        cached_tokens = cache_event.get("cached_tokens") if isinstance(cache_event, dict) else None
        bank_evidence = {
            "accepted": optional_bank(accepted_event),
            "cache_status": optional_bank(cache_event),
            "done": optional_bank(done_event),
        }
        bank_values = {value for value in bank_evidence.values() if value is not None}
        bank_event_consistent = len(bank_values) <= 1
        reported_bank = next((value for value in bank_evidence.values() if value is not None), None)
        # Current bank-worker live waves set dual=true and assign each request
        # by input position modulo four. Prefer an explicit worker bank field if
        # a future/updated worker reports it.
        bank_source = "worker_event"
        if reported_bank is None:
            reported_bank = position % 4
            bank_source = "fixed_live_wave_order_index_mod_4"
        mapping[request_id] = reported_bank
        semantic_pass = actual == expected
        finish_reason = done_event.get("finish_reason") if isinstance(done_event, dict) else None
        natural_stop = finish_reason == "stop"
        cache_observed = isinstance(cached_tokens, int) and cached_tokens >= 0
        cache_pass = (cache_observed and cached_tokens == 0 if expected_cache in {"cold", "miss"}
                      else cache_observed and cached_tokens > 0)
        context_pass = isinstance(input_tokens, int) and input_tokens + 16 <= bank_context
        row = {
            "id": request_id,
            "document_id": request["_document_id"],
            "prompt_sha256": request["_prompt_sha256"],
            "expected": expected,
            "actual": actual,
            "semantic_pass": semantic_pass,
            "finish_reason": finish_reason,
            "natural_stop": natural_stop,
            "input_tokens": input_tokens,
            "context_limit_tokens": bank_context,
            "context_pass": context_pass,
            "cache_status_present": isinstance(cache_event, dict),
            "cached_tokens": cached_tokens,
            "cache_expectation": expected_cache,
            "cache_pass": cache_pass,
            "bank": reported_bank,
            "bank_event_values": bank_evidence,
            "bank_event_consistent": bank_event_consistent,
            "bank_mapping_source": bank_source,
            "ttft_s": first_tokens.get(request_id),
            "last_emission_s": last_tokens.get(request_id),
        }
        rows.append(row)
        good = good and semantic_pass and natural_stop and context_pass and cache_pass and bank_event_consistent

    enriched = {
        "label": label,
        "request_order": [str(request["id"]) for request in requests],
        "mapping": mapping,
        "mapping_source": "worker event when present; otherwise source-derived position modulo four",
        "bank_mapping_by_request": rows,
        "wave_wall_s": result.get("wall_s"),
        "control_roundtrip_s": result.get("control_roundtrip_s"),
        "first_token_s": result.get("first_token_s", {}),
        "last_token_s": result.get("last_token_s", {}),
        "timings": result.get("timings", {}),
        "errors": errors,
        # Contains only the short generated answers and runtime events, never prompts or credentials.
        "response": result,
        "status": "PASS" if good else "FAIL",
    }
    return enriched, good, mapping


def run_campaign(transport: WaveTransport, source: str, output_path: Path,
                 bank_context: int = DEFAULT_BANK_CONTEXT) -> dict[str, Any]:
    body = make_corpus_body(source)
    waves = campaign_waves(body)
    report: dict[str, Any] = {
        "schema": "ac922-prefix-affinity-acceptance-v1",
        "status": "RUNNING",
        "started_unix": time.time(),
        "mode": "inband whole-wave on already-loaded service",
        "model_load_count_requested": 0,
        "model_load_count_claimed": "parent-managed; harness does not launch or reload model",
        "corpus_characters": len(body),
        "corpus_sha256": sha256_text(body),
        "corpus_source_sha256": sha256_text(source),
        "sampling": "greedy settings are inherited from the loaded worker",
        "max_tokens_per_request": 16,
        "bank_context_tokens": bank_context,
        "cache_expectations": {
            "cold4": "all requests cached_tokens=0",
            "same4/shuffle/subsets": "all requests cached_tokens>0 and stable document-to-bank mapping",
            "edited_early_body2": "each request cached_tokens=0 and changed NOVA value returned exactly",
        },
        "waves": [],
    }
    stable_mapping: dict[str, int] = {}
    all_pass = True
    try:
        for label, requests, expected_cache in waves:
            # Validate prompt shape before any live call, and keep each call at
            # two or more lanes so no request enters the single-request profile.
            if len(requests) < 2:
                raise AcceptanceError(f"{label} would enter the single-request profile")
            for request in requests:
                prompt = str(request["prompt"])
                if body not in prompt:
                    raise AcceptanceError(f"{label}/{request['id']} does not contain exactly 32K corpus characters")
                if (not prompt.startswith(f"DOCUMENT_ID={request['_document_id']}\nNOVA = ") or
                        f"NOVA = {request['_expected']}\n" not in prompt[:160]):
                    raise AcceptanceError(f"{label}/{request['id']} header is not at the prompt front")
            started = time.monotonic()
            raw_result = transport.wave(label, requests)
            raw_result.setdefault("control_roundtrip_s", time.monotonic() - started)
            enriched, passed, mapping = enrich_wave(
                label, requests, raw_result, expected_cache=expected_cache, bank_context=bank_context)
            if label == "cold4":
                stable_mapping = mapping
                enriched["mapping_stability"] = "anchor"
            elif expected_cache == "hit":
                stable = {request_id: mapping.get(request_id) == stable_mapping.get(request_id)
                          for request_id in mapping}
                enriched["mapping_stability_by_request"] = stable
                enriched["mapping_stable"] = all(stable.values())
                passed = passed and enriched["mapping_stable"]
                enriched["status"] = "PASS" if passed else "FAIL"
            report["waves"].append(enriched)
            all_pass = all_pass and passed
            report["status"] = "RUNNING"
            report["finished_wave"] = label
            write_report(output_path, report)
            print(json.dumps({"wave": label, "status": enriched["status"],
                              "wall_s": enriched.get("wave_wall_s"),
                              "mapping": enriched.get("mapping"),
                              "cached_tokens": {row["id"]: row["cached_tokens"]
                                                for row in enriched["bank_mapping_by_request"]}},
                             ensure_ascii=False), flush=True)
        report["status"] = "PASS" if all_pass else "FAIL"
        report["finished_unix"] = time.time()
        report["summary"] = {
            "all_six_waves_pass": all_pass,
            "stable_mapping": stable_mapping,
            "positive_hit_waves": [wave["label"] for wave in report["waves"]
                                    if wave["label"] in {"same4", "shuffle_3_1_0_2", "subset2_2_0", "subset3_1_3_2"}
                                    and wave["status"] == "PASS"],
            "edited_body_miss": next((wave["status"] == "PASS" for wave in report["waves"]
                                       if wave["label"] == "edited_early_body2"), False),
        }
        write_report(output_path, report)
        if not all_pass:
            raise AcceptanceError("one or more exact-prefix affinity gates failed")
        return report
    except BaseException as error:
        report["status"] = "FAIL"
        report["error"] = f"{type(error).__name__}: {error}"
        report["finished_unix"] = time.time()
        write_report(output_path, report)
        raise


class MockTransport:
    """Deterministic in-memory affinity/cache model used only by --self-test."""

    def __init__(self, document_affinity: bool = True):
        self.cache: dict[tuple[str, str], int] = {}
        self.calls: list[tuple[str, list[dict[str, Any]]]] = []
        self.document_affinity = document_affinity
        self.bank_by_document = {doc_id: bank for bank, (_, doc_id, _) in enumerate(DOCUMENTS)}

    def wave(self, label: str, requests: list[dict[str, Any]]) -> dict[str, Any]:
        self.calls.append((label, requests))
        accepted: dict[str, Any] = {}
        done: dict[str, Any] = {}
        cache_status: dict[str, Any] = {}
        texts: dict[str, str] = {}
        first: dict[str, float] = {}
        last: dict[str, float] = {}
        for offset, request in enumerate(requests):
            request_id = request["id"]
            prompt = request["prompt"]
            doc_id = request["_document_id"]
            bank = self.bank_by_document[doc_id] if self.document_affinity else offset % 4
            cache_key = (str(bank), sha256_text(prompt))
            cached = self.cache.get(cache_key, 0)
            # Positive hits use a checkpoint-sized synthetic count. For the edited
            # prompts the hash changes, so the old document state cannot be reused.
            input_tokens = max(8192, len(prompt) // 3)
            if request["cache_prompt"]:
                self.cache[cache_key] = input_tokens - 256
            accepted[request_id] = {"type": "accepted", "id": request_id,
                                    "input_tokens": input_tokens, "bank": bank}
            cache_status[request_id] = {"type": "cache_status", "id": request_id,
                                        "hit": cached > 0, "cached_tokens": cached,
                                        "checkpoint_tokens": cached, "bank": bank}
            texts[request_id] = str(request["_expected"])
            done[request_id] = {"type": "done", "id": request_id,
                                "input_tokens": input_tokens, "output_tokens": 1,
                                "finish_reason": "stop", "bank": bank}
            first[request_id] = 0.25 + offset * 0.01
            last[request_id] = first[request_id]
        return {
            "id": f"mock-{label}", "accepted": accepted, "cache_status": cache_status,
            "done": done, "errors": [], "texts": texts, "first_token_s": first,
            "last_token_s": last, "wall_s": 0.5, "timings": {"mock": True},
        }


def self_test(source: str) -> dict[str, Any]:
    transport = MockTransport()
    with tempfile.TemporaryDirectory(prefix="ac922-affinity-mock-") as temporary:
        output = Path(temporary) / "affinity-result.json"
        report = run_campaign(transport, source, output)
        if report["status"] != "PASS":
            raise AcceptanceError("mock affinity campaign failed")
        if [name for name, _ in transport.calls] != [
                "cold4", "same4", "shuffle_3_1_0_2", "subset2_2_0", "subset3_1_3_2", "edited_early_body2"]:
            raise AcceptanceError("mock campaign wave order changed")
        for label, requests in transport.calls:
            if len(requests) < 2:
                raise AcceptanceError(f"self-test found a single-request profile in {label}")
            for request in requests:
                if not request["prompt"].startswith(f"DOCUMENT_ID={request['_document_id']}\nNOVA = "):
                    raise AcceptanceError("self-test found a malformed front-matter mapping")
        saved = output.read_text(encoding="utf-8")
        if any(request["prompt"] in saved for _, requests in transport.calls for request in requests):
            raise AcceptanceError("report leaked a full input prompt")
        # Prove the gates reject positional bank assignment: it can hit in the
        # original order but loses the stored state under shuffle/subset waves.
        wrong_route = MockTransport(document_affinity=False)
        wrong_output = Path(temporary) / "wrong-affinity-result.json"
        caught_wrong_route = False
        try:
            run_campaign(wrong_route, source, wrong_output)
        except AcceptanceError:
            caught_wrong_route = True
        if not caught_wrong_route:
            raise AcceptanceError("mock did not reject a non-affinity bank assignment")
        wrong_report = json.loads(wrong_output.read_text(encoding="utf-8"))
        if wrong_report.get("status") != "FAIL":
            raise AcceptanceError("wrong-route mock failure was not persisted")
        return {"self_test": "PASS", "transport": "in-memory mock; no network/socket/model calls",
                "waves": len(report["waves"]), "wrong_route_rejected": True,
                "report_mode": report["mode"]}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--self-test", action="store_true", help="run only the in-memory mock; no network")
    modes.add_argument("--run", action="store_true", help="send six waves to the existing control socket")
    parser.add_argument("--socket", default=os.environ.get("AC922_AFFINITY_SOCKET", str(DEFAULT_SOCKET)))
    parser.add_argument("--corpus", default=os.environ.get("AC922_CORPUS_FILE", str(DEFAULT_CORPUS)))
    parser.add_argument("--out", default=os.environ.get("AC922_AFFINITY_RESULT", str(DEFAULT_OUT)))
    parser.add_argument("--bank-context", type=int,
                        default=int(os.environ.get("AC922_BANK_CONTEXT_TOKENS", str(DEFAULT_BANK_CONTEXT))))
    parser.add_argument("--timeout", type=float, default=1500.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    if args.bank_context <= 16:
        raise AcceptanceError("bank context must leave room for the 16-token output reservation")
    source = load_corpus(Path(args.corpus).expanduser())
    if args.self_test:
        print(json.dumps(self_test(source), ensure_ascii=False))
        return 0
    report = run_campaign(UnixWaveTransport(Path(args.socket), args.timeout), source,
                          Path(args.out).expanduser(), args.bank_context)
    print(json.dumps({"status": report["status"], "result": str(Path(args.out).expanduser())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AcceptanceError as error:
        print(f"affinity acceptance failed: {error}", file=sys.stderr)
        raise SystemExit(1)
