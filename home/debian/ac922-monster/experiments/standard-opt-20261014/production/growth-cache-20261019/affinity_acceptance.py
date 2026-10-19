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
DEFAULT_SOCKET = Path("/home/debian/ac922-monster/production/growth-cache-control.sock")
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
                *, revision: str = "baseline", answer: str | None = None,
                expected_bank: int | None = None) -> dict[str, Any]:
    request_id, document_id, original_answer = document
    prompt = make_prompt(document_id, answer or original_answer, body, revision)
    request = {
        "id": request_id,
        "prompt": prompt,
        "max_tokens": 16,
        "cache_prompt": True,
        "_expected": answer or original_answer,
        "_document_id": document_id,
        "_prompt_sha256": sha256_text(prompt),
    }
    if expected_bank is not None:
        request["_expected_bank"] = expected_bank
    return request


def campaign_waves(body: str) -> list[tuple[str, list[dict[str, Any]], str]]:
    baseline = [request_for(document, body) for document in DOCUMENTS]
    return [
        ("cold4", baseline, "cold"),
        ("same4", [request_for(DOCUMENTS[i], body) for i in range(4)], "hit"),
        ("shuffle_3_1_0_2", [request_for(DOCUMENTS[i], body) for i in (3, 1, 0, 2)], "hit"),
        ("subset2_2_0", [request_for(DOCUMENTS[i], body) for i in (2, 0)], "hit"),
        ("subset3_1_3_2", [request_for(DOCUMENTS[i], body) for i in (1, 3, 2)], "hit"),
        ("singleton_D3", [request_for(DOCUMENTS[3], body, expected_bank=3)], "hit"),
        ("singleton_A0", [request_for(DOCUMENTS[0], body, expected_bank=0)], "hit"),
        ("singleton_B1", [request_for(DOCUMENTS[1], body, expected_bank=1)], "hit"),
        ("singleton_D3_repeat", [request_for(DOCUMENTS[3], body, expected_bank=3)], "hit"),
        # Keep this last: each bank owns a single checkpoint, and an edited miss
        # replaces that bank's prior entry. At least two requests avoid the
        # singleton profile and leave both alpha/bravo banks exercised.
        ("edited_early_body2", [
            request_for(DOCUMENTS[0], body, revision="edited-early-v1", answer="7301"),
            request_for(DOCUMENTS[1], body, revision="edited-early-v1", answer="4159"),
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


def geometry_signature(timings: Any) -> dict[int, dict[str, int]] | None:
    """Return every bank's reported target/draft context and target UBatch."""
    if not isinstance(timings, dict):
        return None
    banks = timings.get("banks")
    if not isinstance(banks, list) or len(banks) != 4:
        return None
    result: dict[int, dict[str, int]] = {}
    for item in banks:
        if not isinstance(item, dict):
            return None
        bank_id = item.get("bank", item.get("id"))
        target = item.get("target_context", item.get("target_context_per_sequence"))
        draft = item.get("draft_context", item.get("draft_context_per_sequence"))
        ubatch = item.get("target_ubatch", item.get("ubatch"))
        if not all(isinstance(value, int) and value >= 0 for value in (bank_id, target, draft, ubatch)):
            return None
        if bank_id in result:
            return None
        result[bank_id] = {"target_context": target, "draft_context": draft,
                           "target_ubatch": ubatch}
    return result if set(result) == set(range(4)) else None


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
        reserved_tokens = accepted_event.get("reserved_tokens") if isinstance(accepted_event, dict) else None
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
        expected_bank = request.get("_expected_bank")
        bank_pass = expected_bank is None or reported_bank == expected_bank
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
            "expected_bank": expected_bank,
            "bank_pass": bank_pass,
            "reserved_tokens": reserved_tokens,
            "bank_event_values": bank_evidence,
            "bank_event_consistent": bank_event_consistent,
            "bank_mapping_source": bank_source,
            "ttft_s": first_tokens.get(request_id),
            "last_emission_s": last_tokens.get(request_id),
        }
        rows.append(row)
        good = good and semantic_pass and natural_stop and context_pass and cache_pass and bank_event_consistent and bank_pass

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


def qualify_cached_singleton(enriched: dict[str, Any], request: dict[str, Any],
                            previous_geometry: dict[int, dict[str, int]] | None) -> tuple[bool, dict[str, Any]]:
    timings = enriched.get("timings", {})
    row = enriched["bank_mapping_by_request"][0]
    reasons: list[str] = []
    current_geometry = geometry_signature(timings)
    cached = row.get("cached_tokens")
    input_tokens = row.get("input_tokens")
    bank = row.get("bank")
    tail_tokens = (input_tokens - cached if isinstance(input_tokens, int) and isinstance(cached, int)
                   and 0 <= cached <= input_tokens else None)
    hit_fraction = (cached / input_tokens if isinstance(cached, int) and isinstance(input_tokens, int)
                    and input_tokens > 0 else None)

    exact_hit_pass = isinstance(hit_fraction, float) and hit_fraction >= 0.75
    tail_pass = isinstance(tail_tokens, int) and tail_tokens <= 8192
    if not exact_hit_pass:
        reasons.append("exact_prefix_hit_below_75_percent")
    if not tail_pass:
        reasons.append("uncached_tail_exceeds_8192_or_unreported")

    cap = current_geometry.get(bank, {}).get("target_context") if current_geometry and isinstance(bank, int) else None
    mtp_depth = timings.get("mtp_depth", 3) if isinstance(timings, dict) else 3
    reserved = row.get("reserved_tokens")
    capacity_fit = (isinstance(cap, int) and isinstance(reserved, int) and isinstance(mtp_depth, int)
                    and reserved + mtp_depth + 1 <= cap)
    if not capacity_fit:
        reasons.append("request_does_not_fit_reported_bank_capacity")

    free_by_gpu = timings.get("single_cache_admission_free_mib") if isinstance(timings, dict) else None
    floor = timings.get("production_floor_mib") if isinstance(timings, dict) else None
    free_evidence_ok = (isinstance(free_by_gpu, list) and len(free_by_gpu) == 6 and
                        all(isinstance(value, int) and value >= 0 for value in free_by_gpu) and
                        isinstance(floor, int) and floor >= 0)
    pre_free_margin_ok = (free_evidence_ok and
                          all(value >= floor + 128 for value in free_by_gpu))
    if not pre_free_margin_ok:
        reasons.append("pre_admission_gpu_free_below_floor_plus_128_or_unreported")

    geometry_same = current_geometry is not None and previous_geometry is not None and current_geometry == previous_geometry
    if not geometry_same:
        reasons.append("four_bank_capacity_or_ubatch_changed_from_previous_wave")

    cache_affinity_enabled = isinstance(timings, dict) and timings.get("single_cached_affinity") is True
    if not cache_affinity_enabled:
        reasons.append("single_cached_affinity_not_selected")
    rebuilds = timings.get("context_rebuilds_before_execution") if isinstance(timings, dict) else None
    no_rebuild = rebuilds == 0
    if not no_rebuild:
        reasons.append("context_rebuild_before_singleton_execution")
    migration_supported = timings.get("state_migration_supported") if isinstance(timings, dict) else None
    no_state_migration = migration_supported is False
    if not no_state_migration:
        reasons.append("worker_does_not_explicitly_disable_state_migration")
    reclaimed = timings.get("post_wave_reclaimed_banks") if isinstance(timings, dict) else None
    no_reclaim = reclaimed == 0
    if not no_reclaim:
        reasons.append("post_wave_bank_reclamation_occurred_or_unreported")

    expected_bank = request.get("_expected_bank")
    expected_bank_pass = bank == expected_bank
    if not expected_bank_pass:
        reasons.append("singleton_did_not_retain_expected_document_bank")

    result = {
        "document_id": request["_document_id"],
        "expected_bank": expected_bank,
        "actual_bank": bank,
        "expected_bank_pass": expected_bank_pass,
        "actual_input_tokens": input_tokens,
        "cached_tokens": cached,
        "cached_fraction": hit_fraction,
        "uncached_tail_tokens": tail_tokens,
        "exact_hit_at_least_75_percent": exact_hit_pass,
        "uncached_tail_at_most_8192": tail_pass,
        "reported_bank_capacity": cap,
        "reserved_tokens": reserved,
        "mtp_depth": mtp_depth,
        "capacity_fit": capacity_fit,
        "pre_free_mib_by_gpu": free_by_gpu,
        "production_floor_mib": floor,
        "pre_free_at_least_floor_plus_128_each_gpu": pre_free_margin_ok,
        "geometry_matches_previous_wave": geometry_same,
        "geometry": current_geometry,
        "single_cached_affinity": cache_affinity_enabled,
        "context_rebuilds_before_execution": rebuilds,
        "state_migration_supported": migration_supported,
        "post_wave_reclaimed_banks": reclaimed,
        "reasons": reasons,
    }
    return not reasons, result


def run_campaign(transport: WaveTransport, source: str, output_path: Path,
                 bank_context: int = DEFAULT_BANK_CONTEXT, *, emit_summary: bool = True) -> dict[str, Any]:
    body = make_corpus_body(source)
    waves = campaign_waves(body)
    report: dict[str, Any] = {
        "schema": "ac922-single-cache-affinity-acceptance-v2",
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
            "singleton_D3/A0/B1/D3_repeat": "bank affinity hit >=75%, uncached tail <=8192, cap fit, pre-free >= floor+128, no rebuild/migration/reclaim, identical four-bank geometry",
        },
        "waves": [],
    }
    stable_mapping: dict[str, int] = {}
    previous_geometry: dict[int, dict[str, int]] | None = None
    all_pass = True
    try:
        for label, requests, expected_cache in waves:
            singleton = label in {"singleton_D3", "singleton_A0", "singleton_B1", "singleton_D3_repeat"}
            # Keep the legacy small-wave guard for ordinary requests. Only the
            # four explicitly qualified cache-affinity cases may be singleton.
            if len(requests) < 2 and not singleton:
                raise AcceptanceError(f"{label} would enter the single-request profile")
            if singleton and len(requests) != 1:
                raise AcceptanceError(f"{label} must contain exactly one request")
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
            current_geometry = geometry_signature(enriched.get("timings"))
            enriched["geometry_signature"] = current_geometry
            if singleton:
                singleton_pass, singleton_result = qualify_cached_singleton(
                    enriched, requests[0], previous_geometry)
                enriched["singleton_retention"] = singleton_result
                passed = passed and singleton_pass
                enriched["status"] = "PASS" if passed else "FAIL"
            report["waves"].append(enriched)
            all_pass = all_pass and passed
            report["status"] = "RUNNING"
            report["finished_wave"] = label
            write_report(output_path, report)
            if current_geometry is not None:
                previous_geometry = current_geometry
            if emit_summary:
                print(json.dumps({"wave": label, "status": enriched["status"],
                                  "wall_s": enriched.get("wave_wall_s"),
                                  "mapping": enriched.get("mapping"),
                                  "cached_tokens": {row["id"]: row["cached_tokens"]
                                                    for row in enriched["bank_mapping_by_request"]}},
                                 ensure_ascii=False), flush=True)
        report["status"] = "PASS" if all_pass else "FAIL"
        report["finished_unix"] = time.time()
        report["summary"] = {
            "all_waves_pass": all_pass,
            "stable_mapping": stable_mapping,
            "positive_hit_waves": [wave["label"] for wave in report["waves"]
                                    if wave["label"] in {"same4", "shuffle_3_1_0_2", "subset2_2_0", "subset3_1_3_2"}
                                    and wave["status"] == "PASS"],
            "singleton_waves_pass": [wave["label"] for wave in report["waves"]
                                      if wave["label"].startswith("singleton_") and wave["status"] == "PASS"],
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

    def __init__(self, failure_mode: str | None = None):
        # The production prefix manager stores one checkpoint per bank.
        # A miss overwrites that bank's prior entry; entries are not retained
        # globally by document or prompt hash.
        self.cache: dict[int, tuple[str, int]] = {}
        self.calls: list[tuple[str, list[dict[str, Any]]]] = []
        self.failure_mode = failure_mode
        self.bank_by_document = {doc_id: bank for bank, (_, doc_id, _) in enumerate(DOCUMENTS)}
        self.capacities = [DEFAULT_BANK_CONTEXT] * 4
        self.ubatches = [256] * 4

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
            bank = self.bank_by_document[doc_id]
            singleton = label.startswith("singleton_")
            if self.failure_mode == "singleton_bank0_fallback" and singleton:
                bank = 0
            if self.failure_mode == "geometry_rebuild" and singleton:
                self.capacities[bank] = DEFAULT_BANK_CONTEXT - 256
            prompt_hash = sha256_text(prompt)
            previous_entry = self.cache.get(bank)
            cached = previous_entry[1] if previous_entry and previous_entry[0] == prompt_hash else 0
            if self.failure_mode == "edited_false_hit" and label == "edited_early_body2":
                cached = 1024
            # Positive hits use a checkpoint-sized synthetic count. For the edited
            # prompts the hash changes, so the old document state cannot be reused.
            input_tokens = max(8192, len(prompt) // 3)
            if request["cache_prompt"]:
                # Successful hits refresh the same entry; cold/edited requests
                # replace the old checkpoint for this bank.
                self.cache[bank] = (prompt_hash, input_tokens - 256)
            accepted[request_id] = {"type": "accepted", "id": request_id,
                                    "input_tokens": input_tokens,
                                    "reserved_tokens": input_tokens + request["max_tokens"], "bank": bank}
            cache_status[request_id] = {"type": "cache_status", "id": request_id,
                                        "hit": cached > 0, "cached_tokens": cached,
                                        "checkpoint_tokens": cached, "bank": bank}
            texts[request_id] = str(request["_expected"])
            done[request_id] = {"type": "done", "id": request_id,
                                "input_tokens": input_tokens, "output_tokens": 1,
                                "finish_reason": "stop", "bank": bank}
            first[request_id] = 0.25 + offset * 0.01
            last[request_id] = first[request_id]
        timings = {
            "mock": True,
            "banks": [{"bank": i, "target_context": self.capacities[i],
                       "draft_context": self.capacities[i], "target_ubatch": self.ubatches[i]}
                      for i in range(4)],
            "mtp_depth": 3,
            "single_cached_affinity": singleton and self.failure_mode != "singleton_bank0_fallback",
            "single_cache_admission_free_mib": [512] * 6,
            "production_floor_mib": 256,
            "context_rebuilds_before_execution": 1 if self.failure_mode == "geometry_rebuild" and singleton else 0,
            "state_migration_supported": False,
            "post_wave_reclaimed_banks": 0,
        }
        return {
            "id": f"mock-{label}", "accepted": accepted, "cache_status": cache_status,
            "done": done, "errors": [], "texts": texts, "first_token_s": first,
            "last_token_s": last, "wall_s": 0.5, "timings": timings,
        }


def self_test(source: str) -> dict[str, Any]:
    transport = MockTransport()
    with tempfile.TemporaryDirectory(prefix="ac922-affinity-mock-") as temporary:
        output = Path(temporary) / "affinity-result.json"
        report = run_campaign(transport, source, output, emit_summary=False)
        if report["status"] != "PASS":
            raise AcceptanceError("mock affinity campaign failed")
        if [name for name, _ in transport.calls] != [
                "cold4", "same4", "shuffle_3_1_0_2", "subset2_2_0", "subset3_1_3_2",
                "singleton_D3", "singleton_A0", "singleton_B1", "singleton_D3_repeat",
                "edited_early_body2"]:
            raise AcceptanceError("mock campaign wave order changed")
        for label, requests in transport.calls:
            if len(requests) < 2 and not label.startswith("singleton_"):
                raise AcceptanceError(f"self-test found a single-request profile in {label}")
            for request in requests:
                if not request["prompt"].startswith(f"DOCUMENT_ID={request['_document_id']}\nNOVA = "):
                    raise AcceptanceError("self-test found a malformed front-matter mapping")
        singleton_mapping = {
            wave["label"]: wave["mapping"] for wave in report["waves"]
            if wave["label"].startswith("singleton_")
        }
        expected_singleton_mapping = {
            "singleton_D3": {"doc-delta": 3},
            "singleton_A0": {"doc-alpha": 0},
            "singleton_B1": {"doc-bravo": 1},
            "singleton_D3_repeat": {"doc-delta": 3},
        }
        if singleton_mapping != expected_singleton_mapping:
            raise AcceptanceError("mock did not preserve the requested singleton bank affinity")
        if report["waves"][-1]["label"] != "edited_early_body2":
            raise AcceptanceError("edited checkpoint replacement probe must be last")
        saved = output.read_text(encoding="utf-8")
        if any(request["prompt"] in saved for _, requests in transport.calls for request in requests):
            raise AcceptanceError("report leaked a full input prompt")
        # Model the invalid ordering explicitly: editing alpha/bravo replaces
        # their only bank-local checkpoints; querying the original A/B must be
        # reported as cold misses, never promoted as positive-hit cases.
        order_negative = MockTransport()
        body = make_corpus_body(source)
        waves = campaign_waves(body)
        for label, requests, _ in waves[:5]:
            order_negative.wave(label, requests)
        edit_label, edit_requests, _ = waves[-1]
        order_negative.wave(edit_label, edit_requests)
        cold_after_edit: dict[str, int] = {}
        singleton_by_label = {label: requests for label, requests, _ in waves
                              if label in {"singleton_A0", "singleton_B1"}}
        for label in ("singleton_A0", "singleton_B1"):
            original_request = singleton_by_label[label]
            reordered_label = "after-edit-" + label
            result_after_edit = order_negative.wave(reordered_label, original_request)
            enriched_after_edit, accepted_as_hit, _ = enrich_wave(
                reordered_label, original_request, result_after_edit,
                expected_cache="hit", bank_context=DEFAULT_BANK_CONTEXT)
            hit_tokens = enriched_after_edit["bank_mapping_by_request"][0]["cached_tokens"]
            if hit_tokens != 0 or accepted_as_hit:
                raise AcceptanceError(f"single-bank edit overwrite was not reflected for {label}")
            cold_after_edit[label] = hit_tokens
        # Each negative arm isolates a requested singleton safety gate.
        rejected_modes: list[str] = []
        for failure_mode in ("singleton_bank0_fallback", "geometry_rebuild", "edited_false_hit"):
            bad_transport = MockTransport(failure_mode=failure_mode)
            bad_output = Path(temporary) / f"{failure_mode}.json"
            caught = False
            try:
                run_campaign(bad_transport, source, bad_output, emit_summary=False)
            except AcceptanceError:
                caught = True
            if not caught:
                raise AcceptanceError(f"mock failed to reject {failure_mode}")
            bad_report = json.loads(bad_output.read_text(encoding="utf-8"))
            if bad_report.get("status") != "FAIL":
                raise AcceptanceError(f"{failure_mode} failure was not persisted")
            rejected_modes.append(failure_mode)
        return {"self_test": "PASS", "transport": "in-memory mock; no network/socket/model calls",
                "waves": len(report["waves"]), "singleton_mapping": singleton_mapping,
                "early_edit_then_original_A_B_cold": cold_after_edit,
                "negative_modes_rejected": rejected_modes,
                "report_mode": report["mode"]}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--self-test", action="store_true", help="run only the in-memory mock; no network")
    modes.add_argument("--run", action="store_true", help="send ten waves to the existing single-cache control socket")
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
