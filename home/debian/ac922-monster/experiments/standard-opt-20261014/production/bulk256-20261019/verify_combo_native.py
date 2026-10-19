#!/usr/bin/env python3
"""Gate the R444 combo worker on the original four frozen native canaries."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any


HERE = Path(__file__).resolve().parent
BASE = Path(os.environ.get("AC922_EXPERIMENT_ROOT", str(HERE.parents[1])))
NATIVE_DIR = BASE / "fixtures/dual-context-v35-onebank4-combined-20261016/native-semantic"
NATIVE_CASES = NATIVE_DIR / "case-input.json"
NATIVE_EVALUATOR = NATIVE_DIR / "run_semantic_native.py"
EXPECTED_NATIVE_CASES_SHA256 = "f2487f90c32d0e383494dc5ae68bec24d6bf0c8d31a065dfbf614447fb6b53fd"
EXPECTED_NATIVE_EVALUATOR_SHA256 = "85b56838258de35eb492425a0286f1078ac836b1826b9f8885fe331cd92cc2ad"
EXPECTED_RUNTIME_BUILD_ID = "20261018T092503.472988Z"
EXPECTED_RUNTIME_REPORT_SHA256 = "b993f9a857c73586a9ff39b5b655ec5bc1984ae48924b33ef033abcef5daf4f4"
EXPECTED_CASE_IDS = ("capital-en", "arithmetic-zh", "logic-en", "strict-json-zh")
EXPECTED_PROMPT_TOKENS = {
    "capital-en": 32,
    "arithmetic-zh": 35,
    "logic-en": 41,
    "strict-json-zh": 47,
}
PROMPT_SOURCE = "native-semantics-case-input-chat-text"


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def load_native_evaluator() -> Any:
    if sha_file(NATIVE_CASES) != EXPECTED_NATIVE_CASES_SHA256:
        raise ValueError("frozen native case-input.json SHA-256 mismatch")
    if sha_file(NATIVE_EVALUATOR) != EXPECTED_NATIVE_EVALUATOR_SHA256:
        raise ValueError("frozen run_semantic_native.py SHA-256 mismatch")
    spec = importlib.util.spec_from_file_location("combo_frozen_run_semantic_native", NATIVE_EVALUATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen native evaluator: {NATIVE_EVALUATOR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify_runtime_pin(runtime_build_id: str) -> tuple[str, list[str]]:
    errors: list[str] = []
    if runtime_build_id != EXPECTED_RUNTIME_BUILD_ID:
        errors.append(f"ROOT_RUNTIME_REVISION={runtime_build_id!r}, expected R444 {EXPECTED_RUNTIME_BUILD_ID}")
    build_report = os.environ.get("ROOT_BUILD_REPORT", "")
    if not build_report:
        errors.append("ROOT_BUILD_REPORT is unset; cannot verify the pinned R444 build report")
        return "", errors
    path = Path(build_report)
    try:
        digest = sha_file(path)
        if digest != EXPECTED_RUNTIME_REPORT_SHA256:
            errors.append(f"ROOT_BUILD_REPORT SHA-256={digest}, expected {EXPECTED_RUNTIME_REPORT_SHA256}")
        report_data = json.loads(path.read_text(encoding="utf-8"))
        if report_data.get("build_run_id") != EXPECTED_RUNTIME_BUILD_ID:
            errors.append(
                f"ROOT_BUILD_REPORT build_run_id={report_data.get('build_run_id')!r}, "
                f"expected {EXPECTED_RUNTIME_BUILD_ID}"
            )
    except Exception as exc:
        errors.append(f"cannot verify ROOT_BUILD_REPORT {path}: {type(exc).__name__}: {exc}")
        digest = ""
    return digest, errors


def frozen_four_cases() -> list[dict[str, Any]]:
    rows = json.loads(NATIVE_CASES.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("frozen native case-input.json must be an array")
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in by_id:
            raise ValueError(f"missing or duplicate frozen case_id: {case_id!r}")
        by_id[case_id] = row
    cases = [by_id.get(case_id) for case_id in EXPECTED_CASE_IDS]
    if any(case is None for case in cases):
        raise ValueError("frozen source does not contain the exact original native four cases")
    return [case for case in cases if case is not None]


def check_boot_metadata(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected = {
        # Four-bank boot retains the V3064096-token logical batch.
        # Wide single-prefill contexts use32768 after a drained profile switch.
        "target_n_batch": 17152,
        "draft_n_batch": 17152,
        "target_n_ubatch": 256,
        "configured_target_n_ubatch": 256,
        "draft_n_ubatch": 256,
        "configured_draft_n_ubatch": 256,
        "logical_prefill_batch_capacity": 32768,
        "mtp_depth": 3,
    }
    for field, value in expected.items():
        if record.get(field) != value:
            errors.append(f"boot metadata {field}={record.get(field)!r}, expected {value}")

    if record.get("bank_count") != 4 or record.get("bank_lanes") != 1:
        errors.append("boot must contain four independent banks with one lane per bank")
    if record.get("mtp_unrolled") is not False:
        errors.append("boot must use the verified generic D3 MTP chain")
    for field in ("actual_sequence_count", "actual_kv_streams",
                  "actual_draft_sequence_count", "actual_draft_kv_streams"):
        if record.get(field) != 1:
            errors.append(f"boot metadata {field}={record.get(field)!r}, expected one")
    topology = record.get("bank_topology")
    if not isinstance(topology, str) or not topology.startswith("4_independent_banks_1_lane_each_per_seq_"):
        errors.append(f"unexpected context topology: {topology!r}")

    layout = record.get("banks_kv_layout")
    if not isinstance(layout, list) or len(layout) != 4:
        errors.append("actual per-bank KV layout must report exactly four contexts")
    else:
        bank_ids: list[int] = []
        for row in layout:
            bank_id = row.get("bank")
            if type(bank_id) is not int:
                errors.append(f"invalid bank id in actual KV layout: {bank_id!r}")
                continue
            bank_ids.append(bank_id)
            for field in ("target_sequence_count", "draft_sequence_count",
                          "target_actual_kv_streams", "draft_actual_kv_streams"):
                if row.get(field) != 1:
                    errors.append(f"bank {bank_id} {field}={row.get(field)!r}, expected one")
            for field in ("target_context_per_sequence", "draft_context_per_sequence"):
                if row.get(field) != 17152:
                    errors.append(f"bank {bank_id} {field}={row.get(field)!r}, expected 17152")
        if sorted(bank_ids) != [0, 1, 2, 3]:
            errors.append(f"actual KV contexts must be four independent banks 0..3; got {bank_ids!r}")
    return errors


def evaluate_combo(scenarios: Any, results: Any, runtime_build_id: str,
                   native: Any, source_cases: list[dict[str, Any]] | None = None
                   ) -> tuple[dict[str, Any], Any]:
    runtime_report_sha256, errors = verify_runtime_pin(runtime_build_id)
    source_cases = source_cases if source_cases is not None else frozen_four_cases()
    if not isinstance(scenarios, list):
        errors.append("scenario input must be an array")
        scenarios = []
    semantic_scenarios = [row for row in scenarios if row.get("kind") == "semantic_p4_gate"]
    if len(semantic_scenarios) != 1:
        errors.append("scenario input must contain exactly one semantic_p4_gate")
    scenario = semantic_scenarios[0] if len(semantic_scenarios) == 1 else {}
    if scenario.get("dual") is not True:
        errors.append("native canary must use four independent bank owners")
    requests = scenario.get("requests", [])
    if not isinstance(requests, list):
        errors.append("semantic scenario requests must be an array")
        requests = []
    request_ids = [row.get("id") for row in requests]
    if tuple(request_ids) != EXPECTED_CASE_IDS:
        errors.append(f"native request IDs/order differ from frozen case-input.json: {request_ids!r}")

    cases_by_id = {case["case_id"]: case for case in source_cases}
    requests_by_id: dict[str, dict[str, Any]] = {}
    for request in requests:
        case_id = request.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in requests_by_id:
            errors.append(f"missing or duplicate native scenario request id: {case_id!r}")
            continue
        requests_by_id[case_id] = request
        source = cases_by_id.get(case_id)
        if source is None:
            errors.append(f"scenario request is not in frozen case-input.json: {case_id}")
            continue
        prompt = source.get("prompt")
        prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest() if isinstance(prompt, str) else ""
        for field in ("prompt", "checker", "expected"):
            if request.get(field) != source.get(field):
                errors.append(f"{case_id}: scenario {field} differs from frozen case-input.json")
        if request.get("prompt_source") != PROMPT_SOURCE or request.get("prompt_sha256") != prompt_sha:
            errors.append(f"{case_id}: scenario source marker or exact prompt hash differs")
        if request.get("max_tokens") != 32 or request.get("ignore_eos") is not False:
            errors.append(f"{case_id}: native canary token/termination contract changed")

    if not isinstance(results, list):
        errors.append("result input must be an array")
        results = []
    updated = deepcopy(results)
    scenario_label = scenario.get("label")
    matching = [row for row in updated if row.get("label") == scenario_label]
    if len(matching) != 1:
        errors.append(f"expected exactly one result for semantic scenario {scenario_label!r}")
        record = matching[0] if matching else None
    else:
        record = matching[0]

    case_results: list[dict[str, Any]] = []
    if record is not None:
        if record.get("kind") != "semantic_p4_gate":
            errors.append("native result kind differs from semantic_p4_gate")
        errors.extend(check_boot_metadata(record))
        flows = record.get("flows", [])
        if not isinstance(flows, list):
            errors.append("native result flows must be an array")
            flows = []
        flow_by_id: dict[str, dict[str, Any]] = {}
        for flow in flows:
            case_id = flow.get("id")
            if not isinstance(case_id, str) or not case_id or case_id in flow_by_id:
                errors.append(f"missing or duplicate result flow id: {case_id!r}")
                continue
            flow_by_id[case_id] = flow
        if set(flow_by_id) != set(EXPECTED_CASE_IDS):
            errors.append("native result flow IDs differ from the exact frozen four-case IDs")
        bank_ids = [flow.get("bank") for flow in flows]
        if any(type(bank) is not int for bank in bank_ids) or sorted(bank_ids) != [0, 1, 2, 3]:
            errors.append(f"native flows must map to four distinct banks 0..3; observed {bank_ids!r}")

        for expected_bank, case_id in enumerate(EXPECTED_CASE_IDS):
            source = cases_by_id.get(case_id)
            request = requests_by_id.get(case_id)
            flow = flow_by_id.get(case_id)
            if source is None or request is None or flow is None:
                continue
            prompt_sha = hashlib.sha256(source["prompt"].encode("utf-8")).hexdigest()
            expected_tokens = EXPECTED_PROMPT_TOKENS[case_id]
            for field, expected in (("prompt_sha256", prompt_sha),
                                    ("prompt_source", PROMPT_SOURCE),
                                    ("checker", source["checker"]),
                                    ("expected", source["expected"]),
                                    ("input_tokens", expected_tokens)):
                if flow.get(field) != expected:
                    errors.append(f"{case_id}: flow {field}={flow.get(field)!r}, expected {expected!r}")
            output, bank = flow.get("output"), flow.get("bank")
            if not isinstance(output, str) or type(bank) is not int:
                errors.append(f"{case_id}: result flow lacks text output or integer bank owner")
                continue
            if bank != expected_bank:
                errors.append(f"{case_id}: mapped bank={bank}, expected independent bank {expected_bank}")

            # This local worker reports bank ownership, not HTTP id_slot. Map the
            # verified independent bank ordinal into the evaluator's slot field;
            # the report records this adaptation and does not claim HTTP slot echo.
            evaluator_case = {
                "case_id": case_id,
                "checker": source["checker"],
                "prompt": source["prompt"],
                "expected": source["expected"],
                "id_slot": bank,
            }
            try:
                status, reason, extracted = native.evaluate_case(
                    evaluator_case, {"content": output, "id_slot": bank}
                )
            except Exception as exc:
                status, reason, extracted = "fail", f"{type(exc).__name__}: {exc}", None
            flow["semantic_checked"] = True
            flow["semantic_pass"] = status == "pass"
            flow["semantic_reason"] = reason
            case_results.append({
                "case_id": case_id,
                "bank": bank,
                "checker": source["checker"],
                "prompt_sha256": prompt_sha,
                "expected_input_tokens": expected_tokens,
                "actual_input_tokens": flow.get("input_tokens"),
                "status": status,
                "reason": reason,
                "expected": source["expected"],
                "actual": extracted if extracted is not None else output,
                "token_ids": flow.get("token_ids"),
                "evaluator_id_slot_source": "verified local worker bank ordinal; not an HTTP slot claim",
            })
            if status != "pass":
                errors.append(f"{case_id}: frozen native {source['checker']} evaluator returned {status}")

        passed = not errors and len(case_results) == 4 and all(row["status"] == "pass" for row in case_results)
        record["semantic_status"] = "pass" if passed else "fail"
        record["semantic_gate"] = {
            "schema": "combo-native4-gate-v1",
            "status": record["semantic_status"],
            "scope": "exactly four frozen native cases; not the full semantic suite",
            "native_source_sha256": EXPECTED_NATIVE_CASES_SHA256,
            "native_evaluator_sha256": EXPECTED_NATIVE_EVALUATOR_SHA256,
            "runtime_build_id": runtime_build_id,
            "runtime_report_sha256": runtime_report_sha256,
            "context_mapping": "four independent one-lane banks; local bank ordinal adapted to native evaluator id_slot",
            "actual_target_n_batch": record.get("target_n_batch"),
            "actual_target_n_ubatch": record.get("target_n_ubatch"),
            "cases": case_results,
            "errors": errors,
        }
    else:
        passed = False

    report = {
        "schema": "combo-native4-report-v1",
        "status": "pass" if passed else "fail",
        "scope": "four frozen native cases only; no full-suite or performance claim",
        "runtime_build_id": runtime_build_id,
        "expected_runtime_build_id": EXPECTED_RUNTIME_BUILD_ID,
        "runtime_report_sha256": runtime_report_sha256,
        "native_source": str(NATIVE_CASES),
        "native_source_sha256": EXPECTED_NATIVE_CASES_SHA256,
        "native_evaluator": str(NATIVE_EVALUATOR),
        "native_evaluator_sha256": EXPECTED_NATIVE_EVALUATOR_SHA256,
        "case_count": len(case_results),
        "cases": case_results,
        "errors": errors,
        "model_loaded_by_gate": False,
        "gpu_touched_by_gate": False,
    }
    return report, updated


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        temp = Path(stream.name)
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        native = load_native_evaluator()
        source_cases = frozen_four_cases()
        scenarios = json.loads(args.scenarios.read_text(encoding="utf-8"))
        results = json.loads(args.results.read_text(encoding="utf-8"))
        runtime_build_id = os.environ.get("ROOT_RUNTIME_REVISION", "")
        report, updated = evaluate_combo(scenarios, results, runtime_build_id, native, source_cases)
        write_json_atomic(args.results, updated)
        write_json_atomic(args.report, report)
    except Exception as exc:
        report = {
            "schema": "combo-native4-report-v1",
            "status": "fail",
            "scope": "four frozen native cases only; no full-suite or performance claim",
            "runtime_build_id": os.environ.get("ROOT_RUNTIME_REVISION", ""),
            "errors": [f"{type(exc).__name__}: {exc}"],
            "model_loaded_by_gate": False,
            "gpu_touched_by_gate": False,
        }
        try:
            write_json_atomic(args.report, report)
        except Exception as report_exc:
            print(f"combo native report write failed: {type(report_exc).__name__}: {report_exc}", file=sys.stderr)
        print(f"combo native gate failed closed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"combo-native4 status={report['status']} cases={report['case_count']} runtime={report['runtime_build_id']}")
    print(f"report={args.report}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
