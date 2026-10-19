#!/usr/bin/env python3
"""Gate the same-process clean performance family on native and F32 integrity.

This gate deliberately does not require generated-token parity. The original
post-run comparator still reports all token/prefix failures; this preclean
gate only prevents a clean timing phase after semantic failure, non-finite
logits, or a serial/P4 full-logit difference.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from compare_prefill_logits_v16 import compare_dumps


def _check_semantics(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("status") != "pass":
        errors.append("native semantic gate did not pass")
    if report.get("checker") != "run_semantic_native.evaluate_case":
        errors.append("native semantic report does not name the original checker")
    if report.get("generated_python_executed") is not False:
        errors.append("native semantic report does not prove generated Python was not executed")
    if report.get("id_slot_policy") != "null; local bank/lane IDs are not HTTP slots":
        errors.append("native semantic report has an invalid slot mapping")
    cases = [case for scenario in report.get("scenarios", []) for case in scenario.get("cases", [])]
    if not cases or any(case.get("status") != "pass" for case in cases):
        errors.append("native semantic report has missing or failed cases")
    return errors


def evaluate(
    results: list[dict[str, Any]],
    plan: dict[str, Any],
    semantic_report: dict[str, Any],
    runtime_log: str,
    logit_dir: Path,
) -> dict[str, Any]:
    errors = _check_semantics(semantic_report)
    finite_rows = []
    for line_number, line in enumerate(runtime_log.splitlines(), 1):
        if "first-logits id=" not in line:
            continue
        fields = {key: int(match.group(1)) for key, pattern in (
            ("finite", r"\bfinite=(\d+)"),
            ("nan", r"\bnan=(\d+)"),
            ("positive_inf", r"\bpositive_inf=(\d+)"),
        ) if (match := re.search(pattern, line))}
        finite_rows.append({"line": line_number, **fields})
        if fields.get("finite", 0) <= 0 or fields.get("nan") != 0 or fields.get("positive_inf") != 0:
            errors.append(f"first-logits diagnostics are non-finite or incomplete at log line {line_number}")
    if not finite_rows:
        errors.append("diagnostic family did not emit first-logits finite-value checks")

    indexed = {record.get("label"): record for record in results}
    comparisons = []
    within_flow = []
    for pair in plan.get("pairs", []):
        flow_id = pair["id"]
        serial_label = pair["serial_label"]
        parallel_label = pair["parallel_label"]
        serial = indexed.get(serial_label)
        parallel = indexed.get(parallel_label)
        if serial is None or parallel is None:
            errors.append(f"missing diagnostic serial/P4 record for full-logit pair {flow_id}")
            continue
        serial_dump = logit_dir / f"{serial_label}__{flow_id}.ac922pflogits"
        parallel_dump = logit_dir / f"{parallel_label}__{flow_id}.ac922pflogits"
        serial_before = logit_dir / f"{serial_label}_before-mtp__{flow_id}.ac922pflogits"
        parallel_before = logit_dir / f"{parallel_label}_before-mtp__{flow_id}.ac922pflogits"
        for phase, left, right in (
            ("before-mtp", serial_before, parallel_before),
            ("after-mtp", serial_dump, parallel_dump),
        ):
            try:
                result = compare_dumps(left, right)
            except (OSError, ValueError) as error:
                errors.append(f"{flow_id} {phase} full-logit comparison failed: {error}")
                continue
            comparisons.append({
                "flow_id": flow_id,
                "phase": phase,
                "max_abs_diff": result["max_abs_diff"],
                "rms_diff": result["rms_diff"],
                "argmax_equal": result["argmax_equal"],
            })
            if result["max_abs_diff"] != 0.0:
                errors.append(f"{flow_id} {phase} target logits differ between serial and P4")

    for record in results:
        label = record.get("label", "")
        for flow in record.get("flows", []):
            flow_id = flow.get("id", "")
            before = logit_dir / f"{label}_before-mtp__{flow_id}.ac922pflogits"
            after = logit_dir / f"{label}__{flow_id}.ac922pflogits"
            try:
                result = compare_dumps(before, after)
            except (OSError, ValueError) as error:
                errors.append(f"{label}/{flow_id} before/after-MTP logit comparison failed: {error}")
                continue
            within_flow.append({
                "label": label,
                "flow_id": flow_id,
                "max_abs_diff": result["max_abs_diff"],
                "rms_diff": result["rms_diff"],
                "argmax_equal": result["argmax_equal"],
            })
            if result["max_abs_diff"] != 0.0:
                errors.append(f"{label}/{flow_id} target logits changed during MTP prefill processing")

    # Preserve all known token mismatches as context, but do not use them to
    # skip the separately measured clean performance scenario.
    token_differences = []
    for pair in plan.get("pairs", []):
        serial = indexed.get(pair["serial_label"])
        parallel = indexed.get(pair["parallel_label"])
        if serial is None or parallel is None:
            continue
        sf = {flow.get("id"): flow for flow in serial.get("flows", [])}.get(pair["id"])
        pf = {flow.get("id"): flow for flow in parallel.get("flows", [])}.get(pair["id"])
        if sf and pf and sf.get("token_ids") != pf.get("token_ids"):
            token_differences.append(pair["id"])

    return {
        "schema": "ac922-v22-preclean-gate-v1",
        "status": "pass" if not errors else "fail",
        "clean_phase_allowed": not errors,
        "original_token_parity_failures_preserved_and_not_suppressed": token_differences,
        "native_semantic_status": semantic_report.get("status"),
        "native_case_count": sum(len(s.get("cases", [])) for s in semantic_report.get("scenarios", [])),
        "finite_logit_rows_checked": len(finite_rows),
        "serial_p4_full_logit_pairs_exact": sum(item["max_abs_diff"] == 0.0 for item in comparisons),
        "serial_p4_full_logit_pairs_total": len(comparisons),
        "same_flow_before_after_mtp_pairs_exact": sum(item["max_abs_diff"] == 0.0 for item in within_flow),
        "same_flow_before_after_mtp_pairs_total": len(within_flow),
        "full_logit_comparisons": comparisons,
        "same_flow_before_after_mtp_comparisons": within_flow,
        "errors": errors,
        "reason": "native semantics and full target logits must be clean before the same-process non-diagnostic performance family; token parity is left to the final comparator",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--semantic-gate-report", required=True, type=Path)
    parser.add_argument("--logit-dir", required=True, type=Path)
    parser.add_argument("--runtime-log", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    report = evaluate(
        json.loads(args.results.read_text(encoding="utf-8")),
        json.loads(args.plan.read_text(encoding="utf-8")),
        json.loads(args.semantic_gate_report.read_text(encoding="utf-8")),
        args.runtime_log.read_text(encoding="utf-8", errors="replace"),
        args.logit_dir,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "clean_phase_allowed",
          "native_semantic_status", "native_case_count", "serial_p4_full_logit_pairs_exact",
          "serial_p4_full_logit_pairs_total", "original_token_parity_failures_preserved_and_not_suppressed",
          "errors")}, ensure_ascii=False))
    return 0 if report["clean_phase_allowed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
