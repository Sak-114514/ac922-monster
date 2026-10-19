#!/usr/bin/env python3
"""Read-only C4 evidence joiner. Produces measured-summary.json, never a release/promotion."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

HERE = Path(__file__).resolve().parent
RUN = HERE / "run"
L3_RELEASE = Path("/home/debian/ac922-monster/production/releases/20261019-r444-banked-l3")
L3_FLOW = HERE.parent / "single-cache-20261019/evidence/four512-flow-record.json"
FOUR512_LABEL = "c3-four512"
RAW32_LABEL = "c3-raw32k-cold-fallback"
CACHED_SINGLE_LABELS = (
    "c3-mixed-geometry-single-d", "c3-cached-single-512-prose",
    "c3-single-d-after-prose", "c3-single-capacity-fallback",
)


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise RuntimeError(f"required evidence file missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot parse {path.name}: {type(exc).__name__}") from None


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def receipt_gates(run_dir: Path) -> dict[str, Any]:
    fixture = read_json(HERE / "onecat-fixture-receipt.json")
    require(fixture.get("status") == "PASS" and bool(fixture.get("cases")),
            "OneCAT fixture receipt is not PASS")
    require(all(case.get("passed") and case.get("verified") for case in fixture["cases"]),
            "OneCAT fixture contains a failed/unverified case")
    campaign = read_json(run_dir / "campaign-status.json")
    require(campaign.get("status") == "PASS_PENDING_FULL_TEXT_REVIEW",
            "C3 controller receipt did not finish all four campaigns")
    stages = {stage.get("label"): stage for stage in campaign.get("stages", [])}
    expected_stages = {"growth", "affinity", "single-cache", "http", "cancel-cached", "prefill-pacing"}
    require(expected_stages <= set(stages), "C3 controller receipt is missing a stage")
    require(all(stages[name].get("status") == "PASS" and stages[name].get("returncode") == 0
                for name in expected_stages), "C3 controller receipt contains a failed stage")
    return {"onecat_fixture": {"status": fixture["status"], "run_tag": fixture.get("run_tag"),
                               "case_count": len(fixture["cases"])},
            "campaign": {"status": campaign["status"],
                         "stages": {name: {"status": stages[name].get("status"),
                                           "returncode": stages[name].get("returncode")}
                                    for name in sorted(expected_stages)}}}


def find_http_result(run_dir: Path) -> tuple[Path, dict[str, Any]]:
    preferred = run_dir / "http-result.json"
    if preferred.exists():
        return preferred, read_json(preferred)
    candidates = sorted(run_dir.glob("lastwave-http-result-*.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise RuntimeError("HTTP acceptance result missing")
    return candidates[-1], read_json(candidates[-1])


def find_wave(report: dict[str, Any], label: str) -> dict[str, Any]:
    waves = report.get("waves")
    if not isinstance(waves, list):
        raise RuntimeError("single-cache result has no wave list")
    matches = [wave for wave in waves if wave.get("label") == label]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one wave named {label}")
    return matches[0]


def rawflow_for(model_root: Path, wave_id: str) -> tuple[Path, dict[str, Any]]:
    for path in sorted(model_root.glob("**/lastwave-*.json"), key=lambda p: p.stat().st_mtime):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and data.get("wave_id") == wave_id and isinstance(data.get("flows"), list):
            return path, data
    raise RuntimeError(f"raw flow record absent for wave_id={wave_id}")


def compare_flows(current: list[dict[str, Any]], baseline: list[dict[str, Any]]) -> dict[str, Any]:
    old = {str(flow.get("id")): flow for flow in baseline}
    result: dict[str, Any] = {}
    for flow in current:
        flow_id = str(flow.get("id"))
        if flow_id not in old:
            raise RuntimeError(f"L3 four512 evidence has no flow {flow_id}")
        previous = old[flow_id]
        now_tokens, old_tokens = flow.get("token_ids"), previous.get("token_ids")
        now_text, old_text = flow.get("output"), previous.get("output")
        if not isinstance(now_tokens, list) or not isinstance(old_tokens, list) or not isinstance(now_text, str) or not isinstance(old_text, str):
            raise RuntimeError(f"full token/text evidence absent for flow {flow_id}")
        result[flow_id] = {
            "token_ids_exact": now_tokens == old_tokens,
            "text_exact": now_text == old_text,
            "current_output_tokens": flow.get("output_tokens"),
            "current_text_characters": len(now_text),
            "l3_text_characters": len(old_text),
        }
    if set(result) != set(old):
        raise RuntimeError(f"C3/L3 four512 flow IDs differ: C3={sorted(result)}, L3={sorted(old)}")
    return result


def telemetry_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "MISSING", "path": str(path)}
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and isinstance(row.get("gpu"), list) and len(row["gpu"]) == 6:
            rows.append(row)
    if not rows:
        return {"status": "INVALID_OR_EMPTY", "path": str(path), "samples": 0}
    return {
        "status": "RECORDED", "path": str(path), "samples": len(rows),
        "start_unix": rows[0].get("unix"), "end_unix": rows[-1].get("unix"),
        "minimum_free_mib_by_gpu": [min(float(row["gpu"][i][0]) for row in rows) for i in range(6)],
        "maximum_temperature_c_by_gpu": [max(int(row["gpu"][i][1]) for row in rows) for i in range(6)],
    }


def summarize(run_dir: Path) -> dict[str, Any]:
    receipt = receipt_gates(run_dir)
    affinity = read_json(run_dir / "affinity-result.json")
    single = read_json(run_dir / "single-cache-result.json")
    http_path, http_report = find_http_result(run_dir)
    cancel = read_json(run_dir / "cancel-cached-single.json")
    require(affinity.get("status") == "PASS", "affinity campaign receipt is not PASS")
    require(single.get("status") == "PASS_PENDING_FULL_TEXT_REVIEW", "single-cache IPC receipt is not PASS_PENDING_FULL_TEXT_REVIEW")
    require(http_report.get("status") == "PASS", "HTTP suite receipt is not PASS")
    require(cancel.get("status") == "PASS", "cancel/cached-single receipt is not PASS")

    raw32 = find_wave(single, RAW32_LABEL)
    rt = raw32["timings"]
    require(rt.get("prefill_input_tokens") == 32768 and rt.get("cached_tokens") == 0,
            "C3 raw32K wave is not an uncached exact 32768-token prompt")
    require(raw32.get("accepted", {}).get("raw32k", {}).get("bank") == 0,
            "raw32K fallback did not use the bank-0 legacy profile")
    require(rt.get("banks", [{}])[0].get("target_ubatch") == 1024,
            "raw32K fallback did not preserve the 1024 UBatch profile")

    four = find_wave(single, FOUR512_LABEL)
    done = four.get("done", {})
    flow_tokens = [item.get("output_tokens") for item in done.values()]
    require(len(done) == 4 and all(value == 512 for value in flow_tokens),
            "C3 four512 IPC wave did not complete four actual 512-token outputs")
    raw_path, raw = rawflow_for(run_dir / "model", four["id"])
    flows = raw.get("flows", [])
    require(len(flows) == 4 and all(flow.get("output_tokens") == 512 for flow in flows),
            "C3 four512 raw flow is absent or incomplete")
    require(sum(flow["output_tokens"] for flow in flows) == sum(flow_tokens),
            "IPC done counts differ from raw full-flow counts")
    frozen_receipt = read_json(L3_RELEASE / "lastwave-acceptance.json")
    expected_baseline = frozen_receipt["evidence_sha256"]["evidence/four512-flow-record.json"]
    require(hashlib.sha256(L3_FLOW.read_bytes()).hexdigest() == expected_baseline,
            "readable L3 reference differs from frozen acceptance hash")
    baseline_doc = read_json(L3_FLOW)
    baseline_flows = baseline_doc.get("flows") if isinstance(baseline_doc, dict) else None
    if not isinstance(baseline_flows, list):
        raise RuntimeError("L3 four512-flow-record.json has no flows")
    exactness = compare_flows(flows, baseline_flows)

    http_phases = http_report.get("phases", {})
    http_usage = http_phases.get("four_by_512_usage_aggregate", {})
    usage_rows = http_usage.get("requests", [])
    usage_counts = [row.get("usage", {}).get("completion_tokens") for row in usage_rows]
    http_usage_total = sum(value for value in usage_counts if isinstance(value, int)) if usage_counts else None
    http_usage_elapsed = http_usage.get("arrival_to_last_completion_s")
    stream_phase = http_phases.get("four_by_512_stream_metrics", {})
    stream_rows = {str(row.get("label")): row for row in stream_phase.get("requests", [])}
    usage_by_story = {str(row.get("label", "")).removeprefix("usage-"): row for row in usage_rows}
    c3_flows = {str(flow["id"]): flow for flow in flows}
    http_identity: dict[str, Any] = {}
    for flow_id, flow in c3_flows.items():
        stream = stream_rows.get(flow_id)
        usage_row = usage_by_story.get(flow_id)
        http_identity[flow_id] = {
            "sse_available": isinstance(stream, dict),
            "sse_full_text_exact": stream.get("text") == flow.get("output") if isinstance(stream, dict) else None,
            "usage_response_available": isinstance(usage_row, dict),
            "usage_response_full_text_exact": usage_row.get("text") == flow.get("output") if isinstance(usage_row, dict) else None,
        }

    cache_rows = []
    for label in CACHED_SINGLE_LABELS:
        wave = find_wave(single, label)
        cache_rows.append({
            "label": label, "status": wave.get("semantic_status", "UNADJUDICATED"),
            "wall_s": wave.get("wall_s"), "control_roundtrip_s": wave.get("control_roundtrip_s"),
            "bank_by_request": {key: event.get("bank") for key, event in wave.get("accepted", {}).items()},
            "cached_tokens_by_request": {key: event.get("cached_tokens") for key, event in wave.get("cache_status", {}).items()},
            "output_tokens_by_request": {key: event.get("output_tokens") for key, event in wave.get("done", {}).items()},
            "geometry": wave.get("timings", {}).get("banks"),
            "single_cached_affinity": wave.get("timings", {}).get("single_cached_affinity"),
            "context_rebuilds_before_execution": wave.get("timings", {}).get("context_rebuilds_before_execution"),
            "post_wave_reclaimed_banks": wave.get("timings", {}).get("post_wave_reclaimed_banks"),
        })

    raw_wall = four.get("wall_s")
    ipc_actual_tokens = sum(flow_tokens)
    return {
        "schema": "ac922-growth-cache-measured-summary-v1",
        "status": "PENDING_SEMANTIC_ADJUDICATION",
        "semantic_adjudication": "Parent must review retained full outputs; throughput and exactness are reported separately.",
        "receipts": receipt,
        "acceptance_status": {"affinity": affinity["status"], "single_cache": single["status"],
                              "http": http_report["status"], "cancel_cached_single": cancel["status"],
                              "http_result_path": str(http_path)},
        "c3_raw32k_uncached_prefill": {key: rt.get(key) for key in (
            "prefill_input_tokens", "prefill_wall_seconds", "prefill_aggregate_wall_tok_s",
            "target_prefill_us_sum", "mtp_prefill_us_sum", "thermal_wait_us_sum", "cached_tokens")}
            | {"control_wall_s": raw32.get("wall_s"), "bank": 0, "target_ubatch": 1024},
        "c3_four512": {
            "wave_id": four["id"], "rawflow_path": str(raw_path), "actual_output_tokens": ipc_actual_tokens,
            "actual_tokens_by_request": flow_tokens, "ipc_control_wall_s": raw_wall,
            "ipc_control_actual_tok_s": ipc_actual_tokens / raw_wall if isinstance(raw_wall, (int,float)) and raw_wall > 0 else None,
            "worker_whole_wave_output_tok_s": four.get("timings", {}).get("whole_wave_output_tok_s"),
            "worker_all_emissions_tok_s": four.get("timings", {}).get("all_emissions_tok_s"),
            "http_usage_completion_tokens": http_usage_total,
            "http_usage_elapsed_s": http_usage_elapsed,
            "http_usage_actual_tok_s_recomputed": http_usage_total / http_usage_elapsed if isinstance(http_usage_total,int) and isinstance(http_usage_elapsed,(int,float)) and http_usage_elapsed > 0 else None,
            "http_usage_recorded_tok_s": http_usage.get("aggregate_tok_s"),
            "http_stream_phase_status": stream_phase.get("status"),
            "http_stream_counted_tokens_from_finish_reason": stream_phase.get("stream_max_tokens_verified_count"),
        },
        "l3_four512_exact_comparison": {"baseline_path": str(L3_FLOW), "per_flow": exactness},
        "http_full_text_identity_if_available": http_identity,
        "cached_single_waves": cache_rows,
        "telemetry": telemetry_summary(run_dir / "gpu-samples.jsonl"),
        "source_paths": {"run_dir": str(run_dir), "l3_flow": str(L3_FLOW)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(RUN))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).expanduser()
    output = Path(args.out).expanduser() if args.out else run_dir / "measured-summary.json"
    summary: dict[str, Any] = {"schema": "ac922-growth-cache-measured-summary-v1", "status": "FAIL"}
    try:
        summary = summarize(run_dir)
    except Exception as exc:
        summary.update({"status": "FAIL", "failure": f"{type(exc).__name__}: {exc}", "run_dir": str(run_dir)})
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"status": summary["status"], "summary": str(output)}, ensure_ascii=False))
    return 0 if summary["status"] == "PENDING_SEMANTIC_ADJUDICATION" else 1


if __name__ == "__main__":
    raise SystemExit(main())
