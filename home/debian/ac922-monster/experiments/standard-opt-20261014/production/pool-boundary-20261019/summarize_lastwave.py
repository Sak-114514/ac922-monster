"""Summarize measured windows without conflating cache hits or HTTP latency."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
run = HERE / "run"
inband = json.loads((run / "inband-result.json").read_text())
http = json.loads((run / "http-result.json").read_text())
prefix4 = json.loads((run / "prefix-fourbank-result.json").read_text())
waves = {row["label"]: row for row in inband["waves"]}
summary = {"worker_sha256": json.loads((HERE/"build-lastwave/build-report.json").read_text())["worker_sha256"],
           "status": "PENDING_FULL_TEXT_REVIEW", "prefill": {}, "generation": {},
           "cache_fourbank": prefix4["status"], "http": http["status"], "inband": inband["status"]}
for label in ["lastwave-32k-first", "lastwave-32k-repeat", "lastwave-64k-max262k-UB512", "lastwave-native4-8k"]:
    timing = waves[label]["timings"]
    summary["prefill"][label] = {"input_tokens": timing["prefill_input_tokens"],
        "seconds": timing["prefill_wall_seconds"], "tok_s": timing["prefill_aggregate_wall_tok_s"],
        "cached_tokens": timing["cached_tokens"], "whole_control_wall_s": waves[label]["wall_s"]}
four = waves["lastwave-four512"]
summary["generation"].update({"worker_execute_tok_s": four["timings"]["whole_wave_output_tok_s"],
    "all_emissions_tok_s": four["timings"]["all_emissions_tok_s"],
    "control_arrival_to_last_tok_s": 2048/four["wall_s"], "control_elapsed_s": four["wall_s"],
    "http_actual_usage_tok_s": http["phases"]["four_by_512_usage_aggregate"]["aggregate_tok_s"]})
for path in (run / "model").rglob("lastwave-*.json"):
    data = json.loads(path.read_text())
    if data.get("wave_id") != four["id"]:
        continue
    flows = data["flows"]
    global_decode = max(flow["decode_started_us"] for flow in flows)
    active = [[t for t in flow["emitted_us"] if t >= global_decode] for flow in flows]
    if all(active):
        begin, end = max(times[0] for times in active), min(times[-1] for times in active)
        if end > begin:
            count = sum(sum(begin <= t <= end for t in flow["emitted_us"]) for flow in flows)
            summary["generation"].update(shared_post_all_warmups_tok_s=count*1e6/(end-begin),
                                         shared_tokens=count, shared_span_s=(end-begin)/1e6)
    summary["generation"]["per_flow"] = [{"id": flow["id"],
        "input_tokens": flow["input_tokens"], "output_tokens": flow["output_tokens"],
        "prefill_us": flow["prefill_us"], "proposed": flow["proposed"], "accepted": flow["accepted"],
        "own_all_emissions_tok_s": flow["output_tokens"]*1e6/max(1,flow["emitted_us"][-1]-flow["emitted_us"][0])}
        for flow in flows]
    break
summary["prefix_single"] = [{key: row.get(key) for key in ["label", "actual_prompt_tokens",
    "cache_hit_tokens", "wall_s", "text", "semantic_pass"]}
    for row in http["phases"]["exact_prefix_cache"]["requests"]]
summary["asymmetric_prefill_first_token_s"] = waves["lastwave-asymmetric-prefill-32k-8k-1k-short"]["first_token_s"]
summary["minimum_reported_post_wave_free_mib"] = min(
    min(row["timings"]["post_wave_free_before_mib"]) for row in inband["waves"])
(run / "measured-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
print(json.dumps(summary, ensure_ascii=False, indent=2))
