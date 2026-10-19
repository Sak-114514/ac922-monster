"""Archive the completed C3 campaign after the parent's full-text review."""
import hashlib
import json
from pathlib import Path
import shutil
import time

F = Path(__file__).resolve().parent
R = F / "run"
E = F / "evidence"
E.mkdir(exist_ok=True)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


s = read(R / "measured-summary.json")
assert s["status"] == "PENDING_SEMANTIC_ADJUDICATION"
assert all(x["token_ids_exact"] and x["text_exact"] for x in s["l2_four512_exact_comparison"]["per_flow"].values())
assert all(x["sse_full_text_exact"] and x["usage_response_full_text_exact"] for x in s["http_full_text_identity_if_available"].values())
assert min(s["telemetry"]["minimum_free_mib_by_gpu"]) >= 256
assert max(s["telemetry"]["maximum_temperature_c_by_gpu"]) <= 65
warm = read(R / "raw32k-recheck-result.json")
assert warm["status"] == "PASS_TIMING_ONLY"
for wave in warm["waves"]:
    t = wave["timings"]
    assert t["cached_tokens"] == 0 and t["prefill_input_tokens"] == 32768
s["raw32k_order_recheck"] = [{"label": w["label"], "control_wall_s": w["wall_s"],
    **{k: w["timings"][k] for k in ["prefill_aggregate_wall_tok_s", "cached_tokens", "context_rebuilds_before_execution", "thermal_wait_us_sum"]}}
    for w in warm["waves"]]
single = read(R / "single-cache-result.json")
essay = next(w for w in single["waves"] if w["label"] == "c3-cached-single-512-prose")["texts"]["cached-essay"]
review = {"status": "PASS", "reviewed_unix": time.time(),
    "new_cached_single_512": {"text_sha256": hashlib.sha256(essay.encode()).hexdigest(),
        "review": "Parent read the entire 512-token CPU-cache essay. Coherent topic, grammar and hierarchy; no garbling, repetition loop or cross-document answer leakage. Truncated at requested token cap."},
    "four512": "All 2048 IPC token IDs and full texts equal the fully reviewed L2 four-flow artifact; all eight SSE/nonstream full texts equal those IPC texts.",
    "new_state_paths": "D/A/B/D same-bank hits, mixed geometry, after-prose answer, early-prefix-edit misses, capacity fallback and cancelled cached-single follow-up passed exact expected answers.",
    "limits": "Semantic canaries and full-text review are evidence, not a general quality benchmark. Raw 8-token timing slices were not semantic quality probes."}
(E / "semantic-review.json").write_text(json.dumps(review, indent=2) + "\n")
s["producer_status"] = s["status"]
s["status"] = "PASS"
s["semantic_adjudication"] = "evidence/semantic-review.json"
(F / "MEASURED-SINGLE-CACHE.json").write_text(json.dumps(s, indent=2) + "\n")
for name in ["affinity-result.json", "single-cache-result.json", "http-result.json", "cancel-cached-single.json", "campaign-status.json", "raw32k-recheck-result.json"]:
    shutil.copy2(R / name, E / name)
shutil.copy2(Path(s["c3_four512"]["rawflow_path"]), E / "four512-flow-record.json")
for name in ["c3-postcleanup-api.log", "c3-postcleanup-runner.log", "c3-postcleanup-affinity-mock.log", "build-report.json"]:
    shutil.copy2(F / "build-lastwave" / name, E / name)
for path in (R / "model").rglob("*.semantic-gate.json"):
    shutil.copy2(path, E / "native-semantic-gate.json")
    break
else:
    raise RuntimeError("native startup semantic receipt absent")
report = read(F / "build-lastwave/build-report.json")
receipt = {"status": "PASS", "qualified_unix": time.time(), "worker_sha256": report["worker_sha256"],
    "api_tests": 30, "cuda_tus_rebuilt": 0, "engine_tus_rebuilt": 0, "mtp_depth": 3,
    "quality": "evidence/semantic-review.json", "six_device_first_capture": "onecat-fixture-receipt.json",
    "telemetry": s["telemetry"], "evidence_sha256": {str(p.relative_to(F)): sha(p) for p in sorted(E.iterdir()) if p.is_file()},
    "limits": ["raw300/6000notmet", "cache latency improvement is not raw compute throughput", "full524288pool proof inherited from unchanged L2 allocator; no full-pool input rerun", "fixedD3, samequant/experts/runtime; no hardware changes"],
    "architect_review": "Pending final evidence signoff; parent must obtain approval before promotion"}
(F / "lastwave-acceptance.json").write_text(json.dumps(receipt, indent=2) + "\n")
print("C3_EVIDENCE_ARCHIVED; FINAL_ARCHITECT_SIGNOFF_REQUIRED")
