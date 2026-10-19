"""Same loaded model: four independent recurrent/MTP prefix restores."""
import json
from pathlib import Path

import lastwave_inband_acceptance as harness

harness.OUT = Path(__file__).resolve().parent / "run/prefix-fourbank-result.json"
harness.report = {"status": "RUNNING", "waves": [], "purpose": "four separate same-bank prefix states"}
native = json.loads(harness.CASES.read_text())[0]["requests"]
requests = [{"id": case["id"], "prompt": case["prompt"], "max_tokens": case["max_tokens"],
             "semantic_pad_to_tokens": 8192} for case in native]
try:
    for label in ["prefix-fourbank-cold", "prefix-fourbank-hit"]:
        result = harness.wave(label, requests, cache_prompt=True)
        for case in native:
            text = result["texts"][case["id"]].strip()
            if case["checker"] == "strict_json":
                assert json.loads(text) == case["expected"], (case["id"], text)
            else:
                assert text == str(case["expected"]), (case["id"], text)
            hit = result["cache_status"][case["id"]]["cached_tokens"]
            assert hit == (7680 if label.endswith("hit") else 0), (case["id"], hit)
        result["semantic_status"] = "PASS"
        harness.save()
    harness.report["status"] = "PASS"
except BaseException as error:
    harness.report["status"] = "FAIL"
    harness.report["error"] = repr(error)
    raise
finally:
    harness.save()
