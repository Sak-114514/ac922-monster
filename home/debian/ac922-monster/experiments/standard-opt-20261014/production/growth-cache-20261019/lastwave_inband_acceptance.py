"""Exact stored C12 workloads through the already-loaded candidate's Unix socket."""
from __future__ import annotations

import json
from pathlib import Path
import socket
import subprocess
import time
import uuid

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
OUT = HERE / "run/inband-result.json"
SOCKET = Path("/home/debian/ac922-monster/production/growth-cache-control.sock")
CASES = BASE / "fixtures/fourbank-d3-newruntime-v306-20261017/v221-d3-scenarios.json"
report = {"status": "RUNNING", "started_unix": time.time(), "waves": [],
          "scope": "C12 exact token slices/profiles; no prefix hits in raw performance cases",
          "model_launch_count_requested": 1}


def save():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temp = OUT.with_suffix(".tmp")
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    temp.replace(OUT)


def wave(label, requests, cache_prompt=False):
    epoch = label + "-" + uuid.uuid4().hex[:10]
    requests = [{**request, "cache_prompt": cache_prompt} for request in requests]
    began = time.monotonic()
    with socket.socket(socket.AF_UNIX) as connection:
        connection.settimeout(1510)
        connection.connect(str(SOCKET))
        connection.sendall(json.dumps({"id": epoch, "requests": requests}).encode()+b"\n")
        wire = connection.makefile("rb").readline(8 << 20)
    result = json.loads(wire)
    if result.get("status") == "ERROR":
        raise RuntimeError(result)
    result["label"] = label
    result["control_roundtrip_s"] = time.monotonic()-began
    report["waves"].append(result)
    save()
    print(json.dumps({"label": label, "wall_s": result.get("wall_s"),
                      "timings": result.get("timings"), "errors": result.get("errors")}), flush=True)
    assert not result["errors"], result["errors"]
    assert len(result["done"]) == len(requests), "unfinished accepted request"
    if not cache_prompt:
        assert result["timings"]["cached_tokens"] == 0, "raw workload unexpectedly reused prefix"
    return result


def cool():
    began = time.monotonic()
    while time.monotonic()-began < 120:
        value = subprocess.check_output(["nvidia-smi", "--query-gpu=temperature.gpu",
                                         "--format=csv,noheader,nounits"], text=True)
        if max(map(int, value.split())) <= 54:
            return
        time.sleep(1)
    raise TimeoutError("bounded inter-case cooldown")


def main():
    native = json.loads(CASES.read_text())[0]["requests"]

    def native_wave(label, pad=0):
        requests = [{"id": case["id"], "prompt": case["prompt"], "max_tokens": case["max_tokens"],
                     **({"semantic_pad_to_tokens": pad} if pad else {})} for case in native]
        result = wave(label, requests)
        for case in native:
            text = result["texts"][case["id"]].strip()
            if case["checker"] == "strict_json":
                assert json.loads(text) == case["expected"], (case["id"], text)
            else:
                assert text == str(case["expected"]), (case["id"], text)
        result["semantic_status"] = "PASS"
        save()

    topics = ["how CPU caches and cache coherency work", "photosynthesis and plant metabolism",
              "Roman architectural engineering", "comparison sorting algorithms and complexity"]
    stories = [{"id": f"story-{i}", "prompt": f"Write a coherent technical explanation of {topic}, at least 2000 words. Begin with exactly BEGIN_{i} then continue with the explanation. Do not end early.",
                "max_tokens": 512} for i,topic in enumerate(topics)]
    capacities = [240128,129792,88064,66304]
    allocation = [{**request, "id": "cap-"+request["id"], "max_tokens": 64,
                   "capacity_tokens": capacities[i]-4} for i,request in enumerate(stories)]
    result = wave("lastwave-full524288-allocation", allocation)
    actual = [bank["target_context"] for bank in result["timings"]["banks"]]
    assert actual == capacities and sum(actual) == 524288, actual
    for i,request in enumerate(allocation):
        assert result["texts"][request["id"]].startswith(f"BEGIN_{i}")
    native_wave("lastwave-native4-after-fullpool")
    cool()
    paris = {"id": "paris", "prompt": native[0]["prompt"], "max_tokens": 24}
    result = wave("lastwave-64k-max262k-UB512", [{**paris, "semantic_pad_to_tokens": 65536,
                                                  "capacity_tokens": 262140}])
    assert result["texts"]["paris"].strip() == "Paris", result["texts"]
    assert result["timings"]["banks"][0]["target_context"] == 262144
    result["semantic_status"] = "PASS"
    save()
    cool()
    formal = {"id": "corpus32k", "corpus_token_start": 0, "corpus_token_count": 32768,
              "max_tokens": 8, "ignore_eos": True}
    for phase in ["first", "repeat"]:
        result = wave("lastwave-32k-"+phase, [formal])
        assert result["timings"]["prefill_input_tokens"] == 32768
        assert result["done"]["corpus32k"]["output_tokens"] == 8
        cool()
    result = wave("lastwave-8k-single-semantic", [{**paris, "semantic_pad_to_tokens": 8192}])
    assert result["texts"]["paris"].strip() == "Paris"
    native_wave("lastwave-native4-after-prefill")
    native_wave("lastwave-native4-8k", 8192)
    asymmetric = [{"id": case["id"], "prompt": case["prompt"], "max_tokens": case["max_tokens"],
                   **({"semantic_pad_to_tokens": count} if count else {})}
                  for case,count in zip(native,[32768,8192,1024,0])]
    result = wave("lastwave-asymmetric-prefill-32k-8k-1k-short", asymmetric)
    for case in native:
        text = result["texts"][case["id"]].strip()
        if case["checker"] == "strict_json":
            assert json.loads(text) == case["expected"], (case["id"], text)
        else:
            assert text == str(case["expected"]), (case["id"], text)
    short_first = [result["first_token_s"][case["id"]] for case in native[1:]]
    result["short_first_token_before_long"] = max(short_first) < result["first_token_s"][native[0]["id"]]
    assert result["short_first_token_before_long"], result["first_token_s"]
    result["semantic_status"] = "PASS"
    save()
    result = wave("lastwave-four512", stories)
    for i,request in enumerate(stories):
        assert result["texts"][request["id"]].startswith(f"BEGIN_{i}")
        assert result["done"][request["id"]]["output_tokens"] == 512
    result["semantic_status"] = "MARKER_PASS_FULL_TEXT_REVIEW_REQUIRED"
    native_wave("lastwave-native4-final")
    report["status"] = "INBAND_PASS_PENDING_FULL_TEXT_REVIEW"


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        report["status"] = "FAIL"
        report["error"] = repr(error)
        raise
    finally:
        report["finished_unix"] = time.time()
        save()
