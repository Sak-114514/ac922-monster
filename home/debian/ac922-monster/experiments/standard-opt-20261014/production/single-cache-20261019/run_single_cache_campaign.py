#!/usr/bin/env python3
"""Run the single-cache qualification on one already loaded private worker."""
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
RUN = HERE / "run"
RUN.mkdir(exist_ok=True)
record = {"status": "WAITING_FOR_READY", "started_unix": time.time(), "stages": []}


def save():
    (RUN / "campaign-status.json").write_text(json.dumps(record, indent=2) + "\n")


save()
try:
    deadline = time.monotonic() + 600
    while True:
        try:
            with urlopen("http://127.0.0.1:8004/ready", timeout=3) as response:
                ready = json.load(response)
            if ready.get("status") == "ready":
                record["ready"] = ready
                break
        except Exception:
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError("private candidate did not become ready in 600 seconds")
        time.sleep(2)
    env = dict(os.environ, AC922_BASE_URL="http://127.0.0.1:8004/v1",
               AC922_READY_URL="http://127.0.0.1:8004/ready",
               AC922_API_KEY_FILE="/home/debian/apigateway/api_key",
               AC922_RESULT_JSON=str(RUN / "http-result.json"),
               AC922_MODEL_RUN_ROOT=str(RUN / "model"),
               NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost")
    stages = [
        ("affinity", "/usr/bin/python3", "affinity_acceptance.py", ["--run"]),
        ("single-cache", "/usr/bin/python3", "single_cache_acceptance.py", []),
        ("http", "/home/debian/comfyui/venv/bin/python", "lastwave_http_acceptance.py", []),
        ("cancel-cached", "/home/debian/comfyui/venv/bin/python", "cancel_cached_single.py", ["--run"]),
    ]
    record["status"] = "RUNNING"
    for label, python, script, args in stages:
        stage = {"label": label, "started_unix": time.time(), "status": "RUNNING"}
        record["stages"].append(stage)
        save()
        with (RUN / f"{label}-controller.log").open("w") as output:
            result = subprocess.run([python, str(HERE / script), *args], cwd=HERE,
                                    env=env, stdout=output, stderr=subprocess.STDOUT,
                                    timeout=1200)
        stage.update(returncode=result.returncode, finished_unix=time.time(),
                     status="PASS" if result.returncode == 0 else "FAIL")
        save()
        if result.returncode:
            raise RuntimeError(f"{label} qualification failed; inspect its controller log")
    record["status"] = "PASS_PENDING_FULL_TEXT_REVIEW"
except BaseException as error:
    record["status"] = "FAIL"
    record["error"] = repr(error)
    raise
finally:
    record["finished_unix"] = time.time()
    save()
