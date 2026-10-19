"""Bounded candidate window and reversible promotion; run as root.

Only the inference unit and its existing gateway backend are changed. The
running sensor guard and hardware configuration are untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pwd
from pathlib import Path
import shutil
import subprocess
import sys
import time

BASE = Path("/home/debian/ac922-monster/production")
FALLBACK = BASE / "releases/20261018-r444-banked-combo-c12-r2"
CANDIDATE = Path("/home/debian/ac922-monster/experiments/standard-opt-20261014/production/lastwave-20261018")
RELEASE = BASE / "releases/20261018-r444-lastwave-l1"
OVERRIDE = Path("/etc/systemd/system/ac922-inference.service.d/30-lastwave-candidate.conf")
GATEWAY = Path("/home/debian/apigateway/llm_routes.py")
FAULT = Path("/var/lib/ac922-inference/fault.json")


def command(*args: str, check: bool = True):
    return subprocess.run(list(args), check=check, capture_output=True, text=True)


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_current(target: Path):
    temp = BASE / (".current-lastwave-" + str(os.getpid()))
    temp.symlink_to(target)
    temp.replace(BASE / "current")


def unit_override(directory: Path, staged: bool):
    start = "candidate_server.py" if staged else "api.py"
    verifier = "verify_candidate.py" + ("" if staged else " --require-acceptance")
    runtime = CANDIDATE / "run/api" if staged else BASE / "run/lastwave-api"
    return f"""[Service]
WorkingDirectory={directory}
Environment=AC922_WORKER_COMMAND_JSON={directory}/lastwave-worker-command.json
Environment=AC922_RUN_DIR={runtime}
Environment=AC922_API_PORT=8002
ExecStartPre=
ExecStartPre=/usr/bin/python3 {directory}/{verifier}
ExecStartPre=/usr/bin/python3 {FALLBACK}/wait_safety.py
ExecStart=
ExecStart=/home/debian/comfyui/venv/bin/python {directory}/{start}
Restart={'no' if staged else 'on-failure'}
RestartSec=30
"""


def stage():
    if (BASE / "current").resolve() != FALLBACK or OVERRIDE.exists():
        raise RuntimeError("production changed or an existing candidate override needs inspection")
    if FAULT.exists():
        raise RuntimeError("sensor guard fault is latched")
    command("/usr/sbin/runuser", "-u", "debian", "--", "/usr/bin/python3", str(CANDIDATE / "verify_candidate.py"))
    tag = time.strftime("lastwave-%Y%m%dT%H%M%S", time.gmtime())
    backup = BASE / "backups" / tag
    backup.mkdir(mode=0o700)
    shutil.copy2(GATEWAY, backup / "llm_routes.py")
    command("/usr/bin/systemctl", "cat", "ac922-inference.service")
    (backup / "inference-unit.txt").write_text(command("/usr/bin/systemctl", "cat", "ac922-inference.service").stdout)
    timer = "ac922-" + tag + "-rollback"
    record = {"fallback": str(FALLBACK), "release": str(RELEASE), "override": str(OVERRIDE),
              "gateway_sha256": sha(GATEWAY), "timer": timer, "created_unix": time.time()}
    (backup / "rollback.json").write_text(json.dumps(record, indent=2) + "\n")
    root_script = backup / "stage_lastwave.py"
    shutil.copy2(Path(__file__).resolve(), root_script)
    root_script.chmod(0o700)
    command("/usr/bin/systemd-run", "--unit=" + timer, "--on-active=45m",
            "/usr/bin/python3", str(root_script), "--rollback", str(backup))
    try:
        command("/usr/bin/systemctl", "stop", "ac922-inference.service")
        OVERRIDE.parent.mkdir(parents=True, exist_ok=True)
        OVERRIDE.write_text(unit_override(CANDIDATE, True))
        command("/usr/bin/systemctl", "daemon-reload")
        command("/usr/bin/systemctl", "reset-failed", "ac922-inference.service", check=False)
        command("/usr/bin/systemctl", "start", "ac922-inference.service")
    except Exception:
        rollback(backup)
        raise
    (CANDIDATE / "staging.json").write_text(json.dumps({"backup": str(backup), **record}, indent=2) + "\n")
    print(json.dumps({"status": "PRIVATE_CANDIDATE_STARTED", "port": 8002,
                      "backup": str(backup), "rollback_timer": timer}), flush=True)


def rollback(backup: Path):
    record = json.loads((backup / "rollback.json").read_text())
    if (backup / "promoted.json").exists() and "--force" not in sys.argv:
        print("ROLLBACK_TIMER_DISARMED_AFTER_PROMOTION", flush=True)
        return
    current = (BASE / "current").resolve()
    if current not in {Path(record["fallback"]), Path(record["release"])}:
        raise RuntimeError("another release owns current; refusing unrelated rollback")
    command("/usr/bin/systemctl", "stop", "ac922-inference.service", check=False)
    replace_current(Path(record["fallback"]))
    OVERRIDE.unlink(missing_ok=True)
    shutil.copy2(backup / "llm_routes.py", GATEWAY)
    command("/usr/bin/systemctl", "daemon-reload")
    command("/usr/bin/systemctl", "restart", "ac922-apigateway.service")
    if not FAULT.exists():
        command("/usr/bin/systemctl", "reset-failed", "ac922-inference.service", check=False)
        command("/usr/bin/systemctl", "start", "ac922-inference.service")
    command("/usr/bin/systemctl", "stop", record["timer"] + ".timer", check=False)
    (backup / "rolled-back.json").write_text(json.dumps({"unix": time.time(), "fault_latched": FAULT.exists()}) + "\n")
    print(json.dumps({"status": "C12_RESTORED", "fault_latched": FAULT.exists()}), flush=True)


def promote(backup: Path):
    record = json.loads((backup / "rollback.json").read_text())
    if FAULT.exists() or (BASE / "current").resolve() != FALLBACK:
        raise RuntimeError("fault or external release change prevents promotion")
    command("/usr/sbin/runuser", "-u", "debian", "--", "/usr/bin/python3",
            str(CANDIDATE / "verify_candidate.py"), "--require-acceptance")
    if sha(GATEWAY) != record["gateway_sha256"]:
        raise RuntimeError("gateway changed during private acceptance")
    if RELEASE.exists():
        raise RuntimeError("release already exists; never overwrite a frozen release")
    shutil.copytree(CANDIDATE, RELEASE, symlinks=True,
                    ignore=shutil.ignore_patterns("run", "attempts", "base-objects", "__pycache__", "*.o", "*.compile.log", "*.link.log"))
    # Frozen code stays root-owned; model outputs and bootstrap scenarios need
    # a separate writable directory on the next service start.
    user = pwd.getpwnam("debian")
    model_runtime = BASE / "run/lastwave-model"
    api_runtime = BASE / "run/lastwave-api"
    scenarios = model_runtime / "copy0/fourbank16k-d3/scenarios.json"
    scenarios.parent.mkdir(parents=True, exist_ok=True)
    api_runtime.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CANDIDATE / "run/model/copy0/fourbank16k-d3/scenarios.json", scenarios)
    for root in [model_runtime, api_runtime]:
        os.chown(root, user.pw_uid, user.pw_gid)
        for path in root.rglob("*"):
            os.chown(path, user.pw_uid, user.pw_gid)
    worker_command = json.loads((RELEASE / "lastwave-worker-command.json").read_text())
    worker_command["argv"] = [str(model_runtime) if arg == str(CANDIDATE / "run/model") else
                              arg.replace(str(CANDIDATE), str(RELEASE)) for arg in worker_command["argv"]]
    worker_command["env"] = {key: value.replace(str(CANDIDATE), str(RELEASE))
                             for key, value in worker_command["env"].items()}
    # Production responses keep cache accounting; full generated texts were
    # recorded for private acceptance only.
    worker_command["env"]["AC922_LASTWAVE_RECORD_OUTPUTS"] = "0"
    (RELEASE / "lastwave-worker-command.json").write_text(json.dumps(worker_command, indent=2) + "\n")
    (RELEASE / "worker-command.json").write_text(json.dumps(worker_command, indent=2) + "\n")
    (RELEASE / "verify_release.py").write_text(
        '"""Qualified last-wave release gate."""\nfrom verify_candidate import verify\nverify(require_acceptance=True)\n')
    manifest = json.loads((RELEASE / "candidate-manifest.json").read_text())
    manifest["sha256"]["lastwave-worker-command.json"] = sha(RELEASE / "lastwave-worker-command.json")
    manifest["sha256"]["worker-command.json"] = sha(RELEASE / "worker-command.json")
    manifest["sha256"]["verify_release.py"] = sha(RELEASE / "verify_release.py")
    manifest["status"] = "QUALIFIED_RELEASE"
    release_manifest = {"schema": "ac922-lastwave-release-v1", "version": RELEASE.name,
                        "baseline": str(FALLBACK), "status": "QUALIFIED", "promotion_eligible": True,
                        "worker_sha256": json.loads((RELEASE / "build-lastwave/build-report.json").read_text())["worker_sha256"],
                        "pool_tokens": 524288, "max_context": 262140, "mtp_depth": 3,
                        "acceptance_sha256": sha(RELEASE / "lastwave-acceptance.json"),
                        "model_runtime": str(model_runtime), "api_runtime": str(api_runtime),
                        "backend_port": 8002}
    (RELEASE / "release-manifest.json").write_text(json.dumps(release_manifest, indent=2)+"\n")
    manifest["sha256"]["release-manifest.json"] = sha(RELEASE / "release-manifest.json")
    (RELEASE / "candidate-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    command("/usr/bin/python3", str(RELEASE / "verify_candidate.py"), "--require-acceptance")
    # Restore baseline configuration automatically if the final gateway switch fails.
    try:
        original = GATEWAY.read_text()
        needle = "BACKEND='http://127.0.0.1:8001'"
        if original.count(needle) != 1:
            raise RuntimeError("gateway backend declaration no longer matches the frozen contract")
        GATEWAY.write_text(original.replace(needle, "BACKEND='http://127.0.0.1:8002'"))
        OVERRIDE.write_text(unit_override(RELEASE, False))
        replace_current(RELEASE)
        command("/usr/bin/systemctl", "daemon-reload")
        command("/usr/bin/systemctl", "restart", "ac922-apigateway.service")
    except Exception:
        rollback(backup)
        raise
    promoted = {"unix": time.time(), "release": str(RELEASE), "worker_reloaded": False,
                "running_candidate_identical_to_frozen_release": True, "backend_port": 8002,
                "rollback": f"sudo python3 {backup}/stage_lastwave.py --rollback {backup} --force"}
    (backup / "promoted.json").write_text(json.dumps(promoted, indent=2) + "\n")
    command("/usr/bin/systemctl", "stop", record["timer"] + ".timer", check=False)
    (RELEASE / "promotion.json").write_text(json.dumps(promoted, indent=2) + "\n")
    print(json.dumps({"status": "PROMOTED_WITHOUT_MODEL_RELOAD", **promoted}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--stage", action="store_true")
    group.add_argument("--rollback", type=Path)
    group.add_argument("--promote", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if os.geteuid() != 0:
        raise SystemExit("run with existing administrator authorization")
    if args.stage:
        stage()
    elif args.rollback:
        rollback(args.rollback)
    else:
        promote(args.promote)
