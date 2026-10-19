#!/usr/bin/env python3
"""Stage the private one-window event diagnostic with a 40-minute L5 restore timer."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pwd
import shutil
import subprocess
import time
from urllib.request import urlopen

ROOT = Path("/home/debian/ac922-monster")
PROJECT = ROOT / "experiments/standard-opt-20261014"
PRODUCTION = ROOT / "production"
HERE = PROJECT / "production/stage-event-diagnostic-20261019"
L5 = PRODUCTION / "releases/20261019-r444-banked-l5"
SOURCE_CANDIDATE = PROJECT / "production/safe-pool-reject-20261019"
SCENARIOS = PROJECT / "fixtures/fourbank-d3-newruntime-v306-20261017/v221-d3-scenarios.json"
OVERRIDE = Path("/etc/systemd/system/ac922-inference.service.d/96-stage-event-diagnostic.conf")
UNIT = "ac922-inference.service"
GATEWAY_UNIT = "ac922-apigateway.service"
PRODUCTION_BACKEND_PORT = 8007
PRIVATE_API_PORT = 8012
CONTROL_SOCKET = PRODUCTION / "stage-event-control.sock"
READY = PRODUCTION / "production-ready.json"
FAULT = Path("/var/lib/ac922-inference/fault.json")


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, check=check)


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gateway_check() -> None:
    active = run("/usr/bin/systemctl", "is-active", GATEWAY_UNIT, check=False)
    if active.returncode != 0 or active.stdout.strip() != "active":
        raise RuntimeError("existing public API gateway service is not active")


def ensure_runtime_dirs() -> None:
    user = pwd.getpwnam("debian")
    destination = HERE / "run/model/copy0/fourbank16k-d3/scenarios.json"
    if destination.is_symlink():
        raise RuntimeError(f"diagnostic scenario path must not be a symlink: {destination}")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if destination.exists():
        if sha(destination) != sha(SCENARIOS):
            raise RuntimeError("existing diagnostic scenario JSON differs from the frozen D3 fixture")
    else:
        shutil.copy2(SCENARIOS, destination)
    for directory in (HERE / "run", HERE / "run/model", HERE / "run/api", HERE / "run/waves", HERE / "run/logs"):
        if directory.is_symlink():
            raise RuntimeError(f"diagnostic runtime directory must not be a symlink: {directory}")
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chown(directory, user.pw_uid, user.pw_gid)
        for child in directory.rglob("*"):
            os.chown(child, user.pw_uid, user.pw_gid)
    os.chown(destination, user.pw_uid, user.pw_gid)
    if (HERE / "run/api/worker.log").exists():
        raise RuntimeError("worker.log already exists; refusing to mix diagnostic windows")
    if list((HERE / "run/model").rglob("lastwave-*.json")):
        raise RuntimeError("old native flow records exist; refusing to mix diagnostic windows")
    run("/usr/sbin/runuser", "-u", "debian", "--", "/usr/bin/python3", "-c",
        "import os,sys; raise SystemExit(0 if os.access(sys.argv[1], os.W_OK|os.X_OK) else 1)",
        str(HERE / "run/api"))


def override_text() -> str:
    return f"""[Service]
WorkingDirectory={HERE}
Environment=AC922_WORKER_COMMAND_JSON={HERE}/lastwave-worker-command.json
Environment=AC922_RUN_DIR={HERE}/run/api
Environment=AC922_API_PORT={PRIVATE_API_PORT}
Environment=AC922_ACCEPTANCE_SOCKET={CONTROL_SOCKET}
ExecStartPre=
ExecStartPre=/usr/bin/python3 {HERE}/verify_event_candidate.py --runtime-required
ExecStartPre=/usr/bin/python3 {L5}/wait_safety.py
ExecStart=
ExecStart=/home/debian/comfyui/venv/bin/python {L5}/candidate_server.py
Restart=no
"""


def restore(backup: Path) -> None:
    backup = backup.resolve(strict=True)
    if backup.parent != (PRODUCTION / "backups").resolve(strict=True):
        raise RuntimeError("restore receipt is outside the production backups directory")
    record = json.loads((backup / "rollback.json").read_text(encoding="utf-8"))
    if record.get("release") != str(L5) or record.get("override") != str(OVERRIDE):
        raise RuntimeError("rollback receipt does not belong to this event diagnostic")
    if (PRODUCTION / "current").resolve(strict=True) != L5:
        raise RuntimeError("production current changed from frozen L5; refusing to restore over another release")
    run("/usr/bin/systemctl", "stop", record["timer"] + ".timer", check=False)
    run("/usr/bin/systemctl", "stop", UNIT, check=False)
    if OVERRIDE.exists():
        content = OVERRIDE.read_text(encoding="utf-8")
        if (f"AC922_API_PORT={PRIVATE_API_PORT}" not in content or
                f"AC922_WORKER_COMMAND_JSON={HERE}/lastwave-worker-command.json" not in content or
                "Restart=no" not in content):
            raise RuntimeError("96-stage-event override changed; refusing to remove an unknown unit file")
        OVERRIDE.unlink()
    run("/usr/bin/systemctl", "daemon-reload")
    ready = json.loads((backup / "production-ready.json").read_text(encoding="utf-8"))
    ready.update(status="RESTORING_L5_AFTER_STAGE_EVENT_DIAGNOSTIC", release=str(L5),
                 live_worker_pid=None, unix=time.time(), readiness_still_requires_verification=True)
    write_json(READY, ready)
    if not FAULT.exists():
        run("/usr/bin/systemctl", "reset-failed", UNIT, check=False)
        run("/usr/bin/systemctl", "start", UNIT)
    gateway_check()
    result = {"status": "L5_RESTORE_STARTED" if not FAULT.exists() else "L5_RESTORE_BLOCKED_FAULT_LATCH",
              "fault_latched": FAULT.exists(), "release": str(L5), "api_port": PRIVATE_API_PORT,
              "inference_backend_port": 8007, "public_gateway_unit_active": True, "unix": time.time(),
              "readiness_still_requires_verification": True}
    write_json(backup / "restored.json", result)
    print(json.dumps(result), flush=True)


def preflight() -> None:
    if (PRODUCTION / "current").resolve(strict=True) != L5:
        raise RuntimeError("production current is not the frozen L5 release")
    if FAULT.exists():
        raise RuntimeError("inference fault latch is present")
    if OVERRIDE.exists():
        raise RuntimeError(f"stage-event override already exists: {OVERRIDE}")
    if CONTROL_SOCKET.exists():
        raise RuntimeError(f"new diagnostic control socket already exists: {CONTROL_SOCKET}")
    if not READY.is_file() or not (HERE / "l5-frozen-worker-command.json").is_file():
        raise RuntimeError("frozen L5 readiness or imported worker-command baseline is missing")
    if not (HERE / "stage_event.py").is_file():
        raise RuntimeError(f"stage-event packet missing: {HERE}")
    gateway_check()


def verify_as_worker(runtime_required: bool = False) -> None:
    args = ["/usr/sbin/runuser", "-u", "debian", "--", "/usr/bin/python3",
            str(HERE / "verify_event_candidate.py")]
    if runtime_required:
        args.append("--runtime-required")
    run(*args)


def stage() -> None:
    preflight()
    ensure_runtime_dirs()
    verify_as_worker(runtime_required=True)
    preflight()

    tag = time.strftime("stage-event-%Y%m%dT%H%M%SZ", time.gmtime())
    backup = PRODUCTION / "backups" / tag
    backup.mkdir(mode=0o700, parents=True)
    shutil.copy2(READY, backup / "production-ready.json")
    unit = run("/usr/bin/systemctl", "cat", UNIT)
    (backup / "inference-unit.txt").write_text(unit.stdout, encoding="utf-8")
    timer = "ac922-" + tag + "-restore"
    write_json(backup / "rollback.json", {
        "timer": timer, "release": str(L5), "override": str(OVERRIDE),
        "gateway_unit": GATEWAY_UNIT, "production_backend_port": PRODUCTION_BACKEND_PORT,
        "private_api_port": PRIVATE_API_PORT, "created_unix": time.time(),
    })
    root_script = backup / "stage_event.py"
    shutil.copy2(Path(__file__).resolve(), root_script)
    root_script.chmod(0o700)
    run("/usr/bin/systemd-run", "--unit=" + timer, "--on-active=40m",
        "/usr/bin/python3", str(root_script), "--restore", str(backup))
    try:
        readiness = json.loads(READY.read_text(encoding="utf-8"))
        readiness.update(status="STAGE_EVENT_DIAGNOSTIC_PRIVATE_WINDOW", release=str(L5),
                         candidate=str(HERE), api_port=PRIVATE_API_PORT,
                         control_socket=str(CONTROL_SOCKET), rollback_backup=str(backup),
                         rollback_timer=timer, inference_backend_port_during_window=PRODUCTION_BACKEND_PORT,
                         public_gateway_unit_active_unmodified=GATEWAY_UNIT, unix=time.time())
        write_json(READY, readiness)
        run("/usr/bin/systemctl", "stop", UNIT)
        OVERRIDE.parent.mkdir(parents=True, exist_ok=True)
        OVERRIDE.write_text(override_text(), encoding="utf-8")
        os.chmod(OVERRIDE, 0o644)
        run("/usr/bin/systemctl", "daemon-reload")
        run("/usr/bin/systemctl", "reset-failed", UNIT, check=False)
        run("/usr/bin/systemctl", "start", UNIT)
        gateway_check()
        if run("/usr/bin/systemctl", "is-active", UNIT, check=False).stdout.strip() != "active":
            raise RuntimeError("private diagnostic inference unit did not become active")
    except BaseException:
        restore(backup)
        raise
    receipt = {
        "status": "STAGE_EVENT_DIAGNOSTIC_STARTED_PRIVATE_API_NOT_PROMOTED",
        "release_unchanged": str(L5), "override": str(OVERRIDE),
        "private_api_port": PRIVATE_API_PORT, "inference_backend_port_during_window": PRODUCTION_BACKEND_PORT,
        "public_gateway_unit_active_unmodified": GATEWAY_UNIT,
        "control_socket": str(CONTROL_SOCKET), "restart": "no",
        "rollback_backup": str(backup), "rollback_timer": timer, "unix": time.time(),
    }
    write_json(HERE / "staging-event.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--stage", action="store_true")
    modes.add_argument("--restore", type=Path)
    options = parser.parse_args()
    if os.geteuid() != 0:
        raise SystemExit("stage/restore must run as root")
    stage() if options.stage else restore(options.restore)
