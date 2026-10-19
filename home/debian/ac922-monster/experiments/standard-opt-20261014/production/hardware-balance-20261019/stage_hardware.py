#!/usr/bin/env python3
"""Stage the bounded hardware-balance candidate with a 40-minute L5 restore timer."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pwd
import shutil
import subprocess
import sys
import time

PRODUCTION = Path("/home/debian/ac922-monster/production")
HERE = Path("/home/debian/ac922-monster/experiments/standard-opt-20261014/production/hardware-balance-20261019")
L5 = PRODUCTION / "releases/20261019-r444-banked-l5"
OVERRIDE = Path("/etc/systemd/system/ac922-inference.service.d/98-hardware-balance-candidate.conf")
FAULT = Path("/var/lib/ac922-inference/fault.json")
READY = PRODUCTION / "production-ready.json"
UNIT = "ac922-inference.service"


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, check=check)


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def write_text(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.chmod(temporary, 0o644)
    os.replace(temporary, path)


def restore(backup: Path) -> None:
    backup = backup.resolve(strict=True)
    if backup.parent != (PRODUCTION / "backups").resolve(strict=True):
        raise RuntimeError("rollback backup is outside the production backups directory")
    record = json.loads((backup / "rollback.json").read_text(encoding="utf-8"))
    if record.get("release") != str(L5) or record.get("override") != str(OVERRIDE):
        raise RuntimeError("rollback record does not belong to the hardware-balance candidate")
    if (PRODUCTION / "current").resolve() != L5:
        raise RuntimeError("current no longer points to frozen L5; refusing to change service state")

    run("/usr/bin/systemctl", "stop", record["timer"] + ".timer", check=False)
    run("/usr/bin/systemctl", "stop", UNIT, check=False)
    if OVERRIDE.exists():
        content = OVERRIDE.read_text(encoding="utf-8")
        if (f"WorkingDirectory={HERE}" not in content or "Environment=AC922_API_PORT=8014" not in content or
                "Restart=no" not in content):
            raise RuntimeError("candidate override changed after staging; refusing to remove an unknown unit file")
        OVERRIDE.unlink()
    run("/usr/bin/systemctl", "daemon-reload")
    readiness = json.loads((backup / "production-ready.json").read_text(encoding="utf-8"))
    readiness.update(status="RESTORING_L5", live_worker_pid=None, unix=time.time(),
                     readiness_still_requires_verification=True)
    write_json(READY, readiness)
    if not FAULT.exists():
        run("/usr/bin/systemctl", "reset-failed", UNIT, check=False)
        run("/usr/bin/systemctl", "start", UNIT)
    result = {"status": "L5_RESTORE_STARTED" if not FAULT.exists() else "L5_RESTORE_BLOCKED_FAULT_LATCH",
              "fault_latched": FAULT.exists(), "release": str(L5), "unix": time.time(),
              "readiness_still_requires_verification": True}
    write_json(backup / "restored.json", result)
    print(json.dumps(result), flush=True)


def preflight() -> None:
    current = (PRODUCTION / "current").resolve(strict=True)
    if current != L5:
        raise RuntimeError(f"production current is {current}, expected frozen L5 {L5}")
    if FAULT.exists():
        raise RuntimeError("inference fault latch is present")
    if OVERRIDE.exists():
        raise RuntimeError(f"candidate override already exists and requires inspection: {OVERRIDE}")
    physical_available_kib = 0
    for node in (0, 8):
        for line in Path(f"/sys/devices/system/node/node{node}/meminfo").read_text().splitlines():
            if any(label in line for label in ("MemFree:", "Inactive(file):", "SReclaimable:")):
                physical_available_kib += int(line.split()[-2])
    if physical_available_kib < 4608 * 1024:
        raise RuntimeError("physical CPU DDR free plus conservative reclaimable headroom below 4.5 GiB before adding 3 GiB PLE tail and staging")
    if not READY.is_file():
        raise RuntimeError(f"production ready receipt is missing: {READY}")
    if not (HERE / "stage_hardware.py").is_file():
        raise RuntimeError(f"candidate directory is missing: {HERE}")


def verify_as_worker() -> None:
    run("/usr/sbin/runuser", "-u", "debian", "--", "/usr/bin/python3",
        str(HERE / "verify_hardware.py"))


def unit_override() -> str:
    return f"""[Service]
WorkingDirectory={HERE}
Environment=AC922_WORKER_COMMAND_JSON={HERE}/worker-command.json
Environment=AC922_RUN_DIR={HERE}/run/api
Environment=AC922_API_PORT=8014
Environment=AC922_ACCEPTANCE_SOCKET=/home/debian/ac922-monster/production/hardware-control.sock
ExecStartPre=
ExecStartPre=/usr/bin/python3 {HERE}/verify_hardware.py
ExecStartPre=/usr/bin/python3 {L5}/wait_safety.py
ExecStart=
ExecStart=/home/debian/comfyui/venv/bin/python {L5}/candidate_server.py
Restart=no
"""


def ensure_runtime_dirs() -> None:
    user = pwd.getpwnam("debian")
    for directory in (HERE / "run", HERE / "run/model", HERE / "run/api"):
        if directory.is_symlink():
            raise RuntimeError(f"candidate runtime directory must not be a symlink: {directory}")
        if not directory.exists():
            directory.mkdir(mode=0o700, parents=True)
            os.chown(directory, user.pw_uid, user.pw_gid)
        run("/usr/sbin/runuser", "-u", "debian", "--", "/usr/bin/python3", "-c",
            "import os,sys; raise SystemExit(0 if os.access(sys.argv[1], os.W_OK | os.X_OK) else 1)",
            str(directory))


def stage() -> None:
    preflight()
    verify_as_worker()
    preflight()

    tag = time.strftime("hardware-balance-%Y%m%dT%H%M%SZ", time.gmtime())
    backup = PRODUCTION / "backups" / tag
    backup.mkdir(mode=0o700, parents=True)
    shutil.copy2(READY, backup / "production-ready.json")
    unit = run("/usr/bin/systemctl", "cat", UNIT)
    (backup / "inference-unit.txt").write_text(unit.stdout, encoding="utf-8")
    timer = "ac922-" + tag + "-restore"
    write_json(backup / "rollback.json", {
        "timer": timer,
        "release": str(L5),
        "override": str(OVERRIDE),
        "created_unix": time.time(),
    })
    root_script = backup / "stage_hardware.py"
    shutil.copy2(Path(__file__).resolve(), root_script)
    root_script.chmod(0o700)

    run("/usr/bin/systemd-run", "--unit=" + timer, "--on-active=40m",
        "/usr/bin/python3", str(root_script), "--restore", str(backup))
    try:
        ensure_runtime_dirs()
        run("/usr/bin/systemctl", "stop", UNIT)
        ready = json.loads(READY.read_text(encoding="utf-8"))
        ready.update(status="HARDWARE_BALANCE_PRIVATE_STAGED", release=str(L5),
                     candidate=str(HERE), api_port=8014, live_worker_pid=None,
                     rollback_backup=str(backup), rollback_timer=timer, unix=time.time(),
                     readiness_still_requires_verification=True)
        write_json(READY, ready)

        OVERRIDE.parent.mkdir(parents=True, exist_ok=True)
        write_text(OVERRIDE, unit_override())
        run("/usr/bin/systemctl", "daemon-reload")
        run("/usr/bin/systemctl", "reset-failed", UNIT, check=False)
        run("/usr/bin/systemctl", "start", UNIT)
    except BaseException:
        restore(backup)
        raise

    receipt = {"status": "HARDWARE_BALANCE_CANDIDATE_STARTED", "release_unchanged": str(L5),
               "override": str(OVERRIDE), "api_port": 8014,
               "control_socket": "/home/debian/ac922-monster/production/hardware-control.sock",
               "restart": "no", "rollback_backup": str(backup),
               "rollback_timer": timer, "unix": time.time()}
    write_json(HERE / "staging-prepare.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--stage", action="store_true")
    modes.add_argument("--restore", type=Path)
    options = parser.parse_args()
    if os.geteuid() != 0:
        raise SystemExit("stage/restore must run as root")
    if options.stage:
        stage()
    else:
        restore(options.restore)
