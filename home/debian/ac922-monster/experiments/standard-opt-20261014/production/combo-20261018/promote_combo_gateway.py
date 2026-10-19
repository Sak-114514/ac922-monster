#!/usr/bin/env python3
"""Explicit, reversible gateway-route promotion for an already-active combo release.

Default mode validates all release/smoke/hold gates only. --apply changes only
gateway.py and llm_routes.py, then restarts the API gateway. It does not switch
production/current or start/stop/restart inference services.
"""

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
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


ROOT = Path("/home/debian/ac922-monster/production")
GATEWAY = Path("/home/debian/apigateway/gateway.py")
MODULE = GATEWAY.with_name("llm_routes.py")
GATEWAY_UNIT = "ac922-apigateway.service"
INFERENCE_UNITS = ("ac922-inference.service", "ac922-inference-safety.service")
HTTP_SMOKE = ROOT / "run/combo-http-smoke-result.json"
CLEARANCE = ROOT / "combo-promotion-clearance.json"
MARKER = 'install_browser_access(app, load_user_keys, Path(__file__).parent / "access-state", Path(__file__).parent)'
EXPECTED_HTTP_SMOKE_STATUS = "COMBO_HTTP_REAL_MODEL_QUALITY_STREAM_CANCEL_PASS"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_gates(release_arg: Path) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    release = release_arg.resolve(strict=True)
    current = ROOT / "current"
    require(current.is_symlink(), f"{current} is not a release symlink")
    require(current.resolve(strict=True) == release,
            "production/current must already point to the exact smoke-tested combo release")
    require(release.parent == (ROOT / "releases").resolve(), "release is outside production/releases")

    manifest_path = release / "release-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("schema") == "ac922-combo-release-v1", "unexpected combo release manifest schema")
    require(manifest.get("version") == release.name, "manifest version/path mismatch")
    require(manifest.get("promotion_eligible") is True, "release preparation did not pass its combined smoke gate")
    gate = manifest.get("smoke_gate", {})
    require(gate.get("status") == "PASS", "combined worker smoke is not PASS")
    worker = release / "bank-worker"
    worker_sha = sha256(worker)
    require(worker_sha == manifest.get("worker", {}).get("sha256"), "release worker hash differs from manifest")
    smoke_path = Path(gate.get("path", ""))
    require(smoke_path.resolve(strict=True) == (release / "smoke-acceptance.json").resolve(),
            "worker smoke receipt is not the immutable in-release receipt")
    require(sha256(smoke_path) == gate.get("sha256"), "worker smoke receipt hash mismatch")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    require(smoke.get("status") == "COMBO_NATIVE_PREFILL32K_D3_FOUR512_CAPACITY_PASS" and
            smoke.get("worker_exit") == 0, "worker smoke receipt no longer reports a real PASS")
    require(smoke.get("artifacts", {}).get("worker_sha256") == worker_sha,
            "worker smoke was not run against this exact release binary")

    require(HTTP_SMOKE.is_file(), f"missing combined HTTP smoke report: {HTTP_SMOKE}")
    http_report = json.loads(HTTP_SMOKE.read_text(encoding="utf-8"))
    require(http_report.get("status") == EXPECTED_HTTP_SMOKE_STATUS,
            f"HTTP smoke status is not {EXPECTED_HTTP_SMOKE_STATUS}")
    require(Path(http_report.get("release_path", "")).resolve() == release,
            "HTTP smoke report names a different release")
    require(http_report.get("worker_sha256") == worker_sha,
            "HTTP smoke worker hash differs from release worker")
    if "started" in http_report or "finished" in http_report:
        require(isinstance(http_report.get("started"), (int, float)) and
                isinstance(http_report.get("finished"), (int, float)) and
                http_report["finished"] >= http_report["started"], "HTTP smoke timestamps are invalid")

    require(CLEARANCE.is_file(), f"root must explicitly clear PROMOTION-HOLD in {CLEARANCE}")
    stat = CLEARANCE.stat()
    require(stat.st_uid == 0 and (stat.st_mode & 0o077) == 0,
            "promotion clearance must be root-owned and inaccessible to group/other")
    clearance = json.loads(CLEARANCE.read_text(encoding="utf-8"))
    require(clearance.get("status") == "CLEAR" and clearance.get("cleared_by") == "root" and
            bool(clearance.get("hold_id")), "promotion clearance lacks explicit root HOLD release")
    require(clearance.get("release_path") == str(release) and clearance.get("worker_sha256") == worker_sha,
            "promotion clearance is bound to another release/worker")
    require(clearance.get("http_smoke_sha256") == sha256(HTTP_SMOKE),
            "promotion clearance is bound to another HTTP smoke report")
    require(clearance.get("worker_smoke_sha256") == sha256(smoke_path),
            "promotion clearance is bound to another combined worker smoke report")
    return release, manifest, http_report, clearance


def check_services() -> None:
    for unit in INFERENCE_UNITS:
        subprocess.run(["systemctl", "is-active", "--quiet", unit], check=True, timeout=10)


def build_gateway_patch(release: Path) -> tuple[str, bytes]:
    original = GATEWAY.read_bytes()
    text = original.decode("utf-8")
    require(text.count(MARKER) == 1, "gateway auth installation marker missing or ambiguous")
    require("install_model_routes" not in text, "model routes already installed; refusing duplicate patch")
    addition = '\n\nfrom llm_routes import install_model_routes\ninstall_model_routes(app)'
    updated = text.replace(MARKER, MARKER + addition, 1)
    compile(updated, str(GATEWAY), "exec")
    module = release / "llm_routes.py"
    require(module.is_file(), "combo release does not include llm_routes.py")
    return updated, original


def atomic_write(path: Path, payload: bytes, mode: int, uid: int, gid: int) -> None:
    temp = path.with_name(path.name + ".combo-new")
    temp.write_bytes(payload)
    os.chmod(temp, mode)
    os.chown(temp, uid, gid)
    temp.replace(path)


def smoke_gateway_auth_boundary() -> None:
    # Type=simple can report active before the loopback socket is listening.
    deadline = time.monotonic() + 25
    last_error = None
    while time.monotonic() < deadline:
        try:
            try:
                with urlopen("http://127.0.0.1:8080/v1/models", timeout=2) as response:
                    code = response.status
            except HTTPError as error:
                code = error.code
            require(code == 401, f"unauthenticated /v1/models must remain401, got{code}")
            with urlopen("http://127.0.0.1:8080/health", timeout=2) as response:
                require(response.status == 200, f"public health endpoint returned{response.status}")
            return
        except (URLError, ConnectionError, TimeoutError) as error:
            last_error = str(error)
            time.sleep(.25)
    raise RuntimeError("gateway readiness deadline exceeded: " + str(last_error))


def rollback_deployment(state_path: Path) -> None:
    require(os.geteuid() == 0, "--rollback requires root")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    require(sha256(GATEWAY) == state.get("gateway_deployed_sha256"),
            "gateway changed after combo promotion; refusing to overwrite later edits")
    require(MODULE.is_file() and sha256(MODULE) == state.get("module_deployed_sha256"),
            "llm_routes.py changed after combo promotion; refusing to overwrite later edits")
    backup = Path(state["backup"])
    shutil.copy2(backup / "gateway.py", GATEWAY)
    if state.get("had_module"):
        shutil.copy2(backup / "llm_routes.py", MODULE)
    else:
        MODULE.unlink(missing_ok=True)
    subprocess.run(["systemctl", "restart", GATEWAY_UNIT], check=True, timeout=45)
    subprocess.run(["systemctl", "is-active", "--quiet", GATEWAY_UNIT], check=True, timeout=10)
    state_path.rename(state_path.with_suffix(".rolled-back.json"))
    print(json.dumps({"status": "COMBO_GATEWAY_ROLLED_BACK", "backup": str(backup)}))


def promote(release: Path, manifest: dict[str, Any], http_report: dict[str, Any]) -> None:
    require(os.geteuid() == 0, "--apply requires root after explicit PROMOTION-HOLD clearance")
    state_path = ROOT / "combo-gateway-deployment.json"
    require(not state_path.exists(), f"existing deployment state requires rollback/reconciliation: {state_path}")
    check_services()
    updated, _original = build_gateway_patch(release)
    backup = ROOT / "backups" / ("gateway-combo-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    backup.mkdir(parents=True, exist_ok=False)
    shutil.copy2(GATEWAY, backup / "gateway.py")
    had_module = MODULE.exists()
    if had_module:
        shutil.copy2(MODULE, backup / "llm_routes.py")
    debian = pwd.getpwnam("debian")
    mode = GATEWAY.stat().st_mode & 0o777
    try:
        atomic_write(MODULE, (release / "llm_routes.py").read_bytes(), 0o644, debian.pw_uid, debian.pw_gid)
        atomic_write(GATEWAY, updated.encode("utf-8"), mode, debian.pw_uid, debian.pw_gid)
        import_probe = subprocess.run(
            ["/usr/sbin/runuser", "-u", "debian", "--", "/home/debian/comfyui/venv/bin/python", "-c",
             'import gateway; paths={route.path for route in gateway.app.routes}; '
             'assert {"/v1/models","/v1/chat/completions","/v1/completions"} <= paths; '
             'print("COMBO_GATEWAY_IMPORT_PASS")'],
            cwd=str(GATEWAY.parent), capture_output=True, text=True, timeout=30,
        )
        if import_probe.returncode:
            raise RuntimeError("gateway import/route test failed: " + import_probe.stderr[-1000:])
        subprocess.run(["systemctl", "restart", GATEWAY_UNIT], check=True, timeout=45)
        subprocess.run(["systemctl", "is-active", "--quiet", GATEWAY_UNIT], check=True, timeout=10)
        smoke_gateway_auth_boundary()
    except BaseException:
        # Restore only the files this invocation wrote, then restart the old gateway.
        shutil.copy2(backup / "gateway.py", GATEWAY)
        if had_module:
            shutil.copy2(backup / "llm_routes.py", MODULE)
        else:
            MODULE.unlink(missing_ok=True)
        subprocess.run(["systemctl", "restart", GATEWAY_UNIT], check=False, timeout=45)
        raise
    state = {
        "status": "COMBO_GATEWAY_ROUTES_ADDED",
        "release": str(release),
        "worker_sha256": manifest["worker"]["sha256"],
        "http_smoke_sha256": sha256(HTTP_SMOKE),
        "http_smoke_status": http_report["status"],
        "backup": str(backup),
        "gateway_deployed_sha256": sha256(GATEWAY),
        "had_module": had_module,
        "module_deployed_sha256": sha256(MODULE),
        "unix": time.time(),
        "auth_middleware_marker_preserved": True,
    }
    temp_state = state_path.with_name(state_path.name + ".tmp")
    temp_state.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    os.chmod(temp_state, 0o600)
    os.chown(temp_state, 0, 0)
    temp_state.replace(state_path)
    print(json.dumps({"status": state["status"], "backup": str(backup), "release": str(release)}))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--apply", action="store_true", help="apply the already-gated gateway route change")
    parser.add_argument("--rollback", action="store_true", help="restore the gateway backup from a prior combo route promotion")
    args = parser.parse_args()
    if args.rollback:
        if not args.apply:
            raise SystemExit("--rollback requires --apply")
        rollback_deployment(ROOT / "combo-gateway-deployment.json")
        return 0
    if args.release is None:
        raise SystemExit("--release is required")
    release, manifest, http_report, clearance = load_gates(args.release)
    if not args.apply:
        print(json.dumps({"status": "COMBO_GATEWAY_PROMOTION_GATES_PASS_NO_MUTATION",
                          "release": str(release), "worker_sha256": manifest["worker"]["sha256"],
                          "http_smoke_status": http_report["status"],
                          "promotion_hold": clearance["status"], "gateway_mutated": False,
                          "current_symlink_mutated": False}))
        return 0
    promote(release, manifest, http_report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
