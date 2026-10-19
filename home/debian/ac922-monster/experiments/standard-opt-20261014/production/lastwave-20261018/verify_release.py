"""Start gate for a staged combo release; never promotes a release."""
import hashlib, json, subprocess, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
manifest=json.loads((HERE/"release-manifest.json").read_text())
gate=manifest.get("smoke_gate", {})
if not manifest.get("promotion_eligible") or gate.get("status") != "PASS":
    raise SystemExit("COMBO_SMOKE_GATE_PENDING: current symlink must not point here")
smoke=Path(gate["path"])
if hashlib.sha256(smoke.read_bytes()).hexdigest() != gate.get("sha256"):
    raise SystemExit("combined smoke receipt changed after preparation")
data=json.loads(smoke.read_text())
if data.get("status") != "COMBO_NATIVE_PREFILL32K_D3_FOUR512_CAPACITY_PASS" or data.get("worker_exit") != 0:
    raise SystemExit("combined smoke receipt is not a real PASS")
if data.get("artifacts", {}).get("worker_sha256") != manifest.get("worker", {}).get("sha256"):
    raise SystemExit("combined smoke worker hash does not match this release")
subprocess.run([sys.executable, str(HERE/"verify_release_base.py")], check=True)
