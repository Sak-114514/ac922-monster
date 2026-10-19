"""Update the deployment receipt and keep that receipt reversible with C12."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import time

F = Path(__file__).resolve().parent
P = Path('/home/debian/ac922-monster/production')
BACKUP = P / 'backups/lastwave-20261018T152800'
RELEASE = P / 'releases/20261018-r444-lastwave-l1'

assert (P / 'current').resolve() == RELEASE
acceptance = json.loads((F / 'lastwave-acceptance.json').read_text())
assert acceptance['status'] == 'PASS'
assert json.loads((F / 'gateway-smoke-result.json').read_text())['status'] == 'PASS'
assert json.loads((F / 'post-promotion-static.json').read_text())['status'] == 'PASS'
ready = P / 'production-ready.json'
ready_backup = BACKUP / 'production-ready.json'
if not ready_backup.exists():
    shutil.copy2(ready, ready_backup)
rollback = BACKUP / 'stage_lastwave.py'
original = rollback.read_text()
needle = '    shutil.copy2(backup / "llm_routes.py", GATEWAY)\n'
extra = ('    if (backup / "production-ready.json").exists():\n'
         '        shutil.copy2(backup / "production-ready.json", BASE / "production-ready.json")\n')
if extra not in original:
    assert original.count(needle) == 1
    updated = original.replace(needle, needle + extra)
    ast.parse(updated)
    shutil.copy2(rollback, BACKUP / 'stage_lastwave.before-ready.py')
    rollback.write_text(updated)
    rollback.chmod(0o700)
record = json.loads(ready.read_text())
record.update(
    unix=time.time(), release=str(RELEASE),
    worker_sha256=acceptance['worker_sha256'],
    live_worker_pid=json.loads((F / 'live-runtime-identity.json').read_text())['processes'][0]['pid'],
    ops_promoter_path=str(F / 'stage_lastwave.py'),
    ops_promoter_sha256=hashlib.sha256((F / 'stage_lastwave.py').read_bytes()).hexdigest(),
    private_http_acceptance=str(F / 'evidence/http-result.json'),
    gateway_auth_models_chat_sse_pass=True, services_enabled=True,
    all_seven_dsos_and_r5_private=True, gpu_stop_c=65,
    external_network_validation={'receipt':str(F / 'public-smoke-result.json'),
        'mac_proxy_disabled_tls_models_and_chat':'PASS',
        'independent_mobile_or_cross_operator_retest':False},
    lastwave_acceptance=str(F / 'lastwave-acceptance.json'),
    rollback_script=str(rollback),
    rollback_script_sha256=hashlib.sha256(rollback.read_bytes()).hexdigest(),
    promotion_without_model_reload=True,
)
temp = ready.with_suffix('.lastwave.tmp')
temp.write_text(json.dumps(record, indent=2) + '\n')
temp.chmod(0o644)
temp.replace(ready)
print('PRODUCTION_RECEIPT_UPDATED; C12_RECEIPT_BACKED_UP; ROLLBACK_RESTORES_RECEIPT')
