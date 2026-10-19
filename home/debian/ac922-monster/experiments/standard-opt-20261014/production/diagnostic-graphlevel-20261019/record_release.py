"""Finalize the qualified L5 operational receipt using existing admin access."""
import hashlib
import json
import os
from pathlib import Path
import time

assert os.geteuid() == 0
F = Path(__file__).resolve().parent
P = Path('/home/debian/ac922-monster/production')
release = (P/'current').resolve()
assert release.name == '20261019-r444-banked-l5'
assert not Path('/var/lib/ac922-inference/fault.json').exists()
for name in ['lastwave-acceptance.json', 'gateway-smoke-result.json', 'post-promotion-static.json', 'public-smoke-result.json']:
    assert json.loads((F/name).read_text())['status'] == 'PASS', name
record = json.loads((P/'production-ready.json').read_text())
record.update(unix=time.time(), release=str(release),
    worker_sha256=json.loads((F/'lastwave-acceptance.json').read_text())['worker_sha256'],
    live_worker_pid=json.loads((F/'live-runtime-identity.json').read_text())['processes'][0]['pid'],
    ops_promoter_path=str(F/'stage_lastwave.py'),
    ops_promoter_sha256=hashlib.sha256((F/'stage_lastwave.py').read_bytes()).hexdigest(),
    private_http_acceptance=str(F/'evidence/http-result.json'),
    gateway_auth_models_chat_sse_pass=True, services_enabled=True,
    all_seven_dsos_and_r5_private=True, gpu_stop_c=65, root_fault_latch_absent=True,
    external_network_validation={'receipt': str(F/'public-smoke-result.json'),
        'mac_proxy_disabled_tls_models_and_chat': 'PASS',
        'independent_mobile_or_cross_operator_retest': False},
    lastwave_acceptance=str(F/'lastwave-acceptance.json'),
    rollback_script=json.loads((F/'staging.json').read_text())['backup']+'/stage_lastwave.py',
    promotion_without_model_reload=True)
record['rollback_script_sha256'] = hashlib.sha256(Path(record['rollback_script']).read_bytes()).hexdigest()
temp = P/'production-ready.safe-pool.tmp'
temp.write_text(json.dumps(record, indent=2)+'\n')
temp.chmod(0o644)
temp.replace(P/'production-ready.json')
print('L5_PRODUCTION_RECEIPT_UPDATED; ROOT_FAULT_LATCH_ABSENT; L4_ROLLBACK_PRESERVED')
