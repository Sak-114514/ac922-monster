"""Restore the gateway and stop only the newly added inference services."""
import hashlib,json,shutil,subprocess
from pathlib import Path
ROOT=Path('/home/debian/ac922-monster/production')
for unit in ('ac922-inference.service','ac922-inference-safety.service'):
    subprocess.run(['systemctl','disable','--now',unit],check=True,timeout=75)
state=ROOT/'gateway-deployment.json'
if state.exists():
    d=json.loads(state.read_text());gateway=Path('/home/debian/apigateway/gateway.py');backup=Path(d['backup'])
    if hashlib.sha256(gateway.read_bytes()).hexdigest()!=d['gateway_sha256']:
        raise SystemExit('Inference stopped. Gateway has subsequent edits; backup retained for manual reconciliation: '+str(backup))
    shutil.copy2(backup/'gateway.py',gateway)
    module=gateway.with_name('llm_routes.py')
    if d['had_module']:shutil.copy2(backup/'llm_routes.py',module)
    else:module.unlink(missing_ok=True)
    subprocess.run(['systemctl','restart','ac922-apigateway.service'],check=True,timeout=45)
    state.rename(state.with_suffix('.rolled-back.json'))
print('ROLLBACK_DONE: inference stopped/disabled; previous gateway restored when deployed; model and experiment archives retained.')
