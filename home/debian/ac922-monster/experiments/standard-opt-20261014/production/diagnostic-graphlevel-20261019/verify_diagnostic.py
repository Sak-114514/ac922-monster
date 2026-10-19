"""Private profile identity only; this build can never qualify for promotion."""
import hashlib
import json
from pathlib import Path

from verify_candidate import verify

HERE = Path(__file__).resolve().parent
verify()
command = json.loads((HERE / 'lastwave-worker-command.json').read_text())
env = command['env']
assert env['AC922_DIAG_PREFILL32_PRIME'] == '1'
assert json.loads((HERE / 'tiny-probe-result.json').read_text())['status'] == 'PASS'
assert env['AC922_DIAG_PREFILL32'] == '1' and env['ROOT_CUPTI_PHASE'] == 'prefill1'
assert env['ROOT_CUPTI_DEFER_FAILURE'] == '0'
assert env['GGML_CUDA_GRAPH_OPT'] == '1'
assert env.get('GGML_CUDA_DISABLE_GRAPHS', '0') == '0'
assert env.get('ROOT_NSYS_PREFILL_PROFILE', '0') == '0'
collector = Path(env['ROOT_CUPTI_COLLECTOR'])
identity = json.loads((HERE / 'collector-identity.json').read_text())
assert str(collector) == identity['path']
assert hashlib.sha256(collector.read_bytes()).hexdigest() == identity['sha256']
assert (HERE / 'run').is_dir()
print('PRIVATE_CURRENT32K_DIAGNOSTIC_IDENTITY_PASS; NOT_A_PRODUCTION_BUILD')
