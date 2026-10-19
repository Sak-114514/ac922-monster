"""Read-only service, mapped-runtime and next-start verification after promotion."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

F = Path(__file__).resolve().parent
P = Path('/home/debian/ac922-monster/production')
release = (P / 'current').resolve()
assert release.name == '20261019-r444-banked-l5'
report = json.loads((F / 'build-lastwave/build-report.json').read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(*argv, **kwargs):
    return subprocess.check_output(argv, text=True, **kwargs).strip()


units = ['ac922-inference.service', 'ac922-inference-safety.service', 'ac922-apigateway.service']
for unit in units:
    assert run('systemctl', 'is-active', unit) == 'active'
    assert run('systemctl', 'is-enabled', unit) == 'enabled'
command = json.loads((release / 'lastwave-worker-command.json').read_text())
for arg in command['argv']:
    if arg.startswith('/'):
        assert Path(arg).exists(), arg
assert command['env']['AC922_LASTWAVE_RECORD_OUTPUTS'] == '0'
assert command['env']['AC922_SINGLE_CACHE_AFFINITY'] == '1'
run('python3', str(release / 'verify_candidate.py'), '--require-acceptance')
ldd = run('ldd', str(release / 'build-lastwave/bank-worker'),
          env=dict(os.environ, LD_LIBRARY_PATH=str(release / 'build-lastwave/lib')))
assert 'not found' not in ldd
for name in report['libraries']:
    assert name+' => '+str(release / 'build-lastwave/lib')+'/' in ldd, name
for folder in ['safe-pool-reject-api', 'safe-pool-reject-model']:
    assert os.access(P / 'run' / folder, os.W_OK), folder
workers = []
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():
        continue
    try:
        exe = (proc / 'exe').resolve(strict=True)
        if exe not in [F/'build-lastwave/bank-worker', release/'build-lastwave/bank-worker']:
            continue
        assert sha(exe) == report['worker_sha256']
        maps = {(line.split()[-1]) for line in (proc/'maps').read_text().splitlines()
                if '/' in line and 'build-lastwave/lib/' in line}
        assert maps and all(path.startswith((str(F/'build-lastwave/lib/'), str(release/'build-lastwave/lib/'))) for path in maps)
        mapped_hashes = {sha(path) for path in maps}
        assert set(report['libraries'].values()) <= mapped_hashes
        assert sha(F/'build-lastwave/lib/libac922-tail-bridge.so') in mapped_hashes
        workers.append({'pid': int(proc.name), 'exe': str(exe), 'worker_sha256': sha(exe), 'private_runtime_map_verified': True})
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
assert len(workers) == 1, workers
identity = {'verified_unix': time.time(), 'processes': workers}
(F/'live-runtime-identity.json').write_text(json.dumps(identity, indent=2)+'\n')
safety = json.loads(Path('/run/ac922-inference/safety.json').read_text())
assert safety['ready'] and safety['healthy'] and time.time()-safety['unix'] < 15
# The latch directory is root-only. Its absence is checked separately by the
# privileged release-record writer; user-side permissions must remain intact.
timer = json.loads((F/'staging.json').read_text())['timer']+'.timer'
assert subprocess.run(['systemctl','is-active','--quiet',timer]).returncode != 0
result = {'status': 'PASS', 'verified_unix': time.time(), 'release': str(release),
    'static_next_start_paths_and_ldd': 'PASS', 'writable_runtime': 'PASS',
    'rollback_timer_inactive': True, 'safety': safety, 'services': {u:'enabled/active' for u in units},
    'fault_latch_check': 'required separately in privileged release record',
    'cold_restart_performed': False}
(F/'post-promotion-static.json').write_text(json.dumps(result, indent=2)+'\n')
print('C5G_LIVE_RUNTIME_AND_NEXT_START_PASS')
