"""Attest the live diagnostic process without reading credentials or model bytes."""
import hashlib
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
report = json.loads((HERE / 'run/diagnostic-receipt.json').read_text())
pid = report['worker_pid']
process = Path('/proc') / str(pid)
exe = (process / 'exe').resolve(strict=True)
assert exe == HERE / 'build/bank-worker'
build = json.loads((HERE / 'build/build-report.json').read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(exe) == build['worker_sha256']
maps = set()
for line in (process / 'maps').read_text().splitlines():
    fields = line.split(maxsplit=5)
    if len(fields) == 6 and fields[5].startswith('/'):
        maps.add(Path(fields[5]))
libraries = {}
for name, expected in build['libraries'].items():
    mapped = [path for path in maps if path.name == name]
    assert len(mapped) == 1 and mapped[0] == HERE / 'build/lib' / name, (name, mapped)
    assert sha(mapped[0]) == expected
    libraries[name] = {'path': str(mapped[0]), 'sha256': expected}
helper = HERE.parent / 'captured-argument-observer-20261019/libac922-tail-observer.so'
assert helper in maps and sha(helper) == report['observer_sha256']
result = {'status': 'PASS', 'pid': pid, 'exe': str(exe), 'worker_sha256': sha(exe),
          'all_seven_exact_L5_dsos': libraries, 'observer_loaded': str(helper),
          'observer_sha256': sha(helper), 'unix': time.time(),
          'model_loads_requested': 1, 'hardware_changed': False}
(HERE / 'run/live-identity.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'status': result['status'], 'pid': pid, 'libraries': len(libraries), 'observer_loaded': True}))
