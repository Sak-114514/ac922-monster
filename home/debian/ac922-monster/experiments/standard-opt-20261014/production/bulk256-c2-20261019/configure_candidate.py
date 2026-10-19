"""Freeze the one-host-TU candidate with every original L5 DSO and profile."""
from pathlib import Path
import hashlib
import json
HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
L5=Path('/home/debian/ac922-monster/production/releases/20261019-r444-banked-l5')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
report=json.loads((HERE/'build/build-report.json').read_text())
assert report['status']=='BUILD_PASS' and report['cuda_tus']==report['engine_tus']==0
command=json.loads((L5/'lastwave-worker-command.json').read_text())
i=command['argv'].index(str(L5/'build-lastwave/bank-worker'))
command['argv'][i]=str(HERE/'build/bank-worker')
command['argv'][i+2]=str(HERE/'run/model')
command['argv'][i+3]=str(HERE/'verify_bulk_native.py')
command['env'].update(LD_LIBRARY_PATH=str(HERE/'build/lib'),ROOT_DYNAMIC_LIB_DIR=str(HERE/'build/lib'),GGML_CUDA_GRAPH_DEVICE_TAIL_LIBRARY=str(HERE/'build/lib/libac922-tail-bridge.so'),AC922_LASTWAVE_RECORD_OUTPUTS='1',AC922_MULTI_BULK256='1')
(HERE/'worker-command.json').write_text(json.dumps(command,indent=2)+'\n')
gate=(L5/'verify_combo_native.py').read_text()
for key in ('target_n_batch','draft_n_batch'):
    old='"'+key+'": 4096'
    assert gate.count(old)==1
    gate=gate.replace(old,'"'+key+'": 17152')
(HERE/'verify_combo_native.py').write_text(gate)
# Gate regression: only declared geometry changes, no prompt/checker weakening.
import importlib.util
spec=importlib.util.spec_from_file_location('bulk_native_gate',HERE/'verify_combo_native.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
assert 'target_n_batch=4096' in ' '.join(module.check_boot_metadata({'target_n_batch':4096,'draft_n_batch':4096}))
assert not any('n_batch=' in x for x in module.check_boot_metadata({'target_n_batch':17152,'draft_n_batch':17152}))
for d in ['run/model/copy0/fourbank16k-d3','run/api']:
    (HERE/d).mkdir(parents=True,exist_ok=True)
scenario=BASE/'fixtures/fourbank-d3-newruntime-v306-20261017/v221-d3-scenarios.json'
(HERE/'run/model/copy0/fourbank16k-d3/scenarios.json').write_bytes(scenario.read_bytes())
files=['build/build-report.json','worker-command.json','verify_bulk.py','verify_bulk_native.py','verify_combo_native.py','qualify_bulk.py','reference-v306.json','run/model/copy0/fourbank16k-d3/scenarios.json']
(HERE/'pins.json').write_text(json.dumps(dict(files={name:sha(HERE/name) for name in files}),indent=2)+'\n')
print(json.dumps(dict(status='BULK256_CONFIGURATION_FROZEN',pool=524288,mtp=3,engine_dsos_unchanged=7)))
