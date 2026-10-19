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
command['argv'][i+3]=str(HERE/'verify_multi_native.py')
command['env'].update(LD_LIBRARY_PATH=str(HERE/'build/lib'),ROOT_DYNAMIC_LIB_DIR=str(HERE/'build/lib'),GGML_CUDA_GRAPH_DEVICE_TAIL_LIBRARY=str(HERE/'build/lib/libac922-tail-bridge.so'),AC922_LASTWAVE_RECORD_OUTPUTS='1',AC922_MULTI_PREFILL512='1')
(HERE/'worker-command.json').write_text(json.dumps(command,indent=2)+'\n')
(HERE/'verify_combo_native.py').write_bytes((L5/'verify_combo_native.py').read_bytes())
for d in ['run/model/copy0/fourbank16k-d3','run/api']:
    (HERE/d).mkdir(parents=True,exist_ok=True)
scenario=BASE/'fixtures/fourbank-d3-newruntime-v306-20261017/v221-d3-scenarios.json'
(HERE/'run/model/copy0/fourbank16k-d3/scenarios.json').write_bytes(scenario.read_bytes())
files=['build/build-report.json','worker-command.json','verify_multi.py','verify_multi_native.py','verify_combo_native.py','qualify_multi.py','reference-v306.json','run/model/copy0/fourbank16k-d3/scenarios.json']
(HERE/'pins.json').write_text(json.dumps(dict(files={name:sha(HERE/name) for name in files}),indent=2)+'\n')
print(json.dumps(dict(status='MULTI_CONFIGURATION_FROZEN',pool=524288,mtp=3,engine_dsos_unchanged=7)))
