"""Freeze the two-TU candidate and the unchanged production geometry."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
L5 = Path('/home/debian/ac922-monster/production/releases/20261019-r444-banked-l5')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

report_path = HERE/'build/build-report.json'
report = json.loads(report_path.read_text())
assert report['status'] == 'BUILD_PASS'
assert report['cuda_tus'] == report['cpp_tus'] == 1 and report['template_tus'] == 0
command = json.loads((L5/'lastwave-worker-command.json').read_text())
index = command['argv'].index(str(L5/'build-lastwave/bank-worker'))
command['argv'][index+2] = str(HERE/'run/model')
command['argv'][index+3] = str(HERE/'verify_phase_native.py')
command['env'].update(LD_LIBRARY_PATH=str(HERE/'build/lib'),
    ROOT_DYNAMIC_LIB_DIR=str(HERE/'build/lib'),ROOT_BUILD_REPORT=str(report_path),
    ROOT_RUNTIME_REVISION=report['build_run_id'],
    GGML_CUDA_GRAPH_DEVICE_TAIL_LIBRARY=str(HERE/'build/lib/libac922-tail-bridge.so'),
    LLAMA_PREFILL_DIRECT='1',AC922_LASTWAVE_RECORD_OUTPUTS='1')
(HERE/'worker-command.json').write_text(json.dumps(command,indent=2)+'\n')
gate = (L5/'verify_combo_native.py').read_text()
gate = gate.replace('EXPECTED_RUNTIME_BUILD_ID = "20261018T092503.472988Z"',
                    'EXPECTED_RUNTIME_BUILD_ID = '+repr(report['build_run_id']))
gate = gate.replace('EXPECTED_RUNTIME_REPORT_SHA256 = "b993f9a857c73586a9ff39b5b655ec5bc1984ae48924b33ef033abcef5daf4f4"',
                    'EXPECTED_RUNTIME_REPORT_SHA256 = '+repr(sha(report_path)))
(HERE/'verify_combo_native.py').write_text(gate)
for folder in ['run/model/copy0/fourbank16k-d3','run/api']:
    (HERE/folder).mkdir(parents=True,exist_ok=True)
scenario = BASE/'fixtures/fourbank-d3-newruntime-v306-20261017/v221-d3-scenarios.json'
(HERE/'run/model/copy0/fourbank16k-d3/scenarios.json').write_bytes(scenario.read_bytes())
pins = dict(report_sha256=sha(report_path),command_sha256=sha(HERE/'worker-command.json'),
            source_sha256={name:sha(HERE/name) for name in ['ggml-cuda.cu','llama-context.cpp']},
            gate_sha256=sha(HERE/'verify_combo_native.py'),scenario_sha256=sha(scenario))
(HERE/'phase-pins.json').write_text(json.dumps(pins,indent=2)+'\n')
print(json.dumps(dict(status='PHASE_CONFIGURATION_FROZEN',runtime=report['build_run_id'],
                      report_sha256=pins['report_sha256'],worker_unchanged=True,
                      mtp_depth=command['env']['ROOT_MTP_DEPTH'],pool=command['env']['AC922_POOL_TOKENS'])))
