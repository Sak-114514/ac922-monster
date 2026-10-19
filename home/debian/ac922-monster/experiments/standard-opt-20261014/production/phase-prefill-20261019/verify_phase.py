"""Identity gate for original L5 plus exactly two private host dispatch TUs."""
from pathlib import Path
import hashlib
import json
import os
import subprocess

HERE = Path(__file__).resolve().parent
L5 = Path('/home/debian/ac922-monster/production/releases/20261019-r444-banked-l5')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    pins = json.loads((HERE/'phase-pins.json').read_text())
    assert sha(HERE/'build/build-report.json') == pins['report_sha256']
    assert sha(HERE/'worker-command.json') == pins['command_sha256']
    assert sha(HERE/'verify_combo_native.py') == pins['gate_sha256']
    for name,digest in pins['source_sha256'].items():
        assert sha(HERE/name) == digest
    report = json.loads((HERE/'build/build-report.json').read_text())
    base = json.loads((L5/'build-lastwave/build-report.json').read_text())
    assert report['status'] == 'BUILD_PASS'
    assert report['control_relink_exact_hashes'] == {k:base['libraries'][k] for k in ['libllama.so.0','libggml-cuda.so.0']}
    assert report['worker_sha256'] == sha(L5/'build-lastwave/bank-worker') == base['worker_sha256']
    assert len(report['libraries']) == 7
    for name,digest in report['libraries'].items():
        assert sha(HERE/'build/lib'/name) == digest
        if name not in ['libllama.so.0','libggml-cuda.so.0']:
            assert digest == base['libraries'][name]
    assert report['original_r5_sha256'] == sha(L5/'build-lastwave/lib/libac922-tail-bridge.so')
    assert sha(HERE/'build/lib/libac922-tail-bridge.so') == report['original_r5_sha256']
    command = json.loads((HERE/'worker-command.json').read_text())
    original = json.loads((L5/'lastwave-worker-command.json').read_text())
    expected = json.loads(json.dumps(original))
    index = expected['argv'].index(str(L5/'build-lastwave/bank-worker'))
    expected['argv'][index+2] = str(HERE/'run/model')
    expected['argv'][index+3] = str(HERE/'verify_phase_native.py')
    expected['env'].update(LD_LIBRARY_PATH=str(HERE/'build/lib'),
        ROOT_DYNAMIC_LIB_DIR=str(HERE/'build/lib'),ROOT_BUILD_REPORT=str(HERE/'build/build-report.json'),
        ROOT_RUNTIME_REVISION=report['build_run_id'],
        GGML_CUDA_GRAPH_DEVICE_TAIL_LIBRARY=str(HERE/'build/lib/libac922-tail-bridge.so'),
        LLAMA_PREFILL_DIRECT='1',AC922_LASTWAVE_RECORD_OUTPUTS='1')
    assert command == expected
    assert sha(HERE/'run/model/copy0/fourbank16k-d3/scenarios.json') == pins['scenario_sha256']
    closure = subprocess.check_output(['ldd',str(L5/'build-lastwave/bank-worker')],text=True,
                                     env=dict(os.environ,LD_LIBRARY_PATH=str(HERE/'build/lib')))
    assert 'not found' not in closure
    for name in report['libraries']:
        assert name+' => '+str(HERE/'build/lib')+'/' in closure
    print(json.dumps(dict(status='PHASE_IDENTITY_PASS',runtime=report['build_run_id'],
                          original_worker=True,original_r5=True,unchanged_dsos=5,
                          mtp_depth=3,pool_tokens=524288,quality='PENDING_ACTUAL_MODEL')))

if __name__ == '__main__':
    main()
