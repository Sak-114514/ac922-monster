"""Run only after the protected service stop, then permit a single model load."""
from pathlib import Path
import json
import os
import subprocess
import time
import hashlib

HERE = Path(__file__).resolve().parent
LIB = HERE/'build/lib'
env = {k:v for k,v in os.environ.items() if not k.startswith(('GGML_','LLAMA_','ROOT_'))}
env.update(LD_LIBRARY_PATH=str(LIB),GGML_CUDA_GRAPH_OPT='1',GGML_CUDA_GRAPH_STRUCTURAL='1',
    GGML_CUDA_GRAPH_PATH_LOG='1',GGML_CUDA_GRAPH_BINDING_KEYS='1',
    GGML_CUDA_GRAPH_BINDING_KEYS_CAP='32',GGML_OPTIMIZATION_PATH_LOG='1',
    GGML_CUDA_GRAPH_DEVICE_TAIL='0')
assert subprocess.check_output(['systemctl','show','ac922-inference.service',
                                '-p','ActiveState','--value'],text=True).strip() == 'inactive'
results = []
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

for count in (1,6):
    log = HERE/f'run/backend-fixture-{count}.log'
    saved = HERE/f'run/backend-fixture-{count}-receipt.json'
    identity = dict(report=sha(HERE/'build/build-report.json'),
                    binary=sha(HERE/'build/test_prefill_direct'))
    reuse = False
    if log.exists() and saved.exists():
        prior = json.loads(saved.read_text())
        reuse = prior.get('identity') == identity and prior.get('log_sha256') == sha(log)
    if not reuse:
        with log.open('w') as stream:
            subprocess.run([str(HERE/'build/test_prefill_direct'),str(LIB),str(count)],env=env,
                            stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=45)
    result = subprocess.check_output(['python3',str(HERE/'check_prefill_direct_fixture.py'),
                                     str(log),'--devices',str(count)],text=True)
    checked = json.loads(result)
    results.append(checked)
    saved.write_text(json.dumps(dict(status='PASS',identity=identity,log_sha256=sha(log),
                                    checked=checked,unix=time.time()),indent=2)+'\n')
report = dict(status='PASS',fixtures=results,unix=time.time(),actual_model='PENDING')
(HERE/'run/backend-fixture.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
