"""Qualify the bounded L4 worker using one private model load."""
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.request import urlopen

F = Path(__file__).resolve().parent
R = F/'run'
R.mkdir(exist_ok=True)
d = {'status':'WAITING_FOR_READY', 'started_unix':time.time(), 'stages':[]}


def save():
    (R/'campaign-status.json').write_text(json.dumps(d, indent=2)+'\n')


save()
try:
    deadline = time.monotonic()+600
    while True:
        try:
            with urlopen('http://127.0.0.1:8005/ready', timeout=3) as response:
                ready = json.load(response)
            if ready.get('status') == 'ready':
                d['ready'] = ready
                break
        except Exception:
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError('L4 private worker did not become ready within600seconds')
        time.sleep(2)
    env = dict(os.environ, AC922_BASE_URL='http://127.0.0.1:8005/v1',
        AC922_READY_URL='http://127.0.0.1:8005/ready', AC922_API_KEY_FILE='/home/debian/apigateway/api_key',
        AC922_RESULT_JSON=str(R/'http-result.json'), AC922_MODEL_RUN_ROOT=str(R/'model'),
        NO_PROXY='127.0.0.1,localhost', no_proxy='127.0.0.1,localhost')
    stages = [('growth','/usr/bin/python3','growth_acceptance.py',['--run']),
        ('affinity','/usr/bin/python3','affinity_acceptance.py',['--run']),
        ('single-cache','/usr/bin/python3','single_cache_acceptance.py',[]),
        ('http','/home/debian/comfyui/venv/bin/python','lastwave_http_acceptance.py',[]),
        ('cancel-cached','/home/debian/comfyui/venv/bin/python','cancel_cached_single.py',['--run']),
        ('prefill-pacing','/usr/bin/python3','prefill_pacing_acceptance.py',[])]
    d['status'] = 'RUNNING'
    for label, python, script, args in stages:
        step = {'label':label, 'status':'RUNNING', 'started_unix':time.time()}
        d['stages'].append(step)
        save()
        with (R/(label+'-controller.log')).open('w') as log:
            p = subprocess.run([python, str(F/script), *args], cwd=F, env=env,
                               stdout=log, stderr=subprocess.STDOUT, timeout=1200)
        step.update(returncode=p.returncode, status='PASS' if not p.returncode else 'FAIL', finished_unix=time.time())
        save()
        if p.returncode:
            raise RuntimeError(label+' failed; see stage log')
    d['status'] = 'PASS_PENDING_FULL_TEXT_REVIEW'
except BaseException as error:
    d.update(status='FAIL', error=repr(error))
    raise
finally:
    d['finished_unix'] = time.time()
    save()
