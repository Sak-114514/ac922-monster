"""One loaded model: safe pool rejection, existing caches, full HTTP, cancellation."""
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.request import urlopen

F = Path(__file__).resolve().parent
R = F / 'run'
report = {'status': 'WAITING_FOR_READY', 'started_unix': time.time(), 'stages': []}


def save():
    temp = R / 'campaign-status.tmp'
    temp.write_text(json.dumps(report, indent=2) + '\n')
    temp.replace(R / 'campaign-status.json')


save()
try:
    deadline = time.monotonic() + 600
    while True:
        try:
            with urlopen('http://127.0.0.1:8007/ready', timeout=3) as response:
                ready = json.load(response)
            if ready.get('status') == 'ready':
                report['ready'] = ready
                break
        except Exception:
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError('private safe-pool worker readiness timeout')
        time.sleep(2)
    env = dict(os.environ, AC922_BASE_URL='http://127.0.0.1:8007/v1',
               AC922_READY_URL='http://127.0.0.1:8007/ready',
               AC922_API_KEY_FILE='/home/debian/apigateway/api_key',
               AC922_RESULT_JSON=str(R / 'http-result.json'), AC922_MODEL_RUN_ROOT=str(R / 'model'),
               NO_PROXY='127.0.0.1,localhost', no_proxy='127.0.0.1,localhost')
    report['status'] = 'RUNNING'
    for label, python, script, args in [
        ('safe-pool-affinity', '/usr/bin/python3', 'safe_pool_reject_acceptance.py', ['--run']),
        ('single-cache', '/usr/bin/python3', 'single_cache_acceptance.py', []),
        ('http', '/home/debian/comfyui/venv/bin/python', 'lastwave_http_acceptance.py', []),
        ('cancel-cached', '/home/debian/comfyui/venv/bin/python', 'cancel_cached_single.py', ['--run']),
    ]:
        step = {'label': label, 'status': 'RUNNING', 'started_unix': time.time()}
        report['stages'].append(step)
        save()
        with (R / (label + '-controller.log')).open('w') as log:
            process = subprocess.run([python, str(F / script), *args], cwd=F, env=env,
                                     stdout=log, stderr=subprocess.STDOUT, timeout=1200)
        step.update(status='PASS' if not process.returncode else 'FAIL',
                    returncode=process.returncode, finished_unix=time.time())
        save()
        if process.returncode:
            raise RuntimeError(label + ' failed; see controller log')
    report['status'] = 'PASS_PENDING_FULL_TEXT_REVIEW'
except BaseException as error:
    report.update(status='FAIL', error=repr(error))
    raise
finally:
    report['finished_unix'] = time.time()
    save()
