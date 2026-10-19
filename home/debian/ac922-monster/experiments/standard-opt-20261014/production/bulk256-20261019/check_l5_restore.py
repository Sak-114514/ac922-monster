"""Verify the actual restored L5 process, rather than trusting systemd start."""
from pathlib import Path
import json
import subprocess
import time
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
F5 = HERE.parent / 'safe-pool-reject-20261019'
report = dict(status='WAITING_FOR_L5_READY', phases={}, started_unix=time.time())
out = HERE/'run/l5-restore-verification.json'

def save():
    out.write_text(json.dumps(report, indent=2)+'\n')

try:
    save()
    deadline = time.monotonic()+600
    while True:
        try:
            with urlopen('http://127.0.0.1:8007/ready', timeout=3) as response:
                ready = json.load(response)
            if ready.get('status') == 'ready':
                report['ready'] = ready
                break
        except Exception:
            pass
        state = subprocess.check_output(['systemctl','show','ac922-inference.service',
                                         '-p','ActiveState','--value'],text=True).strip()
        if state in ('failed','inactive') or time.monotonic() > deadline:
            raise RuntimeError('L5 failed or readiness timed out: '+state)
        time.sleep(2)
    for name, interpreter in [('warm_after_restore.py','/home/debian/comfyui/venv/bin/python'),
                               ('verify_live_deployment.py','/usr/bin/python3'),
                               ('gateway_lastwave_smoke.py','/home/debian/comfyui/venv/bin/python')]:
        subprocess.run([interpreter,str(F5/name)],check=True,cwd=F5)
        report['phases'][name] = 'PASS'
        save()
    report['status'] = 'PASS_INTERNAL_PUBLIC_CHECK_PENDING'
except BaseException as error:
    report.update(status='FAIL',error=repr(error))
    raise
finally:
    report['finished_unix'] = time.time()
    save()
