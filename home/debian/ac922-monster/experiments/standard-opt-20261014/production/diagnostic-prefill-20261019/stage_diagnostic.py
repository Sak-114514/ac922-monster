"""Bounded private profiling window; root fallback restores the unchanged L5."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

P = Path('/home/debian/ac922-monster/production')
RELEASE = P / 'releases/20261019-r444-banked-l5'
HERE = Path('/home/debian/ac922-monster/experiments/standard-opt-20261014/production/diagnostic-prefill-20261019')
OVERRIDE = Path('/etc/systemd/system/ac922-inference.service.d/80-current-prefill-diagnostic.conf')
FAULT = Path('/var/lib/ac922-inference/fault.json')


def run(*args, check=True):
    return subprocess.run(args, capture_output=True, text=True, check=check)


def restore(backup):
    record = json.loads((backup / 'rollback.json').read_text())
    assert (P / 'current').resolve() == RELEASE
    run('systemctl', 'stop', 'ac922-inference.service', check=False)
    OVERRIDE.unlink(missing_ok=True)
    run('systemctl', 'daemon-reload')
    readiness = json.loads((backup / 'production-ready.json').read_text())
    readiness.update(status='RESTORING_L5', live_worker_pid=None, unix=time.time())
    (P / 'production-ready.json').write_text(json.dumps(readiness, indent=2) + '\n')
    if not FAULT.exists():
        run('systemctl', 'reset-failed', 'ac922-inference.service', check=False)
        run('systemctl', 'start', 'ac922-inference.service')
    run('systemctl', 'stop', record['timer'] + '.timer', check=False)
    result = {'status': 'L5_RESTORE_STARTED', 'fault_latched': FAULT.exists(),
              'unix': time.time(), 'readiness_still_requires_verification': True}
    (backup / 'restored.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


def stage():
    assert (P / 'current').resolve() == RELEASE and not OVERRIDE.exists() and not FAULT.exists()
    run('runuser', '-u', 'debian', '--', 'python3', str(HERE / 'verify_diagnostic.py'))
    tag = time.strftime('current-prefill-profile-%Y%m%dT%H%M%S', time.gmtime())
    backup = P / 'backups' / tag
    backup.mkdir(mode=0o700)
    shutil.copy2(Path(__file__).resolve(), backup / 'stage_diagnostic.py')
    (backup / 'stage_diagnostic.py').chmod(0o700)
    shutil.copy2(P / 'production-ready.json', backup / 'production-ready.json')
    timer = 'ac922-' + tag + '-restore'
    (backup / 'rollback.json').write_text(json.dumps({'timer': timer, 'release': str(RELEASE)}))
    run('systemd-run', '--unit=' + timer, '--on-active=40m', 'python3',
        str(backup / 'stage_diagnostic.py'), '--restore', str(backup))
    try:
        run('systemctl', 'stop', 'ac922-inference.service')
        readiness = json.loads((P / 'production-ready.json').read_text())
        readiness.update(status='PRIVATE_DIAGNOSTIC_RUNNING', live_worker_pid=None,
                         diagnostic_backup=str(backup), unix=time.time())
        (P / 'production-ready.json').write_text(json.dumps(readiness, indent=2) + '\n')
        OVERRIDE.write_text(f'''[Service]
WorkingDirectory={HERE}
Environment=AC922_WORKER_COMMAND_JSON={HERE}/lastwave-worker-command.json
Environment=AC922_RUN_DIR={HERE}/run/api
Environment=AC922_API_PORT=8008
Environment=AC922_ACCEPTANCE_SOCKET=/home/debian/ac922-monster/production/diag-prefill-control.sock
ExecStartPre=
ExecStartPre=/usr/bin/python3 {HERE}/verify_diagnostic.py
ExecStartPre=/usr/bin/python3 {RELEASE}/wait_safety.py
ExecStart=
ExecStart=/home/debian/comfyui/venv/bin/python {HERE}/candidate_server.py
Restart=no
''')
        run('systemctl', 'daemon-reload')
        run('systemctl', 'reset-failed', 'ac922-inference.service', check=False)
        run('systemctl', 'start', 'ac922-inference.service')
    except BaseException:
        restore(backup)
        raise
    result = {'status': 'PRIVATE_DIAGNOSTIC_STARTED', 'port': 8008,
              'backup': str(backup), 'timer': timer, 'production_release_unchanged': str(RELEASE)}
    (HERE / 'staging-diagnostic.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--stage', action='store_true')
    mode.add_argument('--restore', type=Path)
    args = parser.parse_args()
    assert os.geteuid() == 0
    stage() if args.stage else restore(args.restore)
