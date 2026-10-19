"""Reuse the armed R441 window while preserving the original root restore markers."""
from pathlib import Path
import json
import os
import subprocess
import time

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / 'r441-reuse-20261019'
PRODUCTION = Path('/home/debian/ac922-monster/production')
L5 = PRODUCTION / 'releases/20261019-r444-banked-l5'
OVERRIDE = Path('/etc/systemd/system/ac922-inference.service.d/97-r441-reuse-candidate.conf')


def run(args):
    return subprocess.run(args, capture_output=True, text=True, check=True)


def main():
    assert os.geteuid() == 0 and (PRODUCTION / 'current').resolve() == L5
    assert not Path('/var/lib/ac922-inference/fault.json').exists()
    old = OVERRIDE.read_text()
    assert f'WorkingDirectory={PREVIOUS}' in old and 'Restart=no' in old
    assert 'Environment=AC922_API_PORT=8013' in old
    assert f'AC922_WORKER_COMMAND_JSON={PREVIOUS}/worker-command.json' in old
    stage = json.loads((PREVIOUS / 'staging-prepare.json').read_text())
    backup = Path(stage['rollback_backup'])
    timer = stage['rollback_timer'] + '.timer'
    run(['systemctl', 'is-active', timer])
    created = json.loads((backup / 'rollback.json').read_text())['created_unix']
    assert time.time() < created + 2400 - 600, 'less than ten minutes remain in the finite window'
    report = json.loads((PREVIOUS / 'run/qualification.json').read_text())
    assert report['status'] == 'MODEL_COMPLETE_PENDING_LOG_AND_TEXT_REVIEW'
    assert report['four512_all_2048_token_ids_exact'] and report['four512_full_texts_exact']
    assert json.loads((PREVIOUS / 'run/fourlong-native.json').read_text())['status'] == 'PASS'
    run(['/usr/sbin/runuser', '-u', 'debian', '--', 'python3', str(HERE / 'verify_topology.py')])
    saved = backup / 'before-topology-buckets-override.conf'
    assert not saved.exists(), 'this transition was already attempted'
    saved.write_text(old)
    replacement = old.replace(f'{PREVIOUS}/worker-command.json', f'{HERE}/worker-command.json')
    replacement = replacement.replace(f'AC922_RUN_DIR={PREVIOUS}/run/api', f'AC922_RUN_DIR={HERE}/run/api')
    replacement = replacement.replace(f'{PREVIOUS}/verify_r441_candidate.py', f'{HERE}/verify_topology.py')
    assert f'WorkingDirectory={PREVIOUS}' in replacement and 'Environment=AC922_API_PORT=8013' in replacement
    try:
        run(['systemctl', 'stop', 'ac922-inference.service'])
        temporary = OVERRIDE.with_suffix('.topology.tmp')
        temporary.write_text(replacement)
        temporary.chmod(0o644)
        temporary.replace(OVERRIDE)
        ready = json.loads((PRODUCTION / 'production-ready.json').read_text())
        ready.update(status='PRIVATE_R441_TOPOLOGY_LOADING', live_worker_pid=None,
                     candidate=str(HERE), unix=time.time(), readiness_still_requires_verification=True)
        (PRODUCTION / 'production-ready.json').write_text(json.dumps(ready, indent=2) + '\n')
        run(['systemctl', 'daemon-reload'])
        run(['systemctl', 'reset-failed', 'ac922-inference.service'])
        run(['systemctl', 'start', 'ac922-inference.service'])
    except BaseException:
        run(['python3', str(backup / 'stage_r441.py'), '--restore', str(backup)])
        raise
    result = {'status': 'PRIVATE_R441_TOPOLOGY_STARTED', 'api_port': 8013,
              'original_backup': str(backup), 'unchanged_restore_timer': timer,
              'production_release_unchanged': str(L5), 'unix': time.time()}
    (HERE / 'staging-r3.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
