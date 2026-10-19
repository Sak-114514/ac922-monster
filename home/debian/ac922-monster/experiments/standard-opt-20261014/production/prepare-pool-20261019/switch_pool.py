"""Reuse the armed private window; its original root timer still restores L5."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import time

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / 'parallel-prepare-20261019'
PRODUCTION = Path('/home/debian/ac922-monster/production')
L5 = PRODUCTION / 'releases/20261019-r444-banked-l5'
OVERRIDE = Path('/etc/systemd/system/ac922-inference.service.d/95-parallel-prepare-candidate.conf')


def run(args):
    return subprocess.run(args, capture_output=True, text=True, check=True)


def main():
    assert os.geteuid() == 0 and (PRODUCTION / 'current').resolve() == L5
    assert not Path('/var/lib/ac922-inference/fault.json').exists()
    old_override = OVERRIDE.read_text()
    assert f'WorkingDirectory={PREVIOUS}' in old_override and 'Restart=no' in old_override
    assert f'AC922_WORKER_COMMAND_JSON={PREVIOUS}/lastwave-worker-command.json' in old_override
    stage = json.loads((PREVIOUS / 'staging-prepare.json').read_text())
    backup = Path(stage['rollback_backup'])
    timer = stage['rollback_timer'] + '.timer'
    run(['systemctl', 'is-active', timer])
    run(['/usr/sbin/runuser', '-u', 'debian', '--', 'python3', str(HERE / 'verify_pool.py')])
    assert not (backup / 'before-pool-override.conf').exists()
    (backup / 'before-pool-override.conf').write_text(old_override)
    replacement = old_override.replace(f'{PREVIOUS}/lastwave-worker-command.json',
                                        f'{HERE}/lastwave-worker-command.json')
    replacement = replacement.replace(f'AC922_RUN_DIR={PREVIOUS}/run/api', f'AC922_RUN_DIR={HERE}/run/api')
    replacement = replacement.replace('prepare-control.sock', 'prepare-pool-control.sock')
    replacement = replacement.replace(f'{PREVIOUS}/verify_prepare_candidate.py', f'{HERE}/verify_pool.py')
    # Keep the original WorkingDirectory/port/Restart markers. The already
    # armed root-owned fallback script will therefore remove this override too.
    try:
        run(['systemctl', 'stop', 'ac922-inference.service'])
        temporary = OVERRIDE.with_suffix('.pool.tmp')
        temporary.write_text(replacement)
        temporary.chmod(0o644)
        temporary.replace(OVERRIDE)
        ready = json.loads((PRODUCTION / 'production-ready.json').read_text())
        ready.update(status='PRIVATE_PREPARE_POOL_LOADING', live_worker_pid=None,
                     candidate=str(HERE), unix=time.time())
        (PRODUCTION / 'production-ready.json').write_text(json.dumps(ready, indent=2) + '\n')
        run(['systemctl', 'daemon-reload'])
        run(['systemctl', 'reset-failed', 'ac922-inference.service'])
        run(['systemctl', 'start', 'ac922-inference.service'])
    except BaseException:
        run(['python3', str(backup / 'stage_prepare.py'), '--restore', str(backup)])
        raise
    print(json.dumps(dict(status='PRIVATE_POOL_STARTED', api_port=8011,
                           unchanged_original_restore_timer=timer,
                           original_backup=str(backup), production_release_unchanged=str(L5))))


if __name__ == '__main__':
    main()
