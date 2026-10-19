"""Bounded CPU proofs and unchanged L4 policy/runtime identity; no model load."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
BASELINE = Path('/home/debian/ac922-monster/production/releases/20261019-r444-banked-l4')
BUILD = HERE / 'build-lastwave'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(name):
    output = BUILD / ('safe-' + name)
    with output.with_suffix('.log').open('w') as log:
        subprocess.run(['/usr/bin/g++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                        '-pthread', '-I' + str(HERE.parents[1] / 'src/vendor'),
                        str(HERE / (name + '.cpp')), '-o', str(output)],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run([str(output)], stdout=log, stderr=subprocess.STDOUT, check=True)
    return {'fixture': name, 'status': 'PASS'}


def main():
    BUILD.mkdir(exist_ok=True)
    headers = {p.name: sha(p) for p in HERE.glob('*.h')}
    for name, expected in headers.items():
        assert sha(BASELINE / name) == expected, name
    assert set(headers) == {p.name for p in BASELINE.glob('*.h')}
    names = ['test_bank_retention', 'test_bank_growth', 'test_bank_affinity',
             'test_bank_setup_guard', 'test_prefix_checkpoint', 'test_lastwave_runner',
             'test_prefill_thermal']
    with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:
        fixtures = list(pool.map(fixture, names))
    suites = ['test_api', 'test_refill_protocol', 'test_pool_plan_protocol', 'test_private_refill']
    with (BUILD / 'safe-python-tests.log').open('w') as log:
        subprocess.run(['/home/debian/comfyui/venv/bin/python', '-m', 'unittest', '-v', *suites],
                       cwd=HERE, stdout=log, stderr=subprocess.STDOUT, check=True)
    for script in ['affinity_acceptance.py', 'safe_pool_reject_acceptance.py']:
        with (BUILD / ('safe-' + script + '.log')).open('w') as log:
            subprocess.run(['/usr/bin/python3', str(HERE / script), '--self-test'],
                           cwd=HERE, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = {'status': 'PASS', 'fixtures': fixtures, 'python_suites': suites,
              'headers_byte_identical_to_L4': headers, 'model_loaded': False}
    (BUILD / 'safe-cpu-validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
