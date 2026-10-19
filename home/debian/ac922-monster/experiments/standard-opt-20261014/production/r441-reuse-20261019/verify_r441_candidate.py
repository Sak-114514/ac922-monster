"""Check the unchanged L5 worker/DSOs and the already tested R441 R2 helper."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
L5 = Path('/home/debian/ac922-monster/production/releases/20261019-r444-banked-l5')
FIXTURE = PROJECT / 'fixtures/shared-hostexec-v441-r2-root-20261018'
HELPER = FIXTURE / 'helper/build-v441-r2/libac922-tail-bridge-v441-r2.so'
WORKER_SHA = '1c5498bce467858e7c1a4006e16e3f5376988f4d5f12bd8f6daae73f110a49a5'
HELPER_SHA = '75fb5bbe2ba34a5a12d737b4c496830daa5a64cc1b3877d0da230191268037ff'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    command = json.loads((HERE / 'worker-command.json').read_text())
    original = json.loads((L5 / 'lastwave-worker-command.json').read_text())
    expected_argv = original['argv'][:]
    worker = expected_argv.index(str(L5 / 'build-lastwave/bank-worker'))
    expected_argv[worker + 2] = str(HERE / 'run/model')
    assert command['argv'] == expected_argv
    assert sha(Path(expected_argv[worker])) == WORKER_SHA
    expected_env = dict(original['env'],
                        GGML_CUDA_GRAPH_DEVICE_TAIL_LIBRARY=str(HELPER),
                        AC922_HOST_GRAPH_SHARED_EXEC='1',
                        AC922_TAIL_STAGE_TIMING='1',
                        AC922_LASTWAVE_RECORD_OUTPUTS='1')
    assert command['env'] == expected_env
    assert sha(HELPER) == HELPER_SHA
    provenance = json.loads((FIXTURE / 'provenance-r2.json').read_text())
    for name in ['helper/bridge-v441-r2.cu', 'helper/context-gate-v441.h',
                 'helper/update-policy-v441.h', 'helper/bridge-api.h', 'tests/shared-exec-v441-r2.cu']:
        assert sha(FIXTURE / name) == provenance['files'][name], name
    for count, tag in [(1, 'one'), (6, 'six')]:
        text = (FIXTURE / f'run{count}-20261019-c1.log').read_text()
        assert f'V441_SHARED_EXEC_LOG_PASS devices={count} ' in text
        assert 'exact_outputs=fixture-oracle' in text
    assert 'V441_R2_CPU_CONTRACT_PASS' in (FIXTURE / 'cpu-20261019-c1.log').read_text()
    report = json.loads((L5 / 'build-lastwave/build-report.json').read_text())
    for name, expected in report['libraries'].items():
        assert sha(L5 / 'build-lastwave/lib' / name) == expected, name
    assert len(report['libraries']) == 7
    scenario = HERE / 'run/model/copy0/fourbank16k-d3/scenarios.json'
    original_scenario = PROJECT / 'fixtures/fourbank-d3-newruntime-v306-20261017/v221-d3-scenarios.json'
    assert sha(scenario) == sha(original_scenario)
    assert command['env']['LLAMA_QSA_INDEXER_IMPL'] == 'standard'
    assert command['env']['GGML_SCHED_LAYER_PARALLEL_PREPARE'] == '0'
    assert command['env']['ROOT_MTP_DEPTH'] == '3'
    assert command['env']['AC922_POOL_TOKENS'] == '524288'
    assert command['env'].get('LD_PRELOAD', '') == ''
    assert 'AC922_STAGE_EVENT_DIAGNOSTIC' not in command['env']
    assert 'AC922_CAPTURED_ARGUMENT_OBSERVER' not in command['env']
    print(json.dumps({'status': 'R441_R2_IDENTITY_AND_FIXTURE_PASS', 'worker_sha256': WORKER_SHA,
                      'helper_sha256': HELPER_SHA, 'engine_dsos': 7,
                      'engine_or_worker_tus_rebuilt': 0}))


if __name__ == '__main__':
    main()
