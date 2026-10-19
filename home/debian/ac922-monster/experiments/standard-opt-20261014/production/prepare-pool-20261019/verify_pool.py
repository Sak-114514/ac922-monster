"""Validate the private CPU-pool delta before starting another model process."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
L5 = Path('/home/debian/ac922-monster/production/releases/20261019-r444-banked-l5')
PREVIOUS = HERE.parent / 'parallel-prepare-20261019'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    build = json.loads((HERE / 'build/build-report.json').read_text())
    previous = json.loads((PREVIOUS / 'build/build-report.json').read_text())
    l5 = json.loads((L5 / 'build-lastwave/build-report.json').read_text())
    assert build['status'] == 'BUILD_PASS' and build['cuda_tus'] == 0
    assert build['host_tus'] == ['ggml-backend.cpp']
    assert build['control_relink_exact_sha256'] == l5['libraries']['libggml-base.so.0']
    assert sha(HERE / 'ggml-backend.cpp') == build['source_after_sha256']
    assert sha(HERE / 'ggml-backend.before.cpp') == build['source_before_sha256'] == l5['host_source_sha256']['ggml/src/ggml-backend.cpp']
    assert sha(HERE / 'prepare-six-pool-private.hpp') == build['pool_header_sha256']
    actual = {p.name: sha(p) for p in (HERE / 'build/lib').iterdir() if p.is_file() and not p.is_symlink()}
    assert actual == build['libraries']
    for name, expected in previous['libraries'].items():
        if name != 'libggml-base.so.0':
            assert actual[name] == expected
    fixture = json.loads((HERE / 'fixture-result.json').read_text())
    assert fixture['status'] == 'PASS' and fixture['persistent_workers_selected']
    assert fixture['runtime_libraries_sha256'] == actual
    assert {x['copy_depth'] for x in fixture['cases']} == {4, 8}
    assert all(x['status'] == 'PASS' and x['native_frames'] == 24 and
               x['devices'] == list(range(6)) and x['oracle'] == 'exact' for x in fixture['cases'])
    command = json.loads((HERE / 'lastwave-worker-command.json').read_text())
    old_command = json.loads((PREVIOUS / 'lastwave-worker-command.json').read_text())
    expected_argv = list(old_command['argv'])
    index = next(i for i, arg in enumerate(expected_argv) if arg.endswith('/build-lastwave/bank-worker'))
    expected_argv[index + 2] = str(HERE / 'run/model')
    assert command['argv'] == expected_argv
    assert sha(Path(command['argv'][index])) == build['worker_reused_sha256'] == l5['worker_sha256']
    expected_env = dict(old_command['env'])
    expected_env.update(LD_LIBRARY_PATH=str(HERE / 'build/lib'),
                        ROOT_DYNAMIC_LIB_DIR=str(HERE / 'build/lib'),
                        GGML_CUDA_GRAPH_DEVICE_TAIL_LIBRARY=str(HERE / 'build/lib/libac922-tail-bridge.so'),
                        GGML_SCHED_LAYER_PREPARE_PERSISTENT_WORKERS='1',
                        GGML_SCHED_LAYER_GPU_SPANS='0', GGML_META_GPU_SPANS='0',
                        GGML_META_GPU_SPANS_PERIOD='0')
    assert command['env'] == expected_env
    for name in ['api.py', 'candidate_server.py']:
        assert sha(PREVIOUS / name) == sha(L5 / name)
    assert (HERE / 'run/model/copy0/fourbank16k-d3/scenarios.json').is_file()
    print('POOL_PRIVATE_CPU_AND_SIX_GPU_FIXTURE_IDENTITY_PASS; NOT_PROMOTED')


if __name__ == '__main__':
    verify()
