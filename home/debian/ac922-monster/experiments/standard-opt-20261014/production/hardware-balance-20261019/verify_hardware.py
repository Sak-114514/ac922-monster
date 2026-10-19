"""Pin the private host profile to the unchanged L5 numerical libraries and R5."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
L5 = Path('/home/debian/ac922-monster/production/releases/20261019-r444-banked-l5')
WORKER_SHA = '993fe664b11633979dc489cee52b3c2fc7934ce19d02418350b2a7191a7a6ee1'
SOURCE_SHA = 'dfd81e2e55ff1236b62321723e092c9b93113e85d4fe7bcefca481f876405041'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    build = json.loads((HERE / 'build/build-report.json').read_text())
    base = json.loads((L5 / 'build-lastwave/build-report.json').read_text())
    assert build['status'] == 'BUILD_PASS' and build['engine_tus'] == build['cuda_tus'] == 0
    assert sha(HERE / 'build/bank-worker') == build['worker_sha256'] == WORKER_SHA
    assert sha(HERE / 'bank-worker-combo.cpp') == build['source_sha256'] == SOURCE_SHA
    assert build['libraries'] == base['libraries'] and len(build['libraries']) == 7
    for name, expected in build['libraries'].items():
        assert sha(HERE / 'build/lib' / name) == expected, name
    assert sha(HERE / 'build/lib/libac922-tail-bridge.so') == build['original_r5_sha256']
    assert build['original_r5_sha256'] == sha(L5 / 'build-lastwave/lib/libac922-tail-bridge.so')
    for name, expected in build['headers'].items():
        assert sha(HERE / name) == expected, name
    original = json.loads((L5 / 'lastwave-worker-command.json').read_text())
    command = json.loads((HERE / 'worker-command.json').read_text())
    for name, digest in {'verify_hardware_native.py': '6ae3f78c744f2377178201ac9ab83c81683680547a36d3f5a99ed9e63c114405', 'verify_combo_native.py': '0f17ecfbcca3e7b81ed938cd0e28adc1d4661961e652e7dbd406b1cc906bcff9'}.items():
        assert sha(HERE / name) == digest, name
    argv = original['argv'][:]
    index = argv.index('LLAMA_MTP_BASE_DEVICE')
    assert argv[index - 1] == '-u'
    del argv[index - 1:index + 1]
    worker = argv.index(str(L5 / 'build-lastwave/bank-worker'))
    argv[worker], argv[worker + 2] = str(HERE / 'build/bank-worker'), str(HERE / 'run/model')
    argv[worker + 3] = str(HERE / 'verify_hardware_native.py')
    assert command['argv'] == argv
    expected = dict(original['env'], LD_LIBRARY_PATH=str(HERE / 'build/lib'),
                    ROOT_DYNAMIC_LIB_DIR=str(HERE / 'build/lib'),
                    GGML_CUDA_GRAPH_DEVICE_TAIL_LIBRARY=str(HERE / 'build/lib/libac922-tail-bridge.so'),
                    AC922_LASTWAVE_RECORD_OUTPUTS='1', LLAMA_MTP_BASE_DEVICE='1',
                    LLAMA_PLE_NUMA_GPU_GIB='16',
                    LLAMA_PLE_NUMA_GPU_STRIPES_MIB='3968,128,3840,2816,2688,2944',
                    AC922_WIDE_PREFILL='1', AC922_DRAFT_UBATCH='128', LLAMA_MTP_N_UBATCH='128')
    assert command['env'] == expected
    assert sum(map(int, expected['LLAMA_PLE_NUMA_GPU_STRIPES_MIB'].split(','))) == 16 * 1024
    assert not any(flag in expected for flag in ('AC922_HOST_GRAPH_SHARED_EXEC', 'AC922_HOST_GRAPH_TOPOLOGY_BUCKETS'))
    assert expected['LLAMA_QSA_INDEXER_IMPL'] == 'standard' and expected['ROOT_MTP_DEPTH'] == '3'
    assert expected['AC922_POOL_TOKENS'] == '524288' and expected['AC922_REQUEST_TOKENS'] == '262140'
    scenario = PROJECT / 'fixtures/fourbank-d3-newruntime-v306-20261017/v221-d3-scenarios.json'
    assert sha(HERE / 'run/model/copy0/fourbank16k-d3/scenarios.json') == sha(scenario)
    print(json.dumps({'status': 'HARDWARE_PROFILE_IDENTITY_PASS', 'worker_sha256': WORKER_SHA,
                      'engine_dsos_unchanged': 7, 'original_r5': True, 'mtp_depth': 3,
                      'pool_tokens': 524288, 'quality': 'PENDING_ACTUAL_MODEL'}))


if __name__ == '__main__':
    main()
