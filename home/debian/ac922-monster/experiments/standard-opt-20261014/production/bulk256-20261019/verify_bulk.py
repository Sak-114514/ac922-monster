"""Pin one CPU-worker change to every unchanged production numerical DSO."""
from pathlib import Path
import hashlib
import json
HERE = Path(__file__).resolve().parent
L5 = Path('/home/debian/ac922-monster/production/releases/20261019-r444-banked-l5')
WORKER = '5228499d571bd1ea89ad60ef34580257e9eecc5ca9650ce03e4e99712a2a22bb'
SOURCE = '095fdca98a5576749860448a106d9f147a1dbfa16494f022c1f529686a34c315'
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    build = json.loads((HERE/'build/build-report.json').read_text())
    base = json.loads((L5/'build-lastwave/build-report.json').read_text())
    pins = json.loads((HERE/'pins.json').read_text())
    assert build['status']=='BUILD_PASS' and build['cuda_tus']==build['engine_tus']==0
    assert sha(HERE/'build/bank-worker')==build['worker_sha256']==WORKER
    assert sha(HERE/'bank-worker-combo.cpp')==build['source_sha256']==SOURCE
    assert build['libraries']==base['libraries'] and len(build['libraries'])==7
    for name, expected in build['libraries'].items():
        assert sha(HERE/'build/lib'/name)==expected, name
    assert sha(HERE/'build/lib/libac922-tail-bridge.so')==build['original_r5_sha256']==sha(L5/'build-lastwave/lib/libac922-tail-bridge.so')
    for name, expected in build['headers'].items():
        assert sha(HERE/name)==expected, name
    for name, expected in pins['files'].items():
        assert sha(HERE/name)==expected, name
    original=json.loads((L5/'lastwave-worker-command.json').read_text())
    command=json.loads((HERE/'worker-command.json').read_text())
    argv=original['argv'][:]
    index=argv.index(str(L5/'build-lastwave/bank-worker'))
    argv[index]=str(HERE/'build/bank-worker')
    argv[index+2]=str(HERE/'run/model')
    argv[index+3]=str(HERE/'verify_bulk_native.py')
    assert command['argv']==argv
    expected=dict(original['env'],LD_LIBRARY_PATH=str(HERE/'build/lib'),
                  ROOT_DYNAMIC_LIB_DIR=str(HERE/'build/lib'),
                  GGML_CUDA_GRAPH_DEVICE_TAIL_LIBRARY=str(HERE/'build/lib/libac922-tail-bridge.so'),
                  AC922_LASTWAVE_RECORD_OUTPUTS='1',AC922_MULTI_BULK256='1')
    assert command['env']==expected
    assert expected['ROOT_MTP_DEPTH']=='3' and expected['LLAMA_MTP_N_UBATCH']=='256'
    assert expected['AC922_POOL_TOKENS']=='524288' and expected['AC922_REQUEST_TOKENS']=='262140'
    assert expected['LLAMA_QSA_INDEXER_IMPL']=='standard' and expected['LLAMA_PLE_NUMA_GPU_GIB']=='18'
    expected_gate=(L5/'verify_combo_native.py').read_text()
    for key in ('target_n_batch','draft_n_batch'):
        before='"'+key+'": 4096'
        assert expected_gate.count(before)==1
        expected_gate=expected_gate.replace(before,'"'+key+'": 17152')
    assert (HERE/'verify_combo_native.py').read_text()==expected_gate
    print(json.dumps(dict(status='BULK256_PROFILE_IDENTITY_PASS',engine_dsos_unchanged=7,mtp_depth=3,pool_tokens=524288,quality='PENDING_REAL_MODEL')))
if __name__=='__main__':
    main()
