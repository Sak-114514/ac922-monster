"""Build only the private event-selector worker against all seven frozen L5 DSOs."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import time

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
SOURCE = BASE / 'src'
L5 = Path('/home/debian/ac922-monster/production/releases/20261019-r444-banked-l5')
OUT = HERE / 'build'
LIB = OUT / 'lib'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert os.environ.get('AC922_STAGE_EVENT_BUILD') == '1'
    assert not OUT.exists()
    started = time.monotonic()
    OUT.mkdir()
    LIB.mkdir()
    baseline = json.loads((L5 / 'build-lastwave/build-report.json').read_text())
    assert sha(L5 / 'bank-worker-combo.cpp') == 'bc490d7edbba51abc224deebbabf6df4b7e4a7479bba60908b29ee006d53dc2e'
    for path in L5.glob('*.h'):
        shutil.copy2(path, HERE / path.name)
    for name, expected in baseline['libraries'].items():
        original = L5 / 'build-lastwave/lib' / name
        assert sha(original) == expected
        shutil.copy2(original, LIB / name)
        (LIB / name.removesuffix('.0')).symlink_to(name)
    shutil.copy2(L5 / 'build-lastwave/lib/libac922-tail-bridge.so', LIB / 'libac922-tail-bridge.so')
    includes = [SOURCE / 'common', SOURCE / 'src', SOURCE / 'include', SOURCE / 'ggml/include',
                SOURCE / 'ggml/src', SOURCE / 'vendor', SOURCE / 'tools/server', SOURCE / 'tools/mtmd',
                BASE / 'build-reuse/include', BASE / 'build-reuse/ggml/include', BASE / 'fixtures', HERE]
    for child in ['long-prefill-combo-v45-20261016', 'dual-context-v35-onebank4-combined-20261016',
                  'layer6-fourbank-v146-20261017', 'mtp-vocab-fourbank-v221-20261017']:
        includes.append(BASE / 'fixtures' / child)
    names = list(baseline['libraries'])
    args = ['/usr/bin/nice', '-n', '5', '/usr/bin/g++', '-std=c++17', '-O3', '-DNDEBUG', '-pthread',
            *['-D' + value for value in ['GGML_BACKEND_SHARED', 'GGML_SHARED', 'GGML_USE_CPU', 'GGML_USE_CUDA', 'LLAMA_SHARED']],
            '-I/usr/local/cuda-12.4/targets/ppc64le-linux/include', *['-I' + str(x) for x in includes],
            str(HERE / 'bank-worker-combo.cpp'), '-rdynamic', '-fno-omit-frame-pointer', '-o', str(OUT / 'bank-worker'),
            '-Wl,-rpath,$ORIGIN/lib', '-Wl,--no-as-needed', *[str(LIB / name) for name in names],
            '-Wl,--as-needed', '/usr/local/cuda-12.4/targets/ppc64le-linux/lib/stubs/libcuda.so']
    with (OUT / 'worker-compile.log').open('w') as log:
        subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True)
    closure = subprocess.check_output(['ldd', str(OUT / 'bank-worker')],
                                     env=dict(os.environ, LD_LIBRARY_PATH=str(LIB)), text=True)
    assert 'not found' not in closure
    for name, expected in baseline['libraries'].items():
        assert sha(LIB / name) == expected
        assert name + ' => ' + str(LIB) + '/' in closure
    (OUT / 'ldd.txt').write_text(closure)
    report = dict(status='BUILD_PASS', base_release=str(L5), cuda_tus=0, engine_tus=0,
                  worker_sha256=sha(OUT / 'bank-worker'), source_sha256=sha(HERE / 'bank-worker-combo.cpp'),
                  headers={p.name: sha(p) for p in HERE.glob('*.h')},
                  libraries={name: sha(LIB / name) for name in names},
                  original_r5_sha256=sha(LIB / 'libac922-tail-bridge.so'),
                  compile_argv=args, elapsed_s=time.monotonic()-started,
                  production_changed=False, actual_event_selection='PENDING_REAL_RUNTIME')
    (OUT / 'build-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['status', 'cuda_tus', 'engine_tus', 'elapsed_s']}), flush=True)


if __name__ == '__main__':
    main()
