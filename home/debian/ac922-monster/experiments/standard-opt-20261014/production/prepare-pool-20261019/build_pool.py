"""One private scheduler TU atop the qualified L5 and tested prepare candidate."""
from pathlib import Path
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import time

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
L1 = Path('/home/debian/ac922-monster/production/releases/20261018-r444-lastwave-l1')
L5 = Path('/home/debian/ac922-monster/production/releases/20261019-r444-banked-l5')
PREPARE = BASE / 'production/parallel-prepare-20261019'
OBJECTS = BASE / 'production/lastwave-20261018/build-lastwave'
OUT = HERE / 'build'
LIB = OUT / 'lib'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args, log, cwd):
    with log.open('w') as stream:
        subprocess.run(args, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    assert os.environ.get('AC922_PREPARE_POOL_BUILD') == '1'
    assert not OUT.exists()
    started = time.monotonic()
    OUT.mkdir()
    LIB.mkdir()
    old = json.loads((L5 / 'build-lastwave/build-report.json').read_text())
    previous = json.loads((PREPARE / 'build/build-report.json').read_text())
    assert sha(HERE / 'ggml-backend.before.cpp') == old['host_source_sha256']['ggml/src/ggml-backend.cpp']
    for name, expected in previous['libraries'].items():
        original = PREPARE / 'build/lib' / name
        assert sha(original) == expected
        shutil.copy2(original, LIB / name)
        if name.endswith('.0'):
            (LIB / name.removesuffix('.0')).symlink_to(name)

    # L1 froze the non-backend objects used by the exact current base DSO.
    # Reproduce it byte-for-byte before substituting the single private TU.
    recipe = json.loads((L5 / 'V444-build.json').read_text())
    item, = [v for v in recipe['relink_plan'] if v['target_name'] == 'ggml-base']
    args = shlex.split(item['rewritten_link_script_contents'])
    object_index = None
    for index, arg in enumerate(args):
        if arg.endswith('.o') and not Path(arg).is_absolute():
            if Path(arg).name == 'ggml-backend.cpp.o':
                args[index] = str(OBJECTS / 'ggml-backend.o')
                object_index = index
            else:
                args[index] = str(OBJECTS / 'base-objects/ggml-base' / arg)
            assert Path(args[index]).is_file(), args[index]
    assert object_index is not None
    control = OUT / 'libggml-base.control.so'
    args[args.index('-o') + 1] = str(control)
    run(args, OUT / 'control-link.log', Path(item['cwd']))
    assert sha(control) == old['libraries']['libggml-base.so.0'], 'L5 base object closure drifted'
    control_sha = sha(control)
    control.unlink()

    commands = json.loads((BASE / 'build-reuse/compile_commands.json').read_text())
    entry, = [v for v in commands if Path(v['file']).name == 'ggml-backend.cpp']
    compile_args = entry.get('arguments') or shlex.split(entry['command'])
    compile_args = [v.replace('/home/debian/ac922-monster/src/build-monster', str(BASE / 'build-reuse'))
                   .replace('/home/debian/ac922-monster/src', str(BASE / 'src')) for v in compile_args]
    output = OUT / 'ggml-backend.cpp.o'
    compile_args[compile_args.index('-o') + 1] = str(output)
    compile_args[compile_args.index('-c') + 1] = str(HERE / 'ggml-backend.cpp')
    compile_args += ['-I' + str(BASE / 'src/ggml/src'), '-I' + str(HERE)]
    run(compile_args, OUT / 'backend-compile.log', BASE / 'build-reuse/ggml/src')
    args[object_index] = str(output)
    args[args.index('-o') + 1] = str(LIB / 'libggml-base.so.0')
    run(args, OUT / 'backend-link.log', Path(item['cwd']))
    for name, expected in previous['libraries'].items():
        if name != 'libggml-base.so.0':
            assert sha(LIB / name) == expected
    report = dict(status='BUILD_PASS', cuda_tus=0, host_tus=['ggml-backend.cpp'],
                  worker_reused_sha256=old['worker_sha256'],
                  control_relink_exact_sha256=control_sha,
                  libraries={p.name: sha(p) for p in LIB.iterdir() if p.is_file() and not p.is_symlink()},
                  source_before_sha256=sha(HERE / 'ggml-backend.before.cpp'),
                  source_after_sha256=sha(HERE / 'ggml-backend.cpp'),
                  pool_header_sha256=sha(HERE / 'prepare-six-pool-private.hpp'),
                  compile_argv=compile_args, link_argv=args, elapsed_s=time.monotonic()-started)
    (OUT / 'build-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['status', 'cuda_tus', 'host_tus', 'elapsed_s']}), flush=True)


if __name__ == '__main__':
    main()
