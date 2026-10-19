"""One CUDA umbrella TU; preserve and prove the frozen L5 object closure."""
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
L5 = Path('/home/debian/ac922-monster/production/releases/20261019-r444-banked-l5')
OUT = HERE / 'build'
LIB = OUT / 'lib'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args, log, cwd):
    with log.open('w') as stream:
        subprocess.run(args, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    assert os.environ.get('AC922_PARALLEL_PREPARE_BUILD') == '1'
    assert not OUT.exists(), 'fresh isolated build directory required'
    started = time.monotonic()
    OUT.mkdir()
    LIB.mkdir()
    old = json.loads((L5 / 'build-lastwave/build-report.json').read_text())
    recipe = json.loads((L5 / 'V444-build.json').read_text())
    compile_recipe, = [x for x in recipe['compile_plan'] if x['source'].endswith('/ggml-cuda.cu')]
    link_recipe, = [x for x in recipe['relink_plan'] if x['target_name'] == 'ggml-cuda']
    original_source = BASE / 'src/ggml/src/ggml-cuda/ggml-cuda.cu'
    assert sha(original_source) == sha(HERE / 'ggml-cuda.before.cu')
    for name, expected in old['libraries'].items():
        assert sha(L5 / 'build-lastwave/lib' / name) == expected, name
        shutil.copy2(L5 / 'build-lastwave/lib' / name, LIB / name)
        (LIB / name.removesuffix('.0')).symlink_to(name)

    # Reproduce the old shared library before replacing any object. A mismatch
    # means the reusable object closure has drifted and this build must stop.
    link = shlex.split(link_recipe['rewritten_link_script_contents'])
    control = OUT / 'libggml-cuda.control.so'
    link[link.index('-o') + 1] = str(control)
    run(link, OUT / 'control-link.log', Path(link_recipe['cwd']))
    assert sha(control) == old['libraries']['libggml-cuda.so.0'], 'V444 CUDA object closure drifted'
    control_sha = sha(control)
    control.unlink()

    args = list(compile_recipe['argv'])
    output = OUT / 'ggml-cuda.cu.o'
    args[args.index('-o') + 1] = str(output)
    args[args.index('-c') + 1] = str(HERE / 'ggml-cuda.cu')
    for flag, destination in [('-MF', str(OUT / 'ggml-cuda.cu.o.d')), ('-MT', str(output))]:
        if flag in args:
            args[args.index(flag) + 1] = destination
    args += ['-I' + str(original_source.parent), '--threads', '4']
    run(args, OUT / 'cuda-compile.log', Path(compile_recipe['cwd']))
    replaced = 0
    for index, arg in enumerate(link):
        if Path(arg).name == 'ggml-cuda.cu.o':
            link[index] = str(output)
            replaced += 1
    assert replaced == 1
    link[link.index('-o') + 1] = str(LIB / 'libggml-cuda.so.0')
    run(link, OUT / 'cuda-link.log', Path(link_recipe['cwd']))

    helper = BASE / 'fixtures/six-cuda-prepare-v436-root-20261018/helper/build-v436/libac922-tail-bridge-v3.so'
    exported = subprocess.check_output(['nm', '-D', str(helper)], text=True)
    for name in ['ac922_tail_bridge_v2_launch_captured', 'ac922_tail_bridge_v3_prepare_captured',
                 'ac922_tail_bridge_v3_can_prepare', 'ac922_tail_bridge_drop_context']:
        assert name in exported, name
    shutil.copy2(helper, LIB / 'libac922-tail-bridge.so')
    for name, expected in old['libraries'].items():
        if name != 'libggml-cuda.so.0':
            assert sha(LIB / name) == expected, name
    assert sha(original_source) == sha(HERE / 'ggml-cuda.before.cu')
    report = dict(status='BUILD_PASS', baseline=str(L5), cuda_tus=1, template_tus=0,
                  worker_reused_sha256=old['worker_sha256'], control_relink_exact_sha256=control_sha,
                  source_before_sha256=sha(HERE / 'ggml-cuda.before.cu'),
                  source_after_sha256=sha(HERE / 'ggml-cuda.cu'),
                  libraries={p.name: sha(p) for p in LIB.iterdir() if p.is_file() and not p.is_symlink()},
                  compile_argv=args, link_argv=link, elapsed_s=time.monotonic() - started)
    (OUT / 'build-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['status', 'cuda_tus', 'template_tus', 'elapsed_s']}), flush=True)


if __name__ == '__main__':
    main()
