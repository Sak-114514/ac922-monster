"""Replace two host-dispatch TUs after proving both frozen object closures."""
from concurrent.futures import ThreadPoolExecutor
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
OUT = HERE/'build'
LIB = OUT/'lib'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def run(args, path, cwd):
    with path.open('w') as stream:
        subprocess.run(args, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, check=True)

def main():
    assert os.environ.get('AC922_PHASE_PREFILL_BUILD') == '1'
    assert not OUT.exists(), 'fresh isolated output required'
    started = time.monotonic()
    OUT.mkdir()
    LIB.mkdir()
    old = json.loads((L5/'build-lastwave/build-report.json').read_text())
    recipe = json.loads((L5/'V444-build.json').read_text())
    sources = [('cuda','ggml-cuda.cu','ggml-cuda.before.cu',
                'ggml/src/ggml-cuda/ggml-cuda.cu','ggml-cuda','libggml-cuda.so.0'),
               ('context','llama-context.cpp','llama-context.before.cpp',
                'src/llama-context.cpp','llama','libllama.so.0')]
    for name, expected in old['libraries'].items():
        path = L5/'build-lastwave/lib'/name
        assert sha(path) == expected
        shutil.copy2(path, LIB/name)
        (LIB/name.removesuffix('.0')).symlink_to(name)
    helper = L5/'build-lastwave/lib/libac922-tail-bridge.so'
    shutil.copy2(helper, LIB/helper.name)
    closure = {}
    plans = []
    for label, name, before, relative, target, library in sources:
        if label == 'context':
            expected = old['host_source_sha256'][relative]
            original = L5/'host-src'/relative
        else:
            expected, = [x['current_sha256'] for x in recipe['source_changes']
                         if x.get('path') == relative]
            original = BASE/'src'/relative
        assert sha(original) == sha(HERE/before) == expected, relative
        pool = recipe['compile_plan'] if label == 'cuda' else recipe['source_closure_compile_plan']
        compile_recipe, = [x for x in pool if x['source'].endswith('/'+name)]
        link_recipe, = [x for x in recipe['relink_plan'] if x['target_name'] == target]
        link = shlex.split(link_recipe['rewritten_link_script_contents'])
        object_slot, = [i for i,arg in enumerate(link) if Path(arg).name == name+'.o']
        if label == 'context':
            # L1's scoped scheduler geometry is a required production delta
            # after V444. Its frozen TU and $ORIGIN/private dependency closure
            # must reproduce L5, rather than the older build-reuse object.
            link[object_slot] = str(BASE/'production/lastwave-20261018/build-lastwave/llama-context.o')
            for i,arg in enumerate(link):
                if arg.startswith('-Wl,-rpath,'):
                    link[i] = '-Wl,-rpath,$ORIGIN'
                for libname in old['libraries']:
                    if Path(arg).name == libname or Path(arg).name.startswith(libname+'.'):
                        link[i] = str(L5/'build-lastwave/lib'/libname)
        control = OUT/(library+'.control')
        link[link.index('-o')+1] = str(control)
        run(link, OUT/(label+'-control-link.log'), Path(link_recipe['cwd']))
        assert sha(control) == old['libraries'][library], label+' original object closure drifted'
        closure[library] = sha(control)
        control.unlink()
        args = list(compile_recipe['argv'])
        obj = OUT/(name+'.o')
        args[args.index('-o')+1] = str(obj)
        args[args.index('-c')+1] = str(HERE/name)
        for flag, value in [('-MF',str(obj)+'.d'),('-MT',str(obj))]:
            if flag in args:
                args[args.index(flag)+1] = value
        args += ['-I'+str(original.parent)]
        if label == 'cuda':
            args += ['--threads','4']
        plans.append(dict(label=label,name=name,before=before,relative=relative,
                          library=library,obj=obj,compile=args,link=link,
                          object_slot=object_slot,
                          compile_cwd=Path(compile_recipe['cwd']),link_cwd=Path(link_recipe['cwd'])))
    physical_kib = 0
    for node in (0,8):
        for line in Path(f'/sys/devices/system/node/node{node}/meminfo').read_text().splitlines():
            if any(label in line for label in ('MemFree:','Inactive(file):','SReclaimable:')):
                physical_kib += int(line.split()[-2])
    jobs = 2 if physical_kib >= 6*1024*1024 else 1
    assert physical_kib >= 4*1024*1024, 'insufficient physical DDR for bounded incremental compile'
    print(json.dumps(dict(status='CONTROL_RELINKS_EXACT', hashes=closure,
                          concurrent_tus=jobs, physical_ddr_gib=physical_kib/1024**2)),flush=True)
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        tasks = [executor.submit(run,p['compile'],OUT/(p['label']+'-compile.log'),p['compile_cwd']) for p in plans]
        for task in tasks:
            task.result()
    for p in plans:
        link = list(p['link'])
        link[p['object_slot']] = str(p['obj'])
        link[link.index('-o')+1] = str(LIB/p['library'])
        run(link,OUT/(p['label']+'-link.log'),p['link_cwd'])
        p['link'] = link
    changed = {'libllama.so.0','libggml-cuda.so.0'}
    for name, expected in old['libraries'].items():
        if name not in changed:
            assert sha(LIB/name) == expected
    assert sha(LIB/helper.name) == sha(helper)
    worker = L5/'build-lastwave/bank-worker'
    assert sha(worker) == old['worker_sha256']
    ldd = subprocess.check_output(['ldd',str(worker)],text=True,
                                 env=dict(os.environ,LD_LIBRARY_PATH=str(LIB)))
    assert 'not found' not in ldd
    for name in old['libraries']:
        assert name+' => '+str(LIB)+'/' in ldd, name
    (OUT/'ldd.txt').write_text(ldd)
    report = dict(status='BUILD_PASS',build_run_id='20261019-phase-prefill-direct-v1',
        baseline=str(L5),cuda_tus=1,cpp_tus=1,template_tus=0,worker_sha256=sha(worker),
        control_relink_exact_hashes=closure,concurrent_tus=jobs,
        sources={p['relative']:dict(before=sha(HERE/p['before']),after=sha(HERE/p['name'])) for p in plans},
        libraries={name:sha(LIB/name) for name in old['libraries']},
        original_r5_sha256=sha(LIB/helper.name),
        compile_argvs={p['label']:p['compile'] for p in plans},
        elapsed_s=time.monotonic()-started,production_changed=False,actual_model='PENDING')
    (OUT/'build-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['status','build_run_id','cuda_tus','cpp_tus','template_tus','elapsed_s']}),flush=True)

if __name__ == '__main__':
    main()
