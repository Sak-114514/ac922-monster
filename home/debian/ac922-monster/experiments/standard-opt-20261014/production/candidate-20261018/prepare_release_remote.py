"""Freeze a candidate release without activating services or gateway routes."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess

BASE=Path('/home/debian/ac922-monster/experiments/standard-opt-20261014')
SOURCE=BASE/'production/candidate-20261018'
VERSION='20261018-r299-banked-api-v6-r2'
ROOT=Path('/home/debian/ac922-monster/production')
RELEASE=ROOT/'releases'/VERSION
if RELEASE.exists(): raise SystemExit('Refusing to overwrite release '+str(RELEASE))
RELEASE.mkdir(parents=True)
files=['api.py','test_api.py','safety_monitor.py','wait_safety.py','verify_release.py',
       'llm_routes.py','ac922-inference.service','ac922-inference-safety.service',
       'bank-worker.cpp','production-ipc.h','production-shape-policy.h','prepare_worker.py','build-worker.sh','README.md','rollback.py','promote_gateway.py']
for name in files: shutil.copy2(SOURCE/name,RELEASE/name)
shutil.copy2(SOURCE/'build-worker-v6/bank-worker',RELEASE/'bank-worker')
shutil.copy2(BASE/'notes/V299-build.json',RELEASE/'frozen-runtime-build.json')
lib=RELEASE/'lib';lib.mkdir()
names=['libllama-common.so.0','libmtmd.so.0','libllama.so.0','libggml.so.0',
       'libggml-cpu.so.0','libggml-cuda.so.0','libggml-base.so.0']
for name in names:
    shutil.copy2(BASE/'runtime-perf-v299/lib'/name,lib/name)
    (lib/name.removesuffix('.0')).symlink_to(name)
run=ROOT/'run';out=run/'model'
(run/'model/copy0/fourbank16k-d3').mkdir(parents=True,exist_ok=True)
profile_dir=BASE/'fixtures/fourbank-d3-newruntime-v306-20261017'
shutil.copy2(profile_dir/'v221-d3-scenarios.json',out/'copy0/fourbank16k-d3/scenarios.json')
profile=json.loads((SOURCE/'v306-launch-profile.json').read_text())
env={k:str(v) for k,v in profile['environment_overrides'].items() if v is not None}
env.update(LD_LIBRARY_PATH=str(lib),ROOT_DYNAMIC_LIB_DIR=str(lib),ROOT_BUILD_REPORT=str(RELEASE/'frozen-runtime-build.json'),
           ROOT_RUNTIME_REVISION=profile['runtime']['build_id'],ROOT_CUPTI_PHASE='off',ROOT_NSYS_PREFILL_PROFILE='0',
           LD_PRELOAD='',AC922_SYNC_AUDIT='0',AC922_POOL_TOKENS='524288',AC922_MIN_GPU_FREE_MIB='256',AC922_REQUEST_TOKENS='262140')
model=Path('/home/debian/models/qwen3.8-flash-next-gsq-rco/integrated/IQ3_XXS-MTP-Q8.gguf')
argv=['/usr/bin/env','-u','LLAMA_QSA_INDEXER_TRACE','-u','LLAMA_MTP_BASE_DEVICE','-u','GGML_CUDA_ALLREDUCE',
      '-u','LLAMA_ISLAND_SPLIT','-u','GGML_META_CPU_ASYNC_COPY_BUDGET_MIB','-u','GGML_META_CPU_ASYNC_COPY_MAX_BYTES',
      '/usr/bin/numactl','--cpunodebind=0','--membind=0,8',str(RELEASE/'bank-worker'),str(model),str(out),
      str(profile_dir/'verify_v221_native4.py'),str(BASE/'fixtures/dual-context-v35-onebank4-combined-20261016/verify-v35-preclean.py')]
(RELEASE/'worker-command.json').write_text(json.dumps({'argv':argv,'env':env},indent=2)+'\n')
resolved=subprocess.run(['ldd','-r',str(RELEASE/'bank-worker')],env={**os.environ,'LD_LIBRARY_PATH':str(lib)},text=True,capture_output=True,check=True).stdout
if 'not found' in resolved or 'undefined symbol' in resolved: raise SystemExit('Release linkage failed')
for name in names:
    if name+' => '+str(lib/name) not in resolved: raise SystemExit('Unexpected runtime resolution '+name)
(RELEASE/'ldd.txt').write_text(resolved)
checks={}
for p in sorted(RELEASE.rglob('*')):
    if p.is_file(): checks[str(p.relative_to(RELEASE))]=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
external=[profile_dir/'verify_v221_native4.py',profile_dir/'v146-scenarios-source.json',
          BASE/'fixtures/layer6-fourbank-v146-20261017/verify_v146_native4.py',
          BASE/'fixtures/dual-context-v35-onebank4-combined-20261016/native-semantic/case-input.json',
          BASE/'fixtures/dual-context-v35-onebank4-combined-20261016/native-semantic/run_semantic_native.py',
          BASE/'notes/inband-guard-v222-20261017/model_guard.py',BASE/'notes/inband-guard-v222-20261017/guard_policy.py',
          BASE/'notes/inband-guard-v222-20261017/sdr.cache']
for p in external: checks[str(p)]=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
stat=model.stat()
manifest={'version':VERSION,'status':'FROZEN_NOT_PROMOTED','sha256':checks,
          'model_identity':[{'path':str(model),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}],
          'capacity_candidate':{'max_context':262140,'pool_cells':524288,'max_parallel':4},
          'baseline_runtime':'R299','gateway_mutated':False}
(RELEASE/'release-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'status':'FROZEN_NOT_PROMOTED','release':str(RELEASE),'sha_count':len(checks),'gpu_touched':False}))
