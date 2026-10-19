import hashlib, json, os, queue, subprocess, sys, threading, time
from pathlib import Path

BASE=Path('/home/debian/ac922-monster/experiments/standard-opt-20261014')
HERE=BASE/'production/combo-20261018'
OUT=Path(sys.argv[1])
OUT.mkdir(parents=True,exist_ok=True)
rd,wr=os.pipe()
env=dict(os.environ,AC922_API_EVENT_FD=str(wr),AC922_POOL_TOKENS='524288',AC922_MIN_GPU_FREE_MIB='256',AC922_REQUEST_TOKENS='262140')
profile=BASE/'fixtures/fourbank-d3-newruntime-v306-20261017'
argv=[str(HERE/'build-combo-c10/bank-worker'),'/home/debian/models/qwen3.8-flash-next-gsq-rco/integrated/IQ3_XXS-MTP-Q8.gguf',str(OUT),str(HERE/'verify_combo_native.py'),str(BASE/'fixtures/dual-context-v35-onebank4-combined-20261016/verify-v35-preclean.py')]
log=(OUT/'runtime.log').open('w')
p=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=log,stderr=log,pass_fds=(wr,),env=env,text=True)
os.close(wr)
events=queue.Queue()
def reader():
    try:
        with os.fdopen(rd) as f:
            for line in f:
                events.put(json.loads(line))
    finally: events.put({'type':'eof'})
threading.Thread(target=reader,daemon=True).start()
report={'started':time.time(),'worker_pid':p.pid,'waves':[],'status':'RUNNING','pool_claim':'candidate; allocation and long workload under test'}
report['artifacts'] = {name:hashlib.sha256((HERE/rel).read_bytes()).hexdigest() for name,rel in {
    'worker_sha256':'build-combo-c10/bank-worker','source_sha256':'bank-worker-combo.cpp',
    'environment_sha256':'combo-environment.json','thermal_header_sha256':'thermal-pacing.h',
    'thermal_policy_sha256':'combo_temperature_policy.py'}.items()}
report['runtime_build_id']='20261018T092503.472988Z'
report['thermal_policy']={'gpu_stop_c':65,'pause_c':59,'resume_c':54}
def save(): (OUT/'worker-smoke-result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
def send(obj):
    p.stdin.write(json.dumps(obj,ensure_ascii=False)+'\n');p.stdin.flush()
def wave(wid,requests,cancel_after=None):
    start=time.monotonic(); send({'type':'wave','id':wid,'requests':requests})
    output={r['id']:bytearray() for r in requests}; record={'id':wid,'accepted':{},'done':{},'errors':[], 'first_token_s':{}}
    while time.monotonic()-start<1500:
        try: e=events.get(timeout=20)
        except queue.Empty:
            if p.poll() is not None: raise RuntimeError('worker died '+str(p.returncode))
            continue
        kind=e['type']; rid=e.get('id')
        if kind=='accepted': record['accepted'][rid]=e; print(json.dumps(e),flush=True)
        elif kind=='token':
            record['first_token_s'].setdefault(rid,time.monotonic()-start)
            output[rid].extend(e['piece_bytes'])
            if cancel_after and len(output[rid])>=cancel_after:
                send({'type':'cancel','id':rid});cancel_after=None
        elif kind=='done': record['done'][rid]=e
        elif kind=='error': record['errors'].append(e)
        elif kind=='wave_done':
            assert rid==wid
            record['timings']=e.get('timings',{})
            record['wall_s']=time.monotonic()-start
            record['texts']={k:bytes(v).decode('utf-8',errors='replace') for k,v in output.items()}
            report['waves'].append(record);save();print(json.dumps({'wave':wid,'wall_s':record['wall_s'],'done':record['done'],'errors':record['errors']},ensure_ascii=False),flush=True)
            return record
        elif kind=='eof': raise RuntimeError('worker event EOF')
    raise TimeoutError(wid)
try:
    while True:
        e=events.get(timeout=600)
        if e['type']=='eof': raise RuntimeError('worker failed during native boot gate')
        if e['type']=='ready': report['ready']=e;save();print(json.dumps(e),flush=True);break
    native_cases=json.loads((profile/'v221-d3-scenarios.json').read_text())[0]['requests']
    def cool():
        began=time.monotonic()
        while time.monotonic()-began<120:
            raw=subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],text=True)
            if max(int(v) for v in raw.split())<=54:return
            time.sleep(1)
        raise TimeoutError('pre-case cooldown')
    paris={'id':'combo-paris-8192','prompt':native_cases[0]['prompt'],'semantic_pad_to_tokens':8192,'max_tokens':24}
    def native_wave(label):
        r=wave(label,[{'id':v['id'],'prompt':v['prompt'],'max_tokens':v['max_tokens']} for v in native_cases])
        assert not r['errors'],r
        for v in native_cases:
            text=r['texts'][v['id']].strip()
            if v['checker']=='strict_json':assert json.loads(text)==v['expected'],(v['id'],text)
            else:assert text==str(v['expected']),(v['id'],text)
        return r
    topics=['how CPU caches and cache coherency work','photosynthesis and plant metabolism','Roman architectural engineering','comparison sorting algorithms and complexity']
    requests=[{'id':f'story-{i}', 'prompt':f'Write a coherent technical explanation of {topic}, at least 2000 words. Begin with exactly BEGIN_{i} then continue with the explanation. Do not end early.', 'max_tokens':512} for i,topic in enumerate(topics)]
    caps=[240128,129792,88064,66304]
    caprequests=[{**v,'id':'cap-'+v['id'],'max_tokens':64,'capacity_tokens':caps[i]-4} for i,v in enumerate(requests)]
    r=wave('combo-524288-capacity-multi',caprequests);assert not r['errors'],r
    for i,v in enumerate(caprequests):assert r['texts'][v['id']].startswith(f'BEGIN_{i}'),r['texts']
    native_wave('combo-native4-after-large')
    cool()
    large={**paris,'id':'combo-max-single','semantic_pad_to_tokens':65536,'capacity_tokens':262140}
    r=wave('combo-max-single-UB512',[large]);assert not r['errors'] and r['texts'][large['id']].strip()=='Paris',r
    cool()
    formal={'id':'combo-corpus32k','corpus_token_start':0,'corpus_token_count':32768,'max_tokens':8,'ignore_eos':True}
    for label in ('cold','warm'):
        r=wave('combo-formal32k-'+label,[formal]);assert not r['errors'] and r['timings']['prefill_input_tokens']==32768,r
        assert r['done'][formal['id']]['output_tokens']==8,r
        cool()
    r=wave('combo-quality8192-after-cache',[paris]);assert r['texts'][paris['id']].strip()=='Paris',r
    native_wave('combo-native4-after-prefill')
    r=wave('combo-four512',requests);assert not r['errors'],r
    for i,v in enumerate(requests):
        assert r['texts'][v['id']].startswith(f'BEGIN_{i}'),r['texts'][v['id']][:200]
        assert r['done'][v['id']]['output_tokens']==512,r['done']
    native_wave('combo-native4-final')
    report['status']='COMBO_NATIVE_PREFILL32K_D3_FOUR512_CAPACITY_PASS'
    send({'type':'shutdown'})
    assert p.wait(timeout=60)==0
except BaseException as e:
    report['status']='FAIL';report['error']=repr(e)
    if p.poll() is None:
        p.terminate()
        try:p.wait(timeout=20)
        except subprocess.TimeoutExpired:p.kill();p.wait(timeout=20)
    raise
finally:
    report['finished']=time.time();report['worker_exit']=p.poll();save();log.close()
