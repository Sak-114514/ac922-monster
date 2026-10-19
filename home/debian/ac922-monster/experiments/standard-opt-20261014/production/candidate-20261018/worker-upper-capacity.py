import json, os, queue, subprocess, sys, threading, time
from pathlib import Path

BASE=Path('/home/debian/ac922-monster/experiments/standard-opt-20261014')
HERE=BASE/'production/candidate-20261018'
OUT=Path(sys.argv[1])
OUT.mkdir(parents=True,exist_ok=True)
rd,wr=os.pipe()
env=dict(os.environ,AC922_API_EVENT_FD=str(wr),AC922_POOL_TOKENS='1048576',AC922_MIN_GPU_FREE_MIB='256',AC922_REQUEST_TOKENS='262140')
profile=BASE/'fixtures/fourbank-d3-newruntime-v306-20261017'
argv=[str(HERE/'build-worker-v6/bank-worker'),'/home/debian/models/qwen3.8-flash-next-gsq-rco/integrated/IQ3_XXS-MTP-Q8.gguf',str(OUT),str(profile/'verify_v221_native4.py'),str(BASE/'fixtures/dual-context-v35-onebank4-combined-20261016/verify-v35-preclean.py')]
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
    topics=['how CPU caches and cache coherency work','photosynthesis and plant metabolism','Roman architectural engineering','comparison sorting algorithms and complexity']
    levels=[[262144,131072,65536,65536], [262144,262144,65536,65536],
            [262144,262144,131072,65536], [262144,262144,131072,131072],
            [262144,262144,196608,131072], [262144,262144,196608,196608],
            [262144,262144,262144,196608], [262144,262144,262144,262144]]
    for caps in levels:
        requests=[{'id':f'cap-{sum(caps)}-{i}',
            'prompt':f'Write a coherent technical explanation of {topic}, at least 2000 words. Begin with exactly BEGIN_{i} then continue with the explanation. Do not end early.',
            'max_tokens':512,'capacity_tokens':caps[i]-4} for i,topic in enumerate(topics)]
        result=wave('capacity-'+str(sum(caps)),requests)
        assert not result['errors'],result['errors']
        for i,r in enumerate(requests):
            assert result['texts'][r['id']].startswith(f'BEGIN_{i}'),result['texts'][r['id']][:200]
            assert result['done'][r['id']]['output_tokens']>=480,result['done']
        report['highest_pass_pool_cells']=sum(caps)
        report['highest_pass_shape']=caps
        save()
        sample=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,temperature.gpu','--format=csv,noheader,nounits'],text=True)
        values=[[int(v.strip()) for v in line.split(',')] for line in sample.splitlines()]
        report.setdefault('steady_samples',[]).append({'pool':sum(caps),'gpu':values});save()
        if any(free<384 for free,temp in values):
            report['stop_reason']='steady reserve below384MiB after successful output; keep prior qualified pool for production'
            break
    report['status']='UPPER_CAPACITY_SWEEP_FINISHED'
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
