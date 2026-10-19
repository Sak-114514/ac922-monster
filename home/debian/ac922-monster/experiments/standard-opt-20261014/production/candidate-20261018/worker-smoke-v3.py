import json, os, queue, subprocess, sys, threading, time
from pathlib import Path

BASE=Path('/home/debian/ac922-monster/experiments/standard-opt-20261014')
HERE=BASE/'production/candidate-20261018'
OUT=Path(sys.argv[1])
OUT.mkdir(parents=True,exist_ok=True)
rd,wr=os.pipe()
env=dict(os.environ,AC922_API_EVENT_FD=str(wr),AC922_POOL_TOKENS='393216',AC922_REQUEST_TOKENS='262140')
profile=BASE/'fixtures/fourbank-d3-newruntime-v306-20261017'
argv=[str(HERE/'build-worker-v2/bank-worker'),'/home/debian/models/qwen3.8-flash-next-gsq-rco/integrated/IQ3_XXS-MTP-Q8.gguf',str(OUT),str(profile/'verify_v221_native4.py'),str(BASE/'fixtures/dual-context-v35-onebank4-combined-20261016/verify-v35-preclean.py')]
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
            report['waves'].append(record);save();print(json.dumps({'wave':wid,'wall_s':record['wall_s'],'done':record['done'],'errors':record['errors'],'texts':record['texts']},ensure_ascii=False),flush=True)
            return record
        elif kind=='eof': raise RuntimeError('worker event EOF')
    raise TimeoutError(wid)
try:
    while True:
        e=events.get(timeout=600)
        if e['type']=='eof': raise RuntimeError('worker failed during native boot gate')
        if e['type']=='ready': report['ready']=e;save();print(json.dumps(e),flush=True);break
    qs=[('a','What is the capital city of France? Reply with exactly the city name.',32,'Paris'),('b','What is 7 + 4? Reply with exactly the integer.',16,'11'),('c','What is the capital city of Japan? Reply with exactly the city name.',48,'Tokyo'),('d','What is the capital city of Germany? Reply with exactly the city name.',24,'Berlin')]
    r=wave('short-asymmetric',[{'id':i,'prompt':q,'max_tokens':n} for i,q,n,_ in qs])
    assert not r['errors']
    for i,_,_,expected in qs: assert expected in r['texts'][i], (i,r['texts'][i])
    caps=[262140,65532,32764,32764]
    r=wave('capacity-393216',[{'id':'cap'+str(i),'prompt':q,'max_tokens':n,'capacity_tokens':caps[i]} for i,(_,q,n,_) in enumerate(qs)])
    assert not r['errors'],r
    for i,(_,_,_,expected) in enumerate(qs): assert expected in r['texts']['cap'+str(i)]
    report['allocation_probe']='PASS_393216_Q8_target_and_draft_cells_with_512MiB_guard; not full-occupancy workload proof';save()
    r=wave('cancel',[{'id':'cancel','prompt':'Write a very long detailed history of computing and continue until you reach the output limit.','max_tokens':2048}],cancel_after=30)
    assert r['done']['cancel']['finish_reason']=='cancelled',r
    r=wave('reuse',[{'id':'reuse','prompt':qs[0][1],'max_tokens':32}])
    assert 'Paris' in r['texts']['reuse'] and not r['errors']
    corpus=json.loads(Path(env['ROOT_V131_CORPUS_SOURCE']).read_text())
    text='\n\n'.join(s['text'] for s in corpus['corpus']['segments'])
    long=text+'\n'+text+'\n'+text[:len(text)//3]
    prompt='The following is reference material. Read it, then answer only the final question.\n'+long+'\nFINAL QUESTION: What is the capital city of France? Reply with exactly the city name, no explanation.'
    r=wave('long-over200k',[{'id':'long','prompt':prompt,'max_tokens':32}])
    assert not r['errors'],r
    assert r['accepted']['long']['input_tokens']>=200000,r['accepted']
    assert 'Paris' in r['texts']['long'],r['texts']
    report['long_input_tokens']=r['accepted']['long']['input_tokens']
    report['status']='NATIVE4_ASYMMETRIC_CAPACITY_CANCEL_REUSE_LONG_PASS'
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
