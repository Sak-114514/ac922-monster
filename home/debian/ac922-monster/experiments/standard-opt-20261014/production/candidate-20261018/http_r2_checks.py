import asyncio,json,re,time
from pathlib import Path
import httpx
BASE=Path('/home/debian/ac922-monster/experiments/standard-opt-20261014')
ROOT=Path('/home/debian/ac922-monster/production')
URL='http://127.0.0.1:8001';MODEL='flash-next-iq3-xxs'
report={'status':'WAITING_READY','started':time.time()}
OUT=ROOT/'run/http-r2-checks.json'
def save():OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
def marker():
    log=(ROOT/'run/worker.log').read_text(errors='replace')
    lines=[line for line in log.splitlines() if line.startswith('production-context-repartition ')]
    return lines[-1] if lines else ''
async def main():
    async with httpx.AsyncClient(timeout=1800) as client:
        for _ in range(300):
            try:
                r=await client.get(URL+'/ready')
                if r.status_code==200:report['ready']=r.json();break
            except httpx.RequestError:pass
            await asyncio.sleep(2)
        else:raise TimeoutError('readiness')
        report['status']='RUNNING';save()
        corpus=json.loads((BASE/'fixtures/long-prefill-combo-v45-20261016/v45-real-corpus-source.json').read_text())
        source='\n\n'.join(s['text'] for s in corpus['corpus']['segments'])
        targets=[210000,124000,87000,63000]
        prompts=[]
        for target in targets:
            chars=int(target*len(source)/92037)
            text=((source+'\n')*(chars//len(source)+1))[:chars]
            prompts.append('Read the following reference and then summarize it in one sentence.\n'+text)
        old=marker();close=asyncio.Event();opened=asyncio.Queue();heartbeats=[]
        async def cancel_after_allocation(i):
            async with client.stream('POST',URL+'/v1/chat/completions',json={'model':MODEL,'messages':[{'role':'user','content':prompts[i]}],'max_tokens':32,'stream':True}) as r:
                assert r.status_code==200,r.status_code
                async for line in r.aiter_lines():
                    if line.startswith('data: '):
                        await opened.put(i)
                        await close.wait()
                    elif line.startswith(': keep-alive'):
                        heartbeats.append(i)
                        return
        tasks=[asyncio.create_task(cancel_after_allocation(i)) for i in range(4)]
        for _ in range(4):await asyncio.wait_for(opened.get(),30)
        deadline=time.monotonic()+60
        while time.monotonic()<deadline:
            now=marker()
            if now and now!=old:
                values=[int(v) for v in re.search(r'bank_contexts=([0-9,]+)',now)[1].strip(',').split(',')]
                if sum(values)>400000:
                    report['allocated_context_marker']=now;break
            await asyncio.sleep(.2)
        else:raise TimeoutError('large asymmetric allocation marker')
        await asyncio.sleep(16)
        close.set();await asyncio.wait_for(asyncio.gather(*tasks),20)
        assert sorted(heartbeats)==[0,1,2,3],heartbeats
        report['real_quiet_sse_heartbeats']=heartbeats;save()
        async def good(i):
            r=await client.post(URL+'/v1/chat/completions',json={'model':MODEL,'messages':[{'role':'user','content':f'What is {i}+10? Reply with exactly the integer.'}],'max_tokens':16})
            assert r.status_code==200,r.text[:200]
            d=r.json();assert d['choices'][0]['message']['content'].strip()==str(i+10),d
            return d['usage']
        report['post_cancel_reuse']=await asyncio.gather(*(good(i) for i in range(4)))
        now=marker();assert 'bank_contexts=17152,17152,17152,17152,' in now,now
        report['shrink_marker']=now;save()
    process=await asyncio.create_subprocess_exec('/home/debian/comfyui/venv/bin/python',str(BASE/'production/candidate-20261018/decode_pool_smoke.py'),stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
    stdout,stderr=await process.communicate()
    assert process.returncode==0,stderr.decode()[-1000:]
    report['four512']=json.loads((ROOT/'run/decode-pool-smoke-result.json').read_text())
    report['status']='RESTART_NATIVE_GATE_PREFILL_CANCEL_SHRINK_REUSE_FOUR512_PASS';report['finished']=time.time();save()
    print(json.dumps({'status':report['status'],'ready':report['ready'],'shrink':report['shrink_marker'],'four512_wall_s':report['four512']['four_512']['wall_s']}),flush=True)
try:asyncio.run(main())
except BaseException as e:
    report['status']='FAIL';report['error']=repr(e);save();raise
