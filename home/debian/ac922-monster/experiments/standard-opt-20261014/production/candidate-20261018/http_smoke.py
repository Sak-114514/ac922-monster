"""Real-model loopback API acceptance; prompts contain test data only."""
import asyncio,json,time
from pathlib import Path
import httpx

BASE=Path('/home/debian/ac922-monster/experiments/standard-opt-20261014')
OUT=Path('/home/debian/ac922-monster/production/run/http-smoke-result.json')
MODEL='flash-next-iq3-xxs'
URL='http://127.0.0.1:8001'
report={'started':time.time(),'status':'WAITING_READY','tests':[]}
def save():OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')

async def request(client,prompt,limit=32):
    start=time.monotonic()
    response=await client.post(URL+'/v1/chat/completions',json={'model':MODEL,'messages':[{'role':'user','content':prompt}],'max_tokens':limit,'temperature':0})
    if response.status_code!=200: raise RuntimeError(str((response.status_code,response.text[:500])))
    body=response.json()
    return {'text':body['choices'][0]['message']['content'],'usage':body['usage'],'wall_s':time.monotonic()-start}

async def main():
    async with httpx.AsyncClient(timeout=1810) as client:
        for _ in range(300):
            try:
                r=await client.get(URL+'/ready')
                if r.status_code==200:report['ready']=r.json();break
            except httpx.RequestError:pass
            await asyncio.sleep(2)
        else:raise TimeoutError('model readiness')
        report['status']='RUNNING';save();print(json.dumps({'ready':report['ready']}),flush=True)
        q=[('What is the capital city of France? Reply with exactly the city name.','Paris',32),
           ('What is 2847+6958? Reply with exactly the integer.','9805',16),
           ('Reply with this JSON object only: {"ok":true,"count":3}','{"ok": true, "count": 3}',48),
           ('What is the capital city of Japan? Reply with exactly the city name.','Tokyo',24)]
        rows=await asyncio.gather(*(request(client,p,limit) for p,_,limit in q))
        for i,row in enumerate(rows):
            if i==2:assert json.loads(row['text'])=={'ok':True,'count':3},row
            else:assert q[i][1] in row['text'],row
        report['tests'].append({'name':'four_asymmetric_short','results':rows});save()
        stream_start=time.monotonic();chunks=[]
        async with client.stream('POST',URL+'/v1/chat/completions',json={'model':MODEL,'messages':[{'role':'user','content':'Write a long detailed history of computers, including many examples and dates.'}],'max_tokens':2048,'temperature':0,'stream':True}) as r:
            assert r.status_code==200
            async for line in r.aiter_lines():
                if line.startswith('data: ') and line!='data: [DONE]':
                    payload=json.loads(line[6:]);delta=payload.get('choices',[{}])[0].get('delta',{})
                    if delta.get('content'):
                        chunks.append(delta['content']);break
        await asyncio.sleep(.5)
        reuse=await request(client,q[0][0]);assert 'Paris' in reuse['text']
        report['tests'].append({'name':'real_sse_disconnect_cancel_reuse','first_content_s':time.monotonic()-stream_start,'sample':chunks,'reuse':reuse});save()
        wrong=await client.post(URL+'/v1/completions',json={'model':'wrong','prompt':'test','max_tokens':1})
        assert wrong.status_code==400
        unsupported=await client.post(URL+'/v1/completions',json={'model':MODEL,'prompt':'test','temperature':.7})
        assert unsupported.status_code==400
        report['tests'].append({'name':'model_and_sampling_rejection','status':'PASS'});save()
        corpus=json.loads((BASE/'fixtures/long-prefill-combo-v45-20261016/v45-real-corpus-source.json').read_text())
        source='\n\n'.join(s['text'] for s in corpus['corpus']['segments'])
        targets=[214000,125000,88000,64000]
        codes=['ALPHA728419','BETA935217','GAMMA461827','DELTA381659']
        prompts=[]
        for target,code in zip(targets,codes):
            chars=int(target*len(source)/92037)
            text=(source+'\n')*(chars//len(source)+1)
            text=text[:chars]
            prompts.append('Verification code for this request: '+code+'. Preserve this code.\nREFERENCE MATERIAL:\n'+text+'\nEND REFERENCE MATERIAL.\nFINAL QUESTION: What is the verification code for this request? Return exactly the code and no explanation.')
        start=time.monotonic()
        rows=await asyncio.gather(*(request(client,p,32) for p in prompts))
        total=sum(row['usage']['prompt_tokens'] for row in rows)
        for row,code in zip(rows,codes):assert row['text'].strip()==code,(code,row)
        assert rows[0]['usage']['prompt_tokens']>=200000
        assert total>=480000,(total,rows)
        report['tests'].append({'name':'real_four_asymmetric_long_sharedpool','wall_s':time.monotonic()-start,'total_prompt_tokens':total,'results':rows});save()
        follow=await request(client,'请计算 7+4，只输出整数。');assert follow['text'].strip()=='11',follow
        report['tests'].append({'name':'post_long_reuse','result':follow})
        report['status']='HTTP_REAL_MODEL_QUALITY_STREAM_CANCEL_LONG4_PASS';save()
        print(json.dumps(report,ensure_ascii=False),flush=True)

try:asyncio.run(main())
except BaseException as error:
    report['status']='FAIL';report['error']=repr(error);raise
finally:
    report['finished']=time.time();save()
