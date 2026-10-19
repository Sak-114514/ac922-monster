import asyncio,json,time
from pathlib import Path
import httpx

OUT=Path('/home/debian/ac922-monster/production/run/decode-pool-smoke-result.json')
BASE='http://127.0.0.1:8001';MODEL='flash-next-iq3-xxs'
async def main():
    result={'started':time.time(),'status':'RUNNING'};OUT.write_text(json.dumps(result))
    async with httpx.AsyncClient(timeout=600) as client:
        async def good():
            r=await client.post(BASE+'/v1/chat/completions',json={'model':MODEL,'messages':[{'role':'user','content':'What is the capital of France? Reply with exactly the city name.'}],'max_tokens':16})
            assert r.status_code==200,r.text
            d=r.json();assert 'Paris' in d['choices'][0]['message']['content'];return d
        async def bad():
            return await client.post(BASE+'/v1/completions',json={'model':MODEL,'prompt':'z '*270000,'max_tokens':1})
        invalid,valid=await asyncio.gather(bad(),good());assert invalid.status_code==400,(invalid.status_code,invalid.text[:200])
        result['oversized_isolation']={'invalid_status':invalid.status_code,'valid':valid}
        topics=['how CPU caches and cache coherency work','photosynthesis and plant metabolism','Roman architectural engineering','comparison sorting algorithms and complexity']
        async def run(i,topic):
            prompt=f'Write a coherent technical explanation of {topic}, at least 2000 words. Begin with exactly BEGIN_{i} then continue with the explanation. Do not end early.'
            start=time.monotonic()
            r=await client.post(BASE+'/v1/chat/completions',json={'model':MODEL,'messages':[{'role':'user','content':prompt}],'max_tokens':512,'temperature':0})
            assert r.status_code==200,r.text[:500]
            d=r.json();text=d['choices'][0]['message']['content']
            assert text.startswith(f'BEGIN_{i}'),text[:300]
            assert d['usage']['completion_tokens']>=480,d['usage']
            assert 'военно' not in text and '!!!!' not in text
            return {'usage':d['usage'],'finish_reason':d['choices'][0]['finish_reason'],'text':text,'wall_s':time.monotonic()-start}
        start=time.monotonic();rows=await asyncio.gather(*(run(i,t) for i,t in enumerate(topics)))
        result['four_512']={'wall_s':time.monotonic()-start,'results':rows}
        result['followup']=await good();result['status']='RETAINED_LARGE_CONTEXT_FOUR512_AND_ISOLATION_PASS'
    result['finished']=time.time();OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False),flush=True)
asyncio.run(main())
