"""Same loaded worker: cold fallback, mixed geometry, cached prose and raw TPS."""
import json
from pathlib import Path
import affinity_acceptance as a
import lastwave_inband_acceptance as h

HERE=Path(__file__).resolve().parent
h.OUT=HERE/'run/single-cache-result.json'
h.report={'status':'RUNNING','waves':[],'purpose':'conditional singleton cache reuses existing geometry; cold path unchanged'}
cases=json.loads(h.CASES.read_text())[0]['requests']
body=a.make_corpus_body(a.load_corpus(a.DEFAULT_CORPUS))
doc=a.request_for(a.DOCUMENTS[3],body)
base={k:v for k,v in doc.items() if not k.startswith('_')}

def native(label):
 r=h.wave(label,[{'id':c['id'],'prompt':c['prompt'],'max_tokens':c['max_tokens']} for c in cases])
 for c in cases:
  text=r['texts'][c['id']].strip()
  assert json.loads(text)==c['expected'] if c['checker']=='strict_json' else text==str(c['expected']),(c['id'],text)
 r['semantic_status']='PASS';h.save()

try:
 raw=h.wave('c3-raw32k-cold-fallback',[{'id':'raw32k','corpus_token_start':0,'corpus_token_count':32768,'max_tokens':8,'ignore_eos':True}])
 assert raw['timings']['cached_tokens']==0 and not raw['timings']['single_cached_affinity']
 assert raw['accepted']['raw32k']['bank']==0 and raw['timings']['banks'][0]['target_ubatch']==1024
 geometry=raw['timings']['banks']
 def cached(label,request):
  r=h.wave(label,[request],cache_prompt=True);rid=request['id']
  assert r['accepted'][rid]['bank']==3 and r['cache_status'][rid]['cached_tokens']>0
  t=r['timings'];assert t['single_cached_affinity'] and t['context_rebuilds_before_execution']==0
  assert t['banks']==geometry and t['post_wave_reclaimed_banks']==0
  return r
 r=cached('c3-mixed-geometry-single-d',dict(base,id='mixed-doc-d'))
 assert r['texts']['mixed-doc-d'].strip()=='9274';r['semantic_status']='PASS';h.save()
 prefix=base['prompt'].rsplit('QUESTION:',1)[0]
 essay={'id':'cached-essay','prompt':prefix+'QUESTION: Write a coherent detailed technical explanation of CPU caches and cache coherency, at least 2000 words. Begin exactly BEGIN_CACHE_D then continue. Do not end early.','max_tokens':512}
 r=cached('c3-cached-single-512-prose',essay)
 assert r['texts']['cached-essay'].startswith('BEGIN_CACHE_D') and r['done']['cached-essay']['output_tokens']==512
 r['semantic_status']='FULL_TEXT_REVIEW_REQUIRED';h.save()
 r=cached('c3-single-d-after-prose',dict(base,id='doc-d-after-prose'))
 assert r['texts']['doc-d-after-prose'].strip()=='9274';r['semantic_status']='PASS';h.save()
 large=dict(base,id='capacity-fallback-d',max_tokens=8192)
 r=h.wave('c3-single-capacity-fallback',[large],cache_prompt=True)
 assert not r['timings']['single_cached_affinity'] and r['accepted'][large['id']]['bank']==0
 assert r['cache_status'][large['id']]['cached_tokens']==0 and r['texts'][large['id']].strip()=='9274'
 assert r['timings']['banks'][0]['target_ubatch']==1024;r['semantic_status']='PASS';h.save()
 native('c3-native4-after-single-profile-changes')
 topics=['how CPU caches and cache coherency work','photosynthesis and plant metabolism','Roman architectural engineering','comparison sorting algorithms and complexity']
 stories=[{'id':f'story-{i}','prompt':f'Write a coherent technical explanation of {topic}, at least 2000 words. Begin with exactly BEGIN_{i} then continue with the explanation. Do not end early.','max_tokens':512} for i,topic in enumerate(topics)]
 r=h.wave('c3-four512',stories)
 for i in range(4):assert r['texts'][f'story-{i}'].startswith(f'BEGIN_{i}') and r['done'][f'story-{i}']['output_tokens']==512
 r['semantic_status']='FULL_TEXT_REVIEW_REQUIRED';h.save()
 native('c3-native4-final')
 h.report['status']='PASS_PENDING_FULL_TEXT_REVIEW'
except BaseException as error:
 h.report['status']='FAIL';h.report['error']=repr(error);raise
finally:h.save()
