"""One load: long multi prefill, independent decode, affinity, and capacity gates."""
from pathlib import Path
import hashlib
import json
import socket
import subprocess
import time
from urllib.request import urlopen

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
SOCKET='/home/debian/ac922-monster/production/bulk256-c2-control.sock'
OUT=HERE/'run/qualification.json'
report=dict(status='WAITING_FOR_READY',started_unix=time.time(),waves=[],model_loads_requested=1,
            quality_reduction_requested=False,mtp_depth=3,throughput_is_not_a_semantic_gate=True)

def save():
    OUT.parent.mkdir(parents=True,exist_ok=True)
    temp=OUT.with_suffix('.tmp')
    temp.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    temp.replace(OUT)

def wave(label,requests):
    with socket.socket(socket.AF_UNIX) as connection:
        connection.settimeout(360)
        connection.connect(SOCKET)
        connection.sendall(json.dumps(dict(id='bulk256-c2-'+label+'-c1',requests=requests,refill=False)).encode()+b'\n')
        result=json.loads(connection.makefile('rb').readline(12<<20))
    report['waves'].append(result)
    save()
    assert result.get('status')!='ERROR',result.get('error')
    assert not result.get('errors') and len(result.get('done',{}))==len(requests),result
    if len(report['waves'])>1:
        assert result['worker_pid']==report['waves'][0]['worker_pid']
    if not any(x.get('cache_prompt',False) for x in requests):
        assert result['timings']['cached_tokens']==0
    t=result['timings']
    print(json.dumps(dict(label=label,worker_pid=result['worker_pid'],
        prefill_tok_s=t.get('prefill_aggregate_wall_tok_s'),actual_input=t.get('prefill_input_tokens'),
        cached_tokens=t.get('cached_tokens'),banks=t.get('banks'),
        rebuilds=t.get('context_rebuilds_before_execution'),reclaimed=t.get('post_wave_reclaimed_banks'),
        all_emissions_tok_s=t.get('all_emissions_tok_s'))),flush=True)
    return result

def check_cases(result,cases,ids=None):
    for i,c in enumerate(cases):
        key=ids[i] if ids else c['id']
        text=result['texts'][key].strip()
        assert json.loads(text)==c['expected'] if c['checker']=='strict_json' else text==str(c['expected']),(key,text)

def native_record(result):
    matches=[]
    for path in (HERE/'run/model/copy0/fourbank16k-d3').glob('lastwave-*.json'):
        value=json.loads(path.read_text())
        if value.get('wave_id')==result['id']:
            matches.append((path,value))
    assert len(matches)==1,'missing unique full-output record'
    return matches[0]

def main():
    save()
    deadline=time.monotonic()+900
    while True:
        try:
            with urlopen('http://127.0.0.1:8018/ready',timeout=3) as response:
                ready=json.load(response)
            if ready.get('status')=='ready':
                report['ready']=ready
                break
        except Exception:
            pass
        state=subprocess.check_output(['systemctl','show','ac922-inference.service','-p','ActiveState','--value'],text=True).strip()
        if state in ('failed','inactive') or time.monotonic()>deadline:
            raise RuntimeError('private model service failed or timed out before readiness: '+state)
        time.sleep(2)
    report['status']='RUNNING'
    save()
    cases=json.loads((BASE/'fixtures/fourbank-d3-newruntime-v306-20261017/v221-d3-scenarios.json').read_text())[0]['requests']
    long_requests=[dict(id=c['id'],prompt=c['prompt'],max_tokens=c['max_tokens'],semantic_pad_to_tokens=8192,cache_prompt=False) for c in cases]
    result=wave('fourlong-native',long_requests)
    check_cases(result,cases)
    assert result['timings']['prefill_input_tokens']==32768
    assert all(b['target_ubatch']==256 for b in result['timings']['banks'])
    report['bulk_long_native']='PASS'
    save()

    baseline_path=HERE/'reference-v306.json'
    assert hashlib.sha256(baseline_path.read_bytes()).hexdigest()=='de1c2788e2ad6961c7c036b5dd11d30a61853b3190bcfe6e3a2c4874b2d005ff'
    baseline=json.loads(baseline_path.read_text())[1]
    assert baseline['label']=='v221-d3-fourbank-4x16k-512'
    requests=[dict(id=f'v221-d3-flow-{i}',corpus_token_start=i*16384,corpus_token_count=16384,max_tokens=512,ignore_eos=True,cache_prompt=False) for i in range(4)]
    result=wave('aligned-four16k512',requests)
    assert result['timings']['prefill_input_tokens']==65536
    assert all(result['done'][r['id']]['output_tokens']==512 for r in requests)
    assert all(b['target_ubatch']==256 for b in result['timings']['banks'])
    path,full=native_record(result)
    boot=json.loads((HERE/'run/model/copy0/fourbank16k-d3/results.json').read_text())[0]
    assert boot['v45_source_corpus_sha256']==baseline['v45_source_corpus_sha256']
    old={f['id']:f for f in baseline['flows']}
    report['aligned']=dict(native_record=str(path),baseline_all_emissions=baseline['all_emissions_tok_s'],
        actual_all_emissions=full['all_emissions_tok_s'],actual_prefill=full['prefill_aggregate_wall_tok_s'],
        parity={f['id']:dict(text_exact=f['output']==old[f['id']]['output'],token_ids_exact=f['token_ids']==old[f['id']]['token_ids']) for f in full['flows']},
        proposed=sum(f['proposed'] for f in full['flows']),accepted=sum(f['accepted'] for f in full['flows']))
    save()

    # Long enough unique marker prefixes make per-bank affinity observable.
    prefix=[dict(id='prefix-'+str(i),prompt='Ignore the reference markers. '+(('reference_CASE'+str(i)+' ')*900)+'\nAnswer only the following question:\n'+c['prompt'],max_tokens=c['max_tokens'],semantic_pad_to_tokens=8192,cache_prompt=True) for i,c in enumerate(cases)]
    filled=wave('prefix-fill',prefix)
    check_cases(filled,cases,[r['id'] for r in prefix])
    order=[2,0,3,1]
    hit=wave('prefix-reordered-hit',[prefix[i] for i in order])
    check_cases(hit,[cases[i] for i in order],[prefix[i]['id'] for i in order])
    assert hit['timings']['cached_tokens']>0
    assert hit['timings']['context_rebuilds_before_execution']==0
    report['prefix_affinity_reordered']=dict(status='PASS',cached_tokens=hit['timings']['cached_tokens'],
        filled_banks={key:v['bank'] for key,v in filled['done'].items()},
        hit_banks={key:v['bank'] for key,v in hit['done'].items()})
    assert report['prefix_affinity_reordered']['filled_banks']==report['prefix_affinity_reordered']['hit_banks']
    save()

    paris=dict(id='single32k',prompt=cases[0]['prompt'],semantic_pad_to_tokens=32768,max_tokens=24,cache_prompt=True)
    for phase in ['fill','hit']:
        result=wave('single32k-'+phase,[paris])
        assert result['texts']['single32k'].strip()=='Paris'
    assert result['timings']['cached_tokens']>0
    report['single_profile_transition_and_prefix']='PASS'
    save()

    topics=['how CPU caches and cache coherency work','photosynthesis and plant metabolism',
            'Roman architectural engineering','comparison sorting algorithms and complexity']
    stories=[dict(id=f'story-{i}',prompt=f'Write a coherent technical explanation of {topic}, at least 2000 words. Begin with exactly BEGIN_{i} then continue with the explanation. Do not end early.',max_tokens=512,semantic_pad_to_tokens=8192,cache_prompt=False) for i,topic in enumerate(topics)]
    result=wave('long-topic-four512',stories)
    for i,r in enumerate(stories):
        assert result['done'][r['id']]['output_tokens']==512
        assert result['texts'][r['id']].startswith(f'BEGIN_{i}'),result['texts'][r['id']][:200]
    report['topic_outputs']='PENDING_READABILITY_REVIEW'
    save()

    full=wave('full524288-native',[dict(id=c['id'],prompt=c['prompt'],max_tokens=c['max_tokens'],capacity_tokens=131068,cache_prompt=False) for c in cases])
    check_cases(full,cases)
    assert sum(x['target_context'] for x in full['timings']['banks'])==524288
    solo=wave('single262140-native',[dict(id='max-single',prompt=cases[0]['prompt'],max_tokens=24,capacity_tokens=262140,cache_prompt=False)])
    assert solo['texts']['max-single'].strip()=='Paris'
    report.update(status='MODEL_COMPLETE_PENDING_LOG_AND_TEXT_REVIEW',native_semantics='PASS',full_pool_and_single_capacity_semantics='PASS',prefix_reuse='PASS')

if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        report.update(status='FAIL',error=repr(error))
        raise
    finally:
        report['finished_unix']=time.time()
        save()
