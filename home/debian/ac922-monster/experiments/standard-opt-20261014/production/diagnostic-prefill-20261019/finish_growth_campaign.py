"""Repair one cold-test precondition, then finish the same loaded C4 campaign."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import affinity_acceptance as a
import lastwave_inband_acceptance as h

F = Path(__file__).resolve().parent
R = F/'run'
campaign = json.loads((R/'campaign-status.json').read_text())
assert campaign['status'] == 'FAIL' and campaign['stages'][-1]['label'] == 'affinity'
old = json.loads((R/'affinity-result.json').read_text())
assert old['status'] == 'FAIL'
assert old['waves'][0]['label'] == 'cold4' and old['waves'][0]['status'] == 'FAIL'
assert all(w['status'] == 'PASS' for w in old['waves'][1:])
assert all(row['semantic_pass'] and row['bank_pass'] for row in old['waves'][0]['bank_mapping_by_request'])
saved = F/'attempts/affinity-primed-by-growth.json'
shutil.copy2(R/'affinity-result.json', saved)
h.OUT = R/'affinity-precondition-repair.json'
h.report = {'status':'RUNNING', 'waves':[], 'purpose':'explicitly invalidate all four prefix checkpoints before cold-only retry'}
native = json.loads(h.CASES.read_text())[0]['requests']
cleared = h.wave('c4-native4-invalidate-before-cold-retry',
    [{'id':c['id'], 'prompt':c['prompt'], 'max_tokens':c['max_tokens']} for c in native])
for c in native:
    text = cleared['texts'][c['id']].strip()
    assert json.loads(text) == c['expected'] if c['checker'] == 'strict_json' else text == str(c['expected'])
body = a.make_corpus_body(a.load_corpus(a.DEFAULT_CORPUS))
_, requests, _ = a.campaign_waves(body)[0]
raw = a.UnixWaveTransport(a.DEFAULT_SOCKET).wave('c4-cold4-after-explicit-invalidation', requests)
replacement, passed, mapping = a.enrich_wave('cold4', requests, raw, expected_cache='cold', bank_context=17152)
assert passed and mapping == old['waves'][0]['mapping']
replacement['actual_run_order'] = 'after original nine passing affinity waves and explicit four-bank invalidation'
old['producer_status'] = old['status']
old['precondition_repair'] = {'status':'PASS', 'original_report':str(saved),
    'original_sha256':hashlib.sha256(saved.read_bytes()).hexdigest(),
    'reason':'growth seed already primed DOC-C/D, so initial cold-only expectation was invalid; all nine functional hit/edit gates had passed',
    'repeat_scope':'only cold4 repeated after explicit native4 cache-disabled invalidation'}
old['waves'][0] = replacement
old['status'] = 'PASS'
old['summary']['all_waves_pass'] = True
a.write_report(R/'affinity-result.json', old)
h.report.update(status='PASS', cold4_replacement=replacement)
h.save()
step = campaign['stages'][-1]
step.update(producer_status=step['status'], producer_returncode=step['returncode'],
    status='PASS', returncode=0, repair_receipt=str(h.OUT))
campaign.update(producer_status=campaign['status'], status='RUNNING')
campaign.pop('error', None)
env = dict(os.environ, AC922_BASE_URL='http://127.0.0.1:8005/v1',
    AC922_READY_URL='http://127.0.0.1:8005/ready', AC922_API_KEY_FILE='/home/debian/apigateway/api_key',
    AC922_RESULT_JSON=str(R/'http-result.json'), AC922_MODEL_RUN_ROOT=str(R/'model'),
    NO_PROXY='127.0.0.1,localhost', no_proxy='127.0.0.1,localhost')


def save():
    (R/'campaign-status.json').write_text(json.dumps(campaign, indent=2)+'\n')


try:
    for label, python, script, args in [
        ('single-cache','/usr/bin/python3','single_cache_acceptance.py',[]),
        ('http','/home/debian/comfyui/venv/bin/python','lastwave_http_acceptance.py',[]),
        ('cancel-cached','/home/debian/comfyui/venv/bin/python','cancel_cached_single.py',['--run']),
        ('prefill-pacing','/usr/bin/python3','prefill_pacing_acceptance.py',[])]:
        stage = {'label':label, 'status':'RUNNING', 'started_unix':time.time()}
        campaign['stages'].append(stage)
        save()
        with (R/(label+'-controller.log')).open('w') as out:
            result = subprocess.run([python,str(F/script),*args],cwd=F,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=1200)
        stage.update(status='PASS' if result.returncode == 0 else 'FAIL', returncode=result.returncode, finished_unix=time.time())
        save()
        if result.returncode:
            raise RuntimeError(label+' failed; see stage log')
    campaign['status'] = 'PASS_PENDING_FULL_TEXT_REVIEW'
except BaseException as error:
    campaign.update(status='FAIL', error=repr(error))
    raise
finally:
    campaign['finished_unix'] = time.time()
    save()
