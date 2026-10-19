"""One-loaded-worker uncached prefill measurements with the fixed61/57 profile."""
import json
from pathlib import Path
import lastwave_inband_acceptance as h

F = Path(__file__).resolve().parent
h.OUT = F / 'run/prefill-pacing-result.json'
h.report = {'status': 'RUNNING', 'waves': [], 'purpose': 'raw timing and semantic4x8K after ordinary32K warming; soft61/57, hard65'}
cases = json.loads(h.CASES.read_text())[0]['requests']
try:
    for suffix in ['reconstructed', 'repeat-retained']:
        r = h.wave('c4-raw32k-'+suffix, [{'id':'raw32k-'+suffix, 'corpus_token_start':0,
            'corpus_token_count':32768, 'max_tokens':8, 'ignore_eos':True}])
        assert r['timings']['prefill_input_tokens'] == 32768
    requests = [{'id':c['id'], 'prompt':c['prompt'], 'max_tokens':c['max_tokens'],
                 'semantic_pad_to_tokens':8192} for c in cases]
    r = h.wave('c4-native4-8k-soft61', requests)
    assert r['timings']['prefill_input_tokens'] == 32768
    for c in cases:
        text = r['texts'][c['id']].strip()
        assert json.loads(text) == c['expected'] if c['checker'] == 'strict_json' else text == str(c['expected']), (c['id'], text)
    r['semantic_status'] = 'PASS'
    paths = list((F/'run/api').glob('worker.log*'))
    marker = 'mode=relaxed trigger_c=61 resume_c=57 hard_stop_c=65'
    assert any(marker in path.read_text(errors='replace') for path in paths), 'actual fixed thermal profile not logged'
    h.report['thermal_profile'] = {'actual_log_marker':marker, 'hardware_configuration_modified':False}
    h.report['status'] = 'PASS'
except BaseException as error:
    h.report.update(status='FAIL', error=repr(error))
    raise
finally:
    h.save()
