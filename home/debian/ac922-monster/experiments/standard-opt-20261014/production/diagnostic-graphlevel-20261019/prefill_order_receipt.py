"""Two consecutive uncached 32K waves in the same private worker, before/after reuse."""
from pathlib import Path

import lastwave_inband_acceptance as harness

HERE = Path(__file__).resolve().parent
harness.OUT = HERE / 'run/prefill-order-result.json'
harness.report = {'status': 'RUNNING', 'waves': [],
                  'purpose': 'separate reconstructed and retained contexts; prefix hits disabled'}
try:
    for label in ['c5g-raw32k-profile-reconstruction', 'c5g-raw32k-profile-retained']:
        result = harness.wave(label, [{'id': 'raw32k', 'corpus_token_start': 0,
                                      'corpus_token_count': 32768, 'max_tokens': 8,
                                      'ignore_eos': True}])
        timing = result['timings']
        assert timing['cached_tokens'] == 0 and timing['prefill_input_tokens'] == 32768
        assert timing['banks'][0]['target_ubatch'] == 1024
        if label.endswith('retained'):
            assert timing['context_rebuilds_before_execution'] == 0
    harness.report['status'] = 'PASS'
except BaseException as error:
    harness.report.update(status='FAIL', error=repr(error))
    raise
finally:
    harness.save()
