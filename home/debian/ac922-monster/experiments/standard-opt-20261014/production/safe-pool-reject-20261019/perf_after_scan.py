"""Repeat only affected performance slices after the audited read storm stopped."""
import asyncio
import json
from pathlib import Path
import time

import lastwave_http_acceptance as http
import lastwave_inband_acceptance as ipc

HERE = Path(__file__).resolve().parent
RUN = HERE / 'run'
ipc.OUT = RUN / 'perf-after-scan-prefill.json'
ipc.report = {'status': 'RUNNING', 'waves': [], 'started_unix': time.time(),
              'purpose': 'same worker; unbounded audit grep stopped before measurement'}


async def output_wave():
    output = RUN / 'perf-after-scan-http.json'
    report = {'status': 'RUNNING', 'phases': {}, 'started_utc': http.utc_now(),
              'purpose': 'same worker after disk-scan termination; SSE then usage-counted four512'}
    base, root = http.api_urls('http://127.0.0.1:8007/v1')
    runner = http.Runner(http.StandardHttpClient(), base, root, 'flash-next-iq3-xxs',
                         http.load_keyfile(Path('/home/debian/apigateway/api_key')),
                         output, report, None)
    try:
        await runner.discover('http://127.0.0.1:8007/ready')
        await runner.stories_sse()
        await runner.stories_usage_wave()
        single = json.loads((RUN / 'single-cache-result.json').read_text())
        old = next(w for w in single['waves'] if w['label'] == 'c3-four512')['texts']
        report['fulltext_identity_to_reviewed_ipc'] = {
            name: all(row['text'] == old[row['label'].removeprefix('usage-')]
                      for row in report['phases'][name]['requests'])
            for name in ['four_by_512_stream_metrics', 'four_by_512_usage_aggregate']}
        report['status'] = 'PASS'
    except BaseException as error:
        report.update(status='FAIL', error=repr(error))
        raise
    finally:
        report['finished_utc'] = http.utc_now()
        http.write_report(output, report)


try:
    ipc.wave('c5g-after-scan-raw32k-retained', [{'id': 'raw32k', 'corpus_token_start': 0,
              'corpus_token_count': 32768, 'max_tokens': 8, 'ignore_eos': True}])
    timing = ipc.report['waves'][0]['timings']
    assert timing['context_rebuilds_before_execution'] == 0
    assert timing['prefill_input_tokens'] == 32768 and timing['cached_tokens'] == 0
    ipc.report['status'] = 'PASS'
except BaseException as error:
    ipc.report.update(status='FAIL', error=repr(error))
    raise
finally:
    ipc.save()

asyncio.run(output_wave())
