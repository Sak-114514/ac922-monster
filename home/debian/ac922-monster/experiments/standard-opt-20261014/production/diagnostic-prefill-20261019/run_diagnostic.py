"""One private model load, one exact32K target trace, then native semantic canaries."""
import json
from pathlib import Path
import socket
import subprocess
import time
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
RUN = HERE / 'run'
SOCKET = '/home/debian/ac922-monster/production/diag-prefill-control.sock'
report = {'status': 'WAITING_FOR_READY', 'started_unix': time.time(), 'waves': []}


def save():
    temp = RUN / 'diagnostic-status.tmp'
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    temp.replace(RUN / 'diagnostic-status.json')


def wave(epoch, requests):
    with socket.socket(socket.AF_UNIX) as connection:
        connection.settimeout(300)
        connection.connect(SOCKET)
        connection.sendall(json.dumps({'id': epoch, 'requests': requests, 'refill': False}).encode() + b'\n')
        result = json.loads(connection.makefile('rb').readline(8 << 20))
    report['waves'].append(result)
    save()
    assert result.get('status') != 'ERROR', result.get('error')
    assert not result.get('errors') and len(result['done']) == len(requests)
    return result


save()
try:
    until = time.monotonic() + 900
    while True:
        try:
            with urlopen('http://127.0.0.1:8008/ready', timeout=3) as response:
                ready = json.load(response)
            if ready.get('status') == 'ready':
                report['ready'] = ready
                break
        except Exception:
            pass
        if time.monotonic() >= until:
            raise TimeoutError('diagnostic readiness timeout')
        time.sleep(2)
    report['status'] = 'RUNNING'
    save()
    result = wave('diag-current-raw32k', [{'id': 'current32k-input', 'corpus_token_start': 0,
                  'corpus_token_count': 32768, 'max_tokens': 8, 'ignore_eos': True, 'cache_prompt': False}])
    timing = result['timings']
    assert timing['prefill_input_tokens'] == 32768 and timing['cached_tokens'] == 0
    proof = timing['diagnostic_prefill32']
    assert proof['request_id'] == 'current32k-input' and proof['input_tokens'] == 32768
    assert proof['cached_tokens'] == 0 and proof['chunk_index'] == 0 and proof['mtp_excluded']
    assert proof['prepare_status'] == proof['start_status'] == proof['stop_status'] == 0
    cases = json.loads((HERE.parents[1] / 'fixtures/fourbank-d3-newruntime-v306-20261017/v221-d3-scenarios.json').read_text())[0]['requests']
    native = wave('diag-post-native4', [{'id': c['id'], 'prompt': c['prompt'],
                                       'max_tokens': c['max_tokens'], 'cache_prompt': False} for c in cases])
    for case in cases:
        text = native['texts'][case['id']].strip()
        assert json.loads(text) == case['expected'] if case['checker'] == 'strict_json' else text == str(case['expected'])
    assert native['worker_pid'] == result['worker_pid']
    lines = (RUN / 'api/worker.log').read_text(errors='replace').splitlines()
    starts = [x for x in lines if 'diag-prefill32 event=target-start ' in x]
    stops = [x for x in lines if 'diag-prefill32 event=target-stop ' in x]
    mtp = [x for x in lines if 'diag-prefill32 event=mtp-entry ' in x]
    assert len(starts) == len(stops) == len(mtp) == 1
    assert lines.index(starts[0]) < lines.index(stops[0]) < lines.index(mtp[0])
    report['actual_marker_proof'] = {'start': starts[0], 'stop': stops[0], 'mtp_entry': mtp[0]}
    collector = [x for x in lines if 'cupti-marker configured phase=prefill1' in x]
    assert len(collector) == 1 and 'prepared_before_cuda_init=1' in collector[0]
    summary = HERE.parents[1] / 'fixtures/layer6-cupti32k-v223-20261017/trace_summary.py'
    with (RUN / 'trace-summary.log').open('w') as log:
        subprocess.run(['python3', str(summary), str(RUN / 'current32k-target-c1.jsonl')],
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
    report.update(status='CAPTURE_COMPLETE_PENDING_TRACE_REVIEW', post_native_semantics='PASS',
                  actual_worker_pid=result['worker_pid'])
except BaseException as error:
    report.update(status='FAIL', error=repr(error))
    raise
finally:
    report['finished_unix'] = time.time()
    save()
