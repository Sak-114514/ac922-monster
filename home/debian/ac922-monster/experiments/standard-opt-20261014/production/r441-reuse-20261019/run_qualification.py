"""One loaded model: exact raw workloads, semantic probes, and prefix reuse."""
from pathlib import Path
import json
import socket
import time
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
OUT = HERE / 'run/qualification.json'
SOCKET = '/home/debian/ac922-monster/production/r441-control.sock'
report = dict(status='WAITING_FOR_READY', started_unix=time.time(), waves=[],
              cuda_event_diagnostic_enabled=False, clean_speedup_claimed=False,
              model_loads_requested=1, mtp_depth=3, quality_reduction_requested=False)


def save():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temp = OUT.with_suffix('.tmp')
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    temp.replace(OUT)


def wave(label, requests):
    with socket.socket(socket.AF_UNIX) as connection:
        connection.settimeout(300)
        connection.connect(SOCKET)
        connection.sendall(json.dumps(dict(id=label, requests=requests, refill=False)).encode() + b'\n')
        result = json.loads(connection.makefile('rb').readline(8 << 20))
    report['waves'].append(result)
    save()
    assert result.get('status') != 'ERROR', result.get('error')
    assert not result.get('errors') and len(result['done']) == len(requests)
    if len(report['waves']) > 1:
        assert result['worker_pid'] == report['waves'][0]['worker_pid']
    if not any(x.get('cache_prompt', False) for x in requests):
        assert result['timings']['cached_tokens'] == 0
    timing = result['timings']
    print(json.dumps(dict(label=label, worker_pid=result['worker_pid'],
                          prefill_tok_s=timing.get('prefill_aggregate_wall_tok_s'),
                          prefill_input_tokens=timing.get('prefill_input_tokens'),
                          prefill_wall_s=timing.get('prefill_wall_seconds'),
                          cache_hits=timing.get('cached_tokens'), banks=timing.get('banks'),
                          graph_context_rebuilds=timing.get('context_rebuilds_before_execution'),
                          output_tok_s=timing.get('whole_wave_output_tok_s'),
                          all_emissions_tok_s=timing.get('all_emissions_tok_s'))), flush=True)
    return result


def native():
    cases = json.loads((BASE / 'fixtures/fourbank-d3-newruntime-v306-20261017/v221-d3-scenarios.json').read_text())[0]['requests']
    result = wave('r441-native4-c1', [dict(id=c['id'], prompt=c['prompt'],
                  max_tokens=c['max_tokens'], cache_prompt=False) for c in cases])
    for case in cases:
        text = result['texts'][case['id']].strip()
        assert json.loads(text) == case['expected'] if case['checker'] == 'strict_json' else text == str(case['expected'])
    return cases


def main():
    save()
    deadline = time.monotonic() + 900
    while True:
        try:
            with urlopen('http://127.0.0.1:8013/ready', timeout=3) as response:
                ready = json.load(response)
            if ready.get('status') == 'ready':
                report['ready'] = ready
                break
        except Exception:
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError('private prepare readiness timeout')
        time.sleep(2)
    report['status'] = 'RUNNING'
    cases = native()
    formal = dict(id='raw32k', corpus_token_start=0, corpus_token_count=32768,
                  max_tokens=8, ignore_eos=True, cache_prompt=False)
    for phase in ['cold', 'retained']:
        result = wave('r441-raw32k-' + phase + '-c1', [formal])
        assert result['timings']['prefill_input_tokens'] == 32768
        assert result['done']['raw32k']['output_tokens'] == 8
    topics = ['how CPU caches and cache coherency work', 'photosynthesis and plant metabolism',
              'Roman architectural engineering', 'comparison sorting algorithms and complexity']
    stories = [dict(id=f'story-{i}', prompt=f'Write a coherent technical explanation of {topic}, at least 2000 words. Begin with exactly BEGIN_{i} then continue with the explanation. Do not end early.',
                    max_tokens=512, cache_prompt=False) for i, topic in enumerate(topics)]
    result = wave('r441-four512-c1', stories)
    baseline = json.loads((BASE / 'production/safe-pool-reject-20261019/evidence/four512-flow-record.json').read_text())
    expected = {x['id']: x for x in baseline['flows']}
    # Production IPC deliberately omits full token arrays; the opt-in native
    # record is written before wave_done and retains their original source.
    records = HERE / 'run/model/copy0/fourbank16k-d3'
    matching = [(p, json.loads(p.read_text())) for p in records.glob('lastwave-*.json')]
    matches = [(p, value) for p, value in matching if value.get('wave_id') == result['id']]
    assert len(matches) == 1, 'missing unique on-disk four512 flow record'
    path, full_record = matches[0]
    report['four512_native_record'] = str(path)
    actual_flows = {x['id']: x for x in full_record['flows']}
    parity = {}
    for request in stories:
        key = request['id']
        assert result['done'][key]['output_tokens'] == 512
        assert result['texts'][key].startswith('BEGIN_' + key.rsplit('-', 1)[1])
        parity[key] = dict(full_text_exact=result['texts'][key] == expected[key]['output'],
                           token_ids_exact=actual_flows[key]['token_ids'] == expected[key]['token_ids'])
    report['four512_parity'] = parity
    report['four512_full_texts_exact'] = all(x['full_text_exact'] for x in parity.values())
    report['four512_all_2048_token_ids_exact'] = all(x['token_ids_exact'] for x in parity.values())
    save()
    formal64 = dict(formal, id='raw64k', corpus_token_count=65536)
    for phase in ['cold', 'retained']:
        result = wave('r441-raw64k-' + phase + '-c1', [formal64])
        assert result['timings']['prefill_input_tokens'] == 65536
        assert result['done']['raw64k']['output_tokens'] == 8
    paris = dict(id='long-paris', prompt=cases[0]['prompt'], semantic_pad_to_tokens=32768,
                 max_tokens=24, cache_prompt=True)
    for phase in ['fill', 'hit']:
        result = wave('r441-semantic-prefix-' + phase + '-c1', [paris])
        assert result['texts']['long-paris'].strip() == 'Paris'
    assert result['timings']['cached_tokens'] > 0
    report.update(status='MODEL_COMPLETE_PENDING_LOG_AND_TEXT_REVIEW', native_semantics='PASS',
                  long_semantics='PASS', prefix_reuse='PASS')


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        report.update(status='FAIL', error=repr(error))
        raise
    finally:
        report['finished_unix'] = time.time()
        save()
