"""Select the existing 32K/UB1024 profile by reservation, without reloading."""
from pathlib import Path
import json
import socket
import time
from diagnostic_prerequisites import check

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
main = json.loads((HERE / 'run/qualification.json').read_text())
diagnostic_only = check(main)
assert json.loads((HERE / 'run/aligned-generation.json').read_text())['status'].startswith('THROUGHPUT_COMPLETE')
out = HERE / 'run/prefill-1024-probe.json'
report = dict(status='RUNNING', diagnostic_only=diagnostic_only, waves=[], unchanged_model_pid=main['waves'][0]['worker_pid'],
              reservation_tokens=60000, actual_corpus_tokens=32768,
              pool_tokens=524288, requested_mtp_depth=3)

def save():
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')

def wave(label, request):
    with socket.socket(socket.AF_UNIX) as connection:
        connection.settimeout(300)
        connection.connect('/home/debian/ac922-monster/production/hardware-control.sock')
        connection.sendall(json.dumps(dict(id=label, requests=[request], refill=False)).encode()+b'\n')
        result = json.loads(connection.makefile('rb').readline(8 << 20))
    report['waves'].append(result)
    save()
    assert not result.get('errors') and len(result['done']) == 1
    assert result['worker_pid'] == report['unchanged_model_pid']
    assert result['timings']['cached_tokens'] == 0
    print(json.dumps(dict(label=label, timings=result['timings'])), flush=True)
    return result

try:
    save()
    request = dict(id='raw32k-1024', corpus_token_start=0, corpus_token_count=32768,
                   max_tokens=8, ignore_eos=True, cache_prompt=False, capacity_tokens=60000)
    for phase in ['cold', 'retained']:
        result = wave('hardware-raw32k-1024-'+phase+'-c4', request)
        assert result['timings']['prefill_input_tokens'] == 32768
        assert result['done'][request['id']]['output_tokens'] == 8
    cases = json.loads((BASE/'fixtures/fourbank-d3-newruntime-v306-20261017/v221-d3-scenarios.json').read_text())[0]['requests']
    request = dict(id='paris-1024', prompt=cases[0]['prompt'], semantic_pad_to_tokens=32768,
                   max_tokens=24, cache_prompt=True, capacity_tokens=60000)
    # The raw waves remain uncached. This separate prefix pair tests whether
    # the narrower profile avoids the observed low-memory checkpoint eviction.
    for phase in ['fill', 'hit']:
        with socket.socket(socket.AF_UNIX) as connection:
            connection.settimeout(300)
            connection.connect('/home/debian/ac922-monster/production/hardware-control.sock')
            connection.sendall(json.dumps(dict(id='hardware-paris32k-1024-'+phase+'-c4',
                                              requests=[request], refill=False)).encode()+b'\n')
            result = json.loads(connection.makefile('rb').readline(8 << 20))
        report['waves'].append(result)
        save()
        assert not result.get('errors')
        assert result['worker_pid'] == report['unchanged_model_pid']
        assert result['texts'][request['id']].strip() == 'Paris'
    report['prefix_tokens'] = result['timings']['cached_tokens']
    assert report['prefix_tokens'] > 0
    report.update(status='COMPLETE_PENDING_PROFILE_AND_SPEED_REVIEW', long_semantics='PASS')
except BaseException as error:
    report.update(status='FAIL', error=repr(error))
    raise
finally:
    report['finished_unix'] = time.time()
    save()
