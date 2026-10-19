"""Same loaded model: exercise independent bank execution during four concurrent long prefills."""
from pathlib import Path
import json
import socket
import time

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
NATIVE = PROJECT / 'fixtures/fourbank-d3-newruntime-v306-20261017/v221-d3-scenarios.json'
SOCKET = '/home/debian/ac922-monster/production/phase-prefill-control.sock'
main = json.loads((HERE / 'run/qualification.json').read_text())
assert main['status'] == 'MODEL_COMPLETE_PENDING_LOG_AND_TEXT_REVIEW'
diagnostic_only = False
cases = json.loads(NATIVE.read_text())[0]['requests']
requests = [dict(id=c['id'], prompt=c['prompt'], max_tokens=c['max_tokens'],
                 semantic_pad_to_tokens=8192, cache_prompt=False) for c in cases]
started = time.time()
with socket.socket(socket.AF_UNIX) as connection:
    connection.settimeout(300)
    connection.connect(SOCKET)
    connection.sendall(json.dumps({'id': 'phase-fourlong-native-c1', 'requests': requests,
                                   'refill': False}).encode() + b'\n')
    result = json.loads(connection.makefile('rb').readline(8 << 20))
receipt = {'status': 'RUN_COMPLETE_PENDING_CHECK', 'diagnostic_only': diagnostic_only, 'started_unix': started,
           'finished_unix': time.time(), 'wave': result}
(HERE / 'run/fourlong-native.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + '\n')
assert not result.get('errors') and len(result.get('done', {})) == 4
assert result['worker_pid'] == main['waves'][0]['worker_pid']
for case in cases:
    actual = result['texts'][case['id']].strip()
    assert (json.loads(actual) == case['expected'] if case['checker'] == 'strict_json'
            else actual == str(case['expected'])), (case['id'], actual)
assert result['timings']['prefill_input_tokens'] == 4 * 8192
assert result['timings']['cached_tokens'] == 0
receipt['status'] = 'PASS'
(HERE / 'run/fourlong-native.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + '\n')
print(json.dumps({'status': 'PASS', 'worker_pid': result['worker_pid'],
                  'actual_uncached_prompt_tokens': 32768,
                  'fourlong_prefill_tok_s': result['timings']['prefill_aggregate_wall_tok_s']}))
