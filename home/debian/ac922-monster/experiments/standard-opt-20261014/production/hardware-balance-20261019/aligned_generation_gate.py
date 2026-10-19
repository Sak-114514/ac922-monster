"""Reuse V306's exact uncached 4x16K/512 corpus slices in the loaded candidate."""
from pathlib import Path
import json
import socket
import time
import sys
from diagnostic_prerequisites import check

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
main = json.loads((HERE / 'run/qualification.json').read_text())
diagnostic_only = check(main)
assert json.loads((HERE / 'run/fourlong-native.json').read_text())['status'] == 'PASS'
baseline_path = HERE / 'reference-v306.json'
import hashlib
assert hashlib.sha256(baseline_path.read_bytes()).hexdigest() == 'de1c2788e2ad6961c7c036b5dd11d30a61853b3190bcfe6e3a2c4874b2d005ff'
baseline = json.loads(baseline_path.read_text())[1]
assert baseline['label'] == 'v221-d3-fourbank-4x16k-512'
requests = [dict(id=f'v221-d3-flow-{i}', corpus_token_start=i*16384,
                 corpus_token_count=16384, max_tokens=512,
                 ignore_eos=True, cache_prompt=False) for i in range(4)]
out = HERE / 'run/aligned-generation.json'
if '--finalize-existing' in sys.argv:
    receipt = json.loads(out.read_text())
    result = receipt['wave']
else:
    started = time.time()
    with socket.socket(socket.AF_UNIX) as connection:
        connection.settimeout(300)
        connection.connect('/home/debian/ac922-monster/production/hardware-control.sock')
        connection.sendall(json.dumps(dict(id='hardware-aligned-v306-4x16k512-c4',
                                          requests=requests, refill=False)).encode()+b'\n')
        result = json.loads(connection.makefile('rb').readline(8 << 20))
    receipt = dict(status='RUN_COMPLETE_PENDING_CHECK', diagnostic_only=diagnostic_only, wave=result,
                   baseline_path=str(baseline_path), started_unix=started,
                   finished_unix=time.time(), correctness_scored=False,
                   throughput_is_not_a_semantic_gate=True)
    out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
assert not result.get('errors') and len(result.get('done', {})) == 4
assert result['worker_pid'] == main['waves'][0]['worker_pid']
assert result['timings']['prefill_input_tokens'] == 65536
assert result['timings']['cached_tokens'] == 0
assert all(result['done'][r['id']]['output_tokens'] == 512 for r in requests)
records = []
for path in (HERE / 'run/model/copy0/fourbank16k-d3').glob('lastwave-*.json'):
    value = json.loads(path.read_text())
    if value.get('wave_id') == result['id']:
        records.append((path, value))
assert len(records) == 1
path, record = records[0]
boot = json.loads((HERE / 'run/model/copy0/fourbank16k-d3/results.json').read_text())[0]
assert boot['v45_source_corpus_sha256'] == baseline['v45_source_corpus_sha256']
assert len(record['flows']) == 4
for flow in record['flows']:
    assert flow['input_tokens'] == 16384 and flow['output_tokens'] == 512
    assert flow['id'] in {r['id'] for r in requests}
start = max(f['warmup_finished_us'] for f in record['flows'])
end = min(f['decode_finished_us'] for f in record['flows'])
assert end > start
shared_tokens = sum(sum(start <= t <= end for t in f['emitted_us']) for f in record['flows'])
previous = {f['id']: f for f in baseline['flows']}
receipt.update(status='THROUGHPUT_COMPLETE_PENDING_TEXT_REVIEW', native_record=str(path),
               corpus_sha256=boot['v45_source_corpus_sha256'], requests=requests,
               corpus_proof='frozen boot corpus digest plus unchanged worker token-slice preparation; live slim flow record omits offsets',
               common_post_warmup_window=dict(start_us=start, end_us=end, tokens=shared_tokens,
                                              tok_s=shared_tokens*1e6/(end-start)),
               baseline_output_parity={f['id']: dict(text_exact=f['output']==previous[f['id']]['output'],
                                                     token_ids_exact=f['token_ids']==previous[f['id']]['token_ids'])
                                       for f in record['flows']},
               baseline={k: baseline.get(k) for k in ['all_emissions_tok_s', 'shared_tg',
                           'scenario_whole_wall_us', 'parallel_seconds']},
               actual={k: record.get(k) for k in ['all_emissions_tok_s', 'shared_tg',
                       'scenario_whole_wall_us', 'parallel_seconds', 'prefill_aggregate_wall_tok_s']},
               accepted=sum(f['accepted'] for f in record['flows']),
               proposed=sum(f['proposed'] for f in record['flows']))
out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({k: receipt[k] for k in ['status', 'actual', 'baseline', 'accepted', 'proposed']}))
