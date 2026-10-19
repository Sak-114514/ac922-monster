"""Join real receipts, then freeze them only after the parent's full-text review."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import summarize_single_cache as evidence

F = Path(__file__).resolve().parent
R = F / 'run'
E = F / 'evidence'
L4 = F.parent / 'growth-cache-20261019'
FROZEN_L4 = Path('/home/debian/ac922-monster/production/releases/20261019-r444-banked-l4')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def prepare():
    campaign = read(R / 'campaign-status.json')
    assert campaign['status'] == 'PASS_PENDING_FULL_TEXT_REVIEW'
    assert {x['label'] for x in campaign['stages']} == {
        'safe-pool-affinity', 'single-cache', 'http', 'cancel-cached'}
    assert all(x['status'] == 'PASS' and x['returncode'] == 0 for x in campaign['stages'])
    safe = read(R / 'safe-pool-reject-result.json')
    single = read(R / 'single-cache-result.json')
    http = read(R / 'http-result.json')
    clean_http = read(R / 'perf-after-scan-http.json')
    clean_prefill = read(R / 'perf-after-scan-prefill.json')
    cancel = read(R / 'cancel-cached-single.json')
    assert read(R / 'prefill-order-result.json')['status'] == 'PASS'
    assert safe['status'] == http['status'] == cancel['status'] == 'PASS'
    assert clean_http['status'] == clean_prefill['status'] == 'PASS'
    assert single['status'] == 'PASS_PENDING_FULL_TEXT_REVIEW'
    assert read(F / 'build-lastwave/safe-cpu-validation.json')['status'] == 'PASS'
    baseline_receipt = read(FROZEN_L4 / 'lastwave-acceptance.json')
    for name in ['four512-flow-record.json', 'single-cache-result.json']:
        assert sha(L4 / 'evidence' / name) == baseline_receipt['evidence_sha256']['evidence/' + name]
    four = evidence.find_wave(single, 'c3-four512')
    flow_path, raw = evidence.rawflow_for(R / 'model', four['id'])
    assert len(raw['flows']) == 4 and all(x['output_tokens'] == 512 for x in raw['flows'])
    exact = evidence.compare_flows(raw['flows'], read(L4 / 'evidence/four512-flow-record.json')['flows'])
    phases = http['phases']
    usage = phases['four_by_512_usage_aggregate']
    streams = {x['label']: x for x in phases['four_by_512_stream_metrics']['requests']}
    nonstreams = {x['label'].removeprefix('usage-'): x for x in usage['requests']}
    assert len(usage['requests']) == 4 and all(x['usage']['completion_tokens'] == 512 for x in usage['requests'])
    http_exact = {x['id']: {'stream': streams[x['id']]['text'] == x['output'],
                           'nonstream': nonstreams[x['id']]['text'] == x['output']}
                  for x in raw['flows']}
    clean_usage = clean_http['phases']['four_by_512_usage_aggregate']
    assert len(clean_usage['requests']) == 4 and all(
        x['usage']['completion_tokens'] == 512 for x in clean_usage['requests'])
    clean_rt = clean_prefill['waves'][0]['timings']
    assert clean_rt['prefill_input_tokens'] == 32768 and clean_rt['cached_tokens'] == 0
    assert clean_rt['context_rebuilds_before_execution'] == 0
    essay = evidence.find_wave(single, 'c3-cached-single-512-prose')['texts']['cached-essay']
    old_essay = evidence.find_wave(read(L4 / 'evidence/single-cache-result.json'),
                                   'c3-cached-single-512-prose')['texts']['cached-essay']
    rt = evidence.find_wave(single, 'c3-raw32k-cold-fallback')['timings']
    assert rt['prefill_input_tokens'] == 32768 and rt['cached_tokens'] == 0
    assert rt['banks'][0]['target_ubatch'] == 1024
    telemetry = evidence.telemetry_summary(R / 'gpu-samples.jsonl')
    assert min(telemetry['minimum_free_mib_by_gpu']) >= 256
    assert max(telemetry['maximum_temperature_c_by_gpu']) <= 65
    review_texts = {'cached_essay': essay, 'four_flows': raw['flows'],
                    'http_stream': streams, 'http_nonstream': nonstreams,
                    'clean_http': clean_http['phases']}
    write(R / 'review-texts.json', review_texts)
    result = {'status': 'PENDING_FULL_TEXT_REVIEW', 'prepared_unix': time.time(),
              'safe_pool_status': safe['status'], 'safe_pool_probe': safe['safe_pool_boundary_probe'],
              'four512_exact_to_reviewed_L4': exact, 'http_full_text_identity': http_exact,
              'cached_essay_exact_to_reviewed_L4': essay == old_essay,
              'four512_raw_path': str(flow_path), 'actual_output_tokens': 2048,
              'io_polluted_ipc_actual_output_tok_s': 2048 / four['wall_s'],
              'http_actual_output_tok_s': 2048 / clean_usage['arrival_to_last_completion_s'],
              'clean_http_full_text_identity': clean_http['fulltext_identity_to_reviewed_ipc'],
              'clean_raw32k_uncached': {k: clean_rt[k] for k in ['prefill_input_tokens', 'cached_tokens',
                  'prefill_aggregate_wall_tok_s', 'thermal_wait_us_sum', 'context_rebuilds_before_execution']},
              'io_interference': read(R / 'io-audit-interference.json'),
              'io_polluted_http_output_tok_s': 2048 / usage['arrival_to_last_completion_s'],
              'primary_metrics_scope': 'Same PID after terminating the unbounded audit disk scan; earlier timing retained and labelled.',
              'io_polluted_http_per_request': [{'label': x['label'], 'wall_s': x['wall_s'],
                                    'output_tokens': x['usage']['completion_tokens']}
                                   for x in usage['requests']],
              'http_per_request': [{'label': x['label'], 'wall_s': x['wall_s'],
                                    'output_tokens': x['usage']['completion_tokens']}
                                   for x in clean_usage['requests']],
              'io_polluted_initial_raw32k': {k: rt[k] for k in ['prefill_input_tokens', 'cached_tokens',
                  'prefill_aggregate_wall_tok_s', 'thermal_wait_us_sum', 'context_rebuilds_before_execution']},
              'io_polluted_raw32k_reuse_order': read(R / 'prefill-order-result.json'),
              'telemetry': telemetry, 'review_texts_sha256': sha(R / 'review-texts.json'),
              'quality_and_throughput_separate': True,
              'limits': ['No new tiny1280 context allocation; near-boundary cohort intentionally refused.',
                         'No claim that C5 CUDA cause is fixed or raw300/6000 targets are met.',
                         'Unchanged L4 accepted layouts, standard indexer, fixed D3 and quantization.']}
    write(R / 'measured-summary.json', result)
    print(json.dumps({k: result[k] for k in ['status', 'http_actual_output_tok_s', 'io_polluted_ipc_actual_output_tok_s',
                                           'clean_raw32k_uncached', 'cached_essay_exact_to_reviewed_L4']}))


def qualify():
    summary = read(R / 'measured-summary.json')
    review = read(R / 'parent-semantic-review.json')
    assert review['status'] == 'PASS'
    assert review['measured_summary_sha256'] == sha(R / 'measured-summary.json')
    assert review['review_texts_sha256'] == sha(R / 'review-texts.json')
    E.mkdir(exist_ok=True)
    for name in ['campaign-status.json', 'single-cache-result.json']:
        value = read(R / name)
        write(E / ('producer-original-' + name), value)
        value['producer_status'] = value['status']
        value['status'] = 'PASS'
        value['semantic_adjudication'] = 'parent-semantic-review.json'
        write(R / name, value)
    for name in ['campaign-status.json', 'single-cache-result.json', 'safe-pool-reject-result.json',
                 'http-result.json', 'cancel-cached-single.json', 'prefill-order-result.json',
                 'perf-after-scan-http.json', 'perf-after-scan-prefill.json', 'io-audit-interference.json',
                 'parent-semantic-review.json', 'measured-summary.json']:
        shutil.copy2(R / name, E / name)
    shutil.copy2(Path(summary['four512_raw_path']), E / 'four512-flow-record.json')
    for path in (F / 'build-lastwave').glob('safe-*.log'):
        shutil.copy2(path, E / path.name)
    shutil.copy2(F / 'build-lastwave/safe-cpu-validation.json', E / 'safe-cpu-validation.json')
    report = read(F / 'build-lastwave/build-report.json')
    summary.update(status='PASS', semantic_adjudication='evidence/parent-semantic-review.json')
    write(F / 'MEASURED-SAFE-POOL.json', summary)
    receipt = {'status': 'PASS', 'qualified_unix': time.time(), 'worker_sha256': report['worker_sha256'],
               'quality': 'evidence/parent-semantic-review.json', 'cpu_python_tests': 34,
               'strict_cpp_fixtures': 7, 'cuda_tus_rebuilt': 0, 'engine_tus_rebuilt': 0,
               'baseline': str(FROZEN_L4), 'mtp_depth': 3, 'telemetry': summary['telemetry'],
               'evidence_sha256': {str(p.relative_to(F)): sha(p) for p in sorted(E.iterdir()) if p.is_file()},
               'limits': summary['limits'], 'architect_review': 'Pending final real evidence signoff'}
    write(F / 'lastwave-acceptance.json', receipt)
    print('SAFE_POOL_EVIDENCE_ARCHIVED; FINAL_SIGNOFF_REQUIRED')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--qualify', action='store_true')
    options = parser.parse_args()
    qualify() if options.qualify else prepare()
