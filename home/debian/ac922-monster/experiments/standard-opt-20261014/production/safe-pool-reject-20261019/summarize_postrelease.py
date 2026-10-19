"""Summarize actual windows, routing hits, and setup overlap independently."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
run = HERE / 'run'
inband = json.loads((run/'inband-result.json').read_text())
http = json.loads((run/'http-result.json').read_text())
affinity = json.loads((run/'affinity-result.json').read_text())
waves = {row['label']:row for row in inband['waves']}
summary = {'status':'PENDING_SEMANTIC_ADJUDICATION','worker_sha256':json.loads((HERE/'build-lastwave/build-report.json').read_text())['worker_sha256'],'http_status':http['status'],'affinity_status':affinity['status'],'prefill':{},'generation':{},'affinity':[]}
for label in ['lastwave-32k-first','lastwave-32k-repeat','lastwave-64k-max262k-UB512','lastwave-native4-8k']:
 t=waves[label]['timings'];summary['prefill'][label]={key:t[key] for key in ['prefill_input_tokens','cached_tokens','prefill_wall_seconds','prefill_aggregate_wall_tok_s','thermal_wait_us_sum']}
four=waves['lastwave-four512'];t=four['timings']
summary['generation']={key:t[key] for key in ['whole_wave_output_tok_s','all_emissions_tok_s','wall_seconds']}
summary['generation'].update(ipc_control_tok_s=2048/four['wall_s'],ipc_control_wall_s=four['wall_s'],http_usage_tok_s=http['phases']['four_by_512_usage_aggregate']['aggregate_tok_s'],http_wall_s=http['phases']['four_by_512_usage_aggregate']['arrival_to_last_completion_s'])
for path in (run/'model').rglob('lastwave-*.json'):
 d=json.loads(path.read_text())
 if d.get('wave_id')!=four['id']:continue
 flows=d['flows']; intervals=[(x['warmup_started_us'],x['warmup_finished_us']) for x in flows]
 events=sorted([(a,1) for a,b in intervals]+[(b,-1) for a,b in intervals]);active=peak=0;last=events[0][0];overlap2=overlap4=0
 for timestamp,delta in events:
  if active>=2:overlap2+=timestamp-last
  if active>=4:overlap4+=timestamp-last
  active+=delta;peak=max(peak,active);last=timestamp
 summary['setup']={'scope':'host request-phase wall intervals, not proof of GPU kernel overlap','max_simultaneous_windows':peak,'overlap_ge2_ms':overlap2/1000,'overlap_all4_ms':overlap4/1000,'per_flow':[{'id':x['id'],'parallel':x['setup_parallel'],'wait_us':x['setup_wait_us'],'warmup_frames':x['warmup_frames'],'warmup_ms':(x['warmup_finished_us']-x['warmup_started_us'])/1000,'proposed':x['proposed'],'accepted':x['accepted']} for x in flows]}
 assert all(x['setup_parallel'] and x['warmup_frames']==4 for x in flows)
 assert peak>=2
 baseline_path=Path('/home/debian/ac922-monster/experiments/standard-opt-20261014/production/lastwave-20261018/evidence/four512-flow-record.json')
 baseline={x['id']:x for x in json.loads(baseline_path.read_text())['flows']}
 summary['l1_output_comparison']={x['id']:{'token_ids_exact':x['token_ids']==baseline[x['id']]['token_ids'],'text_exact':x['output']==baseline[x['id']]['output']} for x in flows}
 (run/'four512-flow-record.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
 break
else:raise RuntimeError('four512 raw flow data absent')
for w in affinity['waves']:
 summary['affinity'].append({'label':w['label'],'status':w['status'],'wall_s':w['wave_wall_s'],'mapping':w['mapping'],'cache_tokens':{r['id']:r['cached_tokens'] for r in w['bank_mapping_by_request']}})
rows=[json.loads(line) for line in (run/'gpu-samples.jsonl').read_text().splitlines()]
summary['telemetry']={'samples':len(rows),'minimum_free_mib':min(g[0] for r in rows for g in r['gpu']),'max_gpu_core_c':max(g[1] for r in rows for g in r['gpu'])}
summary['asymmetric_ttft']=waves['lastwave-asymmetric-prefill-32k-8k-1k-short']['first_token_s']
(run/'measured-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
