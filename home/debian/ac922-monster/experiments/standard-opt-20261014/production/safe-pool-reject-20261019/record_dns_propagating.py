
from pathlib import Path
import json,hashlib,time,os
p=Path('/home/debian/ac922-monster/production');f=Path('/home/debian/ac922-monster/experiments/standard-opt-20261014/production/safe-pool-reject-20261019')
assert os.geteuid()==0 and (p/'current').resolve().name=='20261019-r444-banked-l5'
assert not Path('/var/lib/ac922-inference/fault.json').exists()
for n in ['lastwave-acceptance.json','gateway-smoke-result.json','post-promotion-static.json']:assert json.loads((f/n).read_text())['status']=='PASS'
r=json.loads((p/'production-ready.json').read_text());public=json.loads((f/'public-smoke-result.json').read_text())
assert all(x['status']=='PASS' for x in public['origins']['https://ac922.tail58d5eb.ts.net'].values())
r.update(status='READY_DNS_PROPAGATING',release=str((p/'current').resolve()),unix=time.time(),live_worker_pid=json.loads((f/'live-runtime-identity.json').read_text())['processes'][0]['pid'],worker_sha256=json.loads((f/'lastwave-acceptance.json').read_text())['worker_sha256'],lastwave_acceptance=str(f/'lastwave-acceptance.json'),private_http_acceptance=str(f/'evidence/http-result.json'),backend_port=8007,pool_tokens=524288,max_context=262140,root_fault_latch_absent=True,external_network_validation={'receipt':str(f/'public-smoke-result.json'),'funnel_authenticated_models_chat':'PASS','direct_tls_correct_SNI_new_actual_address':'PASS','authoritative_AAAA':'240e:3bb:2eb4:670:72e2:84ff:fe14:474f','local_recursive_resolver_still_old_prefix':True,'independent_mobile_or_cross_operator_retest':False},ops_promoter_path=str(f/'stage_lastwave.py'),rollback_script='/home/debian/ac922-monster/production/backups/safe-pool-reject-20261018T200709/stage_lastwave.py',promotion_without_model_reload=True)
for key,pathkey in [('ops_promoter_sha256','ops_promoter_path'),('rollback_script_sha256','rollback_script')]:r[key]=hashlib.sha256(Path(r[pathkey]).read_bytes()).hexdigest()
tmp=p/'production-ready.dns-propagating.tmp';tmp.write_text(json.dumps(r,indent=2)+'\n');tmp.chmod(0o644);tmp.replace(p/'production-ready.json');print('L5_CURRENT_PID_READY; PUBLIC_FUNNEL_PASS; DIRECT_DNS_PROPAGATING_EXPLICIT')
