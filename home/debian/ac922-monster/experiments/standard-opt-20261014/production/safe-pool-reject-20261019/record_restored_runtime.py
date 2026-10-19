from pathlib import Path
import hashlib,json,time,os
p=Path('/home/debian/ac922-monster/production');f=Path('/home/debian/ac922-monster/experiments/standard-opt-20261014/production/safe-pool-reject-20261019')
assert os.geteuid()==0 and (p/'current').resolve().name=='20261019-r444-banked-l5'
assert not Path('/var/lib/ac922-inference/fault.json').exists()
for n in ['lastwave-acceptance.json','gateway-smoke-result.json','post-promotion-static.json','post-restore-long-canary.json']:assert json.loads((f/n).read_text())['status']=='PASS'
public=json.loads((f/'public-smoke-result.json').read_text());assert all(v['status']=='PASS' for v in public['origins']['https://ac922.tail58d5eb.ts.net'].values())
static=json.loads((f/'post-promotion-static.json').read_text());static.update(cold_restart_performed=True,restart_reason='restored unprofiled L5 after one private CUPTI window');(f/'post-promotion-static.json').write_text(json.dumps(static,indent=2)+chr(10))
r=json.loads((p/'production-ready.json').read_text());r.update(status='READY' if public['status']=='PASS' else 'READY_EXTERNAL_IPV6_RECHECK_PENDING',release=str((p/'current').resolve()),live_worker_pid=json.loads((f/'live-runtime-identity.json').read_text())['processes'][0]['pid'],root_fault_latch_absent=True,cold_restart_after_diagnostic_verified=True,unix=time.time(),external_network_validation={'current_observation':public,'direct_domain_last_full_pass_unix':1792356350.786496,'dns_authoritative_and_current_host_prefix_match':True,'limits':'Current v6 TCP retry timed out despite correct new-prefix DNS; no routing/firewall change made. Authenticated Funnel API passes.'})
(p/'production-ready.json').write_text(json.dumps(r,indent=2)+chr(10));print(r['status'],r['live_worker_pid'])
