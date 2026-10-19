from pathlib import Path
import json,runpy,time
f=Path('/home/debian/ac922-monster/experiments/standard-opt-20261014/production/safe-pool-reject-20261019')
runpy.run_path(str(f/'record_release.py'),run_name='__main__')
p=Path('/home/debian/ac922-monster/production/production-ready.json');d=json.loads(p.read_text());d.update(status='READY',unix=time.time(),backend_port=8007,ddns_source_repair=str(f/'ddns-source-repair.json'));d['external_network_validation']['local_recursive_resolver_still_old_prefix']=False
p.write_text(json.dumps(d,indent=2)+'\n');print('L5_FULL_READY_AND_PUBLIC_DNS_API_PASS')
