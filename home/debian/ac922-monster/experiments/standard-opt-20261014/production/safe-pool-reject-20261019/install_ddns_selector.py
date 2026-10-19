
import ast,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
f=Path('/home/debian/apigateway/aliyun_ddns.py')
p=Path('/home/debian/ac922-monster/experiments/standard-opt-20261014/production/safe-pool-reject-20261019')
b=Path('/home/debian/ac922-monster/production/backups')/time.strftime('ddns-source-%Y%m%dT%H%M%S',time.gmtime());b.mkdir(mode=0o700)
shutil.copy2(f,b/'aliyun_ddns.py.before')
s=f.read_text();node=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name=='current_ipv6')
assert 'best = addr' in ast.get_source_segment(s,node)
lines=s.splitlines(keepends=True);replacement=(p/'ddns-selector.txt').read_text().rstrip()+'\n'
new=''.join(lines[:node.lineno-1])+replacement+''.join(lines[node.end_lineno:]);ast.parse(new)
tmp=f.with_suffix('.py.prefix-candidate');tmp.write_text(new);shutil.copystat(f,tmp);st=f.stat();os.chown(tmp,st.st_uid,st.st_gid)
r=subprocess.run(['python3',str(p/'test_ipv6_source.py'),str(tmp)],capture_output=True,text=True)
(p/'run/ddns-after-tests.log').write_text(r.stdout+r.stderr)
if r.returncode:tmp.unlink();raise RuntimeError(r.stderr)
tmp.replace(f)
record={'status':'INSTALLED','backup':str(b),'before_sha256':hashlib.sha256(s.encode()).hexdigest(),'after_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'cpu_tests':9,'only_address_selector_changed':True,'rollback':'sudo cp '+str(b/'aliyun_ddns.py.before')+' '+str(f),'installed_unix':time.time()}
(p/'ddns-source-repair.json').write_text(json.dumps(record,indent=2)+'\n');os.chown(p/'ddns-source-repair.json',st.st_uid,st.st_gid)
(b/'repair.json').write_text(json.dumps(record,indent=2)+'\n')
r=subprocess.run(['systemctl','restart','ac922-ddns.service'],capture_output=True,text=True)
print(json.dumps(record));print('DDNS_SERVICE_RETURN',r.returncode);print(r.stderr)
raise SystemExit(r.returncode)
