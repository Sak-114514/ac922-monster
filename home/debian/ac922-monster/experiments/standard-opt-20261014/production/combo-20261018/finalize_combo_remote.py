"""Promote the qualified private combo through the existing authenticated gateway."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
from urllib.request import Request,urlopen
ROOT=Path('/home/debian/ac922-monster/production')
release=(ROOT/'current').resolve()
assert release.name=='20261018-r444-banked-combo-c12-r2'
manifest=json.loads((release/'release-manifest.json').read_text())
worker_sha=manifest['worker']['sha256']
report_path=ROOT/'run/combo-http-smoke-result.json'
for _ in range(180):
 if report_path.exists():
  report=json.loads(report_path.read_text())
  if report.get('release_path')==str(release):
   if report.get('status')=='FAIL':raise RuntimeError('Private HTTP acceptance failed')
   if report.get('status')=='COMBO_HTTP_REAL_MODEL_QUALITY_STREAM_CANCEL_PASS' and isinstance(report.get('finished'),(int,float)):break
 time.sleep(2)
else:raise TimeoutError('Private HTTP acceptance')
assert report['worker_sha256']==worker_sha
safety=json.loads(Path('/run/ac922-inference/safety.json').read_text())
assert safety['ready'] and safety['healthy'] and safety['gpu_stop_c']==65 and time.time()-safety['unix']<15
for unit in ('ac922-inference.service','ac922-inference-safety.service'):
 subprocess.run(['systemctl','is-active','--quiet',unit],check=True)
pids={int(v) for v in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).split()}
assert len(pids)==1,pids
pid=pids.pop();assert Path(f'/proc/{pid}/exe').resolve()==release/'bank-worker'
maps=Path(f'/proc/{pid}/maps').read_text()
for name in manifest['runtime']['private_dsos']+['libac922-tail-bridge.so']:
 assert str(release/'lib'/name) in maps,'Non-private library or unused helper: '+name
hold=ROOT/'PROMOTION-HOLD'
promoter=Path(__file__).resolve().with_name('promote_combo_gateway.py')
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
clear=ROOT/'combo-promotion-clearance.json'
if hold.is_file():
 clearance={'status':'CLEAR','cleared_by':'root','hold_id':sha(hold),'release_path':str(release),
  'worker_sha256':worker_sha,'http_smoke_sha256':sha(report_path),
  'worker_smoke_sha256':sha(release/'smoke-acceptance.json'),'unix':time.time()}
 clear.write_text(json.dumps(clearance,indent=2)+'\n');os.chmod(clear,0o600)
 archived=ROOT/'backups'/('promotion-hold-'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'.txt');hold.replace(archived)
else:
 clearance=json.loads(clear.read_text())
 assert clear.stat().st_uid==0 and clearance['status']=='CLEAR' and clearance['release_path']==str(release)
 assert clearance['worker_sha256']==worker_sha and clearance['http_smoke_sha256']==sha(report_path)

try:
 subprocess.run([sys.executable,str(promoter),'--release',str(release),'--apply'],check=True,timeout=120)
 key=Path('/home/debian/apigateway/api_key').read_text().strip()
 def request(path,body=None):
  payload=None if body is None else json.dumps(body).encode()
  req=Request('http://127.0.0.1:8080'+path,data=payload,headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
  with urlopen(req,timeout=60) as response:return response.status,response.read()
 code,body=request('/v1/models');models=json.loads(body)
 assert code==200 and any(x['id']=='flash-next-iq3-xxs' for x in models['data'])
 prompt={'model':'flash-next-iq3-xxs','messages':[{'role':'user','content':'请计算 7+4，只输出整数。'}],'temperature':0,'max_tokens':16}
 code,body=request('/v1/chat/completions',prompt);answer=json.loads(body)
 assert code==200 and answer['choices'][0]['message']['content'].strip()=='11'
 prompt['messages']=[{'role':'user','content':'What is the capital city of France? Reply with exactly the city name.'}];prompt['stream']=True
 code,body=request('/v1/chat/completions',prompt)
 assert code==200 and b'data: [DONE]' in body and b'Paris' in body
 del key
 subprocess.run(['systemctl','enable','ac922-inference.service','ac922-inference-safety.service'],check=True)
 for unit in ('ac922-inference.service','ac922-inference-safety.service','ac922-apigateway.service'):
  subprocess.run(['systemctl','is-active','--quiet',unit],check=True)
 record={'status':'PRODUCTION_COMBO_LOCAL_GATEWAY_VERIFIED','release':str(release),'worker_sha256':worker_sha,
  'live_worker_pid':pid,'all_seven_dsos_and_r5_private':True,'private_http_acceptance':str(report_path),
  'gateway_auth_models_chat_sse_pass':True,'services_enabled':True,'gpu_stop_c':65,
  'ops_promoter_path':str(promoter),'ops_promoter_sha256':sha(promoter),'external_network_validation':'pending','unix':time.time()}
 (ROOT/'production-ready.json').write_text(json.dumps(record,indent=2)+'\n')
 print(json.dumps(record),flush=True)
except BaseException:
 if (ROOT/'combo-gateway-deployment.json').exists():
  subprocess.run([sys.executable,str(promoter),'--rollback','--apply'],check=False,timeout=120)
 raise
