"""Add only model routes after the private acceptance report passes."""
import hashlib,json,os,py_compile,shutil,subprocess,time
from pathlib import Path

ROOT=Path('/home/debian/ac922-monster/production')
RELEASE=(ROOT/'current').resolve()
report=json.loads((ROOT/'run/http-smoke-result.json').read_text())
if report.get('status')!='HTTP_REAL_MODEL_QUALITY_STREAM_CANCEL_LONG4_PASS':
    raise SystemExit('Private real-model acceptance has not passed; gateway unchanged.')
for unit in ('ac922-inference.service','ac922-inference-safety.service'):
    subprocess.run(['systemctl','is-active','--quiet',unit],check=True)
gateway=Path('/home/debian/apigateway/gateway.py')
module=gateway.with_name('llm_routes.py')
backup=ROOT/'backups'/('gateway-'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime()))
backup.mkdir(parents=True)
shutil.copy2(gateway,backup/'gateway.py')
had_module=module.exists()
if had_module:shutil.copy2(module,backup/'llm_routes.py')
marker='install_browser_access(app, load_user_keys, Path(__file__).parent / "access-state", Path(__file__).parent)'
text=gateway.read_text()
if text.count(marker)!=1 or 'install_model_routes' in text:
    raise SystemExit('Unexpected gateway source; manual reconciliation required.')
new=text.replace(marker,marker+'\n\nfrom llm_routes import install_model_routes\ninstall_model_routes(app)')
compile(new,str(gateway),'exec')
shutil.copy2(RELEASE/'llm_routes.py',module)
temp=gateway.with_suffix('.production-new.py');temp.write_text(new);shutil.copymode(gateway,temp);os.chown(temp,1000,1000);temp.replace(gateway);os.chown(module,1000,1000)
try:
    test=subprocess.run(['/usr/sbin/runuser','-u','debian','--','/home/debian/comfyui/venv/bin/python','-c',
        'import gateway; paths={r.path for r in gateway.app.routes}; assert {"/v1/models","/v1/completions","/v1/chat/completions"} <= paths; print("GATEWAY_IMPORT_PASS")'],
        cwd=str(gateway.parent),capture_output=True,text=True,timeout=30)
    if test.returncode:raise RuntimeError('Gateway import failed: '+test.stderr[-500:])
    subprocess.run(['systemctl','restart','ac922-apigateway.service'],check=True,timeout=45)
    subprocess.run(['systemctl','is-active','--quiet','ac922-apigateway.service'],check=True)
except BaseException:
    shutil.copy2(backup/'gateway.py',gateway)
    if had_module:shutil.copy2(backup/'llm_routes.py',module)
    else:module.unlink(missing_ok=True)
    subprocess.run(['systemctl','restart','ac922-apigateway.service'],check=False,timeout=45)
    raise
state={'backup':str(backup),'gateway_sha256':hashlib.sha256(gateway.read_bytes()).hexdigest(),
       'had_module':had_module,'release':str(RELEASE),'unix':time.time()}
(ROOT/'gateway-deployment.json').write_text(json.dumps(state,indent=2)+'\n')
print(json.dumps({'status':'GATEWAY_MODEL_ROUTES_ADDED','backup':str(backup),'release':str(RELEASE)}))
