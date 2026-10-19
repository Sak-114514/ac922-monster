"""Switch only the private inference service after combined acceptance; gateway untouched."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,time
ROOT=Path('/home/debian/ac922-monster/production')
release=Path(sys.argv[1]).resolve(strict=True)
smoke=Path(sys.argv[2]).resolve(strict=True)
assert release.parent==ROOT/'releases'
manifest=json.loads((release/'release-manifest.json').read_text())
report=json.loads(smoke.read_text())
assert report['status']=='COMBO_NATIVE_PREFILL32K_D3_FOUR512_CAPACITY_PASS',report['status']
assert report['artifacts']['worker_sha256']==hashlib.sha256((release/'bank-worker').read_bytes()).hexdigest()
assert report['thermal_policy']['gpu_stop_c']==65
assert report.get('worker_exit')==0
p=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,timeout=15,check=True)
assert not p.stdout.strip(),'GPU occupied; release switch refused'
fault=Path('/var/lib/ac922-inference/fault.json')
assert not fault.exists(),'Persisted safety latch exists; inspect before activation'
backup=ROOT/'backups'/('private-combo-'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime()))
backup.mkdir(parents=True)
old=(ROOT/'current').resolve(strict=True)
(backup/'previous-current.txt').write_text(str(old)+'\n')
for name in ('ac922-inference.service','ac922-inference-safety.service'):
 dest=Path('/etc/systemd/system')/name
 if dest.exists():shutil.copy2(dest,backup/name)
 subprocess.run(['systemctl','stop',name],check=True,timeout=65)
 shutil.copy2(release/name,dest)
temp=ROOT/'current.combo-new'
temp.unlink(missing_ok=True);temp.symlink_to(release,target_is_directory=True);temp.replace(ROOT/'current')
try:
 subprocess.run(['systemctl','daemon-reload'],check=True)
 # Inactive units can be unloaded; reset-failed then returns1 although start is valid.
 subprocess.run(['systemctl','reset-failed','ac922-inference.service','ac922-inference-safety.service'],check=False,capture_output=True)
 subprocess.run(['systemctl','start','ac922-inference-safety.service'],check=True,timeout=30)
 subprocess.run(['systemctl','start','ac922-inference.service'],check=True,timeout=90)
except BaseException:
 subprocess.run(['systemctl','stop','ac922-inference.service','ac922-inference-safety.service'],check=False,timeout=65)
 temp.unlink(missing_ok=True);temp.symlink_to(old,target_is_directory=True);temp.replace(ROOT/'current')
 for name in ('ac922-inference.service','ac922-inference-safety.service'):
  if (backup/name).exists():shutil.copy2(backup/name,Path('/etc/systemd/system')/name)
 subprocess.run(['systemctl','daemon-reload'],check=False)
 raise
state={'release':str(release),'previous_release':str(old),'backup':str(backup),'combined_acceptance':str(smoke),'unix':time.time(),'gateway_mutated':False,'services_enabled':False}
(ROOT/'private-combo-deployment.json').write_text(json.dumps(state,indent=2)+'\n')
print(json.dumps({'status':'COMBO_PRIVATE_SERVICES_STARTED_NOT_PUBLIC','release':str(release),'backup':str(backup)}))
