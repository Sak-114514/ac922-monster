"""Verify the immutable release and refuse a second model process."""
import hashlib,json,re,subprocess
from pathlib import Path

HERE=Path(__file__).resolve().parent
manifest=json.loads((HERE/'release-manifest.json').read_text())
for name,digest in manifest['sha256'].items():
    p=Path(name)
    if not p.is_absolute(): p=HERE/p
    h=hashlib.sha256()
    with p.open('rb') as f:
        while data:=f.read(8<<20): h.update(data)
    if h.hexdigest()!=digest: raise SystemExit('Release checksum mismatch: '+str(p))
for item in manifest['model_identity']:
    p=Path(item['path']);s=p.stat()
    if s.st_size!=item['bytes'] or s.st_mtime_ns!=item['mtime_ns']:
        raise SystemExit('Model identity changed; explicit checksum verification required.')
physical_available_kib=0
for node in (0,8):
    values={}
    for line in Path(f'/sys/devices/system/node/node{node}/meminfo').read_text().splitlines():
        match=re.search(r'Node \d+ (\S+):\s+(\d+) kB',line)
        if match: values[match[1]]=int(match[2])
    physical_available_kib+=max(0,values['MemFree']+values.get('FilePages',0)-values.get('Unevictable',0))
if physical_available_kib<4*1024*1024:
    raise SystemExit('Less than4GiB reclaimable physical DDR on NUMA0/8; model notstarted.')
r=subprocess.run(['/usr/bin/nvidia-smi' ,'--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,timeout=15,check=True)
if r.stdout.strip(): raise SystemExit('GPU already in use; refusing a second model load.')
print('RELEASE_IDENTITY_PASS GPU_IDLE model_metadata_matches_release')
