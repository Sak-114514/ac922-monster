import json,time
from pathlib import Path
p=Path('/run/ac922-inference/safety.json')
for _ in range(45):
    try:
        d=json.loads(p.read_text())
        if d.get('ready') and d.get('healthy') and time.time()-d['unix']<8:
            raise SystemExit(0)
        if d.get('latched'): raise SystemExit('Safety monitor fault is latched; inspect before restart.')
    except (OSError,ValueError,KeyError): pass
    time.sleep(1)
raise SystemExit('Safety monitor not ready; model not started.')
