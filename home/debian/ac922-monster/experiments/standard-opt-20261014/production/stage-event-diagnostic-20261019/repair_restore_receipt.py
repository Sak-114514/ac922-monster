"""Repair only the post-start receipt; do not restart the already restoring L5."""
import json
import os
from pathlib import Path
import subprocess
import time

assert os.geteuid() == 0
root = Path('/home/debian/ac922-monster/production')
backup = root / 'backups/stage-event-20261018T230336Z'
assert (root / 'current').resolve() == root / 'releases/20261019-r444-banked-l5'
assert not Path('/etc/systemd/system/ac922-inference.service.d/96-stage-event-diagnostic.conf').exists()
assert not Path('/var/lib/ac922-inference/fault.json').exists()
for unit in ['ac922-inference.service', 'ac922-apigateway.service']:
    assert subprocess.check_output(['systemctl', 'is-active', unit], text=True).strip() == 'active'
result = {'status': 'L5_RESTORE_STARTED', 'fault_latched': False,
          'readiness_still_requires_verification': True, 'unix': time.time(),
          'diagnostic_override_removed': True, 'root_timer_stopped': True,
          'original_restore_error_after_start': 'NameError only while constructing final receipt, after L5 unit start; no repeated stop/start',
          'inference_backend_port': 8007}
(backup / 'restored.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
