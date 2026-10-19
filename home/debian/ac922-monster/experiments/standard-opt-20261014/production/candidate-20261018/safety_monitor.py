"""Read existing thermal/tach sensors; stop only this model service on fault."""
import json, os, subprocess, sys, time
from pathlib import Path

BASE = Path('/home/debian/ac922-monster/experiments/standard-opt-20261014')
sys.path.insert(0, str(BASE/'notes/inband-guard-v222-20261017'))
from model_guard import collect, run
from guard_policy import fitted_inventory, inspect

STATE = Path('/run/ac922-inference/safety.json')
FAULT = Path('/var/lib/ac922-inference/fault.json')
UNIT = 'ac922-inference.service'
MIN_GPU_FREE_MIB = int(os.environ.get('AC922_MIN_GPU_FREE_MIB','256'))
if not 256 <= MIN_GPU_FREE_MIB <= 4096: raise ValueError('Invalid GPU free memory floor')

def publish(record):
    STATE.parent.mkdir(mode=0o755, exist_ok=True)
    temp = STATE.with_suffix('.tmp')
    temp.write_text(json.dumps(record,allow_nan=False)+'\n')
    os.chmod(temp,0o644)
    temp.replace(STATE)

def main():
    if FAULT.exists():
        publish({'unix':time.time(),'ready':False,'healthy':False,'latched':True,'error':'Previous safety fault requires manual inspection and acknowledgment.'})
        raise RuntimeError('Persisted safety latch exists')
    expected = None
    lows = {}
    good_since = time.monotonic()
    failures = 0
    try:
        while True:
            try:
                sample = collect()
                if expected is None:
                    expected = fitted_inventory(sample)
                errors, low = inspect(sample,expected,time.time())
                gpu = [[int(v.strip()) for v in line.split(',')] for line in run([
                    '/usr/bin/nvidia-smi','--query-gpu=memory.free,temperature.gpu',
                    '--format=csv,noheader,nounits']).splitlines()]
                if len(gpu)!=6 or any(free<MIN_GPU_FREE_MIB or temp>=58 for free,temp in gpu):
                    errors.append('GPU headroom or temperature guard')
                for name in ('fan0_0','fan2_0','fan3_0'):
                    lows[name] = lows.get(name,0)+1 if name in low else 0
                if any(count>=3 for count in lows.values()):
                    errors.append('persistent main fan low tach guard')
                if errors:
                    raise RuntimeError('; '.join(errors))
                failures=0
                if low: good_since=time.monotonic()
                ready = not low and time.monotonic()-good_since >= 15
                temps=[v for name,v in sample['hwmon_readings'].items() if name in expected]
                record={'unix':time.time(),'ready':ready,'healthy':not bool(low),
                        'temperature_max_c':max(temps),'gpu':gpu,'fans':sample['ipmi_readings'],
                        'hardware_configuration_modified':False}
                publish(record)
                print(json.dumps(record),flush=True)
                time.sleep(1 if low else 3)
            except Exception as error:
                failures+=1
                good_since=time.monotonic()
                publish({'unix':time.time(),'ready':False,'healthy':False,'error':str(error)})
                # Temperature or memory guard faults stop immediately; read faults retry twice.
                immediate = 'guard' in str(error).lower() or 'temperature' in str(error).lower()
                if immediate or failures>=3:
                    raise
                time.sleep(1)
    except BaseException as error:
        fault={'unix':time.time(),'ready':False,'healthy':False,'latched':True,'error':str(error)}
        FAULT.parent.mkdir(parents=True,exist_ok=True)
        FAULT.write_text(json.dumps(fault)+'\n')
        os.chmod(FAULT,0o600)
        publish(fault)
        subprocess.run(['/usr/bin/systemctl','stop',UNIT],timeout=55,check=False)
        raise

if __name__=='__main__': main()
