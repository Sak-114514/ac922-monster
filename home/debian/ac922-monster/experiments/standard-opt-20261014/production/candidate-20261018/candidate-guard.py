"""Temporary model guard. Only IPMI reads; model unit starts/stops as debian."""
from pathlib import Path
import json,math,os,subprocess,sys,time
sys.path.insert(0,'/home/debian/ac922-monster/experiments/standard-opt-20261014/notes/inband-guard-v222-20261017')
from guard_policy import fitted_inventory,inspect

HERE=Path('/home/debian/ac922-monster/experiments/standard-opt-20261014/notes/inband-guard-v222-20261017')

def run(args, timeout=8):
    p=subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if p.returncode: raise RuntimeError('command failed: '+str(args[0])+': '+p.stderr[-500:])
    return p.stdout


def collect():
    started=time.time()
    raw=run(['/usr/bin/ipmitool','-S',str(HERE/'sdr.cache'),'sensor','reading','fan0_0','fan2_0','fan3_0','ambient','pcie'])
    ipmi={}
    for line in raw.splitlines():
        if not line.strip(): continue
        key,value=line.split('|',1)
        value=float(value.strip()); assert math.isfinite(value)
        ipmi[key.strip()]=value
    hwmon={}
    for folder in Path('/sys/class/hwmon').glob('hwmon*'):
        if not (folder/'name').exists() or (folder/'name').read_text().strip()!='ibmpowernv': continue
        for p in folder.glob('temp*_label'):
            label=p.read_text().strip()
            hwmon[label]=int(p.with_name(p.name.replace('_label','_input')).read_text())/1000
    return dict(sample_started_unix=started,sample_completed_unix=time.time(),ipmi_readings=ipmi,
                hwmon_readings=hwmon,errors=[],provenance='inband_ipmi_sdr_cache_and_powernv_hwmon',
                bmc_mask_status='unavailable_not_synthesized',bmc_fan_target='unavailable_not_synthesized')


def user_command(args):
    return ['/usr/sbin/runuser','-u','debian','--','/usr/bin/env',
            'XDG_RUNTIME_DIR=/run/user/1000','DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus',*args]


def stop(unit):
    from stop_model import stop as verified_stop
    return verified_stop(unit)


def main():
    manifest=json.loads(Path(sys.argv[1]).read_text())
    unit=manifest['unit']
    gpu_floor=int(manifest.get('gpu_free_floor_mib',512))
    assert 256 <= gpu_floor <= 4096
    from stop_model import command as verified_stop_command
    verified_stop_command(unit, ['show'])  # Validate the exact same allowlist as the stop hook.
    output=Path(manifest['output']); assert any(str(output).startswith(str(HERE.parents[1])+'/fixtures/'+name+'/results/') for name in ('layer6-binding-prefill-v219-20261017','layer6-cupti32k-v223-20261017','layer6-props-pinned-fourbank-v229-20261017','single64k-props-pinned-v233-20261017','layer6-smallj-fourbank-v234-20261017','single64k-smallj512-v237-20261017','smallj-fourbank-d4-v238-20261017','mtp-base-balanced-v244-20261017','base-balanced-combo-v253-20261017','mtp-clones-headroom-v255-20261017','mtp-unrolled-originalmap-v259-20261017','layer6-gpu-ple-v271-20261017','layer6-gpu-ple-copy-v277-20261017','ple-stream-combo-v281-20261017','ple-pipeline-combo-v289-20261017','layer6-twobank-twolane-v284-20261017','layer6-twobank-hybrid-v296-20261017','single32k-largeub-v300-20261017','fourbank-d4-combo-v301-20261017','fourbank-d3-newruntime-v306-20261017','single64k-golden-fresh-v307-20261018','crossbank-draft-v355-20261018','crossbank-progress-batch-v370-20261018','single64k-ub-matrix-v361-20261018','prefill-current-trace-v371-20261018','op-chunks-single64k-v377-root-20261018','op-chunks-single64k-v378-root-20261018','tail-relief-spans-v382-root-20261018','tail-relief-spans-v383-root-20261018','prefill-upload-stream-v386-root-20261018','prefill-upload-combo-v392-root-20261018','fourbank-compact-argmax-v395-root-20261018','prefill-host-child-v397-root-20261018','fourbank-eightcopy-v401-root-20261018','prefill-plan-rebind-v406-root-r2-20261018', 'prefill-eightplan-v412-root-20261018', 'fourbank-paced-target-tail-v413-root-20261018', 'prefill-macro2048-v417-root-20261018', 'fourbank-wave-paced-v418-root-20261018', 'mmq-jcap-v421-root-20261018', 'mmq-decode-jcap-v424-root-20261018', 'prefill64k-ub1024-v427-root-20261018', 'tail-stage-audit-v428-root-20261018', 'prefill-tail-stage-v429-root-20261018', 'layer6-onebank4-v430-root-20261018', 'tail-cpu-audit-v432-root-20261018', 'prefill-tail-cpu-v433-root-20261018', 'layer6-onebank4-v434-root-20261018', 'prefill-first1536-v435-root-20261018', 'six-cuda-prepare-v436-root-20261018', 'prefill-parallelprepare-v436-root-20261018', 'layer6-onebank4-v437-root-20261018', 'fourbank-hostchild-v438-root-20261018', 'six-hostchild-prepare-v439-root-20261018', 'prefill-hostchild-coldwarm-v440-root-20261018', 'shared-hostexec-v441-root-20261018', 'fourbank-numa-owner-v443-root-20261018', 'graph-lease-v444-root-20261018', 'layer6-onebank4-v426-root-20261018', 'prefill-midub-v420-root-20261018', 'fourbank-target-tail-v410-root-20261018'))
    log=output/'inband-guard.jsonl'; expected=None; launched=False; low_count={}; started=time.monotonic()
    def emit(record):
        with log.open('a') as f: f.write(json.dumps(record,allow_nan=False)+'\n')
    try:
        s=collect(); expected=fitted_inventory(s); errors,low=inspect(s,expected,time.time())
        if errors or low: raise RuntimeError('preflight '+str((errors,low)))
        gpus=run(['/usr/bin/nvidia-smi','--query-gpu=memory.free,temperature.gpu,utilization.gpu','--format=csv,noheader,nounits'])
        rows=[[int(v.strip()) for v in row.split(',')] for row in gpus.splitlines()]
        if len(rows)!=6 or any(free<gpu_floor or temp>=58 or util>2 for free,temp,util in rows): raise RuntimeError('GPU preflight')
        pids=run(['/usr/bin/nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).strip()
        if pids: raise RuntimeError('GPU busy before model')
        emit(dict(event='PREFLIGHT_PASS',sample=s,hwmon_inventory=sorted(expected),gpu=rows,
                  limitations='BMC mask/target unavailable; inband readings only; no BMC/fan configuration changed'))
        command=manifest['model_launch_command']
        if command[0]!='systemd-run' or '--unit='+unit not in command: raise RuntimeError('unit launch mismatch')
        run(user_command(command),timeout=20); launched=True
        emit(dict(event='MODEL_STARTED',unit=unit,unix=time.time()))
        while True:
            raw=run(user_command(['/usr/bin/systemctl','--user','show',unit,'-p','ActiveState','-p','Result','-p','ExecMainStatus']))
            state=dict(line.split('=',1) for line in raw.splitlines() if '=' in line)
            if state.get('ActiveState') in ('inactive','failed'):
                emit(dict(event='MODEL_FINISHED',state=state,unix=time.time())); return 0 if state.get('Result')=='success' and state.get('ExecMainStatus')=='0' else 1
            if time.monotonic()-started>1840: raise RuntimeError('guard runtime bound')
            s=collect(); errors,low=inspect(s,expected,time.time())
            for fan in ('fan0_0','fan2_0','fan3_0'): low_count[fan]=low_count.get(fan,0)+1 if fan in low else 0
            rows=[[int(v.strip()) for v in row.split(',')] for row in run(['/usr/bin/nvidia-smi','--query-gpu=memory.free,temperature.gpu','--format=csv,noheader,nounits']).splitlines()]
            if len(rows)!=6 or any(free<gpu_floor or temp>=58 for free,temp in rows): errors.append('GPU temperature/free guard')
            if any(v>=3 for v in low_count.values()): errors.append('three consecutive low tach samples')
            emit(dict(event='SAMPLE',sample=s,gpu=rows,low_counts=low_count,errors=errors))
            if errors: raise RuntimeError(str(errors))
            time.sleep(1 if low else 3)
    except Exception as e:
        emit(dict(event='GUARD_STOP',reason=str(e),unix=time.time()))
        if launched: stop(unit)
        raise
    finally:
        if launched: stop(unit)

if __name__=='__main__': raise SystemExit(main())
