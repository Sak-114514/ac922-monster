"""One extra long multi wave, then the existing full HTTP behavior gate."""
from pathlib import Path
import json
import socket
import subprocess
import time
import sys
HERE=Path(__file__).resolve().parent
F5=HERE.parent/'safe-pool-reject-20261019'
q=json.loads((HERE/'run/qualification.json').read_text())
assert q['status']=='MODEL_COMPLETE_PENDING_LOG_AND_TEXT_REVIEW'
assert all(v['token_ids_exact'] and v['text_exact'] for v in q['aligned']['parity'].values())
report=dict(status='RUNNING',started_unix=time.time(),model_reloaded=False,worker_pid=q['waves'][0]['worker_pid'])
out=HERE/'run/extended-acceptance.json'
def save():out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
save()
telemetry=subprocess.Popen(['python3',str(F5/'gpu_telemetry.py'),str(HERE/'run/extension-gpu-telemetry.jsonl'),'1200'],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True)
try:
    requests=[dict(id=f'long32k-flow-{i}',corpus_token_start=i*16384,corpus_token_count=32768,max_tokens=512,ignore_eos=True,cache_prompt=False) for i in range(4)]
    with socket.socket(socket.AF_UNIX) as conn:
        conn.settimeout(480)
        conn.connect('/home/debian/ac922-monster/production/bulk256-c2-control.sock')
        conn.sendall(json.dumps(dict(id='bulk256-c2-four32k512-c1',requests=requests,refill=False)).encode()+b'\n')
        result=json.loads(conn.makefile('rb').readline(12<<20))
    report['long32k_wave']=result
    save()
    assert not result.get('errors') and len(result.get('done',{}))==4
    assert result['worker_pid']==report['worker_pid']
    assert result['timings']['prefill_input_tokens']==131072 and result['timings']['cached_tokens']==0
    assert all(result['done'][r['id']]['output_tokens']==512 for r in requests)
    print(json.dumps(dict(status='FOUR32K_MEASURED',prefill_tok_s=result['timings']['prefill_aggregate_wall_tok_s'],all_emissions_tok_s=result['timings']['all_emissions_tok_s'],worker_pid=result['worker_pid'])),flush=True)
    subprocess.run([sys.executable,str(F5/'lastwave_http_acceptance.py'),'--base-url','http://127.0.0.1:8018/v1','--key-file','/home/debian/apigateway/api_key','--out',str(HERE/'run/http-acceptance.json')],check=True,timeout=600)
    report['http']='PASS'
    report['status']='PASS_PENDING_LONG_TEXT_REVIEW'
except BaseException as error:
    report.update(status='FAIL',error=repr(error))
    raise
finally:
    report['finished_unix']=time.time();save()
    telemetry.terminate()
    try:telemetry.wait(timeout=5)
    except subprocess.TimeoutExpired:telemetry.kill();telemetry.wait()
