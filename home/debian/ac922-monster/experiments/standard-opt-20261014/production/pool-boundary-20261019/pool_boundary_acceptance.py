#!/usr/bin/env python3
"""Check the 508928-cell tiny-bank boundary and terminal overflow recovery.

Default execution is an in-memory CPU mock. Live execution requires --run and
uses only the already-loaded worker's private Unix control socket.
"""
from __future__ import annotations
import argparse, json, os, sys, time, uuid
from pathlib import Path
from typing import Any, Protocol

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import affinity_acceptance as aff  # noqa: E402

SOCKET=Path("/home/debian/ac922-monster/production/pool-boundary-control.sock")
OUT=HERE/"run/pool-boundary-result.json"
CASES=HERE.parents[1]/"fixtures/fourbank-d3-newruntime-v306-20261017/v221-d3-scenarios.json"
FIT=[169212,169212,169212,1276]
OVER=[174588,174588,174588,1276]
TARGET=[169216,169216,169216,1280]
POOL=524288

class GateError(RuntimeError): pass
class Transport(Protocol):
    def wave(self,label:str,requests:list[dict[str,Any]])->dict[str,Any]: ...

def need(ok:bool,msg:str)->None:
    if not ok: raise GateError(msg)

def native_cases(path:Path)->list[dict]:
    doc=json.loads(path.read_text(encoding="utf-8")); rows=doc[0]["requests"]
    need(len(rows)==4,"expected four original C12 native canaries")
    return [{"id":r["id"],"prompt":r["prompt"],"max_tokens":24,"_expected":r["expected"],
             "_checker":r["checker"],"_expected_bank":i} for i,r in enumerate(rows)]

def public_request(r:dict,capacity:int|None=None)->dict:
    row={k:v for k,v in r.items() if not k.startswith("_")}
    if capacity is not None: row["capacity_tokens"]=capacity
    return row

def geometry(result:dict)->dict[int,dict[str,int]]:
    value=aff.geometry_signature(result.get("timings",{}))
    need(value is not None,"worker omitted four-bank target/draft geometry")
    return value

def semantic(text:str,checker:str,expected:Any)->bool:
    if checker=="strict_json":
        try: return json.loads(text)==expected
        except json.JSONDecodeError: return False
    return text.strip()==str(expected)

def worker_path(explicit:str|None)->str:
    candidate=explicit or os.environ.get("AC922_WORKER_PATH")
    if candidate: return str(Path(candidate))
    for manifest in (HERE/"worker-command.json",HERE/"lastwave-worker-command.json"):
        try: cmd=json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError,json.JSONDecodeError): continue
        for item in cmd.get("argv",[]):
            if isinstance(item,str) and Path(item).name=="bank-worker": return item
    raise GateError("expected worker path missing; pass --worker-path or set AC922_WORKER_PATH")

def proc_pids(expected_path:str)->list[int]:
    found=[]
    proc=Path("/proc")
    for entry in proc.iterdir():
        if not entry.name.isdigit(): continue
        try: argv=[x.decode(errors="replace") for x in (entry/"cmdline").read_bytes().split(b"\0") if x]
        except (OSError,PermissionError): continue
        if expected_path in argv: found.append(int(entry.name))
    return sorted(found)

def check_native(result:dict,cases:list[dict],label:str)->dict:
    need(not result.get("errors"),f"{label}: native canary errors")
    need(len(result.get("accepted",{}))==4 and len(result.get("done",{}))==4,f"{label}: not all canaries completed")
    actual_input=0; rows={}
    for case in cases:
        rid=case["id"]; accepted=result["accepted"][rid]; done=result["done"][rid]
        text=result.get("texts",{}).get(rid,"")
        need(semantic(text,case["_checker"],case["_expected"]),f"{label}/{rid}: original semantic canary failed")
        need(accepted.get("bank")==case["_expected_bank"],f"{label}/{rid}: expected bank mapping changed")
        need(done.get("finish_reason")=="stop",f"{label}/{rid}: canary did not stop naturally")
        actual_input+=int(accepted.get("input_tokens",0))
        rows[rid]={"bank":accepted.get("bank"),"actual_input_tokens":accepted.get("input_tokens"),
                   "output_tokens":done.get("output_tokens"),"text":text}
    return {"actual_input_tokens_total":actual_input,"requests":rows}

def profile_values(waves:list[tuple[str,dict]])->dict[str,Any]:
    aliases={"maxpool":("production_pool","pool_capacity","pool_tokens","max_pool_tokens"),
             "maxrequest":("production_limit","request_limit","max_request_tokens","max_context_tokens"),
             "fixed_d3":("mtp_depth","draft_n_max","fixed_mtp_depth")}
    report={}
    for group,keys in aliases.items():
        observations=[]
        for name,result in waves:
            t=result.get("timings",{})
            for key in keys:
                if key in t and isinstance(t[key],(int,float)):
                    observations.append({"wave":name,"field":key,"value":t[key]}); break
        values={row["value"] for row in observations}
        report[group]={"observations":observations,"unchanged_when_reported":len(values)<=1,
                       "observed":bool(observations)}
        if group=="fixed_d3" and observations and any(row["value"]!=3 for row in observations):
            raise GateError("fixed MTP depth was not D3")
        if len(values)>1: raise GateError(f"{group} changed between waves")
    return report

def capacity_request(cases:list[dict],values:list[int])->list[dict]:
    return [public_request(case,values[i]) for i,case in enumerate(cases)]

def validate_fit(result:dict,cases:list[dict])->dict:
    need(not result.get("errors"),"fit-capacity wave returned errors")
    accepted=result.get("accepted",{}); done=result.get("done",{}); banks=geometry(result)
    actual_input=sum(int(accepted.get(case["id"],{}).get("input_tokens",0)) for case in cases)
    need(len(accepted)==4 and len(done)==4,"fit wave did not accept/complete four requests")
    target=[banks[i]["target_context"] for i in range(4)]
    draft=[banks[i]["draft_context"] for i in range(4)]
    need(target==TARGET and draft==TARGET,"actual target/draft capacities differ from expected aligned vector")
    need(sum(target)==508928 and sum(draft)==508928,"capacity totals are not 508928 per target/draft")
    need([banks[i]["target_ubatch"] for i in range(4)]==[128,128,128,256],
         "capacity wave differs from the existing large/small-context UBatch profiles")
    need(accepted[cases[3]["id"]].get("bank")==3 and target[3]==1280 and draft[3]==1280,
         "bank3 1280-cell active-kernel case was not actually used")
    for case in cases:
        rid=case["id"]; need(semantic(result["texts"][rid],case["_checker"],case["_expected"]),f"fit/{rid}: semantic mismatch")
        need(accepted[rid].get("bank")==case["_expected_bank"],f"fit/{rid}: bank mapping mismatch")
    return {"actual_input_tokens_total":actual_input,"capacity_tokens_requested":FIT,
            "capacity_tokens_requested_sum":sum(FIT),"target_context_by_bank":target,
            "draft_context_by_bank":draft,"target_context_total":sum(target),"draft_context_total":sum(draft),
            "active_tiny_bank3":True,"requests":{case["id"]:{"input_tokens":accepted[case["id"]].get("input_tokens"),
                "reserved_tokens":accepted[case["id"]].get("reserved_tokens"),"bank":accepted[case["id"]].get("bank"),
                "output_tokens":done[case["id"]].get("output_tokens"),"text":result["texts"][case["id"]]} for case in cases}}

def validate_overflow(result:dict,cases:list[dict])->dict:
    errs=result.get("errors",[]); ids={r["id"] for r in cases}
    need(len(errs)==4 and {e.get("id") for e in errs}==ids and all(e.get("status")==400 for e in errs),
         "overpool wave did not return four correlated HTTP-400 worker errors")
    need(not result.get("accepted") and not result.get("done"),"overpool wave incorrectly admitted/completed a request")
    need("timings" in result and isinstance(result["timings"],dict),"overpool wave_done was not observed")
    return {"capacity_tokens_requested":OVER,"capacity_tokens_requested_sum":sum(OVER),
            "aligned_pool_demand_cells":525056,"errors":errs,"accepted_count":0,"done_count":0,
            "wave_done_observed":True}

def run_campaign(transport:Transport,cases:list[dict],pid_reader)->dict:
    report={"schema":"ac922-pool-boundary-acceptance-v1","status":"RUNNING","waves":[],
            "capacity_interpretation":"capacity_tokens are context reservations; actual_input_tokens are reported separately."}
    try:
        pids_before=pid_reader(); need(len(pids_before)==1,"expected exactly one loaded worker PID before test")
        native0=transport.wave("pool-boundary-native-before",[public_request(c) for c in cases])
        report["waves"].append({"label":"native-before","wave_id":native0.get("id"),"accepted":native0.get("accepted"),
            "done":native0.get("done"),"errors":native0.get("errors"),"texts":native0.get("texts"),"timings":native0.get("timings")})
        native0_sem=check_native(native0,cases,"native-before"); geo0=geometry(native0)
        report["waves"][-1].update(semantics=native0_sem,geometry=geo0,wall_s=native0.get("wall_s"))
        pids_after_native0=pid_reader(); need(pids_after_native0==pids_before,"worker PID changed after baseline native wave")

        fit_reqs=capacity_request(cases,FIT)
        fit=transport.wave("pool-boundary-fit-508928",fit_reqs)
        report["waves"].append({"label":"fit-capacity","wave_id":fit.get("id"),"accepted":fit.get("accepted"),
            "done":fit.get("done"),"errors":fit.get("errors"),"texts":fit.get("texts"),"timings":fit.get("timings"),
            "wall_s":fit.get("wall_s"),"capacity_tokens_requested":FIT,"capacity_tokens_requested_sum":sum(FIT)})
        fit_data=validate_fit(fit,cases); report["waves"][-1]["data"]=fit_data
        pids_after_fit=pid_reader(); need(pids_after_fit==pids_before,"worker PID changed after fit-capacity wave")

        over_reqs=capacity_request(cases,OVER)
        over=transport.wave("pool-boundary-overpool-400",over_reqs)
        report["waves"].append({"label":"overpool-terminal-errors","wave_id":over.get("id"),
            "errors":over.get("errors"),"accepted":over.get("accepted"),"done":over.get("done"),
            "timings":over.get("timings"),"capacity_tokens_requested":OVER,"capacity_tokens_requested_sum":sum(OVER)})
        over_data=validate_overflow(over,cases); report["waves"][-1]["data"]=over_data
        pids_after_over=pid_reader(); need(pids_after_over==pids_before,"worker exited/restarted on terminal overpool errors")

        native1=transport.wave("pool-boundary-native-after-overflow",[public_request(c) for c in cases])
        report["waves"].append({"label":"native-after-overflow","wave_id":native1.get("id"),
            "accepted":native1.get("accepted"),"done":native1.get("done"),"errors":native1.get("errors"),
            "texts":native1.get("texts"),"timings":native1.get("timings"),"wall_s":native1.get("wall_s")})
        native1_sem=check_native(native1,cases,"native-after-overflow")
        report["waves"][-1].update(semantics=native1_sem,geometry=geometry(native1))
        pids_after_native1=pid_reader(); need(pids_after_native1==pids_before,"worker PID changed after recovery native wave")
        profiles=profile_values([(name,row) for name,row in (("native-before",native0),("fit-capacity",fit),
            ("overpool",over),("native-after-overflow",native1))])
        report.update(status="PASS",worker_pid={"path":getattr(pid_reader,"worker_path",None),
            "before":pids_before,"after_native_before":pids_after_native0,"after_fit":pids_after_fit,
            "after_overpool":pids_after_over,"after_native_recovery":pids_after_native1},
            profile_invariants=profiles)
    except Exception as exc:
        report.update(status="FAIL",failure=f"{type(exc).__name__}: {exc}")
    return report

class MockTransport:
    def __init__(self,cases:list[dict],failure:str|None=None):
        self.failure=failure; self.geometry=[17152]*4; self.wave_count=0
        self.expected={case["id"]:(case.get("_checker",case.get("checker")),case["_expected"])
                       for case in cases}
    def wave(self,label:str,requests:list[dict])->dict:
        self.wave_count+=1; accepted={}; done={}; errors=[]; texts={}; cache={}
        is_over="overpool" in label
        if is_over:
            errors=[{"type":"error","id":r["id"],"status":400,"message":"pool exceeded"} for r in requests]
            if self.failure=="bad_overflow": errors[0]["status"]=500
            if self.failure=="missing_wave_done": return {"id":label,"errors":errors,"accepted":{},"done":{}}
            return {"id":label,"errors":errors,"accepted":{},"done":{},"timings":{},"wall_s":0.01}
        if "fit-508928" in label:
            self.geometry=TARGET.copy()
            if self.failure=="bad_geometry": self.geometry[3]=1536
        banks=[{"bank":i,"target_context":self.geometry[i],"draft_context":self.geometry[i],
                "target_ubatch":128 if self.geometry[i]>66048 else 256} for i in range(4)]
        if self.failure=="bad_ubatch" and "fit-508928" in label: banks[0]["target_ubatch"]=256
        for i,r in enumerate(requests):
            rid=r["id"]; bank=i
            checker,expected=self.expected.get(rid,("exact_text",None))
            for line in str(r.get("prompt"," ")).splitlines():
                if line.startswith("NOVA = "):
                    checker,expected="exact_text",line.removeprefix("NOVA = ").strip(); break
            if expected is None: raise AssertionError(f"mock missing expected answer for {rid}")
            text=json.dumps(expected,ensure_ascii=False) if checker=="strict_json" else str(expected)
            if self.failure=="bad_answer" and "native-after" in label: text="wrong"
            inp=40
            accepted[rid]={"bank":bank,"input_tokens":inp,"reserved_tokens":inp+r["max_tokens"]}
            done[rid]={"bank":bank,"input_tokens":inp,"output_tokens":1,"finish_reason":"stop"}; texts[rid]=text
        return {"id":label,"accepted":accepted,"done":done,"errors":errors,"texts":texts,"wall_s":0.1,
            "timings":{"banks":banks,"mtp_depth":3,"production_pool":POOL,"production_limit":200000}}

def self_test(cases:list[dict])->dict:
    good=run_campaign(MockTransport(cases),cases,MockPidReader())
    need(good["status"]=="PASS","positive pool-boundary mock failed")
    rejected=[]
    for mode in ("bad_geometry","bad_ubatch","bad_overflow","bad_answer","worker_exit","missing_wave_done"):
        result=run_campaign(MockTransport(cases,mode),cases,MockPidReader(mode))
        need(result["status"]=="FAIL",f"negative mock was not rejected: {mode}")
        rejected.append(mode)
    fit=good["waves"][1]["data"]
    return {"self_test":"PASS","transport":"in-memory mock; no socket/API/GPU/proc access",
            "waves":len(good["waves"]),"actual_input_tokens_fit_total":fit["actual_input_tokens_total"],
            "requested_capacity_sum":fit["capacity_tokens_requested_sum"],
            "target_capacity_sum":fit["target_context_total"],"negative_modes_rejected":rejected}

class MockPidReader:
    worker_path="/mock/bank-worker"
    def __init__(self,failure:str|None=None): self.failure=failure; self.calls=0
    def __call__(self)->list[int]:
        self.calls+=1
        return [9002 if self.failure=="worker_exit" and self.calls>=4 else 9001]

def main()->int:
    p=argparse.ArgumentParser(description=__doc__); mode=p.add_mutually_exclusive_group()
    mode.add_argument("--run",action="store_true"); mode.add_argument("--self-test",action="store_true")
    p.add_argument("--socket",default=str(SOCKET)); p.add_argument("--cases",default=str(CASES))
    p.add_argument("--worker-path",default=os.environ.get("AC922_WORKER_PATH")); p.add_argument("--out",default=str(OUT))
    args=p.parse_args(); cases=native_cases(Path(args.cases).expanduser())
    if not args.run:
        report=self_test(cases); print(json.dumps(report,ensure_ascii=False)); return 0
    try:
        expected=worker_path(args.worker_path)
        pid_reader=lambda:proc_pids(expected)
        pid_reader.worker_path=expected
        report=run_campaign(aff.UnixWaveTransport(Path(args.socket),timeout_s=600),cases,pid_reader)
    except Exception as exc: report={"schema":"ac922-pool-boundary-acceptance-v1","status":"FAIL","failure":f"{type(exc).__name__}: {exc}"}
    aff.write_report(Path(args.out).expanduser(),report); print(json.dumps({"status":report["status"],"result":args.out},ensure_ascii=False))
    return 0 if report["status"]=="PASS" else 1

if __name__=="__main__": raise SystemExit(main())
