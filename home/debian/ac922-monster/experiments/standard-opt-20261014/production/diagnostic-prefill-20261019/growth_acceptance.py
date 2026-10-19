#!/usr/bin/env python3
"""Growth-cache acceptance harness. Default is CPU mock; only --run uses the control socket."""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
from typing import Any, Protocol

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import affinity_acceptance as aff  # noqa: E402

SOCKET=Path("/home/debian/ac922-monster/production/safe-pool-reject-control.sock")
OUT=HERE/"run/growth-result.json"
FLOOR_MIB=256
FIXED_MARGIN_MIB=256
UNIT_MIB=128
BUCKET=8192

class GateError(RuntimeError): pass
class Transport(Protocol):
    def wave(self,label:str,requests:list[dict[str,Any]])->dict[str,Any]: ...

def need(ok:bool,msg:str)->None:
    if not ok: raise GateError(msg)

def vector(value:Any)->list[int]|None:
    if isinstance(value,list) and len(value)==4 and all(isinstance(x,int) and x>=0 for x in value): return value
    if isinstance(value,dict):
        try:
            result=[int(value.get(str(i),value.get(i))) for i in range(4)]
            return result if all(x>=0 for x in result) else None
        except (TypeError,ValueError): return None
    return None

def requests_for(body:str,indices:tuple[int,...],label:str,*,max_tokens:int=16,edited:bool=False)->list[dict]:
    values=("7301","4159")
    result=[]
    for n,index in enumerate(indices):
        doc=aff.DOCUMENTS[index]
        req=aff.request_for(doc,body,revision=label if edited else "baseline",
                            answer=values[n] if edited else None,expected_bank=index)
        req["id"]=f"{label}-{doc[0]}" if label!="growth-seed4" else doc[0]
        req["max_tokens"]=max_tokens
        result.append(req)
    return result

def request_summary(label:str,requests:list[dict],result:dict)->dict:
    return {"label":label,"wave_id":result.get("id"),"wall_s":result.get("wall_s"),
        "control_roundtrip_s":result.get("control_roundtrip_s"),
        "events":{r["id"]:{"accepted":result.get("accepted",{}).get(r["id"],{}),
            "cache":result.get("cache_status",{}).get(r["id"],{}),"done":result.get("done",{}).get(r["id"]),
            "text":result.get("texts",{}).get(r["id"],"")} for r in requests},
        "timings":result.get("timings",{})}

def public_wave(transport:Transport,label:str,requests:list[dict],report:dict|None=None)->dict:
    result=transport.wave(label,requests)
    if report is not None: report["waves"].append(request_summary(label,requests,result))
    need(not result.get("errors"),f"{label}: worker errors: {result.get('errors')}")
    need(len(result.get("done",{}))==len(requests),f"{label}: incomplete wave")
    for req in requests:
        rid=req["id"]
        need(result.get("texts",{}).get(rid,"").strip()==req["_expected"],f"{label}/{rid}: NOVA answer mismatch")
        need(result["done"][rid].get("finish_reason")=="stop",f"{label}/{rid}: output did not stop naturally")
        expected_bank=req.get("_expected_bank")
        if expected_bank is not None: need(result["accepted"][rid].get("bank")==expected_bank,f"{label}/{rid}: owner bank mismatch")
    return result

def geometry(timing:dict)->dict[int,dict[str,int]]:
    value=aff.geometry_signature(timing)
    need(value is not None,"worker omitted complete four-bank geometry")
    return value

def events(result:dict,requests:list[dict])->dict[str,dict]:
    return {req["id"]:{"accepted":result.get("accepted",{}).get(req["id"],{}),
        "cache":result.get("cache_status",{}).get(req["id"],{}),
        "done":result.get("done",{}).get(req["id"]),"text":result.get("texts",{}).get(req["id"],"")}
        for req in requests}

def validate_growth(first:dict,requests:list[dict],seed_geometry:dict[int,dict[str,int]])->dict:
    t=first.get("timings",{})
    exact=vector(t.get("capacity_growth_exact_capacity")); candidate=vector(t.get("capacity_growth_candidate_capacity"))
    need(t.get("capacity_growth_enabled") is True,"7900 wave growth option was not enabled")
    need(t.get("capacity_growth_applied") is True and t.get("capacity_growth_reason")=="applied",
         "7900 wave did not apply optional capacity growth")
    need(exact is not None and candidate is not None,"growth exact/candidate capacities were not reported as four-bank numbers")
    reserved={req["id"]:first["accepted"][req["id"]].get("reserved_tokens") for req in requests}
    need(all(isinstance(v,int) for v in reserved.values()),"7900 actual reserved_tokens missing")
    for req in requests:
        bank=first["accepted"][req["id"]].get("bank")
        need(first["cache_status"][req["id"]].get("cached_tokens")==0,
             f"bank{bank}: a rebuilt context must cold-start its first checkpoint")
        need(bank in (0,1),f"7900 {req['id']} mapped outside bank0/1")
        need(exact[bank]>=reserved[req["id"]],f"bank{bank}: exact capacity is below actual reserved_tokens")
        need(exact[bank]>seed_geometry[bank]["target_context"],f"exact capacity for bank{bank} did not exceed 17152")
        need(candidate[bank]>=exact[bank],f"bank{bank}: candidate capacity is below exact aligned capacity")
        need(candidate[bank]==24576,f"bank{bank} candidate was not rounded to 24576")
    for bank in (2,3):
        need(candidate[bank]==seed_geometry[bank]["target_context"],f"candidate changed unrelated bank{bank}")
    actual=geometry(t)
    need(all(actual[b]["target_context"]==24576 and actual[b]["target_ubatch"]==256 for b in (0,1)),
         "two grown banks are not 24576/UB256")
    need(all(actual[b]==seed_geometry[b] for b in (2,3)),"growth changed an unrelated bank")
    need(t.get("context_rebuilds_before_execution")==2,"growth wave did not perform exactly two bank rebuilds")
    units=t.get("capacity_growth_units")
    calculated=sum((max(0,candidate[i]-seed_geometry[i]["target_context"])+BUCKET-1)//BUCKET for i in range(4))
    need(isinstance(units,int) and units==calculated,f"growth units mismatch: logged={units}, calculated={calculated}")
    required=FLOOR_MIB+FIXED_MARGIN_MIB+UNIT_MIB*calculated
    need(t.get("production_floor_mib",FLOOR_MIB)==FLOOR_MIB,"production floor is not 256 MiB")
    need(t.get("capacity_growth_required_free_mib")==required,f"growth headroom budget mismatch: expected {required}")
    free=t.get("capacity_growth_preflight_free_mib")
    need(isinstance(free,list) and len(free)==6 and all(isinstance(x,int) for x in free),
         "growth preflight must report free MiB for all six GPUs")
    need(all(x>=required and x>=FLOOR_MIB for x in free),"one or more GPUs fail growth headroom/floor")
    return {"exact_capacity":exact,"candidate_capacity":candidate,"reserved_tokens":reserved,
            "reserved_tokens_by_document":{req["_document_id"]:reserved[req["id"]] for req in requests},
            "bank_by_document":{req["_document_id"]:first["accepted"][req["id"]]["bank"] for req in requests},
            "prompt_sha256_by_document":{req["_document_id"]:req["_prompt_sha256"] for req in requests},
            "growth_units_logged":units,"growth_units_calculated":calculated,
            "required_free_mib":required,"preflight_free_mib":free,"geometry":actual,
            "context_rebuilds_before_execution":t["context_rebuilds_before_execution"]}

def validate_larger_repeat(first:dict,second:dict,requests:list[dict],first_gate:dict)->dict:
    t=second.get("timings",{}); exact=vector(t.get("capacity_growth_exact_capacity"))
    reserved={req["id"]:second["accepted"][req["id"]].get("reserved_tokens") for req in requests}
    old=first_gate["reserved_tokens_by_document"]
    need(exact is not None and all(isinstance(v,int) for v in reserved.values()),
         "larger-repeat actual reserved/exact capacity numbers unavailable")
    for req in requests:
        rid=req["id"]; bank=second["accepted"][rid].get("bank")
        doc=req["_document_id"]
        need(req["_prompt_sha256"]==first_gate["prompt_sha256_by_document"][doc],f"{doc}: repeat prompt changed")
        need(reserved[rid]>old[doc],f"{doc}: max_tokens=8192 did not increase actual allocation")
        required_aligned=(reserved[rid]+4+255)//256*256
        need(required_aligned>first_gate["exact_capacity"][bank],
             f"bank{bank}: repeat would still fit the first exact capacity; avoided-growth benefit unproven")
        need(exact[bank]>first_gate["exact_capacity"][bank],f"bank{bank}: exact capacity did not increase from 7900 case")
        need(second["accepted"][rid].get("bank")==first_gate["bank_by_document"][doc],f"{doc}: bank changed")
    need(t.get("context_rebuilds_before_execution")==0,"larger repeat unexpectedly rebuilt contexts")
    banks=geometry(t); need(banks==first_gate["geometry"],"larger repeat changed the grown geometry")
    for req in requests:
        need(second.get("cache_status",{}).get(req["id"],{}).get("cached_tokens")==10496,
             f"{req['id']}: expected 10496-token prefix hit after growth")
    return {"exact_capacity":exact,"reserved_tokens":reserved,"strictly_larger_than_7900":True,
            "geometry":banks,"context_rebuilds_before_execution":0,"cache_tokens":
            {req["id"]:second["cache_status"][req["id"]]["cached_tokens"] for req in requests}}

def build_campaign(transport:Transport,source:str)->dict:
    body=aff.make_corpus_body(source); seed_reqs=requests_for(body,(0,1,2,3),"growth-seed4")
    grow_reqs=requests_for(body,(0,1),"growth-7900-ab",max_tokens=7900)
    repeat_reqs=requests_for(body,(0,1),"growth-8192-ab",max_tokens=8192)
    edit_reqs=requests_for(body,(0,1),"growth-edit-ab",edited=True)
    summary={"schema":"ac922-growth-cache-acceptance-v1","status":"RUNNING","started_unix":time.time(),
        "corpus_characters":len(body),"waves":[]}
    try:
        seed=public_wave(transport,"growth-seed4",seed_reqs,summary); seed_geo=geometry(seed.get("timings",{}))
        need(all(seed_geo[i]["target_context"]==17152 and seed_geo[i]["target_ubatch"]==256 for i in range(4)),
             "seed profile is not four banks at 17152/UB256")
        grow=public_wave(transport,"growth-7900-ab",grow_reqs,summary); grow_gate=validate_growth(grow,grow_reqs,seed_geo)
        summary["growth_gate"]={"wave":"growth-7900-ab",**grow_gate}
        repeat=public_wave(transport,"growth-8192-ab",repeat_reqs,summary)
        repeat_gate=validate_larger_repeat(grow,repeat,repeat_reqs,grow_gate)
        summary["repeat_gate"]={"wave":"growth-8192-ab",**repeat_gate}
        edit=public_wave(transport,"growth-edit-ab",edit_reqs,summary)
        edit_geo=geometry(edit.get("timings",{})); need(edit_geo==repeat_gate["geometry"],"early edit changed bank geometry")
        need(edit.get("timings",{}).get("context_rebuilds_before_execution")==0,"early edit rebuilt an unchanged profile")
        for req in edit_reqs:
            rid=req["id"]; need(edit["cache_status"][rid].get("cached_tokens")==0,f"{rid}: edited prefix falsely hit")
            need(edit["accepted"][rid].get("bank")==req["_expected_bank"],f"{rid}: edited route changed bank")
            need(edit["texts"][rid].strip()==req["_expected"],f"{rid}: edited NOVA answer wrong")
        summary["early_edit_gate"]={"geometry_unchanged":True,"cache_tokens":
            {req["id"]:edit["cache_status"][req["id"]]["cached_tokens"] for req in edit_reqs},
            "answers":{req["id"]:edit["texts"][req["id"]].strip() for req in edit_reqs},
            "prompt_prefixes_changed":{req["_document_id"]:req["_prompt_sha256"]!=grow_reqs[i]["_prompt_sha256"]
                                       for i,req in enumerate(edit_reqs)}}
        need(all(summary["early_edit_gate"]["prompt_prefixes_changed"].values()),
             "early-edit probes did not change their prompt prefix")
        summary["status"]="PASS"
    except Exception as exc:
        summary.update(status="FAIL",failure=f"{type(exc).__name__}: {exc}")
    summary["finished_unix"]=time.time()
    return summary

class MockTransport:
    def __init__(self,failure:str|None=None):
        self.failure=failure; self.cache={}; self.geometry=[17152]*4
        self.docs={d[1]:d for d in aff.DOCUMENTS}; self.bank={d[1]:i for i,d in enumerate(aff.DOCUMENTS)}
    def wave(self,label:str,requests:list[dict])->dict:
        accepted={}; done={}; cache={}; texts={}
        exact=self.geometry.copy(); candidate=self.geometry.copy(); units=0; applied=False
        reason="no_eligible_growth"; rebuilds=0; free=[4096]*6
        if label=="growth-7900-ab":
            exact=[18944,18944,17152,17152]; candidate=[24576,24576,17152,17152]
            units=2; applied=True; reason="applied"; rebuilds=2; free=[2048]*6
            self.geometry=candidate.copy(); self.cache.clear()
            if self.failure=="bad_geometry": self.geometry[0]=23040
        elif label=="growth-8192-ab":
            exact=[19200,19200,17152,17152]; candidate=self.geometry.copy(); free=[2048]*6
        for r in requests:
            rid=r["id"]; key=r["prompt"]; docid=r["_document_id"]
            # The seed owns original DOCs in banks 0..3; A/B growth requests target 0/1.
            b=self.bank[docid]
            inp=10800; reserved=inp+r["max_tokens"]
            cached=10496 if self.cache.get(b)==key else 0
            if self.failure=="bad_cache" and label=="growth-8192-ab": cached=0
            if label=="growth-edit-ab": cached=0
            answer=str(r["_expected"])
            if self.failure=="bad_answer" and label=="growth-edit-ab": answer="6842"
            if r["cache_prompt"]: self.cache[b]=key
            accepted[rid]={"input_tokens":inp,"reserved_tokens":reserved,"bank":b}
            cache[rid]={"cached_tokens":cached,"hit":cached>0,"bank":b}
            done[rid]={"output_tokens":1,"finish_reason":"stop","bank":b}; texts[rid]=answer
        t={"banks":[{"bank":i,"target_context":self.geometry[i],"draft_context":self.geometry[i],"target_ubatch":256} for i in range(4)],
           "context_rebuilds_before_execution":rebuilds,"capacity_growth_enabled":label=="growth-7900-ab",
           "capacity_growth_applied":applied,"capacity_growth_reason":reason,
           "capacity_growth_exact_capacity":exact,"capacity_growth_candidate_capacity":candidate,
           "capacity_growth_units":units,"capacity_growth_required_free_mib":768 if label=="growth-7900-ab" else 0,
           "capacity_growth_preflight_free_mib":free,"production_floor_mib":256}
        return {"id":label+"-mock","accepted":accepted,"cache_status":cache,"done":done,
                "texts":texts,"errors":[],"wall_s":1.0,"timings":t}

def self_test(source:str)->dict:
    good=build_campaign(MockTransport(),source)
    assert good["status"]=="PASS"
    rejected=[]
    for mode in ("bad_geometry","bad_cache","bad_answer"):
        failure=build_campaign(MockTransport(mode),source)
        if failure["status"]!="FAIL": raise AssertionError(f"negative arm not rejected: {mode}")
        rejected.append(mode)
    assert rejected==["bad_geometry","bad_cache","bad_answer"],rejected
    return {"self_test":"PASS","transport":"in-memory mock; no socket/API/model/GPU",
            "waves":len(good["waves"]),"negative_modes_rejected":rejected,
            "growth_units":good["growth_gate"]["growth_units_calculated"],
            "required_free_mib":good["growth_gate"]["required_free_mib"]}

def main()->int:
    p=argparse.ArgumentParser(description=__doc__); mode=p.add_mutually_exclusive_group()
    mode.add_argument("--run",action="store_true"); mode.add_argument("--self-test",action="store_true")
    p.add_argument("--socket",default=str(SOCKET))
    p.add_argument("--corpus",default=str(HERE.parents[1]/"fixtures/long-prefill-combo-v45-20261016/v45-real-corpus-source.json"))
    p.add_argument("--out",default=str(OUT)); args=p.parse_args()
    source=aff.load_corpus(Path(args.corpus).expanduser())
    if args.self_test or not args.run: print(json.dumps(self_test(source),ensure_ascii=False)); return 0
    out=Path(args.out).expanduser(); report={"schema":"ac922-growth-cache-acceptance-v1","status":"RUNNING"}
    try: report=build_campaign(aff.UnixWaveTransport(Path(args.socket),timeout_s=1200),source)
    except Exception as exc: report.update(status="FAIL",failure=f"{type(exc).__name__}: {exc}")
    aff.write_report(out,report); print(json.dumps({"status":report["status"],"result":str(out)},ensure_ascii=False))
    return 0 if report["status"]=="PASS" else 1

if __name__=="__main__": raise SystemExit(main())
