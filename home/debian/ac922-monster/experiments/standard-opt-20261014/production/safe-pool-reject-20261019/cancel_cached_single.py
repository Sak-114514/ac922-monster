#!/usr/bin/env python3
"""HTTP disconnect/cache-reuse regression. Default mode is offline; live calls need --run."""
from __future__ import annotations
import argparse, asyncio, json, os, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import affinity_acceptance as aff  # noqa: E402
import lastwave_http_acceptance as http  # noqa: E402

MODEL = "flash-next-iq3-xxs"
Q_SUFFIX = "QUESTION: For the document identified at the beginning, what is NOVA? Return only its decimal value.\n"
Q_LONG = ("QUESTION: Write a detailed, coherent explanation of the reference material in at least "
          "2000 words. Develop the main ideas in connected prose and do not end early.\n")

def seed(source: str, sock: Path) -> tuple[dict, str, dict]:
    body = aff.make_corpus_body(source)
    reqs = [aff.request_for(doc, body, expected_bank=i) for i, doc in enumerate(aff.DOCUMENTS)]
    transport, deadline, attempts = aff.UnixWaveTransport(sock, timeout_s=30), time.monotonic()+2, 0
    while True:
        attempts += 1
        try: result = transport.wave("cancel-cached-single-seed4", reqs); break
        except aff.AcceptanceError as exc:
            if "HTTP scheduler must be ready and fully drained" not in str(exc) or time.monotonic() >= deadline: raise
            time.sleep(.1)
    for i, (rid, _, expected) in enumerate(aff.DOCUMENTS):
        if result.get("texts", {}).get(rid, "").strip() != expected or result.get("accepted", {}).get(rid, {}).get("bank") != i:
            raise RuntimeError(f"seed semantic/owner mismatch for {rid}")
    geom = aff.geometry_signature(result.get("timings"))
    if geom is None: raise RuntimeError("seed did not report four-bank capacity/UBatch geometry")
    prompt = reqs[3]["prompt"]
    if not prompt.endswith(Q_SUFFIX): raise RuntimeError("unexpected DOC-D question suffix")
    docs={rid:{"expected":expected,"actual":result["texts"][rid].strip(),"bank":result["accepted"][rid]["bank"]}
          for rid,_,expected in aff.DOCUMENTS}
    return result, prompt[:-len(Q_SUFFIX)]+Q_LONG, {"body_characters": len(body),"documents":docs,
        "geometry":geom,"mtp_depth":result.get("timings",{}).get("mtp_depth"),"seed_attempts":attempts}

async def cancel_first(client, runner, prompt: str) -> dict:
    rec = {"arrival_s":runner.t(), "http_status":None, "response_id":None, "first_content_s":None,
           "cancel_requested_s":None, "partial_text":"", "completed_s":None, "max_tokens":512,
           "prompt_characters":len(prompt), "prompt_sha256":http.sha256_text(prompt)}
    parts=[]
    async with client.stream("POST", runner.base_url.rstrip("/")+"/chat/completions", headers=runner.headers,
                             json=runner.public_prompt_payload(prompt,512,True)) as response:
        rec["http_status"], rec["headers_s"] = response.status_code, runner.t()
        if response.status_code != 200: raise RuntimeError(f"cancel SSE HTTP {response.status_code}")
        async for line in response.aiter_lines():
            event=http.parse_json_sse_line(line)
            if event is None: continue
            if "error" in event: raise RuntimeError("cancel SSE returned an error event")
            rec["response_id"] = rec["response_id"] or event.get("id")
            choices=event.get("choices",[]); choice=choices[0] if choices else {}
            delta=choice.get("delta",{}) if isinstance(choice,dict) else {}
            piece=delta.get("content") if isinstance(delta,dict) else None
            if isinstance(piece,str) and piece:
                rec["first_content_s"] = rec["first_content_s"] or runner.t()
                parts.append(piece); rec["cancel_requested_s"]=runner.t(); break
    rec["completed_s"], rec["partial_text"] = runner.t(), "".join(parts)
    if not rec["partial_text"] or not rec["response_id"]: raise RuntimeError("no correlated SSE content to cancel")
    return rec

def find_flows(root: Path, ids: set[str]) -> dict[str,dict]:
    deadline=time.monotonic()+2; found={}
    while time.monotonic()<deadline:
        if root.exists():
            for path in root.glob("**/lastwave-*.json"):
                try: doc=json.loads(path.read_text(encoding="utf-8"))
                except (OSError,json.JSONDecodeError): continue
                for flow in doc.get("flows",[]) if isinstance(doc,dict) else []:
                    if isinstance(flow,dict) and str(flow.get("id","")) in ids:
                        found[str(flow["id"]) ]={"path":str(path),"flow":flow,"banks":doc.get("banks")}
        if ids <= set(found): return found
        time.sleep(.1)
    return found

async def live(args) -> dict:
    base=args.base_url or os.environ.get("AC922_BASE_URL"); keyfile=args.key_file or os.environ.get("AC922_API_KEY_FILE")
    if not base or not keyfile: raise RuntimeError("--run requires AC922_BASE_URL and AC922_API_KEY_FILE")
    out=Path(args.out).expanduser(); report={"schema":"ac922-cancel-cached-single-v1","status":"RUNNING",
        "endpoint":http.safe_endpoint(base),"model_runs_root":str(Path(args.model_run_root).expanduser()),"phases":{}}
    try:
        source=aff.load_corpus(Path(args.corpus).expanduser())
        _, long_prompt, seed_meta=seed(source,Path(args.socket)); report["seed"]=seed_meta
        key=http.load_keyfile(Path(keyfile)); client=http.StandardHttpClient(timeout=1805)
        api_base,root_url=http.api_urls(base); runner=http.Runner(client,api_base,root_url,args.model,key,out,report)
        cancel=await cancel_first(client,runner,long_prompt); report["phases"]["cancel"]=cancel
        await asyncio.sleep(.3)
        follow_prompt=aff.make_prompt(aff.DOCUMENTS[3][1],aff.DOCUMENTS[3][2],aff.make_corpus_body(source))
        follow=await runner.nonstream(follow_prompt,16,"doc-delta-follow"); report["phases"]["follow"]=follow
        usage=follow.get("usage",{}); details=usage.get("prompt_tokens_details",{}) if isinstance(usage,dict) else {}
        cached=details.get("cached_tokens") if isinstance(details,dict) else None
        if follow["text"].strip()!="9274" or not isinstance(cached,int) or cached<=0:
            raise RuntimeError("follow-up must answer 9274 with explicit positive cache usage")
        cid=str(cancel["response_id"]).removeprefix("chatcmpl-"); fid=str(follow["id"]).removeprefix("chatcmpl-")
        if cid==fid: raise RuntimeError("cancel and follow response IDs unexpectedly match")
        flows=find_flows(Path(args.model_run_root).expanduser(),{cid,fid})
        if {cid,fid}-set(flows): raise RuntimeError("cancel/follow internal flows absent after bounded 2s wait")
        cf,ff=flows[cid]["flow"],flows[fid]["flow"]
        for name,flow in (("cancel",cf),("follow",ff)):
            if flow.get("bank")!=3 or int(flow.get("cached_tokens",0))<=0: raise RuntimeError(f"{name} flow not cached on bank 3")
        if not 0<int(cf.get("output_tokens",512))<512: raise RuntimeError("cancel flow did not stop partially below 512 tokens")
        if str(ff.get("output","")).strip()!="9274": raise RuntimeError("internal follow output differs from HTTP")
        cap=seed_meta["geometry"][3]["target_context"]; depth=seed_meta["mtp_depth"]
        if int(cf.get("input_tokens",0))+512+depth+1>cap or int(ff.get("input_tokens",0))+16+depth+1>cap:
            raise RuntimeError("512-token cancel or follow reservation exceeds seeded 17K bank context")
        for evidence in flows.values():
            if aff.geometry_signature({"banks":evidence.get("banks")})!=seed_meta["geometry"]:
                raise RuntimeError("cancel/follow four-bank context/UBatch differs from seed")
        report["phases"].update({"internal_flows":flows,"follow_cached_tokens":cached}); report["status"]="PASS"
        return report
    except Exception as exc:
        report["status"],report["failure"]="FAIL",f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["finished_unix"]=time.time(); http.write_report(out,report)

def main() -> int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--run",action="store_true")
    p.add_argument("--base-url",default=os.environ.get("AC922_BASE_URL")); p.add_argument("--key-file",default=os.environ.get("AC922_API_KEY_FILE"))
    p.add_argument("--socket",default=os.environ.get("AC922_AFFINITY_SOCKET",str(aff.DEFAULT_SOCKET)))
    p.add_argument("--corpus",default=os.environ.get("AC922_CORPUS_FILE",str(aff.DEFAULT_CORPUS)))
    p.add_argument("--model-run-root",default=os.environ.get("AC922_MODEL_RUN_ROOT",str(HERE/"run/model")))
    p.add_argument("--out",default=os.environ.get("AC922_CANCEL_CACHE_RESULT",str(HERE/"run/cancel-cached-single.json")))
    p.add_argument("--model",default=MODEL); args=p.parse_args()
    if not args.run: print("offline: use --run after HTTP suite completes"); return 0
    try: result=asyncio.run(live(args))
    except Exception as exc: print(f"cancel/cache regression FAIL: {type(exc).__name__}: {exc}",file=sys.stderr); return 1
    print(json.dumps({"status":result["status"],"result":args.out},ensure_ascii=False)); return 0

if __name__=="__main__": raise SystemExit(main())
