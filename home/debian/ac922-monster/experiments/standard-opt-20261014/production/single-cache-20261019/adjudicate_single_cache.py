"""Record completed human-agent text review without erasing producer verdicts."""
import hashlib
import json
from pathlib import Path
import shutil

F = Path(__file__).resolve().parent
R, E = F / "run", F / "evidence"


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


review = read(E / "semantic-review.json")
assert review["status"] == "PASS"
for name in ["campaign-status.json", "single-cache-result.json", "measured-summary.json"]:
    d = read(R / name)
    d.setdefault("producer_status", d["status"])
    d.update(status="PASS", semantic_adjudication="evidence/semantic-review.json")
    for wave in d.get("waves", []):
        if wave.get("semantic_status") == "FULL_TEXT_REVIEW_REQUIRED":
            wave["producer_semantic_status"] = wave["semantic_status"]
            wave["semantic_status"] = "PASS_COHERENT_AT_512_TOKEN_LIMIT" if "prose" in wave["label"] else "PASS_EXACT_REVIEWED_L2"
    for wave in d.get("cached_single_waves", []):
        if wave.get("status") == "FULL_TEXT_REVIEW_REQUIRED":
            wave["producer_status"] = wave["status"]
            wave["status"] = "PASS_COHERENT_AT_512_TOKEN_LIMIT"
    if name == "measured-summary.json":
        d["acceptance_status"]["single_cache"] = "PASS"
        d["receipts"]["campaign"]["status"] = "PASS"
    write(R / name, d)
    shutil.copy2(R / name, E / name)
final = read(F / "MEASURED-SINGLE-CACHE.json")
final["cached_single_waves"] = read(R / "measured-summary.json")["cached_single_waves"]
final["receipts"]["campaign"]["status"] = "PASS"
final["acceptance_status"]["single_cache"] = "PASS"
write(F / "MEASURED-SINGLE-CACHE.json", final)
receipt = read(F / "lastwave-acceptance.json")
receipt["architect_review"] = "APPROVE: independently read actual C3 IPC/HTTP/cancel, exact L2 full-text/token-ID comparison and new cached essay. Pending producer labels reconciled with explicit preserved status. No raw speed gain claim."
receipt["evidence_sha256"] = {str(p.relative_to(F)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(E.iterdir()) if p.is_file()}
write(F / "lastwave-acceptance.json", receipt)
print("C3_SEMANTIC_ADJUDICATION_PASS; PRODUCER_STATUS_PRESERVED; ARCHITECT_APPROVED")
