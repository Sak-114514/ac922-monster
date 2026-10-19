#!/usr/bin/env python3
"""Single-loaded-service HTTP acceptance harness for the AC922 last wave.

The normal mode talks only to an already-running API.  --self-test uses an
in-process HTTP mock and never reads credentials or contacts a model. API
keys and prompts are never written to the report.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import sys
import threading
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
DEFAULT_CANARIES = BASE / "fixtures/fourbank-d3-newruntime-v306-20261017/v221-d3-scenarios.json"
DEFAULT_CORPUS = BASE / "fixtures/long-prefill-combo-v45-20261016/v45-real-corpus-source.json"
DEFAULT_MODEL = "flash-next-iq3-xxs"
CALIBRATION_CHARACTERS = 121_049
CALIBRATION_PROMPT_TOKENS = 36_646
MAX_REPORT_TEXT_BYTES = 2_000_000


class AcceptanceError(RuntimeError):
    pass


class HttpResponse:
    def __init__(self, status_code: int, headers: dict[str, str], body: bytes):
        self.status_code = status_code
        self.headers = {str(k).lower(): str(v) for k, v in headers.items()}
        self._body = body

    def json(self) -> dict[str, Any]:
        value = json.loads(self._body.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("HTTP JSON response is not an object")
        return value


class _SseResponse:
    def __init__(self, url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float = 1805.0):
        self.url = url
        self.headers_in = headers
        self.payload = payload
        self.timeout = timeout
        self.status_code = 0
        self.headers: dict[str, str] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[tuple[str, Any]] | None = None
        self._stop = threading.Event()
        self._response: Any = None
        self._thread: threading.Thread | None = None

    def _put(self, item: tuple[str, Any]) -> None:
        if self._loop is not None and self._queue is not None:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, item)

    def _read(self) -> None:
        req_headers = dict(self.headers_in)
        req_headers.setdefault("Content-Type", "application/json")
        req_headers.setdefault("Accept", "text/event-stream")
        req_headers.setdefault("Accept-Encoding", "identity")
        request = Request(self.url, data=json.dumps(self.payload, ensure_ascii=False).encode("utf-8"),
                          headers=req_headers, method="POST")
        try:
            response = urlopen(request, timeout=self.timeout)
        except HTTPError as error:
            response = error
        except Exception as error:
            self._put(("error", type(error).__name__))
            self._put(("eof", None))
            return
        self._response = response
        self.status_code = int(getattr(response, "status", getattr(response, "code", 0)))
        self.headers = {str(k).lower(): str(v) for k, v in response.headers.items()}
        self._put(("response", None))
        try:
            while not self._stop.is_set():
                raw = response.readline()
                if not raw:
                    break
                self._put(("line", raw.decode("utf-8", errors="replace").rstrip("\r\n")))
        except Exception as error:
            if not self._stop.is_set():
                self._put(("error", type(error).__name__))
        finally:
            try:
                response.close()
            except Exception:
                pass
            self._put(("eof", None))

    async def __aenter__(self) -> "_SseResponse":
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()
        self._thread = threading.Thread(target=self._read, name="lastwave-sse-reader", daemon=True)
        self._thread.start()
        assert self._queue is not None
        while True:
            kind, value = await self._queue.get()
            if kind == "response":
                return self
            if kind == "error":
                raise AcceptanceError(f"SSE transport failed: {value}")

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        self._stop.set()
        if self._response is not None:
            try:
                self._response.close()
            except Exception:
                pass
        if self._thread is not None:
            await asyncio.to_thread(self._thread.join, 1.0)
        return False

    async def aiter_lines(self):
        assert self._queue is not None
        while True:
            kind, value = await self._queue.get()
            if kind == "line":
                yield value
            elif kind == "error":
                raise AcceptanceError(f"SSE transport failed: {value}")
            elif kind == "eof":
                return


class StandardHttpClient:
    """Dependency-free HTTP/1.1 client; one connection per request."""

    def __init__(self, timeout: float = 1805.0):
        self.timeout = timeout

    def stream(self, method: str, url: str, *, headers: dict[str, str], json: dict[str, Any]) -> _SseResponse:
        if method != "POST":
            raise ValueError("only POST SSE is supported")
        return _SseResponse(url, headers, json, self.timeout)

    def _request(self, method: str, url: str, headers: dict[str, str], body: bytes | None = None) -> HttpResponse:
        req_headers = dict(headers)
        req_headers.setdefault("Accept-Encoding", "identity")
        request = Request(url, data=body, headers=req_headers, method=method)
        try:
            response = urlopen(request, timeout=self.timeout)
        except HTTPError as error:
            response = error
        except URLError as error:
            raise AcceptanceError(f"HTTP transport failed: {type(error.reason).__name__}") from None
        with response:
            payload = response.read()
            return HttpResponse(int(getattr(response, "status", getattr(response, "code", 0))),
                                dict(response.headers.items()), payload)

    async def get(self, url: str, *, headers: dict[str, str]) -> HttpResponse:
        return await asyncio.to_thread(self._request, "GET", url, headers, None)

    async def post(self, url: str, *, headers: dict[str, str], json: dict[str, Any]) -> HttpResponse:
        body = json_module_dumps(json)
        req_headers = dict(headers)
        req_headers.setdefault("Content-Type", "application/json")
        return await asyncio.to_thread(self._request, "POST", url, req_headers, body)


def json_module_dumps(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def safe_endpoint(url: str) -> str:
    """Drop userinfo, query strings, and fragments before storing endpoint metadata."""
    p = urlsplit(url)
    host = p.hostname or ""
    if p.port is not None:
        host += f":{p.port}"
    return urlunsplit((p.scheme, host, p.path.rstrip("/"), "", ""))


def api_urls(base_url: str) -> tuple[str, str]:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        return root, root[:-3]
    return root + "/v1", root


def prompt_text_from_messages(messages: list[dict[str, Any]]) -> str:
    return "\n".join(str(m.get("content", "")) for m in messages if isinstance(m, dict))


def parse_json_sse_line(line: str) -> dict[str, Any] | None:
    if not line.startswith("data:"):
        return None
    data = line[5:].strip()
    if not data or data == "[DONE]":
        return None
    try:
        value = json.loads(data)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def prefix_declared_enabled(document: Any) -> bool:
    if not isinstance(document, dict):
        return False
    if document.get("prefix_cache_enabled") is True:
        return True
    for key in ("prefix_cache", "capabilities", "features"):
        nested = document.get(key)
        if isinstance(nested, dict):
            if key == "prefix_cache" and nested.get("enabled") is True:
                return True
            if nested.get("prefix_cache_enabled") is True:
                return True
            pc = nested.get("prefix_cache")
            if isinstance(pc, dict) and pc.get("enabled") is True:
                return True
    return False


def cached_token_evidence(body: Any, headers: Any = None) -> int | None:
    """Return explicit cache-hit token count only; never infer from latency."""
    if isinstance(body, dict):
        usage = body.get("usage")
        if isinstance(usage, dict):
            details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details")
            if isinstance(details, dict):
                for key in ("cached_tokens", "cache_hit_tokens", "prefix_cache_hit_tokens"):
                    val = details.get(key)
                    if isinstance(val, int) and val >= 0:
                        return val
            for key in ("cached_prompt_tokens", "cache_hit_tokens", "prefix_cache_hit_tokens"):
                val = usage.get(key)
                if isinstance(val, int) and val >= 0:
                    return val
        for key in ("cached_tokens", "cache_hit_tokens", "prefix_cache_hit_tokens"):
            val = body.get(key)
            if isinstance(val, int) and val >= 0:
                return val
    if headers is not None:
        for key in ("x-ac922-prefix-cache-hit-tokens", "x-prefix-cache-hit-tokens", "x-cached-prompt-tokens"):
            raw = headers.get(key)
            if raw is not None:
                try:
                    val = int(raw)
                except (TypeError, ValueError):
                    continue
                if val >= 0:
                    return val
    return None


def evidence_from_file(path: Path | None, response_id: str | None) -> int | None:
    """Read optional cache evidence after a request, matching its public or worker ID."""
    if path is None or not path.exists() or not response_id:
        return None
    candidates: list[Any] = []
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            candidates.extend(parsed)
        elif isinstance(parsed, dict):
            events = parsed.get("events")
            candidates.extend(events if isinstance(events, list) else [parsed])
    except json.JSONDecodeError:
        for line in raw.splitlines():
            try:
                candidates.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    ids = {response_id, response_id.removeprefix("chatcmpl-"), response_id.removeprefix("cmpl-")}
    for item in candidates:
        if not isinstance(item, dict):
            continue
        rid = str(item.get("request_id", item.get("id", item.get("public_id", ""))))
        if rid not in ids and not any(rid.endswith(i) or i.endswith(rid) for i in ids if rid):
            continue
        for key in ("cached_tokens", "cache_hit_tokens", "prefix_cache_hit_tokens", "reused_prefix_tokens"):
            val = item.get(key)
            if isinstance(val, int) and val >= 0:
                return val
    return None


def extract_chat_text(document: dict[str, Any]) -> str:
    choices = document.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message", {})
    return str(message.get("content", "")) if isinstance(message, dict) else ""


def semantic_canary_result(case: dict[str, Any], text: str) -> tuple[bool, str]:
    expected = case.get("expected")
    checker = case.get("checker")
    if checker == "strict_json":
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            return False, f"invalid_json:{exc.msg}"
        return parsed == expected, "exact_json" if parsed == expected else "json_value_mismatch"
    actual = text.strip()
    return actual == str(expected), "exact_text" if actual == str(expected) else "text_mismatch"


def story_cases() -> list[dict[str, Any]]:
    topics = [
        "how CPU caches and cache coherency work",
        "photosynthesis and plant metabolism",
        "Roman architectural engineering",
        "comparison sorting algorithms and complexity",
    ]
    keywords = [
        ("cache", "memory"),
        ("photosynthesis", "light"),
        ("roman", "concrete"),
        ("sorting", "comparison"),
    ]
    result = []
    for i, topic in enumerate(topics):
        prompt = (
            f"Write a coherent technical explanation of {topic}, at least 2000 words. "
            f"Begin with exactly BEGIN_{i} then continue with the explanation. Do not end early."
        )
        result.append({"id": f"story-{i}", "topic": topic, "prompt": prompt, "max_tokens": 512,
                       "marker": f"BEGIN_{i}", "keywords": keywords[i]})
    return result


def refill_cases() -> list[dict[str, Any]]:
    return [
        {"id": "refill-long", "prompt": "Write a detailed, coherent technical explanation of CPU caches and memory coherency. Begin with exactly BEGIN_REFILL_LONG. Explain cache levels, cache lines, and a standard coherence protocol. Continue in readable prose without ending early.", "max_tokens": 512, "expected_marker": "BEGIN_REFILL_LONG", "kind": "long"},
        {"id": "refill-short-paris", "prompt": "What is the capital city of France? Reply with exactly the city name, without punctuation or explanation.", "max_tokens": 64, "expected": "Paris", "kind": "short"},
        {"id": "refill-short-math", "prompt": "请计算 2847 + 6958。只输出十进制整数，不要解释。", "max_tokens": 64, "expected": "9805", "kind": "short"},
        {"id": "refill-short-logic", "prompt": "All dax are lamn. Some lamn are red. Does it logically follow that some dax are red? Reply with exactly YES or NO.", "max_tokens": 64, "expected": "NO", "kind": "short"},
    ]


def followup_cases() -> list[dict[str, Any]]:
    return [
        {"id": "refill-followup-rome", "prompt": "What is the capital city of Italy? Reply with exactly the city name, without punctuation.", "max_tokens": 64, "expected": "Rome"},
        {"id": "refill-followup-multiply", "prompt": "Calculate 18 multiplied by 19. Reply with only the integer.", "max_tokens": 64, "expected": "342"},
    ]


def load_corpus_text(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    corpus = data.get("corpus", {}) if isinstance(data, dict) else {}
    segments = corpus.get("segments", []) if isinstance(corpus, dict) else []
    parts = [str(x.get("text", "")) for x in segments if isinstance(x, dict) and x.get("text")]
    text = "\n\n".join(parts)
    if not text:
        raise AcceptanceError(f"corpus source contains no text: {path}")
    return text


def build_real_corpus_prompt(source: str, target_tokens: int) -> tuple[str, int]:
    # C12's existing API payload used this same corpus and template; its
    # 121,049-char reference slice produced 36,646 actual API prompt tokens.
    char_count = max(1024, round(CALIBRATION_CHARACTERS * target_tokens / CALIBRATION_PROMPT_TOKENS))
    repeated = (source + "\n") * (char_count // (len(source) + 1) + 1)
    body = repeated[:char_count]
    prompt = (
        "Verification code for this request: EPSILON72941. Preserve this code.\n"
        "REFERENCE MATERIAL:\n" + body +
        "\nEND REFERENCE MATERIAL.\nFINAL QUESTION: What is the verification code for this request? "
        "Return exactly the code and no explanation."
    )
    return prompt, char_count


def assert_safe_text(text: str, label: str) -> None:
    if "\ufffd" in text:
        raise AcceptanceError(f"{label}: decoded output contains Unicode replacement characters")


def story_semantics(case: dict[str, Any], text: str) -> dict[str, Any]:
    lowered = text.lower()
    checks = {
        "marker": text.startswith(case["marker"]),
        "readable_length": len(text) >= 1200,
        "no_replacement_char": "\ufffd" not in text,
        "topic_keywords": all(word.lower() in lowered for word in case["keywords"]),
    }
    return {"pass": all(checks.values()), "checks": checks, "characters": len(text)}


def report_file_from_env(value: str | None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return HERE / f"lastwave-http-result-{stamp}.json"


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    data = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if len(data.encode("utf-8")) > MAX_REPORT_TEXT_BYTES:
        raise AcceptanceError("result JSON exceeded the 2 MiB safety limit")
    temp.write_text(data, encoding="utf-8")
    os.chmod(temp, 0o600)
    os.replace(temp, path)


class Runner:
    def __init__(self, client: Any, base_url: str, root_url: str, model: str,
                 api_key: str, report_path: Path, report: dict[str, Any], prefix_evidence_file: Path | None = None):
        self.client = client
        self.base_url = base_url
        self.root_url = root_url
        self.model = model
        self.report_path = report_path
        self.report = report
        self.prefix_evidence_file = prefix_evidence_file
        self.clock0 = time.monotonic()
        self.headers = {"Authorization": "Bearer " + api_key, "Accept": "application/json"}

    def t(self) -> float:
        return time.monotonic() - self.clock0

    def save(self) -> None:
        write_report(self.report_path, self.report)

    def public_prompt_payload(self, prompt: str, max_tokens: int, stream: bool = False) -> dict[str, Any]:
        return {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                "temperature": 0, "max_tokens": max_tokens, "stream": stream}

    async def discover(self, ready_url: str | None) -> dict[str, Any]:
        discovery: dict[str, Any] = {"ready_endpoint": "unavailable", "prefix_cache_enabled": False}
        candidate = ready_url or self.root_url.rstrip("/") + "/ready"
        try:
            response = await self.client.get(candidate, headers=self.headers)
            if response.status_code == 200:
                doc = response.json()
                if isinstance(doc, dict):
                    discovery["ready_endpoint"] = "available"
                    discovery["ready_fields"] = {k: doc[k] for k in ("status", "model", "max_context", "pool_tokens", "max_parallel") if k in doc}
                    discovery["prefix_cache_enabled"] = prefix_declared_enabled(doc)
                    discovery["prefix_cache_declaration"] = "enabled" if discovery["prefix_cache_enabled"] else "not_declared_enabled"
            else:
                discovery["ready_http_status"] = response.status_code
        except Exception:
            discovery["ready_endpoint"] = "unavailable"

        response = await self.client.get(self.base_url.rstrip("/") + "/models", headers=self.headers)
        if response.status_code != 200:
            raise AcceptanceError(f"authenticated model discovery returned HTTP {response.status_code}")
        doc = response.json()
        models = [str(row.get("id")) for row in doc.get("data", []) if isinstance(row, dict)] if isinstance(doc, dict) else []
        discovery["models"] = models
        if self.model not in models:
            raise AcceptanceError(f"expected model {self.model!r} not present in /v1/models")
        if discovery.get("ready_fields", {}).get("model") not in (None, self.model):
            raise AcceptanceError("/ready model differs from /v1/models")
        self.report["discovery"] = discovery
        self.save()
        return discovery

    async def nonstream(self, prompt: str, max_tokens: int, label: str) -> dict[str, Any]:
        start = self.t()
        response = await self.client.post(self.base_url.rstrip("/") + "/chat/completions",
                                          headers=self.headers,
                                          json=self.public_prompt_payload(prompt, max_tokens, False))
        headers_at = self.t()
        if response.status_code != 200:
            raise AcceptanceError(f"{label}: API returned HTTP {response.status_code}")
        try:
            body = response.json()
        except ValueError as exc:
            raise AcceptanceError(f"{label}: API response was not JSON") from exc
        choices = body.get("choices", [])
        choice = choices[0] if isinstance(choices, list) and choices else {}
        text = extract_chat_text(body)
        usage = body.get("usage", {}) if isinstance(body, dict) else {}
        assert_safe_text(text, label)
        return {
            "label": label, "id": body.get("id"), "http_status": response.status_code,
            "arrival_s": start, "response_received_s": headers_at, "completed_s": self.t(),
            "wall_s": self.t() - start, "text": text, "text_sha256": sha256_text(text),
            "usage": usage if isinstance(usage, dict) else {},
            "finish_reason": choice.get("finish_reason") if isinstance(choice, dict) else None,
            "prompt_sha256": sha256_text(prompt), "prompt_characters": len(prompt),
            "cache_hit_tokens": cached_token_evidence(body, response.headers),
        }

    async def stream(self, prompt: str, max_tokens: int, label: str,
                     events: asyncio.Queue[dict[str, Any]] | None = None,
                     cancel_after_first_content: bool = False) -> dict[str, Any]:
        record: dict[str, Any] = {
            "label": label, "arrival_s": self.t(), "accepted_s": None,
            "first_content_s": None, "completed_s": None, "finish_reason": None,
            "max_tokens": max_tokens, "text_parts": [], "http_status": None,
            "prompt_sha256": sha256_text(prompt), "prompt_characters": len(prompt),
        }
        async with self.client.stream("POST", self.base_url.rstrip("/") + "/chat/completions",
                                      headers=self.headers,
                                      json=self.public_prompt_payload(prompt, max_tokens, True)) as response:
            record["http_status"] = response.status_code
            record["accepted_s"] = self.t()
            if response.status_code != 200:
                raise AcceptanceError(f"{label}: SSE API returned HTTP {response.status_code}")
            async for line in response.aiter_lines():
                payload = parse_json_sse_line(line)
                if payload is None:
                    continue
                if "error" in payload:
                    raise AcceptanceError(f"{label}: SSE API returned an error event")
                choices = payload.get("choices", [])
                if not isinstance(choices, list) or not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta", {}) if isinstance(choice, dict) else {}
                content = delta.get("content") if isinstance(delta, dict) else None
                if isinstance(content, str) and content:
                    if record["first_content_s"] is None:
                        record["first_content_s"] = self.t()
                    record["text_parts"].append(content)
                    if cancel_after_first_content:
                        record["cancelled_after_first_content"] = True
                        record["cancel_requested_s"] = self.t()
                        break
                finish = choice.get("finish_reason") if isinstance(choice, dict) else None
                if finish:
                    record["finish_reason"] = finish
            if not cancel_after_first_content or not record.get("cancelled_after_first_content"):
                record["completed_s"] = self.t()
        record["text"] = "".join(record.pop("text_parts"))
        assert_safe_text(record["text"], label)
        if record.get("cancelled_after_first_content"):
            record["completed_s"] = self.t()
        if events is not None:
            await events.put({"id": label, "record": record})
        return record

    async def canaries(self, canary_file: Path) -> None:
        cases_doc = json.loads(canary_file.read_text(encoding="utf-8"))
        cases = cases_doc[0]["requests"] if isinstance(cases_doc, list) else cases_doc["requests"]
        if len(cases) != 4:
            raise AcceptanceError(f"expected four frozen C12 canaries; found {len(cases)}")

        async def one(case: dict[str, Any]) -> dict[str, Any]:
            item = await self.nonstream(case["prompt"], int(case.get("max_tokens", 32)), case["id"])
            passed, verdict = semantic_canary_result(case, item["text"])
            item["semantic"] = {"pass": passed, "verdict": verdict, "expected": case.get("expected")}
            item.pop("prompt_characters", None)
            return item

        results = await asyncio.gather(*(one(c) for c in cases))
        phase = {"status": "PASS" if all(x["semantic"]["pass"] for x in results) else "FAIL", "requests": results}
        self.report["phases"]["native_canaries"] = phase
        self.save()
        if phase["status"] != "PASS":
            raise AcceptanceError("one or more C12 semantic canaries failed")

    async def refill_wave(self) -> None:
        initial = refill_cases()
        long_id = "refill-long"
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        records: dict[str, dict[str, Any]] = {}
        tasks: dict[str, asyncio.Task[dict[str, Any]]] = {}

        async def launch(case: dict[str, Any]) -> None:
            async def run_one() -> None:
                try:
                    await self.stream(case["prompt"], case["max_tokens"], case["id"], queue)
                except BaseException as exc:
                    await queue.put({"id": case["id"], "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
            tasks[case["id"]] = asyncio.create_task(run_one())

        for case in initial:
            await launch(case)

        followups_sent = False
        expected = {c["id"]: c.get("expected") for c in initial if c.get("expected")}
        followup_expected = {c["id"]: c["expected"] for c in followup_cases()}
        cases_by_id = {c["id"]: c for c in initial}
        cases_by_id.update({c["id"]: c for c in followup_cases()})

        while len(records) < len(initial) + (len(followup_cases()) if followups_sent else 0):
            event = await queue.get()
            if "error" in event:
                raise AcceptanceError(f"{event['id']}: {event['error']}")
            record = event["record"]
            rid = event["id"]
            records[rid] = record
            if rid in expected:
                record["semantic"] = {"pass": record["text"].strip() == expected[rid],
                                      "expected": expected[rid], "actual": record["text"].strip()}
            elif rid == long_id:
                lowered = record["text"].lower()
                long_checks = {
                    "marker": record["text"].startswith("BEGIN_REFILL_LONG"),
                    "cache_topic": "cache" in lowered,
                    "memory_topic": "memory" in lowered,
                    "readable_length": len(record["text"]) >= 1200,
                }
                record["semantic"] = {"pass": all(long_checks.values()), "checks": long_checks}
            if rid in followup_expected:
                record["semantic"] = {"pass": record["text"].strip() == followup_expected[rid],
                                      "expected": followup_expected[rid], "actual": record["text"].strip()}

            if rid != long_id and cases_by_id[rid].get("kind") == "short" and not followups_sent:
                long_done = records.get(long_id, {}).get("completed_s")
                if long_done is None:
                    followups_sent = True
                    for case in followup_cases():
                        await launch(case)

        long_done = records.get(long_id, {}).get("completed_s")
        checks = {}
        for case in followup_cases():
            r = records.get(case["id"])
            if r is None:
                checks[case["id"]] = {
                    "http_accepted_before_long_done": False,
                    "first_content_before_long_done": False,
                    "semantic_pass": False,
                }
                continue
            checks[case["id"]] = {
                "http_accepted_before_long_done": r["http_status"] == 200 and r["accepted_s"] is not None and long_done is not None and r["accepted_s"] < long_done,
                "first_content_before_long_done": r["first_content_s"] is not None and long_done is not None and r["first_content_s"] < long_done,
                "semantic_pass": bool(r.get("semantic", {}).get("pass")),
            }
        all_pass = (followups_sent and all(all(v.values()) for v in checks.values()) and
                    all(bool(r.get("semantic", {}).get("pass")) for r in records.values()))
        phase = {
            "status": "PASS" if all_pass else "FAIL",
            "interpretation": "HTTP 200 is queued/accepted only; first nonempty SSE content before the original 512-token request completes proves refill execution.",
            "long_request": records.get(long_id),
            "requests": records,
            "followup_gates": checks,
            "followups_dispatched_after_short_completion": followups_sent,
        }
        self.report["phases"]["asymmetric_refill"] = phase
        self.save()
        if not all_pass:
            raise AcceptanceError("requests five and six did not both emit before the original long request completed")

    async def cancel_and_reuse(self) -> None:
        prompt = "CANCEL-ME: Write a detailed, long history of computing, naming generations, inventions, and dates. Keep writing in connected prose without concluding early."
        began = self.t()
        cancelled = await self.stream(prompt, 2048, "cancel-stream", cancel_after_first_content=True)
        if not cancelled.get("cancelled_after_first_content") or not cancelled.get("text"):
            raise AcceptanceError("cancel test did not receive content before disconnect")
        await asyncio.sleep(0.15)
        reuse = await self.nonstream("What is the capital city of France? Reply with exactly Paris.", 16, "post-cancel-paris")
        phase = {
            "status": "PASS" if reuse["text"].strip() == "Paris" else "FAIL",
            "cancel_request": {k: cancelled.get(k) for k in ("http_status", "arrival_s", "accepted_s", "first_content_s", "cancel_requested_s", "completed_s", "cancelled_after_first_content")},
            "cancel_sample_text": cancelled["text"],
            "reuse_request": reuse,
            "cancel_to_reuse_wall_s": reuse["completed_s"] - began,
        }
        self.report["phases"]["sse_cancel_reuse"] = phase
        self.save()
        if phase["status"] != "PASS":
            raise AcceptanceError("API did not recover correctly after SSE cancellation")

    async def prefix_cache(self, enabled: bool, corpus_source: str) -> None:
        if not enabled:
            self.report["phases"]["exact_prefix_cache"] = {
                "status": "SKIPPED_NOT_ENABLED",
                "reason": "worker readiness/capabilities did not declare prefix caching enabled",
            }
            self.save()
            return
        shared_body = (corpus_source * (32_768 // max(1, len(corpus_source)) + 1))[:32_768]
        prefix = ("SHARED-PREFIX-BEGIN\nNOVA = 3179\nORION = 6842\nREFERENCE MATERIAL BEGIN\n" + shared_body +
                  "\nREFERENCE MATERIAL END\nInstruction: answer from the reference exactly.\n")
        edited_prefix = ("EDITED-PREFIX-BEGIN\nNOVA = 5301\nORION = 6842\nREFERENCE MATERIAL BEGIN\n" + shared_body +
                         "\nREFERENCE MATERIAL END\nInstruction: answer from the reference exactly.\n")
        question_a = "\nQUESTION: What is NOVA? Output only its value."
        question_b = "\nQUESTION: What is ORION? Output only its value."
        requests = [
            ("prefix-cold-A", prefix + question_a, "3179"),
            ("prefix-hit-B", prefix + question_b, "6842"),
            ("prefix-hit-A", prefix + question_a, "3179"),
            ("prefix-edited-new-key", edited_prefix + question_a, "5301"),
        ]
        rows = []
        for label, prompt, expected in requests:
            row = await self.nonstream(prompt, 32, label)
            body_hit = row.get("cache_hit_tokens")
            file_hit = evidence_from_file(self.prefix_evidence_file, row.get("id"))
            row["cache_hit_tokens"] = body_hit if body_hit is not None else file_hit
            row["cache_evidence_source"] = "response" if body_hit is not None else ("evidence_file" if file_hit is not None else None)
            usage = row.get("usage", {})
            row["actual_prompt_tokens"] = usage.get("prompt_tokens") if isinstance(usage, dict) else None
            row["expected"] = expected
            row["natural_eog"] = row.get("finish_reason") == "stop"
            row["semantic_pass"] = row["text"].strip() == expected and row["natural_eog"]
            rows.append(row)
        hits = [r["cache_hit_tokens"] for r in rows]
        if (any(not r["semantic_pass"] for r in rows) or
                any((r.get("actual_prompt_tokens") or 0) < 8192 for r in rows)):
            status = "FAIL"
        elif any(v is None for v in hits):
            status = "NOT_PROVABLE" if all(r["semantic_pass"] for r in rows) else "FAIL"
        else:
            status = "PASS" if hits[0] == 0 and hits[1] > 0 and hits[2] > 0 and hits[3] == 0 else "FAIL"
        self.report["phases"]["exact_prefix_cache"] = {
            "status": status,
            "rules": "Native chat requests test shared-prefix A/B/A (NOVA=3179, ORION=6842, NOVA=3179) with suffix-only edits, exact answers, and natural EOG; an early common-body edit introduces NOVA=5301 and must not reuse the old prefix. Explicit cached-token evidence is required. Expected cold=0, B/A hits>0, edited-body=0.",
            "requests": rows,
        }
        self.save()
        if status == "FAIL":
            raise AcceptanceError("exact prefix cache semantic or explicit hit-count gate failed")

    async def real_corpus_prefill(self, source: str) -> None:
        rows = []
        for target in (32_768, 65_536):
            prompt, chars = build_real_corpus_prompt(source, target)
            label = f"http-real-corpus-approx-{target}"
            row = await self.nonstream(prompt, 24, label)
            usage = row.get("usage", {})
            actual = usage.get("prompt_tokens") if isinstance(usage, dict) else None
            row["requested_class_token_target"] = target
            row["input_length_status"] = "APPROXIMATE_CHAR_SCALED_REAL_CORPUS"
            row["corpus_characters_used"] = chars
            row["actual_prompt_tokens_from_api"] = actual
            row["ttft_s"] = None
            row["ttft_note"] = "Non-streaming API does not expose first-content time; exact token timing is performed by the parent's in-band run."
            row["semantic_pass"] = row["text"].strip() == "EPSILON72941"
            rows.append(row)
        phase = {"status": "PASS" if all(r["semantic_pass"] for r in rows) else "FAIL",
                 "note": "These HTTP prompt lengths are approximate; actual API usage is authoritative. Do not call them exact 32K/64K tests.",
                 "requests": rows}
        self.report["phases"]["http_real_corpus_prefill"] = phase
        self.save()
        if phase["status"] != "PASS":
            raise AcceptanceError("real-corpus verification-code prompt did not return its code")

    async def stories_sse(self) -> None:
        cases = story_cases()
        started = self.t()
        tasks = [asyncio.create_task(self.stream(c["prompt"], c["max_tokens"], c["id"])) for c in cases]
        rows = await asyncio.gather(*tasks)
        finish = max(r["completed_s"] for r in rows)
        by_id = {x["label"]: x for x in rows}
        for case in cases:
            row = by_id[case["id"]]
            row["inferred_output_tokens"] = 512 if row.get("finish_reason") == "length" else None
            row["token_count_basis"] = "max_tokens_verified_by_finish_reason" if row["inferred_output_tokens"] else "unavailable"
            row["semantic"] = story_semantics(case, row["text"])
            row["ttft_s"] = (row["first_content_s"] - row["arrival_s"]) if row.get("first_content_s") is not None else None
            row["decode_window_s"] = (row["completed_s"] - row["first_content_s"]) if row.get("first_content_s") is not None else None
            row["decode_window_tok_s"] = (512 / row["decode_window_s"]) if row.get("decode_window_s", 0) and row.get("inferred_output_tokens") == 512 else None
        first_emit = min((r["first_content_s"] for r in rows if r.get("first_content_s") is not None), default=None)
        all_emissions_window = (finish - first_emit) if first_emit is not None else None
        total_known = sum(r["inferred_output_tokens"] or 0 for r in rows)
        phase = {
            "status": "PASS" if all(r["semantic"]["pass"] and r.get("finish_reason") == "length" for r in rows) else "FAIL",
            "window_start_s": started,
            "last_completion_s": finish,
            "arrival_to_last_completion_s": finish - started,
            "first_content_to_last_completion_all_emissions_window_s": all_emissions_window,
            "stream_max_tokens_verified_count": total_known,
            "stream_all_emissions_tok_s_supplemental": (total_known / all_emissions_window) if all_emissions_window else None,
            "stream_token_count_note": "SSE event/chunk count is never treated as token count; each request must finish at max_tokens=512 to infer 512 output tokens.",
            "requests": rows,
        }
        self.report["phases"]["four_by_512_stream_metrics"] = phase
        self.save()
        if phase["status"] != "PASS":
            raise AcceptanceError("one or more 512-token SSE stories failed readability/semantic/length gates")

    async def stories_usage_wave(self) -> None:
        cases = story_cases()
        rows = await asyncio.gather(*(self.nonstream(c["prompt"], c["max_tokens"], "usage-" + c["id"]) for c in cases))
        start = min(r["arrival_s"] for r in rows)
        end = max(r["completed_s"] for r in rows)
        completion_tokens = [r.get("usage", {}).get("completion_tokens") for r in rows]
        if any(not isinstance(v, int) for v in completion_tokens):
            aggregate = None
            status = "FAIL"
        else:
            aggregate = sum(completion_tokens) / max(1e-9, end - start)
            status = "PASS" if all(r["text"] and r.get("finish_reason") == "length" for r in rows) else "FAIL"
        phase = {
            "status": status,
            "aggregate_method": "sum(actual usage.completion_tokens) / (last non-stream completion - first request arrival)",
            "arrival_to_last_completion_s": end - start,
            "total_completion_tokens": sum(completion_tokens) if all(isinstance(v, int) for v in completion_tokens) else None,
            "aggregate_tok_s": aggregate,
            "requests": rows,
        }
        self.report["phases"]["four_by_512_usage_aggregate"] = phase
        self.save()
        if status != "PASS":
            raise AcceptanceError("four-way non-stream usage aggregate missing or story wave failed")


async def run_acceptance(base_url: str, api_key: str, output_path: Path, model: str,
                         canary_file: Path, corpus_file: Path, ready_url: str | None,
                         prefix_evidence_file: Path | None, client_override: Any = None) -> dict[str, Any]:
    api_base, api_root = api_urls(base_url)
    report: dict[str, Any] = {
        "schema": "ac922-lastwave-http-acceptance-v1",
        "started_utc": utc_now(),
        "status": "RUNNING",
        "endpoint": safe_endpoint(base_url),
        "model": model,
        "phases": {},
        "limitations": [],
    }
    try:
        if client_override is not None:
            client = client_override
            owns_client = False
        else:
            client = StandardHttpClient()
            owns_client = False
        try:
            runner = Runner(client, api_base, api_root, model, api_key, output_path, report, prefix_evidence_file)
            discovery = await runner.discover(ready_url)
            corpus = load_corpus_text(corpus_file)
            await runner.canaries(canary_file)
            await runner.refill_wave()
            await runner.cancel_and_reuse()
            await runner.prefix_cache(bool(discovery.get("prefix_cache_enabled")), corpus)
            await runner.real_corpus_prefill(corpus)
            await runner.stories_sse()
            await runner.stories_usage_wave()
            prefix_status = report.get("phases", {}).get("exact_prefix_cache", {}).get("status")
            report["status"] = "NOT_PROVABLE" if prefix_status == "NOT_PROVABLE" else "PASS"
        finally:
            if owns_client and hasattr(client, "aclose"):
                await client.aclose()
    except BaseException as exc:
        report["status"] = "FAIL"
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)[:600]}
        if isinstance(exc, asyncio.CancelledError):
            report["status"] = "CANCELLED"
        raise
    finally:
        report["finished_utc"] = utc_now()
        write_report(output_path, report)
    return report


def load_keyfile(path: Path) -> str:
    value = path.expanduser().read_text(encoding="utf-8").strip()
    if not value:
        raise AcceptanceError("API key file is empty")
    if value.lower().startswith("bearer "):
        value = value[7:].strip()
    if not value:
        raise AcceptanceError("API key file has no key after the Bearer prefix")
    return value


def build_mock_client() -> Any:
    """A full-path, CPU-only mock for --self-test; no network sockets are opened."""
    prefix_seen: set[str] = set()
    story_topic_text = [
        "CPU cache memory cache lines and coherency protocol " * 70,
        "Photosynthesis light chlorophyll plant metabolism " * 70,
        "Roman architecture concrete arch engineering " * 70,
        "Comparison sorting algorithms require comparison and O(n log n) " * 70,
    ]

    class MockResponse:
        def __init__(self, status_code: int, body: dict[str, Any], headers: dict[str, str] | None = None):
            self.status_code = status_code
            self._body = body
            self.headers = headers or {}

        def json(self) -> dict[str, Any]:
            return self._body

    class MockStreamResponse:
        def __init__(self, frames: list[tuple[str, float]]):
            self.frames = frames
            self.status_code = 200
            self.headers = {"content-type": "text/event-stream"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def aiter_lines(self):
            for frame, delay in self.frames:
                if delay:
                    await asyncio.sleep(delay)
                for line in frame.splitlines():
                    yield line

    def make_body(payload: dict[str, Any]) -> dict[str, Any]:
        prompt = str(payload.get("prompt", "")) if isinstance(payload.get("prompt"), str) else prompt_text_from_messages(payload.get("messages", []))
        max_tokens = int(payload.get("max_tokens", 16))
        text = "Mock answer"
        finish_reason = "stop"
        usage_prompt = max(1, len(prompt) // 4)
        usage_completion = min(max_tokens, 8)
        cached = 0
        if "SHARED-PREFIX-BEGIN" in prompt or "EDITED-PREFIX-BEGIN" in prompt:
            common_prefix = prompt.split("\nQUESTION:", 1)[0]
            question = prompt.split("\nQUESTION:", 1)[1] if "\nQUESTION:" in prompt else ""
            if "ORION" in question:
                text = "6842"
            elif "NOVA" in question:
                text = "5301" if "NOVA = 5301" in common_prefix else "3179"
            else:
                text = "UNKNOWN"
            usage_completion = 1
            if common_prefix in prefix_seen:
                cached = max(1, usage_prompt - 5)
            else:
                cached = 0
                prefix_seen.add(common_prefix)
        elif "France" in prompt:
            text, usage_completion = "Paris", 1
        elif "2847" in prompt:
            text, usage_completion = "9805", 4
        elif "All dax" in prompt:
            text, usage_completion = "NO", 1
        elif "严格按要求输出" in prompt or '"colors"' in prompt:
            text, usage_completion = '{"ok":true,"count":3,"colors":["红","蓝"]}', 15
        elif "BEGIN_REFILL_LONG" in prompt:
            text = "BEGIN_REFILL_LONG\nCPU cache memory coherency protocol " * 70
            finish_reason, usage_completion = "length", max_tokens
        elif "capital city of Italy" in prompt:
            text, usage_completion = "Rome", 1
        elif "18 multiplied by 19" in prompt:
            text, usage_completion = "342", 1
        elif "CANCEL-ME" in prompt:
            text, usage_completion = "A long history of computing begins with early calculation devices. " * 50, max_tokens
            finish_reason = "length"
        elif "LAMBDA71" in prompt:
            text, usage_completion = "LAMBDA71", 4
        elif "Verification code" in prompt:
            text, usage_completion = "EPSILON72941", 7
        else:
            story_index = next((i for i in range(4) if f"BEGIN_{i}" in prompt), None)
            if story_index is not None:
                text = f"BEGIN_{story_index}\n" + story_topic_text[story_index]
                finish_reason, usage_completion = "length", max_tokens
            elif "What is the capital city of France" in prompt:
                text, usage_completion = "Paris", 1
        if usage_completion == max_tokens:
            finish_reason = "length"
        request_id = "chatcmpl-mock-" + hashlib.sha256(prompt.encode()).hexdigest()[:12]
        return {
            "id": request_id,
            "model": DEFAULT_MODEL,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": usage_prompt, "completion_tokens": usage_completion,
                      "total_tokens": usage_prompt + usage_completion,
                      "prompt_tokens_details": {"cached_tokens": cached}},
        }

    class MockClient:
        async def get(self, url: str, **kwargs: Any) -> MockResponse:
            del kwargs
            if url.endswith("/ready"):
                return MockResponse(200, {"status": "ready", "model": DEFAULT_MODEL,
                                          "max_context": 262140, "pool_tokens": 524288,
                                          "max_parallel": 4, "prefix_cache_enabled": True})
            if url.endswith("/models"):
                return MockResponse(200, {"data": [{"id": DEFAULT_MODEL, "object": "model"}]})
            return MockResponse(404, {"error": "not found"})

        async def post(self, url: str, *, headers: dict[str, str], json: dict[str, Any]) -> MockResponse:
            del headers
            if not url.endswith("/chat/completions"):
                return MockResponse(404, {"error": "not found"})
            return MockResponse(200, make_body(json))

        def stream(self, method: str, url: str, *, headers: dict[str, str], json: dict[str, Any]) -> MockStreamResponse:
            del headers
            if method != "POST" or not url.endswith("/chat/completions"):
                return MockStreamResponse([])
            import json as jsonlib
            body = make_body(json)
            prompt = prompt_text_from_messages(json.get("messages", []))
            text = body["choices"][0]["message"]["content"]
            finish_reason = body["choices"][0]["finish_reason"]
            frames = [
                ("data: " + jsonlib.dumps({"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]}) + "\n\n", 0),
                ("data: " + jsonlib.dumps({"choices": [{"delta": {"content": text}, "finish_reason": None}]}, ensure_ascii=False) + "\n\n", 0.01),
            ]
            if "CANCEL-ME" in prompt:
                return MockStreamResponse(frames)
            delay = 0.25 if "BEGIN_REFILL_LONG" in prompt else 0.01
            frames.append(("data: " + jsonlib.dumps({"choices": [{"delta": {}, "finish_reason": finish_reason}]}) + "\n\n", delay))
            frames.append(("data: [DONE]\n\n", 0))
            return MockStreamResponse(frames)

    return MockClient()


async def self_test() -> None:
    if not DEFAULT_CANARIES.exists() or not DEFAULT_CORPUS.exists():
        raise AcceptanceError("self-test requires the checked-in C12 canary and corpus fixtures")
    mock_client = build_mock_client()

    class MockHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            del fmt, args

        def _write_json(self, response: MockResponse) -> None:
            data = json.dumps(response.json(), ensure_ascii=False).encode("utf-8")
            self.send_response(response.status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            self.wfile.flush()

        def do_GET(self) -> None:
            response = asyncio.run(mock_client.get("http://127.0.0.1" + self.path))
            self._write_json(response)

        def do_POST(self) -> None:
            size = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(size).decode("utf-8"))
            if payload.get("stream") is True and self.path.endswith("/chat/completions"):
                stream = mock_client.stream("POST", self.path, headers={}, json=payload)
                self.send_response(stream.status_code)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                try:
                    for frame, delay in stream.frames:
                        if delay:
                            time.sleep(delay)
                        self.wfile.write(frame.encode("utf-8"))
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                self.close_connection = True
                return
            response = asyncio.run(mock_client.post("http://127.0.0.1" + self.path,
                                                     headers={}, json=payload))
            self._write_json(response)

    with tempfile.TemporaryDirectory(prefix="ac922-lastwave-mock-") as tmp:
        report_path = Path(tmp) / "mock-result.json"
        server = ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
        server_thread = threading.Thread(target=server.serve_forever, name="lastwave-http-mock", daemon=True)
        server_thread.start()
        base_url = f"http://127.0.0.1:{server.server_address[1]}/v1"
        try:
            report = await run_acceptance(base_url, "mock-key-never-written", report_path,
                                          DEFAULT_MODEL, DEFAULT_CANARIES, DEFAULT_CORPUS, None, None,
                                          client_override=StandardHttpClient(timeout=30.0))
        except BaseException:
            if report_path.exists():
                partial = json.loads(report_path.read_text(encoding="utf-8"))
                print(json.dumps({"mock_partial_phases": partial.get("phases", {})}, ensure_ascii=False), file=sys.stderr)
            raise
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=2)
        if report["status"] != "PASS":
            raise AcceptanceError("mock acceptance did not pass")
        assert report["phases"]["asymmetric_refill"]["status"] == "PASS"
        assert report["phases"]["exact_prefix_cache"]["status"] == "PASS"
        assert report["phases"]["four_by_512_usage_aggregate"]["aggregate_tok_s"] is not None
        saved = report_path.read_text(encoding="utf-8")
        if "mock-key-never-written" in saved or "Bearer mock-key-never-written" in saved:
            raise AcceptanceError("self-test detected credential leakage in result JSON")
        print(json.dumps({"self_test": "PASS", "mock_model_requests": "local loopback mock only; no model calls",
                          "phases": list(report["phases"])}, ensure_ascii=False))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("AC922_BASE_URL"))
    parser.add_argument("--key-file", default=os.environ.get("AC922_API_KEY_FILE"))
    parser.add_argument("--ready-url", default=os.environ.get("AC922_READY_URL"))
    parser.add_argument("--out", default=os.environ.get("AC922_RESULT_JSON"))
    parser.add_argument("--canaries", default=os.environ.get("AC922_CANARY_FILE", str(DEFAULT_CANARIES)))
    parser.add_argument("--corpus", default=os.environ.get("AC922_CORPUS_FILE", str(DEFAULT_CORPUS)))
    parser.add_argument("--prefix-evidence-file", default=os.environ.get("AC922_PREFIX_EVIDENCE_FILE"))
    parser.add_argument("--model", default=os.environ.get("AC922_MODEL", DEFAULT_MODEL))
    parser.add_argument("--self-test", action="store_true", help="run CPU-only in-process HTTP mock; no live calls")
    return parser.parse_args(argv)


async def main_async(args: argparse.Namespace) -> int:
    if args.self_test:
        await self_test()
        return 0
    if not args.base_url:
        raise AcceptanceError("set AC922_BASE_URL or pass --base-url")
    if not args.key_file:
        raise AcceptanceError("set AC922_API_KEY_FILE or pass --key-file")
    key_path = Path(args.key_file).expanduser()
    if not key_path.is_file():
        raise AcceptanceError("API key file does not exist")
    api_key = load_keyfile(key_path)
    output = report_file_from_env(args.out)
    report = await run_acceptance(
        args.base_url, api_key, output, args.model, Path(args.canaries).expanduser(),
        Path(args.corpus).expanduser(), args.ready_url,
        Path(args.prefix_evidence_file).expanduser() if args.prefix_evidence_file else None,
    )
    print(json.dumps({"status": report["status"], "result_json": str(output), "model": report["model"],
                      "endpoint": report["endpoint"], "phases": {k: v.get("status") for k, v in report["phases"].items()}}, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else (2 if report["status"] == "NOT_PROVABLE" else 1)


def main() -> int:
    args = parse_args(sys.argv[1:])
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("acceptance interrupted; partial result, if any, remains on disk", file=sys.stderr)
        return 130
    except BaseException as exc:
        # Deliberately avoid dumping request headers, the environment, or key material.
        print(f"acceptance failed: {type(exc).__name__}: {str(exc)[:600]}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
