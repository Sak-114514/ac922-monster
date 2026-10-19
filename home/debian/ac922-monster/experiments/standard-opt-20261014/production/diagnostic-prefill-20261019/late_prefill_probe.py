#!/usr/bin/env python3
"""Diagnose short-request service while a long prefill is in flight.

Default behavior is a local CPU-only mock self-test. Live mode (--run) sends
one long real-corpus chat request and, after 0.5 seconds, three short SSE
requests to an already-loaded API. It does not start or reconfigure a model.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import json as json_module
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any


HERE = Path(__file__).resolve().parent
POSTRELEASE = HERE
sys.path.insert(0, str(POSTRELEASE))

import lastwave_http_acceptance as http_acceptance  # noqa: E402


DEFAULT_CORPUS = POSTRELEASE.parents[1] / "fixtures/long-prefill-combo-v45-20261016/v45-real-corpus-source.json"
DEFAULT_OUT = HERE / "run/late-prefill-result.json"
DEFAULT_MODEL = "flash-next-iq3-xxs"
LONG_CLASS_TOKENS = 32_768
LONG_MAX_TOKENS = 16
SHORT_LAUNCH_DELAY_S = 0.5
VERIFICATION_CODE = "EPSILON72941"

SHORT_CASES = (
    {
        "id": "short-paris",
        "prompt": "What is the capital city of France? Reply with exactly the city name, without punctuation or explanation.",
        "max_tokens": 16,
        "kind": "exact_text",
        "expected": "Paris",
    },
    {
        "id": "short-arithmetic",
        "prompt": "请计算 7 + 4。只输出十进制整数，不要解释。",
        "max_tokens": 16,
        "kind": "exact_text",
        "expected": "11",
    },
    {
        "id": "short-strict-json",
        "prompt": "请严格按要求输出一个 JSON 对象，不要 Markdown、代码围栏或其他文字。对象必须是：{\"ok\":true,\"count\":3,\"colors\":[\"红\",\"蓝\"]}",
        "max_tokens": 32,
        "kind": "strict_json",
        "expected": {"ok": True, "count": 3, "colors": ["红", "蓝"]},
    },
)


class ProbeError(RuntimeError):
    pass


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.chmod(temp, 0o600)
    os.replace(temp, path)


def make_long_prompt(source: str) -> tuple[str, int]:
    prompt, corpus_characters = http_acceptance.build_real_corpus_prompt(source, LONG_CLASS_TOKENS)
    final_question = "\nFINAL QUESTION:"
    if final_question not in prompt:
        raise ProbeError("existing C12 real-corpus prompt is missing its final question marker")
    prompt = prompt.replace(
        final_question,
        f"\nFooter verification code: {VERIFICATION_CODE}.\nFINAL QUESTION:",
        1,
    )
    if VERIFICATION_CODE not in prompt[:256] or VERIFICATION_CODE not in prompt[-512:]:
        raise ProbeError("long prompt does not carry the verification code in both head and tail")
    return prompt, corpus_characters


def load_corpus(path: Path) -> str:
    try:
        return http_acceptance.load_corpus_text(path)
    except Exception as exc:
        raise ProbeError(f"cannot load existing real-corpus fixture: {type(exc).__name__}") from None


class ProbeRunner:
    def __init__(self, client: Any, base_url: str, api_key: str, model: str):
        api_base, _ = http_acceptance.api_urls(base_url)
        self.url = api_base.rstrip("/") + "/chat/completions"
        self.client = client
        self.model = model
        self.headers = {"Authorization": "Bearer " + api_key,
                        "Accept": "application/json"}
        self.clock0 = time.monotonic()

    def elapsed(self) -> float:
        return time.monotonic() - self.clock0

    def payload(self, prompt: str, max_tokens: int) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": True,
        }

    async def stream_one(self, prompt: str, max_tokens: int, label: str,
                         expected: Any, kind: str) -> dict[str, Any]:
        record: dict[str, Any] = {
            "id": label,
            "arrival_s": self.elapsed(),
            "http_status": None,
            "headers_received_s": None,
            "first_nonempty_s": None,
            "completed_s": None,
            "finish_reason": None,
            "response_id": None,
            "max_tokens": max_tokens,
            "prompt_characters": len(prompt),
            "prompt_sha256": sha256_text(prompt),
            "text": "",
            "stream_usage": None,
            "semantic_kind": kind,
            "expected": expected,
        }
        parts: list[str] = []
        try:
            async with self.client.stream(
                    "POST", self.url, headers=self.headers,
                    json=self.payload(prompt, max_tokens)) as response:
                record["http_status"] = response.status_code
                record["headers_received_s"] = self.elapsed()
                async for line in response.aiter_lines():
                    payload = http_acceptance.parse_json_sse_line(line)
                    if payload is None:
                        continue
                    if record["response_id"] is None and isinstance(payload.get("id"), str):
                        record["response_id"] = payload["id"]
                    usage = payload.get("usage")
                    if isinstance(usage, dict):
                        record["stream_usage"] = usage
                    choices = payload.get("choices", [])
                    if not isinstance(choices, list) or not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta", {}) if isinstance(choice, dict) else {}
                    content = delta.get("content") if isinstance(delta, dict) else None
                    if isinstance(content, str) and content:
                        if record["first_nonempty_s"] is None:
                            record["first_nonempty_s"] = self.elapsed()
                        parts.append(content)
                    finish = choice.get("finish_reason") if isinstance(choice, dict) else None
                    if finish:
                        record["finish_reason"] = finish
        except Exception as exc:
            record["transport_error"] = type(exc).__name__
        finally:
            record["completed_s"] = self.elapsed()

        record["text"] = "".join(parts)
        record["semantic_pass"] = semantic_check(record["text"], expected, kind)
        return record


def semantic_check(text: str, expected: Any, kind: str) -> bool:
    if kind == "strict_json":
        try:
            return json.loads(text) == expected
        except json.JSONDecodeError:
            return False
    return text.strip() == str(expected)


async def run_probe(client: Any, base_url: str, api_key: str, model: str,
                    source: str, output_path: Path) -> dict[str, Any]:
    long_prompt, corpus_characters = make_long_prompt(source)
    runner = ProbeRunner(client, base_url, api_key, model)
    report: dict[str, Any] = {
        "schema": "ac922-late-prefill-probe-v1",
        "status": "RUNNING",
        "started_unix": time.time(),
        "endpoint": http_acceptance.safe_endpoint(base_url),
        "model": model,
        "service_mode": "already-loaded API only; no service/model lifecycle operations",
        "additional_model_loads_requested": 0,
        "long_prompt_class": "approximately 32K class using the existing C12 real-corpus character calibration; not exact-token preflight",
        "long_corpus_characters": corpus_characters,
        "long_prompt_characters_including_template": len(long_prompt),
        "long_prompt_sha256": sha256_text(long_prompt),
        "verification_code": VERIFICATION_CODE,
        "long_max_tokens": LONG_MAX_TOKENS,
        "short_launch_delay_s_requested": SHORT_LAUNCH_DELAY_S,
        "short_cases": [{key: value for key, value in case.items() if key != "prompt"}
                         | {"prompt_sha256": sha256_text(case["prompt"]),
                            "prompt_characters": len(case["prompt"])}
                         for case in SHORT_CASES],
        "requests": {},
    }
    long_task = asyncio.create_task(
        runner.stream_one(long_prompt, LONG_MAX_TOKENS, "long-prefill", VERIFICATION_CODE, "exact_text"),
        name="late-prefill-long-request")
    await asyncio.sleep(SHORT_LAUNCH_DELAY_S)
    report["short_wave_launch_s"] = runner.elapsed()
    short_tasks = [asyncio.create_task(
        runner.stream_one(case["prompt"], case["max_tokens"], case["id"], case["expected"], case["kind"]),
        name=f"late-prefill-{case['id']}") for case in SHORT_CASES]
    gathered = await asyncio.gather(long_task, *short_tasks, return_exceptions=True)
    labels = ["long-prefill", *(case["id"] for case in SHORT_CASES)]
    for label, outcome in zip(labels, gathered):
        if isinstance(outcome, BaseException):
            report["requests"][label] = {"id": label, "task_error": type(outcome).__name__,
                                         "completed_s": runner.elapsed(), "text": "",
                                         "semantic_pass": False}
        else:
            report["requests"][label] = outcome

    long = report["requests"]["long-prefill"]
    long_end = long.get("completed_s")
    short_rows = [report["requests"][case["id"]] for case in SHORT_CASES]
    order = {
        row["id"]: (row.get("first_nonempty_s") is not None and long_end is not None and
                    row["first_nonempty_s"] < long_end)
        for row in short_rows
    }
    arrival_times = [row["arrival_s"] for row in report["requests"].values()
                     if isinstance(row.get("arrival_s"), (int, float))]
    completion_times = [row["completed_s"] for row in report["requests"].values()
                        if isinstance(row.get("completed_s"), (int, float))]
    first_arrival = min(arrival_times, default=0.0)
    last_completion = max(completion_times, default=0.0)
    report["timing_analysis"] = {
        "global_time_origin": "long request launch",
        "first_request_arrival_s": first_arrival,
        "short_launch_delay_s_requested": SHORT_LAUNCH_DELAY_S,
        "long_completed_before_short_launch": (long_end is not None and
                                                long_end < report["short_wave_launch_s"]),
        "short_first_nonempty_before_long_completion": order,
        "all_short_first_nonempty_before_long_completion": bool(order) and all(order.values()),
        "last_completion_s": last_completion,
        "arrival_to_last_completion_s": last_completion - first_arrival,
        "interpretation": diagnose(long, short_rows, order,
                                   report["short_wave_launch_s"]),
        "throughput_note": "No token rate is computed from SSE event or chunk counts.",
    }
    report["semantic_status"] = "PASS" if all(
        bool(row.get("semantic_pass")) for row in report["requests"].values()) else "FAIL"
    report["status"] = "RECORDED"
    report["finished_unix"] = time.time()
    write_report(output_path, report)
    return report


def diagnose(long: dict[str, Any], shorts: list[dict[str, Any]],
             order: dict[str, bool], short_launch_s: float) -> str:
    if long.get("completed_s") is not None and long["completed_s"] < short_launch_s:
        return "long request completed before short requests launched; overlap diagnosis unavailable"
    if not long.get("first_nonempty_s"):
        return "long request produced no content; inspect HTTP/stream result before interpreting scheduling"
    if all(order.values()):
        return "all three short requests emitted content while the long request was still active"
    if any(order.values()):
        return "mixed result: some short requests emitted before the long request ended; inspect per-request timing"
    missing = [row["id"] for row in shorts if row.get("first_nonempty_s") is None]
    if missing:
        return "one or more short requests had no first content; inspect HTTP status and transport errors"
    return "all short first-content events followed long completion; consistent with whole-prefill blocking or queueing"


class FakeStreamResponse:
    def __init__(self, text: str, delay_s: float, finish_reason: str = "stop"):
        self.text = text
        self.delay_s = delay_s
        self.finish_reason = finish_reason
        self.status_code = 200

    async def __aenter__(self) -> "FakeStreamResponse":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    async def aiter_lines(self):
        yield "data: " + json.dumps({"id": "mock-response", "choices": [
            {"delta": {"role": "assistant"}, "finish_reason": None}]})
        await asyncio.sleep(self.delay_s)
        yield "data: " + json.dumps({"choices": [
            {"delta": {"content": self.text}, "finish_reason": None}]}, ensure_ascii=False)
        await asyncio.sleep(0.12)
        yield "data: " + json.dumps({"choices": [
            {"delta": {}, "finish_reason": self.finish_reason}]})
        yield "data: [DONE]"


class FakeClient:
    """In-memory response generator; it never opens sockets or contacts a model."""

    def stream(self, method: str, url: str, *, headers: dict[str, str], json: dict[str, Any]):
        del url, headers
        if method != "POST":
            raise AssertionError("mock received unexpected HTTP method")
        prompt = http_acceptance.prompt_text_from_messages(json.get("messages", []))
        if VERIFICATION_CODE in prompt:
            return FakeStreamResponse(VERIFICATION_CODE, 0.95)
        if "capital city of France" in prompt:
            return FakeStreamResponse("Paris", 0.04)
        if "7 + 4" in prompt:
            return FakeStreamResponse("11", 0.06)
        if "strictly" in prompt or "严格按要求输出" in prompt:
            return FakeStreamResponse(json_module.dumps(SHORT_CASES[2]["expected"], ensure_ascii=False), 0.08)
        raise AssertionError("mock received an unknown prompt")


async def self_test(source: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="ac922-late-prefill-mock-") as temporary:
        output = Path(temporary) / "late-prefill-result.json"
        report = await run_probe(FakeClient(), "http://mock.invalid/v1", "mock-key-never-written",
                                 DEFAULT_MODEL, source, output)
        if report["semantic_status"] != "PASS":
            raise ProbeError("mock semantic checks failed")
        if not report["timing_analysis"]["all_short_first_nonempty_before_long_completion"]:
            raise ProbeError("mock timing did not show short responses overlapping the long request")
        saved = output.read_text(encoding="utf-8")
        if "mock-key-never-written" in saved or "Bearer mock-key-never-written" in saved:
            raise ProbeError("mock API key leaked to result JSON")
        if any(case["prompt"] in saved for case in SHORT_CASES):
            raise ProbeError("short prompt leaked to result JSON")
        if report["long_prompt_characters_including_template"] <= 100_000:
            raise ProbeError("existing 32K-class corpus calibration was not reused")
        return {
            "self_test": "PASS",
            "transport": "in-memory fake SSE; no network/socket/model calls",
            "semantic_status": report["semantic_status"],
            "diagnosis": report["timing_analysis"]["interpretation"],
            "result_written": "temporary path only",
        }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--self-test", action="store_true", help="CPU-only mock; this is the default")
    mode.add_argument("--run", action="store_true", help="explicitly call the already-loaded API")
    parser.add_argument("--base-url", default=os.environ.get("AC922_BASE_URL"))
    parser.add_argument("--key-file", default=os.environ.get("AC922_API_KEY_FILE"))
    parser.add_argument("--corpus", default=os.environ.get("AC922_CORPUS_FILE", str(DEFAULT_CORPUS)))
    parser.add_argument("--out", default=os.environ.get("AC922_LATE_PREFILL_RESULT", str(DEFAULT_OUT)))
    parser.add_argument("--model", default=os.environ.get("AC922_MODEL", DEFAULT_MODEL))
    return parser.parse_args(argv)


async def main_async(args: argparse.Namespace) -> int:
    source = load_corpus(Path(args.corpus).expanduser())
    if not args.run:
        print(json.dumps(await self_test(source), ensure_ascii=False))
        return 0
    if not args.base_url or not args.key_file:
        raise ProbeError("live mode requires --base-url/AC922_BASE_URL and --key-file/AC922_API_KEY_FILE")
    api_key = http_acceptance.load_keyfile(Path(args.key_file))
    client = http_acceptance.StandardHttpClient(timeout=1805.0)
    report = await run_probe(client, args.base_url, api_key, args.model, source,
                             Path(args.out).expanduser())
    print(json.dumps({
        "status": report["status"],
        "semantic_status": report["semantic_status"],
        "diagnosis": report["timing_analysis"]["interpretation"],
        "result": str(Path(args.out).expanduser()),
    }, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ProbeError, http_acceptance.AcceptanceError) as error:
        print(f"late prefill probe error: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
