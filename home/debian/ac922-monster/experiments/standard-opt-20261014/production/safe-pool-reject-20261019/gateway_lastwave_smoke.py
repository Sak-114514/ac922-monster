"""Check the promoted gateway using its existing key without recording it."""
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

HERE = Path(__file__).resolve().parent
BASE = "http://127.0.0.1:8080"
MODEL = "flash-next-iq3-xxs"
opener = build_opener(ProxyHandler({}))
key = Path("/home/debian/apigateway/api_key").read_text().strip()
headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}


def request(path, payload=None, authenticated=True):
    data = None if payload is None else json.dumps(payload).encode()
    return opener.open(Request(BASE + path, data=data,
                               headers=headers if authenticated else {}), timeout=120)


def main():
    report = {"status": "RUNNING", "started_unix": time.time(), "base_url": BASE}
    for attempt in range(30):
        try:
            with request("/v1/models") as response:
                models = json.load(response)
            break
        except (HTTPError, URLError):
            if attempt == 29:
                raise
            time.sleep(1)
    assert any(row.get("id") == MODEL for row in models["data"]), models
    report["models"] = "PASS"
    try:
        with request("/v1/models", authenticated=False):
            raise AssertionError("unauthenticated model route was accepted")
    except HTTPError as error:
        assert error.code == 401, error.code
    report["unauthenticated_status"] = 401
    payload = {"model": MODEL, "messages": [{"role": "user", "content":
        "What is the capital city of France? Reply with exactly the city name."}],
        "temperature": 0, "max_tokens": 24}
    with request("/v1/chat/completions", payload) as response:
        answer = json.load(response)
    text = answer["choices"][0]["message"]["content"].strip()
    assert text == "Paris", text
    report["chat"] = {"text": text, "usage": answer["usage"]}
    payload.update(stream=True, messages=[{"role": "user", "content":
        "请计算 7+4，只输出整数。"}])
    pieces, finished = [], False
    with request("/v1/chat/completions", payload) as response:
        for raw in response:
            line = raw.decode().strip()
            if line == "data: [DONE]":
                finished = True
                break
            if line.startswith("data: "):
                chunk = json.loads(line[6:])
                pieces.append(chunk.get("choices", [{}])[0].get("delta", {}).get("content", ""))
    stream_text = "".join(pieces).strip()
    assert finished and stream_text == "11", (finished, stream_text)
    report.update(status="PASS", sse={"text": stream_text, "done": finished},
                  finished_unix=time.time())
    (HERE / "gateway-smoke-result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
