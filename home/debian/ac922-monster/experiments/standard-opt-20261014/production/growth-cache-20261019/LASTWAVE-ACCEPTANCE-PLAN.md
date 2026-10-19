# Last-wave HTTP acceptance plan

This plan qualifies the already-loaded production candidate through its OpenAI-compatible HTTP API. The harness never starts, stops, reloads, or configures the model. The parent owns launch authorization and chooses the endpoint. The run is a single live-service session; `--self-test` uses only a local mock transport and makes no model requests.

## Invocation contract

```sh
AC922_BASE_URL='https://server.wuxuexi.top:8443/v1' \
AC922_API_KEY_FILE='/path/to/api-key' \
AC922_RESULT_JSON='/path/to/lastwave-http-result.json' \
python3 lastwave_http_acceptance.py
```

Required environment:

- `AC922_BASE_URL`: API root ending in `/v1` or the service root.
- `AC922_API_KEY_FILE`: file containing the API key. The program reads it in memory and never places its value in the report or console output.

Optional environment:

- `AC922_READY_URL`: direct readiness endpoint if the chosen API gateway does not expose `/ready`.
- `AC922_RESULT_JSON`: output path. The default is a timestamped JSON file in this directory.
- `AC922_CANARY_FILE`: defaults to the frozen C12 native canaries at `fixtures/fourbank-d3-newruntime-v306-20261017/v221-d3-scenarios.json`.
- `AC922_CORPUS_FILE`: defaults to `fixtures/long-prefill-combo-v45-20261016/v45-real-corpus-source.json`.
- `AC922_PREFIX_EVIDENCE_FILE`: optional worker/service JSON or JSONL with per-request prefix-cache hit-token evidence. Lack of this evidence is reported as `NOT_PROVABLE`.
- `AC922_MODEL`: defaults to `flash-next-iq3-xxs`.

`python3 lastwave_http_acceptance.py --self-test` runs CPU-only checks against an in-process HTTP mock. It does not contact the configured endpoint or read the API key.

## One-load run order

1. Read `/ready` when available and query authenticated `/v1/models`. Require model ID `flash-next-iq3-xxs`; save only the reported model/context/pool/concurrency fields. Readiness is not interpreted as evidence that prefix caching exists.
2. Send the four frozen C12 semantic canaries concurrently: France capital, Chinese arithmetic, logic, and strict JSON. Preserve returned text, token usage, HTTP timings, and each exact semantic verdict.
3. Send four asymmetric SSE requests together: one 512-token long real-text answer and three short 64-token prompts. As soon as one short response completes, submit two more compatible 64-token prompts while the original 512-token request remains active. HTTP 200 is recorded as accepted/queued only. Dynamic refill passes only when both follow-up streams emit their first nonempty content before the original long stream completes. This avoids mistaking a queued HTTP response for a running request.
4. Cancel a separate long SSE request immediately after its first nonempty content, then run the Paris canary and require it to finish correctly. Record cancellation-to-reuse latency and both responses.
5. Run prefix cases only if readiness/capability metadata explicitly says prefix caching is enabled. Use native `/v1/chat/completions` with the exact same at-least-32K-character common body, no request-specific leading ID, and short question suffixes. The shared body defines `NOVA = 3179` and `ORION = 6842`; query them in A/B/A order and require exact outputs `3179`, `6842`, `3179` plus natural EOG to detect cross-request state mixing. Then alter the shared body early so it defines a new NOVA value `5301`; require that new answer with zero cached tokens. A cache hit requires explicit per-request cached-token evidence from response usage/header or the optional evidence file. Latency alone never proves a hit. If declared enabled but no hit evidence is available, report `NOT_PROVABLE`; otherwise report `SKIPPED_NOT_ENABLED`.
6. Send 32K-class and 64K-class long text requests built from the existing C12 real corpus source. Because the public chat API does not expose tokenizer preflight, the prompt lengths are approximate inputs; report each actual `usage.prompt_tokens`, wall latency, and returned semantic code. Do not label character-scaled HTTP prompts as exact 32K/64K token tests. The parent performs exact 32K/64K token-count prefill timing through the existing in-band path under the same model load.
7. Send four independent 512-token C12-topic real-text requests twice: first as SSE to measure per-request TTFT and decode windows while retaining the complete readable output; then as non-streaming requests to obtain authoritative `usage.completion_tokens` for the aggregate rate. Keep those timing samples separate. For streaming, token counts are not inferred from the number of SSE chunks; require `finish_reason=length` at `max_tokens=512` for the per-request 512-token window calculation. The official aggregate is `sum(usage.completion_tokens) / (last completion time - first request arrival)` from the non-streaming wave.

## Report schema and gates

The output JSON includes run metadata, sanitized endpoint origin, readiness/capability evidence, phase status, raw per-request timings and usage, semantic checks, and complete four-story output text. It contains no key and does not include prompts. It is written atomically after each phase and on exit.

Required acceptance evidence:

- All four frozen native semantic canaries pass.
- Refill canaries use at most 64 output tokens and short prompts so their context reservations remain below the current 17,152-token bank profile. The 512-token refill long request remains unfinished when requests five and six begin; requests five and six both return HTTP 200 before it finishes; both emit content before it finishes.
- Cancellation is followed by a successful Paris request.
- Prefix A/B/A outputs must be `3179`, `6842`, `3179`, while the edited early-body variant must return its new key and show zero cached tokens. Exact reuse requires explicit positive cached-token evidence; otherwise it is honestly skipped or marked unprovable.
- Real-corpus HTTP requests report actual API prompt-token usage and are labeled approximate-length workload. Exact 32K/64K timing comes from the separate in-band run.
- All four 512-token stories reach their length limit, pass topic/readability gates, retain complete text, and record individual TTFT/decode windows. A separate non-streaming four-way wave provides the authoritative completion-token aggregate rate.

## C12 comparison anchors

Use these only as matching-scope historical references:

- C12 HTTP: 36,646 actual prompt tokens and 17.1897 seconds end-to-end for one real-corpus verification-code request.
- C12 IPC, not HTTP: first/repeated uncached 32K prefill 4,255.17 / 4,065.87 tok/s; 65,536-token input with the maximum context reservation 1,433.44 tok/s; four 8K requests 3,591.91 aggregate tok/s.
- C12 IPC four-by-512: shared generation window 239.14 tok/s; all-emissions 187.98 tok/s; complete 2,048-token wave 11.2892 seconds, or 181.41 tok/s.

The 36,646-token HTTP value is not called 32K. IPC timings are not mixed with HTTP timings. Token throughput is computed from usage counts and the explicitly named wall-clock window, never from SSE chunk count.
