# AC922 production candidate (2026-10-18)

## Scope and state

This directory is an immutable release candidate. Current deployment/acceptance state is recorded separately under /home/debian/ac922-monster/production/. A configured context limit is not a full-context workload result.

The implementation preserves the V306 four-bank generic-D3 execution path, one shared IQ3_XXS+Q8 MTP model and its frozen R299 libraries. It adds persistent IPC and a loopback API. The published release contains private copies of all seven runtime libraries; it does not mix runtimes or change model precision.

## Context contract

- The model's native context is 262144 tokens. The API reserves four tokens for MTP lookahead and advertises 262140 tokens per request, **prompt plus requested output**.
- Four independent banks share one **logical allocation budget**, repartitioned only at a drained wave boundary. This is not a paged KV arena or four full-size copies of the pool.
- Pool: 524288 target-context cells (matching draft/indexer storage is included in the memory budget). The r1 real HTTP acceptance used502158 input tokens across four banks; the model and compute libraries are unchanged in r2. Rounding and inactive-bank overhead reduce the exact usable sum slightly.
- Requests can have unequal lengths. FIFO requests coalesce for 30ms into waves of at most four. New arrivals queue until that wave drains. A completed lane can return before its peers; it is not refilled mid-wave.
- A request that is too large fails independently. Combined pool overflow splits the wave into smaller batches, respecting queue bounds, cancellation and deadlines.
- Q8_0 target/draft K/V, all experts and the integrated model remain unchanged. Cross-request KV/prefix caching is disabled because the fast single-lane MTP snapshot is not supported.

## API

Internal endpoint: `http://127.0.0.1:8001/v1`.

After gateway qualification and promotion, the existing authenticated gateway provides:

- `https://server.wuxuexi.top/v1`
- Existing IPv4 fallback: `https://ac922.tail58d5eb.ts.net/v1`

Model name: `flash-next-iq3-xxs`. Routes: `/v1/models`, `/v1/chat/completions`, `/v1/completions`; `stream:true` provides token SSE with UTF-8 assembly. Quiet streams emit a 15-second SSE comment heartbeat during long prefill. Use an existing API key in `Authorization: Bearer ...` or `X-API-Key`; keys are not copied into this release.

Supported first-release sampling is greedy: `temperature:0`, `top_p:1`, `n:1`. Output limit is 8192 tokens, additionally constrained by the prompt+output context limit. Text messages support system/user/assistant roles. Images, tools and unsupported sampling/options return explicit errors. This is a text generation API, not an implementation of every OpenAI feature.

Example (substitute an existing key privately):

```sh
curl https://server.wuxuexi.top/v1/chat/completions \
  -H "Authorization: Bearer $AC922_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"flash-next-iq3-xxs","messages":[{"role":"user","content":"请计算7+4，只输出整数。"}],"temperature":0,"max_tokens":32,"stream":true}'
```

## Operations

Services: `ac922-inference.service`, `ac922-inference-safety.service`; existing gateway service remains `ac922-apigateway.service`.

- Startup verifies release/library hashes and model size/mtime, refuses a second GPU model, checks physical DDR NUMA0/8, and waits for 15 consecutive healthy sensor seconds.
- Model loading and the native four-case semantic gate run before `/ready` returns 200. `/health` reports process liveness, not readiness.
- The candidate GPU-free floor is 256MiB following the user's explicit override of the prior 512MiB experiment threshold. Thermal stop remains 58C. The monitor reads the existing sensors; it does not control fans or alter BMC settings.
- A thermal, persistent tach, collector or GPU-memory fault stops this model service and persists a latch under `/var/lib/ac922-inference/fault.json`. Inspect before manually clearing it.
- A worker failure fails pending jobs and exits the API with code70. systemd allows at most two starts per hour, with 30s backoff. There is no watcher that automatically refires benchmarks when GPUs become free.
- Worker logs rotate at4MiB with two backups. Request bodies/prompts are not logged.
- Health: `curl http://127.0.0.1:8001/ready`; logs: `journalctl -u ac922-inference -u ac922-inference-safety`; model detail: `/home/debian/ac922-monster/production/run/worker.log`.

Once installed, rollback is one command:

```sh
sudo python3 /home/debian/ac922-monster/production/rollback.py
```

It stops/disables the new inference services and restores the saved gateway source if that source has not subsequently been edited. It retains model weights, release archives, experiment records and other services. No server shutdown is part of rollback.

## Evidence so far

- Frozen V306 benchmark: shared decode283.44tok/s; inclusive219.94tok/s. These are benchmark metrics, not a claimed production-API throughput result.
- R299 persistent worker: native four cases, asymmetric short requests, 393216-cell asymmetric allocation, cancel/reuse and one actual219189-token input passed. The long input completed in169.99s including its short response; output was Paris. GPU-free minima were1132MiB or greater, GPU peak57C.
- A524288-cell allocation test at the prior512MiB floor was guard-stopped at488MiB free on GPU5; it did not report CUDA OOM. The256MiB candidate then completed502158 real HTTP input tokens over four unequal requests, with all four unique retrieval codes correct.
- API mock suite17/17, safety mock suite7/7 and gateway proxy suite4/4 passed. Real r1 HTTP four-way retrieval used inputs218452/129610/87996/66100 and completed in646.15s including replies. A later retained-context four-way512-token output test completed all2048 tokens in15.69s (130.56tok/s end to end); these prompts differ from the frozen synthetic benchmark. R2 additionally shrinks an oversized idle context family back to the qualified17152-cell four-bank layout at a drained boundary.
- No claim of6000tok/s uncached prefill or300tok/s shared decode. Existing V412 warm32K result5066.44tok/s is a separate R411 recipe and is not attributed to this R299 API.
