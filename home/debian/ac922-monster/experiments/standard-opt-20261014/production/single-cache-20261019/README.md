# AC922 R444 + R5 production combo

## Serving contract

- Model: `flash-next-iq3-xxs`; existing integrated IQ3_XXS main + Q8 MTP head, all experts retained.
- One shared model on six V100 GPUs, four independent target/MTP banks, generic MTP D3.
- Logical pool 524288 cells; per-request prompt plus output at most 262140 tokens. The pool also accounts for alignment and inactive-bank reservations.
- At most four requests in one asymmetric wave. Later requests queue until the current wave drains. This is not mid-wave continuous refill.
- Target and draft KV remain Q8. Greedy sampling only (`temperature: 0`); text chat/completions and SSE supported.
- Private API binds `127.0.0.1:8001`; public model routes use the existing authenticated API gateway.

## Combined paths

Whole R444 libraries + private R5 captured-only helper; standard indexer; graph structural/snapshot caches and V444 bounded retention; six-GPU pinned upload/plan-rebind; eight-copy single-prefill schedulers and four-copy multi-bank schedulers. Multi-bank logical batches remain4096; wide single-prefill logical batches are32768. Target UBatch is 1024 for medium single prompts, 512 for long single prompts, 256 for multi-bank; draft UBatch remains256. Global forced graphs are off while dedicated prefill capture remains enabled.

PLE uses the existing exact striped placement: 18GiB across GPU NUMA nodes plus the remainder on CPU. New contiguous layer split is9/9/8/8/8/8 including tail ownership; PLE stripe MiB4096/4608/4224/3200/2176/128. Full model capacity and cache precision are unchanged.

Relay6 handles AllReduce and legacy TurboPrefill requires Meta3 topology; neither is enabled for this six-CUDA layer pipeline. Prompt/KV prefix reuse is not enabled: per-bank target+MTP snapshot restore is a separate correctness requirement.

## Inference safety policy

GPU core and GPU memory stop threshold:65°C. Long-prefill pacing pauses at59°C and resumes at54°C. CPU/DIMM/VDD/ambient/PCIe stop threshold remains58°C. Missing/stale sensors, repeated tach below2700RPM, and GPU free memory below256MiB stop the model service and persist a fault. Root monitor never writes BMC/fan/driver/power configuration.

## Evidence boundaries

C12 guarded fixture passed524288 total allocated bank cells, maximum262140 single-request allocation with actual65536-token semantic input, native four-way canaries before/after transitions, and four512-token real-text completions. This does not mean524288 input tokens were processed in that fixture. The previous R299 API separately processed502158 real input tokens across four requests; do not present that as a new R444 full-pool run.

C12 IPC worker measurements:32K first/repeated uncached prefill4255.17/4065.87tok/s; actual64K input with maximum262K reserved context1433.44tok/s; four8K native semantic inputs aggregate3591.91tok/s. Four512 shared generation window239.14tok/s, all-emissions187.98tok/s, entire wave2048/11.28924=181.41tok/s. HTTP/client latency is additional. Run order: full-pool allocation,64K/max-context case,32K first/repeat,8K single semantic, native4, four8K semantics, four512, native4. First means a fresh profile context, not a cold driver. These differ from the historical independent prefill5066 record and remain below the300/6000 goals. No repeatability claim from this compact sample. All semantic gates passed.

Production deployment must set systemd LimitMEMLOCK=4G. The qualified user-unit budget is about4GiB; the system default8MiB cannot satisfy the existing32MiB temporary PLE pin fallback. This is a bounded resource limit for this service, not persistent whole-table locking.

## Operations

Use the release's `worker-command.json`, `release-manifest.json`, `smoke-acceptance.json`, private seven-library closure and helper together. Do not mix shared objects from other runtime versions. `systemctl status ac922-inference ac922-inference-safety` and `/run/ac922-inference/safety.json` show model and safety state. A fault under `/var/lib/ac922-inference/fault.json` requires inspection before manual acknowledgment.

The existing gateway key remains the authentication source; do not paste it into logs or this document. The public API version must be verified with real authenticated requests after private acceptance.
