# TurboPrefill pinned-cap staging candidate (2026-10-16)

This is a fixture-local preparation patch. It does not modify V47 source, build a runtime, load a model, or use GPUs.

## Policy

The patch adds `GGML_TURBOPREFILL_PINNED_CAP_MIB` for the combined pinned-byte cap of one `ggml_backend_sched` Turbo state:

- Unset or empty: existing 256 MiB limit.
- Exact `256`: retain 256 MiB.
- Exact `512`: explicit opt-in to 512 MiB.
- Other values, malformed strings, overflow, and values above the 512 MiB hard maximum: reject Turbo admission before work.
- Turbo still requires the existing explicit `LLAMA_TURBOPREFILL=1` gate. The cap environment variable does not activate Turbo by itself.

The new private policy header contains the same strict parser and overflow-safe combined budget helper used by the core patch and CPU tests. Every existing `max_pinned_bytes` use in `ggml-backend.cpp` is routed through the chosen cap. Projected ring+FIFO admission, FIFO live/current checks, actual FIFO allocation sizes, actual ring allocation sizes, active-transaction checks, and high-water reporting continue to use one combined cap. Invalid cap configuration fails before Turbo side effects; it never silently falls back to a larger or looser path.

The cap is per scheduler, matching the existing state lifetime. Multiple concurrently active schedulers can each reserve up to their cap; model-level DDR accounting must include all concurrently active schedulers.

## V43 evidence and bound choice

Exact V43 plan and memory lines are saved in `evidence/v43-pinned-plan-ddr.txt` from `/home/debian/ac922-monster/experiments/standard-semantic-20261014/results/20261015T182643Z/server.log`.

For the actual selected UB1024 plan, V43 logged a 109,469,952-byte frame payload, two ring slots, 93,110,280 bytes projected live FIFO, and 144,277,520 bytes current FIFO. That gives a 312,050,184-byte projected combined requirement (297.59 MiB) and a 363,217,424-byte combined current allocation estimate (346.39 MiB). The existing 256 MiB cap rejects both.

For UB2048, the measured plan needs 622,592,520 bytes (593.75 MiB) projected and later reports 724,173,328 bytes (690.95 MiB) using its larger current FIFO allocation. A 512 MiB hard maximum admits the measured UB1024 plan and continues to refuse UB2048. A 768 MiB ceiling would allow this unqualified UB2048 memory shape, so this staging candidate does not use it.

V43 measured 1,521.905 warm prefill tok/s at UB1024 on the ordinary route; Turbo was declined before work. The 512 MiB per-scheduler ceiling is at most 0.50 GiB of host pinned memory. With the V43 hybrid PLE placement of 18.82 GiB in DDR and 8.00 GiB in HBM, the aggregate DDR accounting is 32 − 18.82 − 0.50 = 12.68 GiB gross for the OS, model, and other allocations before other schedulers. At the observed UB1024 current combined allocation estimate of 346.39 MiB, PLE plus those pinned buffers total about 19.16 GiB, leaving about 12.84 GiB gross.

V43 also logged 26.85 GiB CPU DDR “available” as free plus reclaimable clean file pages before PLE residency. That is not post-load node headroom and is not used to justify this cap. The machine-wide free figure that includes HBM is excluded. The 3 GiB per-node reserve and any concurrent scheduler allocations still need to pass the next model run's actual DDR checks; shrink the ring payload if those checks lack room instead of raising the ceiling.

## Patch and tests

The patch is pinned to the V47 `ggml-backend.cpp` hash in `patches/source-sha256.txt` and adds one new private header. It does not change public GGML headers.

Run the pure CPU tests:

```sh
./fixtures/turbo-pin-budget-20261016/run-cpu-tests.sh
```

They cover default/explicit values, hard ceiling, malformed input, numeric overflow, explicit refusal, V43 UB1024 projected/current combined totals, UB2048 refusal, multiplication/addition overflow, zero payload, and zero slots.

Run syntax-only parsing of the patched backend on AC922:

```sh
./fixtures/turbo-pin-budget-20261016/run-syntax-only.sh
```

This invokes the V47 host compiler with `-fsyntax-only`; it does not link a runtime, execute a model, or use GPUs.

No build or model run was performed for this candidate.
