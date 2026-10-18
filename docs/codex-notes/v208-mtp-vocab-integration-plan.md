# V208 MTP-only vocabulary row parallelism

Default-off LLAMA_MTP_VOCAB_PARALLEL=1. Preserve original Q5_K output matrix, full-K per-row dots and complete F32 logits concatenated on draft GPU5; target graph/model pointers unchanged. V203 actual 437043200-byte head (2560x248320) operator fixture measured 2.77/2.82x D2/D4 and bitwise identical complete logits/argmax. Not a model-level gain.

First implementation context-owned three 104.2MiB copies per MTP context, plus original GPU5 quarter view. Four banks duplicate 1.22GiB total; preserves simple model ownership at the cost of more setup than shared-model copies. No full MTP block cloning and no new quantization. Context registers physical model devices0/1/3 alongside existing5, keeps5 first and scheduler PP off; extend only explicit head-device admission for single-lane chain/dedicated scheduler. Graph params carry immutable per-context weights/backend bindings with reuse equality. Only single-output MTP graphs use new head; adapters/scales/other shapes retain original path.

Backups saved before editing. Proof: source-extracted admission/lifecycle tests, independent review, incremental CPU build/no CUDA compilation; then bounded real full-model four native cases and4x16K512 with actual head+sidecar selection, no stale backend/quality claims. Keep V196 frozen and old flags reversible. No BMC/driver/kernel/power changes.

- Graph order requirement: expand all four head producer nodes before constructing/expanding concat chain. Interleaved nested concat DFS can enqueue GPU5 waits before later peer/head work, erasing parallelism. Add V210 real-scheduler microgate before full model; V203 used independently queued backend graphs.

- Constructor-unwind ownership fix: declare main+sidecar scheduler RAII pointers after graph/weight/backend owners so reverse member destruction releases allocator/events first even if constructor throws before destructor body runs. Normal teardown still explicitly drains and resets sidecar/main.

- V210 actual one-scheduler gate on frozen V196 PASS: PP off/n_copies1/n_splits4, actual owners0/1/3/5, full248320 logits bitdiff0/maxabs0 and argmax41867 exact. Warm median GPU5full769.174us vs scheduler-shard681.713us =1.128x. This supersedes any naive extrapolation of V203 manual-queue2.8x to scheduler integration; fullmodel gain pending. Final distinct-device admission added and CPU ASan/UBSan rerun PASS beforeupload. V208 incremental22CPU TUs/0CUDA compiling, unchanged frozenV196 fallback retained.
