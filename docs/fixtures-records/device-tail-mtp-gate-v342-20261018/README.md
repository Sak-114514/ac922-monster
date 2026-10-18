# V342 independent D3 decode device-tail gate

This fixture is a bounded synthetic GGML backend gate derived from V333. It retains the six physical CUDA devices, two alternating root graphs, four scheduler copies, 20 frames, 120 exact CPU-oracle outputs, owner/UID/canary checks, pinned input/readback fences, and teardown checks. It loads no model or weights and makes no performance claim.

V342 resolves the new `ggml_backend_cuda_set_device_tail_decode_hint(int)` registry procedure. It scopes this independent TLS hint around actual scheduler submissions and restores the prior value on the same submit thread. It does not call the prefill-capture setter. `GGML_CUDA_GRAPHS_FORCE=1` is a separate test setting so the first submission can capture normally; the device-tail hint itself does not change capture policy.

For frame 0 only, the fixture synchronizes after the first capture/update submission and submits the identical bound graph a second time. That stable replay is the one intended to satisfy the V341 guard `device_tail_decode_hint == 1 && !cuda_graph_update_required && !cold && SM70`. The original 20-frame exact-output/lifetime suite remains intact, and the extra replay happens before the same slot’s final readback. The fixture reports `stable_replay_submit=1` only to record that it submitted the second graph; it does not claim the device-tail path was selected.

The root-owned bounded runner must use the frozen V341 runtime, set the helper path, and require the actual CUDA optimization logs to show `path=cuda_graph_device_tail_mtp_decode selected>0` on each of devices 0–5. It should also require 21 decode-hint entry/restore pairs (20 frame submissions plus the one stable replay), the 20-frame `120/120` exact gate, and the existing lifetime/fence gates. A hint log, hostlink, or synthetic output alone does not prove actual device-tail selection.

## Host link

After V341 is frozen, build using the exact runtime report and build ID from that freeze. This compiles the local fixture only; it does not initialize CUDA or execute the binary.

~~~sh
fixtures/device-tail-mtp-gate-v342-20261018/build-host.sh \
  /home/debian/ac922-monster/experiments/standard-opt-20261014/src \
  /home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v341 \
  /home/debian/ac922-monster/experiments/standard-opt-20261014/notes/V341-build.json \
  99554d5605be99c587e9d11f83b7a0511131858c904c65e84212fc75ad9bd700 \
  20261017T202844.343010Z v342-v341
~~~

The hostlink passed against frozen V341 (`notes/V341-build.json` SHA256 `99554d5605be99c587e9d11f83b7a0511131858c904c65e84212fc75ad9bd700`). The binary is `build-v342-v341/ggml-mtp-tail-gate`, SHA256 `97875d6108e56b2c41ea76ae8d61c3fd4353b0a2a0db4bd440815a558c1dbb0d`; as-built fixture source SHA256 is `474c897404f98fcf850d43c1ebc8176c361fb7ad01305125454a700cf30b6db2`. Its 11-file compiler dependency closure SHA256 is `4a99a677357c15c936726f367499060d6a2b9786f73829e03998e711782456ee`. `ldd -r` resolved the pinned V341 GGML libraries. The receipt, source/runtime hashes, and `ldd -r` output are stored beside the binary. The hostlink did not load a model, initialize CUDA, or execute the binary.

## Verification boundary

Only root’s bounded wrapper may execute the binary on GPUs. The meaningful result is the actual per-device `cuda_graph_device_tail_mtp_decode` selection count together with the fixture’s unchanged exact/lifetime gates. This synthetic fixture does not prove that a full Qwen D3 model graph is eligible or faster.
