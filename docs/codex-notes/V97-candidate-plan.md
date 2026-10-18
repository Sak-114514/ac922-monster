# V97 small-relay candidate

Current V43 four-active padded16 target actually replays six CUDA graphs. Its collective kernel sum is 37.90% of the device-summed kernel intervals, not wall time or pure wire time. All reductions in that target slice use one-shot. The existing Relay6 hard floor of 204800 bytes excludes the common 163840-byte 16x2560 F32 panel.

Add an opt-in lower Relay6 minimum, default 204800 unchanged. Clamp requested minimum to at most the historical floor so V96 segmented-arena elision remains conservative; do not change kernel arithmetic, rank order, flag phases, block grid, epoch, or staging layouts. Test 65536 against default on small-payload exact/ABA/canary and graph timing fixtures before any full model. Log the selected minimum.

Also make reserve ceil divisions overflow safe and reject impossible arena byte products before allocation. Existing realistic profiles are unchanged. No BMC/driver/kernel/reboot or quantization change. BF16 wide-panel template reuse remains a separate local gate; combine only after operator results.
