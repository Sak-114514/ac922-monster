# V328 R3 bridge capacity gate

`bridge-fixture-cap32.cu` is the R2 exact-output and lifetime fixture with the isolated cache-capacity case expanded from 8 to 32 entries. It leaves the primary 20-frame A/B/A run, pending invalidation/drain, external H2D/D2H ordering, slot fences, host-node fallback, stream-change decline, context teardown/recreation, and exact integer oracle intact.

The capacity case uses a fresh helper owner lifetime, submits 33 distinct graph keys, requires keys 0–31 to select the bridge, and requires key 32 to return `AC922_TAIL_CACHE_CAPACITY` while the ordinary unchanged graph fallback still passes its oracle. Its success record is `capacity32_gate=PASS limit=32 entries=33 ... thirty_third_decline=1`, and the overall record includes `cap32_33rd_decline=PASS` for the root bounded runner.

The hostlink script targets only the root-provided R3 helper at `build-r3-cap32/libac922-tail-bridge.so` and checks its pinned SHA256 (`c63bcd2d2c15ba983b1d63b68b6427e181f5e53da21c02e425b3825bf0f73730`). It builds `build-fixture-cap32/bridge-fixture`, records the dependency resolution and hashes, and never executes the binary. GPU execution remains controlled by the root bounded wrapper `run-cap32-bounded.py`.

~~~sh
fixtures/device-tail-bridge-v328-20261018/build-cap32-host.sh
~~~

The resulting hostlink receipt proves compilation and DSO resolution only. It does not claim the capacity gate passed until the root wrapper executes the 33-key GPU case.
