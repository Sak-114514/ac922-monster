# V225: bounded raw MTP body clone fixture

The payload contract remains 32 `blk.48.*` tensors totaling 2,775,621,632 bytes. The three Q8_0 expert tensors validate each row-derived expert stride multiplied by the header's 512 experts, and the resulting full-tensor byte counts must match GGUF. All tensor types, dimensions, strides, and bytes remain unchanged.

The fixture reads GGUF metadata with `no_alloc=true`, loads only selected layer-48 payload ranges to a temporary GPU5 tensor bank, and clones them to GPUs0 and1. It does not load the trunk or PLE, create a model context or graph, or run inference. Four 8 MiB verification buffers cap host staging at 32 MiB. Each owner allocation is capped at 3 GiB with a 512 MiB free-memory floor. The runner uses temperature/free-memory checks every two seconds and a 58°C hard cutoff.

## SHA pipe fix and CPU gate

R3 and r4 each timed out after 11 completed `VERIFY` rows; neither passed the 32-tensor/hash gate. The cause was inherited pipe descriptors: each later `sha256sum` fork retained prior stream writers, so the first digest waited forever for EOF. The fixture now sets `FD_CLOEXEC` on all four pipe descriptors before each fork. The extracted `Sha256` CPU regression constructs four simultaneous streams, feeds independent known byte sequences, checks each full SHA-256, and has an eight-second timeout. The passing local artifacts are preserved under `results/v225-sha-streams-r5-local-2/`; the first harness attempt (unequal stream lengths) is also retained separately.

## r5 host link

R5 uses the same frozen V251 runtime with the FD fix, in a unique output directory and log. It does not compile CUDA kernels or load a model. After the local static gate and four-stream regression pass, run:

```sh
MTP_BODY_CLONE_BUILD_AUTHORIZED=1 ./build-v225-r5-host.sh v251
```

Expected outputs: `build-v225-r5-v251/` and `results/v225-host-link-v251-r5-20261017.log`.

## r5 bounded raw-copy gate

The single r5 raw-copy run is authorized only after root hands off the GPU window. It reuses the r5 binary, uses six fixed GPU UUIDs in visible order, and enforces a 180-second timeout, 58°C limit, and 512 MiB free-memory guard. A pass requires all 32 unique tensor `VERIFY` rows, exact byte equality across source and both clones, matching aggregate SHA-256 values, and a staging total no greater than 32 MiB. The r3/r4 timeout results remain partial evidence only.

```sh
MTP_BODY_CLONE_GPU_AUTHORIZED=1 \
CUDA_VISIBLE_DEVICES=<six-fixed-GPU-UUIDs-in-visible-order-0-through-5> \
  ./run-v225-r5-once.sh
```

Static metadata/source checks do not substitute for the full raw-copy gate.

## r5 result

The r5 CPU SHA regression and V251 host link passed. The single guarded raw-copy run verified all 32 tensors byte-for-byte across the GGUF source, GPU5, GPU0, and GPU1. The aggregate SHA-256 is `4e763743cdf6f7db5829f43e29e237df51b3f282c9ac855d4f5d65ec784a56ed`; host staging was 33,554,432 bytes. The guard ended at 41°C maximum and restored all GPUs to about 16.1 GiB free. R3/r4 timeouts remain recorded as partial failures. This validates raw weight copying only; it does not validate model integration, inference quality, or speed.
