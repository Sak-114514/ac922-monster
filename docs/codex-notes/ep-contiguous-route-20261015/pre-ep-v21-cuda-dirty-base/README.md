# Reconstructed pre-EP CUDA source base

This directory contains the seven CUDA files reconstructed from the V22 source by
reversing only the EP edits recorded in `restore-v21-cuda-dirty-base.py`. It is
intended to recover the dirty V21 source state without checking out or replacing
the earlier V100 optimizations.

The original files were not archived before the first edit, so these files are a
deterministic inverse-patch reconstruction, not a separately captured pre-edit
snapshot. The restore script checks every replacement count and fails closed if
the current source no longer matches the EP diff it knows how to reverse.

For a full rollback, copy the seven matching files from this directory over the
private source tree. `revert-ep-cuda-v22.patch` applies to the V22 files before
the later V23 dispatch-log additions; it may need context adjustment against V23.
Do not restore `ggml-ep-private.h` from this backup; that header was created by
the parent task and must be removed only if no other integration code uses it.
