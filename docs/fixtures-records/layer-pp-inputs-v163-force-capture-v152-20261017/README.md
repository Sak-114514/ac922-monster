# V163 structural force-capture tiny candidate

This is a held, no-build/no-model follow-up to the passing V152/V160 tiny gate. It reuses the exact 25-frame V152 runner binary and switches only runtime environment settings for structural CUDA graph forcing and logging. It requires stage-on/cap-256, preserves structural signature validation, unsets `LD_PRELOAD`, and fails the post-run gate unless the runtime log records an actual `compatible_changed_graph_capture` selection and a CUDA graph capture path. Exact outputs and canaries remain mandatory.

`check-v163-profile.py` is CPU-only. `run-v163-tiny.sh` refuses to launch unless `V152_GPU_WINDOW_AUTHORIZED=1`, and refuses to overwrite a previous log. Keep this profile held until the four-bank GPU task releases the cards and root authorizes the tiny window. No runtime compilation or runner rebuild is part of this candidate.
