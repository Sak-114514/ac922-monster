Nsight Systems 2023.4.4.54 completed the V77 probe process successfully. The
probe has no `cudaProfilerStart()` / `cudaProfilerStop()` calls, and the output
directory contains no `.nsys-rep`; the configured `cudaProfilerApi` capture
range therefore produced no active timeline. The application itself completed
all exact/canary checks. This run tests the Nsight-wrapped/injected process path
with capture inactive; it does not test active CUDA collection.

The V77 direct baseline is preserved as `v77-uninstrumented-baseline.log`.
`preflight.log` and `postflight-xid.log` show no new Xid during the run. The
existing GPU0 Xid 31 remains unchanged. No GPU reset was performed.
