# V432 tail instantiate CPU audit

V432 extends the frozen V428 helper with an opt-in per-call host CPU snapshot
around the existing child `cudaGraphInstantiateWithFlags` call. With
`AC922_TAIL_STAGE_TIMING=1` set before process startup, it reads
`CLOCK_THREAD_CPUTIME_ID` and `getrusage(RUSAGE_THREAD)` before and after that
call. It reports thread CPU, user/system time, minor/major faults, and
voluntary/involuntary context switches. `diagnostic_valid=0` means a POSIX
sample failed or a delta was nonmonotonic; it does not change the CUDA result.
The diagnostic preserves `errno` around both POSIX reads. These CPU counters
include time spent actively spinning in the driver and are not proof of useful
CPU computation.

Captured graph node total and type counts come only from the node enumeration
and `cudaGraphNodeGetType` loop already used by R5. No CUDA API was added or
reordered. The fixed-size V428 aggregate cadence is retained and new CPU and
node fields carry last, sum, max, validity, and schema-completeness values.
`k_max_entries` remains 32, the ABI header is unchanged, and both ABI launch
exports remain present.

The V429 formal UB1280 run reported 12.519 s target time, with 7.699 s total
child-instantiation wall time across 60 calls, no done-event sync time, 2.744 ms
upload-drain total, and 2.958 ms controller-instantiation total. V432 isolates
thread CPU from that wall interval; CPU counters can include driver spinning
and do not identify useful CPU computation by themselves.

The candidate is copied from V428 source SHA
`751fd87597e790544cc68bd5ec5f722493afd50dd54ac2549e55ab7e7a0078b7`. The
parent V428 helper binary SHA is recorded in `source-provenance-v432.json`;
the underlying R5 binary SHA is also pinned there. This is staging only: the
helper was not built or run, and no engine or GPU files were changed.
