# Multi-prefill512 verdict, 2026-10-19

**PERFORMANCE_AND_HEADROOM_REJECTED. No production promotion.**

OneCPUworkerTU (63.066s) and privateaffinitypolicy only; all7numericalDSOs/R5/weights/originalPLE18/body5/fixedD3 unchanged. Four8192-token native strictcanaries passed, and all4requestedUB512profiles/copy8planrebind were confirmed.

|Actual uncached wave|Prefill tok/s|All-emissions decode tok/s|
|---|---:|---:|
|4×8192 native canaries (shortanswers)|1970.853|18.062 (not a longdecodebenchmark)|
|4×16384,512outputseach, exactV306corpusslices|2729.016|165.456|

Aligned2048tokens: everyflow differed fromV306; firstdivergences were49/88/303/38. Outputs remain readable corpus continuations, but this is not broadsemantic proof. The separate four8192native canaries passed. Acceptance1362/2038=66.83%. Performance evidence retained separately fromtokenagreement.

During subsequentprefixfill, GPU1 free memory reached165.375MiB, below original256MiBfloor. Originalguard stoppedonlyinference. PeakGPU57°C; no thermal fault, system shutdown, GPUresetorhardwarechanges. Qualification stopped, remainingprefix/longstory/largepoolgate notclaimed. Originalfault was archived, all6GPUsverifiedclean>12GiBfree/temp<60, finitecandidateoverride/timerremoved; only thisidentifieddiagnosticlatchcleared, originalguardunchanged.

OriginalL5 cold-restoredPID583206; actual7DSOs/R5/nextstart/internalAPI/auth/chat/SSE/both Mac proxyoffTLS origins passed, rootREADYrefreshed. 209107-token L5longinput receipt preserved. Further boundedcandidate bulk256 retainsUB256/copy4 and variesonlylogicalbatch; no reuse of this512profile.
