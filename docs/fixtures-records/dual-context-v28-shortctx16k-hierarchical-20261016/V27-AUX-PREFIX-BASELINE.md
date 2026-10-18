# V27 auxiliary probe-prefix baseline (read-only)

Source is the frozen V27 diagnostic `results.json`, not a rerun. The two prefix failures remain active in V28's original comparator and must not be ignored or removed.

Baseline run: `root-p4-v27-shortctx16k-hierarchical-ep6-d3-20261016t022431z`

- V27 diagnostic `results.json` SHA-256: `16cb38688dc2e4581c0f5990dcf432949f431dad842e44ac5560c082cdf6f994`
- V27 clean `results.json` SHA-256: `beee7b03194cea10d8f56225bf2f8bae83c014f461324b8b9603bb746b95ae36`

Both `single8k-pp-probe-32` and `single8k-pp-warm-32` compare their 32-token `real8k0` output against serial flow `real8k-0`. The first differing token is at zero-based index **2**; only IDs 198 and 248069 match first.

| Flow | First 20 token IDs | Decoded output excerpt |
|---|---|---|
| V27 serial `real8k-0` | `[198, 248069, 198, 248069, 271, 550, 25816, 17775, 271, 13962, 220, 16, 13, 387, 17, 47, 1942, 48636, 27718, 528]` | `\n</think>\n</think>\n\n## Technical Analysis\n\n### 1. P2P AllReduce (`allreduce-p2p.cu`) — In-Graph Collective via Peer-to-Peer Writes\n\n**Design.** The kernel implements a *push-based* allreduce...` |
| V27 cold single-8K probe | `[198, 248069, 271, 248068, 198, 1596, 1144, 4087, 1156, 579, 1622, 25, 328, 6065, 25, 14569, 494, 279, 12182, 3558]` | `\n</think>\n\n<think>\nWe need answer user's request: "Task: Continue from the repository material above with a concise, implementation-focused technical analysis. Ground claims...` |
| V27 warm single-8K probe | Same first 20 IDs and same decoded excerpt as the cold probe. | Same as cold probe. |

The full token arrays and output strings remain in the V27 result artifacts. V28 adds a separate case-by-case V67-vs-V65 token-array comparison while preserving the original prefix checker unchanged.
