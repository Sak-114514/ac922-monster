#!/usr/bin/env python3
"""Validate exact topology-bucket CUDA fixture output and helper summaries."""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path


def fail(message: str) -> None:
    raise SystemExit(message)


def main() -> int:
    if len(sys.argv) != 4:
        fail("usage: check_topology_buckets.py devices stdout.log stderr.log")
    devices = int(sys.argv[1])
    if devices not in (1, 6):
        fail("devices must be 1 or 6")

    stdout_path, stderr_path = map(Path, sys.argv[2:])
    if not stdout_path.is_file() or not stderr_path.is_file():
        fail("missing fixture stdout/stderr log")
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
    combined = stdout + "\n" + stderr

    source_path = Path(__file__).with_name("test_topology_buckets.cu")
    if not source_path.is_file():
        fail("missing topology fixture source")
    source = source_path.read_text(encoding="utf-8", errors="replace")
    if '#include "helper/bridge-api.h"' not in source:
        fail("fixture does not use the local V2 bridge API")
    if source.count("cudaStreamSynchronize(") != 1 or "cudaDeviceSynchronize(" in source:
        fail("fixture must have one final stream sync and no device sync")

    marker_re = re.compile(
        r"^V441_TOPOLOGY_BUCKETS_PASS device=(\d+) owner=(\S+) stream=(\S+) "
        r"keys=8 first_sequence=ABBBBCBB same_key_rebind=ABA ready=1 "
        r"ready_device=(\d+) current_unchanged=1 cross_device=(\d+) "
        r"outputs_exact=1 no_inter_launch_sync=1$",
        re.M,
    )
    markers = [
        {
            "device": int(m[0]), "owner": m[1], "stream": m[2],
            "ready_device": int(m[3]), "cross_device": int(m[4]),
        }
        for m in marker_re.findall(stdout)
    ]
    if len(markers) != devices:
        fail(f"expected {devices} per-device PASS markers, found {len(markers)}")
    if sorted(row["device"] for row in markers) != list(range(devices)):
        fail("device PASS markers do not cover the requested device set")
    for row in markers:
        if devices == 6:
            if row["ready_device"] == row["device"] or row["cross_device"] != 1:
                fail("six-device run did not exercise metadata-only readiness cross-device")
        elif row["ready_device"] != row["device"] or row["cross_device"] != 0:
            fail("one-device run has an invalid readiness device marker")

    completion = re.findall(
        rf"^V441_TOPOLOGY_BUCKETS_COMPLETE devices={devices} slots={3 * devices} "
        r"distinct_buckets_per_device=3 outputs_exact=1 middle_updates_required=1$",
        stdout,
        re.M,
    )
    if len(completion) != 1:
        fail("missing unique topology-bucket completion marker")

    summary_re = re.compile(
        r"^ac922-tail-bridge v441_slot_summary owner=(\S+) generation=(\d+) "
        r"device=(\d+) stream=(\S+) committed_key=(\S+) committed_graph=(\S+) "
        r"commit_serial=(\d+) instantiations=(\d+) updates=(\d+) "
        r"update_failures=(\d+) fallback_launches=(\d+) "
        r"topology_bucket=(\S+) slot_nodes=(\d+)$",
        re.M,
    )
    rows = [
        {
            "owner": m[1], "device": int(m[3]), "stream": m[4],
            "key": m[5], "graph": m[6], "serial": int(m[7]),
            "instantiations": int(m[8]), "updates": int(m[9]),
            "failures": int(m[10]), "fallbacks": int(m[11]),
            "bucket": m[12], "nodes": int(m[13]),
        }
        for m in summary_re.finditer(stderr)
    ]
    if len(rows) != 3 * devices:
        fail(f"expected {3 * devices} topology slot summaries, found {len(rows)}")

    by_device: dict[int, list[dict[str, object]]] = defaultdict(list)
    marker_by_device = {row["device"]: row for row in markers}
    for row in rows:
        by_device[row["device"]].append(row)
        marker = marker_by_device.get(row["device"])
        if marker is None or row["owner"] != marker["owner"] or row["stream"] != marker["stream"]:
            fail("slot summary owner/device/stream does not match fixture marker")
        if row["instantiations"] != 1 or row["serial"] < 1:
            fail("each topology bucket must have one initialized executable slot")
        if row["failures"] != 0 or row["fallbacks"] != 0:
            fail("a topology bucket update failed or fell back to per-key execution")

    for device, device_rows in by_device.items():
        nodes = {int(row["nodes"]) for row in device_rows}
        buckets_by_nodes: dict[int, set[str]] = defaultdict(set)
        for row in device_rows:
            buckets_by_nodes[int(row["nodes"])].add(str(row["bucket"]))
        if nodes != {1, 2, 3}:
            fail(f"device {device} summaries do not cover 1/2/3-node topologies")
        if any(len(values) != 1 for values in buckets_by_nodes.values()) or \
                len({str(row["bucket"]) for row in device_rows}) != 3:
            fail(f"device {device} did not retain exactly one bucket per topology")
        middle = [row for row in device_rows if int(row["nodes"]) == 2]
        if sum(int(row["updates"]) for row in middle) <= 0:
            fail(f"device {device} middle topology never used CUDA graph update")
        initial = [row for row in device_rows if int(row["nodes"]) == 1]
        if sum(int(row["updates"]) for row in initial) <= 0:
            fail(f"device {device} A/B/A same-key sequence did not update the A bucket")

    if "v441_update_failure" in combined or "cudaErrorIllegalAddress" in combined or "Xid" in combined:
        fail("fixture logs contain an update failure or CUDA fault marker")

    update_count = sum(int(row["updates"]) for row in rows)
    print(
        f"V441_TOPOLOGY_BUCKETS_LOG_PASS devices={devices} slots={len(rows)} "
        f"buckets_per_device=3 middle_updates={sum(int(r['updates']) for r in rows if int(r['nodes']) == 2)} "
        f"total_updates={update_count} exact_outputs=1 cross_device_ready={devices == 6}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
