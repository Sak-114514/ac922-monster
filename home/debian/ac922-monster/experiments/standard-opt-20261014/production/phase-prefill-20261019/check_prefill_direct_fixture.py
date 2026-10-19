#!/usr/bin/env python3
"""Validate the scoped prefill-direct CUDA fixture's combined stdout/stderr log."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import NoReturn


PHASE_BEGIN = re.compile(
    r"^PREFILL_DIRECT_PHASE_BEGIN device=(\d+) phase=([a-z_]+) calls=(\d+)$"
)
PHASE_PASS = re.compile(
    r"^PREFILL_DIRECT_PHASE_PASS device=(\d+) phase=([a-z_]+) calls=(\d+) exact_oracle=1$"
)
PATH_EVENT = re.compile(
    r"optimization path device=(\d+) path=([a-z0-9_]+) "
    r"scope=host_dispatch_or_capture requested=(\d+) selected=(\d+) reason=([^ ]+)"
)
GRAPH_MODE = re.compile(
    r"CUDA graph path active ctx=(\S+) device=(\d+) uid=(\d+) nodes=(\d+) mode=(capture|replay)"
)
PHASES = {
    "before_capture_replay",
    "scoped_direct",
    "after_capture_replay",
}


def fail(message: str) -> NoReturn:
    print(f"PREFILL_DIRECT_CHECK_FAIL {message}", file=sys.stderr)
    raise SystemExit(1)


def phase_segments(lines: list[str]) -> tuple[dict[tuple[int, str], list[str]], set[tuple[int, str]]]:
    segments: dict[tuple[int, str], list[str]] = {}
    passed: set[tuple[int, str]] = set()
    current: tuple[int, str] | None = None

    for raw in lines:
        line = raw.rstrip("\n")
        begin = PHASE_BEGIN.match(line)
        if begin:
            key = (int(begin.group(1)), begin.group(2))
            if key in segments or key[1] not in PHASES:
                fail(f"duplicate or unknown phase marker: {line}")
            current = key
            segments[key] = []
            continue

        phase_pass = PHASE_PASS.match(line)
        if phase_pass:
            key = (int(phase_pass.group(1)), phase_pass.group(2))
            if current != key or key not in segments:
                fail(f"phase pass without matching begin: {line}")
            if phase_pass.group(3) != "2" and key[1] != "scoped_direct":
                fail(f"normal graph phase did not submit two calls: {line}")
            if phase_pass.group(3) != "1" and key[1] == "scoped_direct":
                fail(f"direct phase did not submit one call: {line}")
            passed.add(key)
            current = None
            continue

        if current is not None:
            segments[current].append(line)

    if current is not None:
        fail(f"unterminated phase {current}")
    return segments, passed


def backend_events(lines: list[str]) -> list[tuple[int, str, int, int, str]]:
    events = []
    for line in lines:
        match = PATH_EVENT.search(line)
        if match:
            events.append((int(match.group(1)), match.group(2),
                           int(match.group(3)), int(match.group(4)), match.group(5)))
    return events


def require_event(events: list[tuple[int, str, int, int, str]], device: int,
        path: str, selected: int, reason: str | None = None) -> int:
    for index, event in enumerate(events):
        ev_device, ev_path, requested, ev_selected, ev_reason = event
        if (ev_device == device and ev_path == path and requested == 1 and
                ev_selected == selected and (reason is None or ev_reason == reason)):
            return index
    suffix = f" reason={reason}" if reason else ""
    fail(f"missing device={device} path={path} requested=1 selected={selected}{suffix}")


def validate(log_path: Path, expected_devices: int) -> dict[str, object]:
    try:
        text = log_path.read_text(errors="replace")
    except OSError as exc:
        fail(f"cannot read {log_path}: {exc}")
    lines = text.splitlines()

    if "PREFILL_DIRECT_FIXTURE_FAIL" in text:
        fail("fixture reported failure")
    if "PREFILL_DIRECT_HOOK_PRESENT direct=1 capture=1 fallback=0" not in text:
        fail("required direct/capture CUDA registry hooks were not both resolved")
    if "PREFILL_DIRECT_TLS_PASS direct_default=0 nested_restore=1 child_default=0 parent_unaffected=1 capture_default=-1" not in text:
        fail("thread-local default, nested restore, or host-thread isolation check failed")

    final = re.search(
        r"^PREFILL_DIRECT_FIXTURE_PASS devices=(\d+) exact_oracle=1 hook_present=1 "
        r"tls_pass=1 before_capture_replay=1 scoped_direct=1 after_replay=1$",
        text, re.MULTILINE,
    )
    if not final:
        fail("fixture completion marker is missing or did not report all required gates")
    if int(final.group(1)) != expected_devices:
        fail(f"fixture reports {final.group(1)} devices, expected {expected_devices}")

    segments, passed = phase_segments(lines)
    expected = {
        (device, phase)
        for device in range(expected_devices)
        for phase in PHASES
    }
    if set(segments) != expected:
        fail(f"phase coverage mismatch: got={sorted(segments)} expected={sorted(expected)}")
    if passed != expected:
        fail(f"not every phase produced an exact-oracle pass: {sorted(passed)}")

    direct_devices: set[int] = set()
    phase_counts: dict[str, dict[int, int]] = {
        phase: {} for phase in PHASES
    }
    for device in range(expected_devices):
        before = backend_events(segments[(device, "before_capture_replay")])
        direct = backend_events(segments[(device, "scoped_direct")])
        after = backend_events(segments[(device, "after_capture_replay")])

        capture_at = require_event(before, device, "cuda_graph_force_capture", 1,
                                   "compatible_changed_graph_capture")
        replay_at = require_event(before, device, "cuda_graph_force_capture", 1,
                                  "compatible_graph_replay")
        if replay_at <= capture_at:
            fail(f"device={device} graph replay did not follow its initial capture")

        if any(event[1] == "cuda_graph_force_capture" for event in direct):
            fail(f"device={device} direct phase entered graph capture/replay policy")
        require_event(direct, device, "prefill_direct", 1)
        direct_devices.add(device)

        # Optimization-path messages are rate limited after the first two
        # calls. Use the actual graph-mode event to prove post-direct replay,
        # including the same backend context/UID/node count as before.
        before_modes = [m.groups() for line in segments[(device, "before_capture_replay")]
                        if (m := GRAPH_MODE.search(line)) and int(m.group(2)) == device]
        after_modes = [m.groups() for line in segments[(device, "after_capture_replay")]
                       if (m := GRAPH_MODE.search(line)) and int(m.group(2)) == device]
        previous = {mode[:4] for mode in before_modes}
        if not any(mode[4] == 'replay' and mode[:4] in previous for mode in after_modes):
            fail(f"device={device} missing same-context/UID graph replay after direct scope")
        if any(mode[4] == 'capture' for mode in after_modes):
            fail(f"device={device} actual graph mode recaptured after direct scope")
        if any(event[1] == "cuda_graph_force_capture" and
                event[4] == "compatible_changed_graph_capture" for event in after):
            fail(f"device={device} graph was recaptured after direct scope; prior entry was not reused")

        for phase, events in (("before_capture_replay", before),
                              ("scoped_direct", direct),
                              ("after_capture_replay", after)):
            phase_counts[phase][device] = sum(
                1 for event in events
                if (event[0] == device and event[2] == 1 and event[3] == 1)
            )

    if direct_devices != set(range(expected_devices)):
        fail(f"direct path admission missing on devices: {sorted(set(range(expected_devices)) - direct_devices)}")

    return {
        "status": "PREFILL_DIRECT_FIXTURE_PASS",
        "devices": expected_devices,
        "phase_oracles": len(passed),
        "per_device_selected_events": phase_counts,
        "capture_replay_before_and_after": True,
        "direct_path_selected_on_all_devices": sorted(direct_devices),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("combined_log", type=Path,
                        help="fixture stdout+stderr captured into one file")
    parser.add_argument("--devices", type=int, choices=(1, 6), default=6,
                        help="expected CUDA device count (default: 6)")
    args = parser.parse_args()
    result = validate(args.combined_log, args.devices)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
