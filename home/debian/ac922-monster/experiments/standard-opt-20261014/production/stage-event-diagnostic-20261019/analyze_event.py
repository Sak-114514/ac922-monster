"""Read-only observer and stream-envelope accounting; never infer SM busy time."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re

FIELDS = re.compile(r"([A-Za-z_]+)=([^\s]+)")
SPAN = re.compile(r"meta-gpu-span device=\d+ trace=0x[0-9a-f]+[^\n]*")
OBSERVER = re.compile(r"captured-argument-observer requested=1[^\n]*")


def merged_us(intervals):
    total = 0.0
    end = float('-inf')
    for first, last in sorted(intervals):
        if last <= first:
            continue
        total += max(0.0, last - max(first, end))
        end = max(end, last)
    return total


def clipped(first, last, low, high):
    a, b = max(first, low), min(last, high)
    return (a, b) if a < b else None


def analyze(log, qualification):
    spans, observer, drops = [], [], {}
    for match in SPAN.finditer(log):
        fields = dict(FIELDS.findall(match.group()))
        if 'dropped' in fields:
            # A freed trace address can be reused by a later context. Preserve
            # any observed drop rather than letting a later zero hide it.
            drops[fields['trace']] = max(drops.get(fields['trace'], 0), int(fields['dropped']))
        elif {'approx_begin_us', 'duration_us', 'initial_calibration_bracket_us'} <= fields.keys():
            spans.append(fields)
    for match in OBSERVER.finditer(log):
        observer.append(dict(FIELDS.findall(match.group())))
    result = {
        'status': 'NO_VALID_WINDOW',
        'scope': 'full_live_wave_stream_envelopes_not_kernel_busy',
        'cross_device_time_is_approximate': True,
        'observer_hash_is_potential_only': True,
        'replay_authorized': False,
        'selected_six_backend_init_logs': len(re.findall(
            r'scheduler layer GPU spans requested=1 selected=1[^\n]*backends=6', log)),
        'all_span_records': len(spans),
        'all_observer_records': len(observer),
        'observer_declines': dict(Counter(row.get('reason', 'missing') for row in observer if row.get('selected') != '1')),
        'all_observer_selected': sum(row.get('selected') == '1' for row in observer),
        'all_observer_potential_duplicates': sum(row.get('seen_hash_before') == '1' for row in observer),
        'windows': {},
    }
    for wave in qualification.get('waves', []):
        name = wave.get('id')
        if name not in ('diag-event-prefill32', 'diag-event-decode4'):
            continue
        low, high = wave.get('host_start_monotonic_us'), wave.get('host_end_monotonic_us')
        if low is None or high is None or high <= low:
            result['windows'][name] = {'status': 'MISSING_VALID_HOST_WINDOW'}
            continue
        stage, traced = {}, []
        calibrations = []
        for row in spans:
            begin = float(row['approx_begin_us'])
            duration = float(row['duration_us'])
            interval = clipped(begin, begin + duration, low, high)
            if interval is not None:
                stage.setdefault(int(row['device']), []).append(interval)
                traced.append(row['trace'])
                calibrations.append(int(row['initial_calibration_bracket_us']))
        selected = []
        prepare_intervals, prepare_by_device = [], {}
        for row in observer:
            if not {'prepare_begin_us', 'prepare_end_us'} <= row.keys():
                continue
            interval = clipped(int(row['prepare_begin_us']), int(row['prepare_end_us']), low, high)
            if interval is not None:
                selected.append(row)
                prepare_intervals.append(interval)
                prepare_by_device.setdefault(int(row['device']), []).append(interval)
        unique_traces = set(traced)
        missing_drop = sorted(unique_traces - drops.keys())
        nonzero_drop = {key: drops[key] for key in unique_traces & drops.keys() if drops[key]}
        devices = sorted(stage)
        passed = (devices == list(range(6)) and not missing_drop and not nonzero_drop
                  and result['selected_six_backend_init_logs'] > 0)
        result['windows'][name] = {
            'status': 'STREAM_ENVELOPE_COVERAGE_PASS' if passed else 'INCOMPLETE_OR_DROPPED',
            'wave_wall_ms': (high - low) / 1000,
            'devices': devices,
            'span_count': sum(map(len, stage.values())),
            'span_union_ms_by_device': {str(d): merged_us(rows) / 1000 for d, rows in stage.items()},
            'span_sum_ms_by_device': {str(d): sum(b - a for a, b in rows) / 1000 for d, rows in stage.items()},
            'any_device_span_union_ms': merged_us([row for rows in stage.values() for row in rows]) / 1000,
            'max_initial_calibration_bracket_us': max(calibrations, default=None),
            'missing_trace_teardown_drop_records': missing_drop,
            'nonzero_drop': nonzero_drop,
            'observer_records': len(selected),
            'observer_selected': sum(row.get('selected') == '1' for row in selected),
            'observer_potential_duplicates': sum(row.get('seen_hash_before') == '1' for row in selected),
            'observer_declines': dict(Counter(row.get('reason', 'missing') for row in selected if row.get('selected') != '1')),
            'observer_wall_sum_ms': sum(int(row.get('observe_us', 0)) for row in selected) / 1000,
            'original_prepare_wall_sum_ms': sum(b - a for a, b in prepare_intervals) / 1000,
            'original_prepare_host_union_ms': merged_us(prepare_intervals) / 1000,
            'original_prepare_union_ms_by_device': {str(d): merged_us(rows) / 1000 for d, rows in prepare_by_device.items()},
            'timings': wave.get('timings', wave.get('timings_raw')),
        }
    if len(result['windows']) == 2 and all(
            row['status'] == 'STREAM_ENVELOPE_COVERAGE_PASS' for row in result['windows'].values()):
        result['status'] = 'STREAM_ENVELOPE_COVERAGE_PASS'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', required=True, type=Path)
    parser.add_argument('--qualification', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    report = analyze(args.log.read_text(errors='replace'), json.loads(args.qualification.read_text()))
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
