"""Streaming analysis of CUDA12.4 graph spans, never labelled kernel/SM busy."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys


def merged(intervals):
    result = []
    for start, end in sorted(intervals):
        if not result or start > result[-1][1]:
            result.append([start, end])
        else:
            result[-1][1] = max(result[-1][1], end)
    return result


def union_ns(intervals):
    return sum(end - start for start, end in merged(intervals))


def summarize(path):
    meta = None
    graphs, children, api_thread, api_name = (defaultdict(list) for _ in range(4))
    counts, api_counts = Counter(), Counter()
    devices, graph_ids = {}, defaultdict(set)
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['record'] == 'trace_meta':
                assert meta is None
                meta = row
                continue
            kind = row['type']
            counts[kind] += 1
            if kind == 'device':
                devices[str(row['device_id'])] = {'uuid': row['uuid'], 'name': row['name']}
            start, end = row.get('start_ns', 0), row.get('end_ns', 0)
            if meta:
                start = max(start, meta['window_start_ns'])
                end = min(end, meta['window_stop_ns'])
            if end <= start or start <= 0:
                continue
            if kind == 'graph_trace':
                dev = str(row['device_id'])
                graphs[dev].append((start, end))
                graph_ids[dev].add(row['graph_id'])
                if row['correlation_id'] == 0xffffffff:
                    children[dev].append((start, end))
            elif kind == 'runtime_api':
                api_thread[str(row['thread_id'])].append((start, end))
                name = row.get('api_name', 'unknown')
                api_name[name].append((start, end))
                api_counts[name] += 1
    assert meta is not None
    wall = meta['window_stop_ns'] - meta['window_start_ns']
    assert wall > 0
    child_union = {dev: union_ns(values) for dev, values in sorted(children.items())}
    edges = [(meta['window_start_ns'], 0), (meta['window_stop_ns'], 0)]
    for spans in children.values():
        for start, end in merged(spans):
            edges.extend([(start, 1), (end, -1)])
    histogram = Counter()
    active = 0
    previous = meta['window_start_ns']
    for at, delta in sorted(edges):
        histogram[active] += at - previous
        active += delta
        previous = at
    assert active == 0 and sum(histogram.values()) == wall
    errors = {name: meta.get(name) for name in ['dropped_selected_window_phase', 'record_overflow',
        'buffer_starvation_selected_window', 'parse_errors_selected_window', 'bad_buffers_selected_window',
        'untimed_activity_records_during_window', 'unknown_activity_kinds']}
    coverage = set(graphs) == set(children) == set(devices) == {str(i) for i in range(6)}
    passed = meta['trace_valid'] and meta.get('device_graph_enable_ok') and coverage and all(v == 0 for v in errors.values())
    return {'status': 'PASS' if passed else 'FAIL_COVERAGE_OR_TRACE', 'metadata': meta,
            'wall_seconds': wall / 1e9, 'events': dict(counts), 'devices': devices,
            'device_graph_span_union_seconds_by_gpu': {dev: value / 1e9 for dev, value in child_union.items()},
            'all_graph_span_union_seconds_by_gpu': {dev: union_ns(values) / 1e9 for dev, values in sorted(graphs.items())},
            'device_graph_span_counts_by_gpu': {dev: len(values) for dev, values in sorted(children.items())},
            'graph_exec_ids_by_gpu': {dev: sorted(values) for dev, values in sorted(graph_ids.items())},
            'mean_devices_with_visible_device_graph_spans': sum(child_union.values()) / wall,
            'device_graph_concurrency_seconds': {str(k): v / 1e9 for k, v in sorted(histogram.items())},
            'host_runtime_api_union_seconds_by_thread': {thread: union_ns(values) / 1e9 for thread, values in api_thread.items()},
            'top_host_api': [{'name': name, 'calls': api_counts[name],
                'duration_sum_seconds': sum(e - s for s, e in values) / 1e9,
                'interval_union_seconds': union_ns(values) / 1e9}
                for name, values in sorted(api_name.items(), key=lambda p: -sum(e - s for s, e in p[1]))[:15]],
            'coverage_gate': {'six_device_graph_sentinel_spans': coverage, 'errors': errors},
            'interpretation': ['Graph execution/dependency spans are not SM busy or per-kernel time.',
                'Device correlation sentinel validated by the tiny child-exec-ID fixture for this local12.4 stack.',
                'CPU API and GPU graph intervals overlap; their times must not be added as wall components.',
                'GRAPH_TRACE hides graph-node copies and kernels; no MoE/QSA/communication percentage is inferred.',
                'All performance here is instrumented diagnosis, not a clean throughput benchmark.']}


if __name__ == '__main__':
    report = summarize(Path(sys.argv[1]))
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['status'] == 'PASS' else 1)
