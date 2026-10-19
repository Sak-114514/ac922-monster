import json
from pathlib import Path
import tempfile
import unittest

from summarize_graphlevel import summarize, union_ns


class GraphSpanTests(unittest.TestCase):
    def sample(self, missing_device_child=False, dropped=0):
        errors = {name: 0 for name in ['record_overflow', 'buffer_starvation_selected_window',
            'parse_errors_selected_window', 'bad_buffers_selected_window',
            'untimed_activity_records_during_window', 'unknown_activity_kinds']}
        rows = [dict(record='trace_meta', trace_valid=True, device_graph_enable_ok=True,
                     window_start_ns=100, window_stop_ns=200, dropped_selected_window_phase=dropped, **errors)]
        for i in range(6):
            rows += [dict(record='event', type='device', device_id=i, uuid=str(i), name='fixture'),
                     dict(record='event', type='graph_trace', device_id=i, graph_id=i+10,
                          start_ns=110+10*i, end_ns=130+10*i,
                          correlation_id=0 if missing_device_child and i == 5 else 0xffffffff)]
        for start, end in [(105, 125), (115, 135)]:
            rows.append(dict(record='event', type='runtime_api', thread_id=1, api_name='fixture',
                             start_ns=start, end_ns=end))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'trace.jsonl'
            path.write_text('\n'.join(json.dumps(x) for x in rows))
            return summarize(path)

    def test_interval_union(self):
        self.assertEqual(union_ns([(5, 9), (1, 6), (9, 12)]), 11)

    def test_six_device_overlap_and_host_union(self):
        result = self.sample()
        self.assertEqual(result['status'], 'PASS')
        self.assertAlmostEqual(result['mean_devices_with_visible_device_graph_spans'], 1.2)
        self.assertEqual(result['device_graph_concurrency_seconds'], {'0': 30e-9, '1': 20e-9, '2': 50e-9})
        self.assertAlmostEqual(result['top_host_api'][0]['duration_sum_seconds'], 40e-9)
        self.assertAlmostEqual(result['host_runtime_api_union_seconds_by_thread']['1'], 30e-9)

    def test_missing_child_or_dropped_records_rejected(self):
        self.assertEqual(self.sample(missing_device_child=True)['status'], 'FAIL_COVERAGE_OR_TRACE')
        self.assertEqual(self.sample(dropped=1)['status'], 'FAIL_COVERAGE_OR_TRACE')


if __name__ == '__main__':
    unittest.main()
