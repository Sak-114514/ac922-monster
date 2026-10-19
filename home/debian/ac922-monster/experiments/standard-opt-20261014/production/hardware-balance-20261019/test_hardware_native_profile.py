#!/usr/bin/env python3
"""CPU regression tests for the combo native gate's boot geometry."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import unittest


sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
GATE_PATH = HERE / "verify_combo_native.py"
SPEC = importlib.util.spec_from_file_location("combo_native_gate", GATE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load combo native gate from {GATE_PATH}")
GATE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GATE
SPEC.loader.exec_module(GATE)


def valid_boot_record() -> dict:
    return {
        "target_n_batch": 4096,
        "draft_n_batch": 4096,
        "target_n_ubatch": 256,
        "configured_target_n_ubatch": 256,
        "draft_n_ubatch": 128,
        "configured_draft_n_ubatch": 128,
        "logical_prefill_batch_capacity": 32768,
        "mtp_depth": 3,
        "bank_count": 4,
        "bank_lanes": 1,
        "mtp_unrolled": False,
        "actual_sequence_count": 1,
        "actual_kv_streams": 1,
        "actual_draft_sequence_count": 1,
        "actual_draft_kv_streams": 1,
        "bank_topology": "4_independent_banks_1_lane_each_per_seq_17152",
        "banks_kv_layout": [
            {
                "bank": bank,
                "target_context_total": 17152,
                "target_context_per_sequence": 17152,
                "target_sequence_count": 1,
                "target_actual_kv_streams": 1,
                "draft_context_total": 17152,
                "draft_context_per_sequence": 17152,
                "draft_sequence_count": 1,
                "draft_actual_kv_streams": 1,
            }
            for bank in range(4)
        ],
    }


class ComboNativeProfileTests(unittest.TestCase):
    def test_4096_actual_batch_and_32768_logical_capacity_pass(self) -> None:
        self.assertEqual(GATE.check_boot_metadata(valid_boot_record()), [])

    def test_wrong_draft_256_is_rejected(self) -> None:
        record = valid_boot_record()
        record['draft_n_ubatch'] = 256
        record['configured_draft_n_ubatch'] = 256
        errors = GATE.check_boot_metadata(record)
        self.assertTrue(any('draft_n_ubatch' in e for e in errors), errors)

    def test_legacy_17152_actual_batch_is_rejected(self) -> None:
        record = valid_boot_record()
        record["target_n_batch"] = 17152
        record["draft_n_batch"] = 17152

        errors = GATE.check_boot_metadata(record)

        self.assertTrue(any("target_n_batch" in error for error in errors), errors)
        self.assertTrue(any("draft_n_batch" in error for error in errors), errors)

    def test_wrong_per_sequence_context_is_rejected(self) -> None:
        record = deepcopy(valid_boot_record())
        record["banks_kv_layout"][2]["target_context_per_sequence"] = 16384

        errors = GATE.check_boot_metadata(record)

        self.assertTrue(any("bank 2 target_context_per_sequence" in error for error in errors), errors)


if __name__ == "__main__":
    unittest.main()
