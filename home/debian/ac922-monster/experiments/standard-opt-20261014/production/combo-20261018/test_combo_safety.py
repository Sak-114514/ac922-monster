"""Hardware-free regression tests for the AC922 safety monitor stop gates."""
from __future__ import annotations

import sys
import unittest
import importlib
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BASE / "notes/inband-guard-v222-20261017"))

import combo_safety_monitor as safety_monitor  # noqa: E402


class StopAfterPublish(BaseException):
    """End a mocked monitor loop after publishing a known number of samples."""


class FakeClock:
    def __init__(self) -> None:
        self.monotonic_seconds = 0.0
        self.wall_seconds = 1_000.0

    def monotonic(self) -> float:
        return self.monotonic_seconds

    def wall(self) -> float:
        return self.wall_seconds

    def sleep(self, seconds: float) -> None:
        self.monotonic_seconds += seconds


def sample(*, fan0: float = 3_000.0, hot_label: str | None = None) -> dict:
    readings = {f"Chip 0 Core {index}": 40.0 for index in range(40)}
    for chip in (0, 8):
        readings[f"Chip {chip} DIMM 5"] = 40.0
        readings[f"Chip {chip} VRM VDD"] = 40.0
        for gpu in range(3):
            readings[f"Chip {chip} GPU {gpu}"] = 40.0
            readings[f"Chip {chip} GPU {gpu} MEM"] = 40.0
    if hot_label is not None:
        readings[hot_label] = 65.0
    return {
        "sample_started_unix": 999.9,
        "sample_completed_unix": 1_000.0,
        "ipmi_readings": {
            "fan0_0": fan0,
            "fan2_0": 3_000.0,
            "fan3_0": 3_000.0,
            "ambient": 32.0,
            "pcie": 40.0,
        },
        "hwmon_readings": readings,
        "errors": [],
    }


class SafetyMonitorTests(unittest.TestCase):
    def run_monitor(
        self,
        collect_side_effect,
        *,
        stop_after_records: int | None = None,
        gpu_free_mib: int = 16_000,
        gpu_temp_c: int = 40,
        env: dict[str, str] | None = None,
        preexisting_fault: bool = False,
        stop_expected: bool = True,
    ):
        clock = FakeClock()
        records: list[dict] = []
        fault_path: Path | None = None

        def publish(record: dict) -> None:
            if record.get("latched"):
                self.assertIsNotNone(fault_path)
                self.assertTrue(fault_path.exists(), "persistent latch must be written before status publication")
            records.append(record)
            sample_records = sum("ready" in item for item in records)
            if (
                stop_after_records is not None
                and not record.get("latched")
                and sample_records >= stop_after_records
            ):
                raise StopAfterPublish()

        def nvidia_smi(_args: list[str]) -> str:
            return "\n".join([f"{gpu_free_mib}, {gpu_temp_c}"] * 6)

        with tempfile.TemporaryDirectory(prefix="ac922-safety-test-") as tempdir:
            fault_path = Path(tempdir) / "fault.json"
            if preexisting_fault:
                fault_path.write_text(json.dumps({"latched": True, "error": "previous fault"}) + "\n")

            def stop_process(*args, **kwargs):
                self.assertTrue(fault_path.exists(), "persistent latch must be written before stopping inference")
                return None

            stop_mock = Mock(side_effect=stop_process)
            with patch.dict(os.environ, env or {}, clear=env is None):
                # The production setting is read when the monitor module loads.
                importlib.reload(safety_monitor)
                with (
                    patch.object(safety_monitor, "FAULT", fault_path),
                    patch.object(safety_monitor, "collect", side_effect=collect_side_effect),
                    patch.object(safety_monitor, "run", side_effect=nvidia_smi),
                    patch.object(safety_monitor, "publish", side_effect=publish),
                    patch.object(safety_monitor.subprocess, "run", stop_mock),
                    patch("builtins.print"),
                    patch.object(safety_monitor.time, "monotonic", clock.monotonic),
                    patch.object(safety_monitor.time, "time", clock.wall),
                    patch.object(safety_monitor.time, "sleep", clock.sleep),
                ):
                    try:
                        safety_monitor.main()
                    except (StopAfterPublish, OSError, RuntimeError):
                        pass
            if stop_expected:
                stop_mock.assert_called_once_with(
                    ["/usr/bin/systemctl", "stop", "ac922-inference.service"],
                    timeout=55,
                    check=False,
                )
            else:
                stop_mock.assert_not_called()
            fault_record = json.loads(fault_path.read_text()) if fault_path.exists() else None
            return records, fault_record

    def test_ready_requires_fifteen_continuous_healthy_seconds_after_low_tach(self) -> None:
        readings = [sample() for _ in range(6)]
        readings += [sample(fan0=2_600.0)]
        readings += [sample() for _ in range(6)]
        next_sample = iter(readings)
        records, _fault = self.run_monitor(lambda: next(next_sample), stop_after_records=13)
        status = [item for item in records if "ready" in item and not item.get("latched")]

        self.assertEqual(len(status), 13)
        self.assertEqual([item["ready"] for item in status[:6]], [False] * 5 + [True])
        self.assertFalse(status[6]["ready"], "low tach sample must clear startup readiness")
        self.assertFalse(status[7]["ready"], "one recovered sample is not 15 seconds of healthy tach")
        self.assertTrue(status[-1]["ready"], "readiness returns after 15 continuous healthy seconds")

    def test_persistent_low_tach_stops_on_third_bad_sample(self) -> None:
        samples = iter([sample(fan0=2_600.0) for _ in range(3)])
        collected = 0

        def collect():
            nonlocal collected
            collected += 1
            return next(samples)

        records, fault = self.run_monitor(collect)
        self.assertEqual(collected, 3)
        self.assertTrue(any("persistent main fan low tach guard" in item.get("error", "") for item in records))
        self.assertTrue(fault and fault.get("latched"))

    def test_temperature_guard_stops_on_first_bad_sample(self) -> None:
        collected = 0

        def collect():
            nonlocal collected
            collected += 1
            return sample(hot_label="Chip 0 GPU 0")

        records, fault = self.run_monitor(collect)
        self.assertEqual(collected, 1)
        self.assertTrue(any("temperature:Chip 0 GPU 0" in item.get("error", "") for item in records))
        self.assertTrue(fault and fault.get("latched"))

    def test_gpu_nvml_boundary_64_allowed_65_stops(self) -> None:
        records, _ = self.run_monitor(lambda: sample(), gpu_temp_c=64, stop_after_records=1)
        self.assertFalse(any("GPU headroom" in item.get("error", "") for item in records))
        records, fault = self.run_monitor(lambda: sample(), gpu_temp_c=65)
        self.assertTrue(any("GPU headroom" in item.get("error", "") for item in records))
        self.assertTrue(fault and fault.get("latched"))

    def test_hwmon_gpu_core_and_memory_boundary_other_sensors_unchanged(self) -> None:
        from combo_temperature_policy import fitted_inventory, inspect
        for label in ("Chip 0 GPU 0", "Chip 8 GPU 2 MEM"):
            for temperature, rejected in ((58, False), (64.9, False), (65, True)):
                record = sample()
                record["hwmon_readings"][label] = temperature
                errors, _ = inspect(record, fitted_inventory(record), 1000.0)
                self.assertEqual(bool(errors), rejected, (label, temperature, errors))
        for label in ("Chip 0 Core 0", "Chip 0 DIMM 5", "Chip 8 VRM VDD"):
            record = sample()
            record["hwmon_readings"][label] = 58
            errors, _ = inspect(record, fitted_inventory(record), 1000.0)
            self.assertIn("temperature:"+label, errors)

    def test_three_consecutive_read_faults_latch_and_stop(self) -> None:
        collected = 0

        def collect():
            nonlocal collected
            collected += 1
            raise OSError("mock sensor read failure")

        records, fault = self.run_monitor(collect)
        self.assertEqual(collected, 3)
        self.assertTrue(any(item.get("latched") is True for item in records))
        self.assertTrue(fault and fault.get("latched"))

    def test_default_gpu_memory_floor_is_256_mib_and_boundary_is_inclusive(self) -> None:
        records, _fault = self.run_monitor(
            lambda: sample(), stop_after_records=1, gpu_free_mib=256, env={}
        )
        self.assertFalse(any("GPU headroom" in item.get("error", "") for item in records))

        collected = 0

        def collect_below_default():
            nonlocal collected
            collected += 1
            return sample()

        records, fault = self.run_monitor(
            collect_below_default, gpu_free_mib=255, env={}
        )
        self.assertEqual(collected, 1)
        self.assertTrue(any("GPU headroom or temperature guard" in item.get("error", "") for item in records))
        self.assertTrue(fault and fault.get("latched"))

    def test_gpu_memory_floor_can_be_raised_by_environment(self) -> None:
        collected = 0

        def collect_sample():
            nonlocal collected
            collected += 1
            return sample()

        records, fault = self.run_monitor(
            collect_sample,
            gpu_free_mib=511,
            env={"AC922_MIN_GPU_FREE_MIB": "512"},
        )
        self.assertEqual(collected, 1)
        self.assertTrue(any("GPU headroom or temperature guard" in item.get("error", "") for item in records))
        self.assertTrue(fault and fault.get("latched"))

    def test_existing_persistent_latch_refuses_start_before_sensor_reads(self) -> None:
        collected = 0

        def collect():
            nonlocal collected
            collected += 1
            return sample()

        records, fault = self.run_monitor(collect, preexisting_fault=True, stop_expected=False)
        self.assertEqual(collected, 0)
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0].get("latched"))
        self.assertEqual(fault, {"latched": True, "error": "previous fault"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
