# Inference thermal policy — 2026-10-18

Human approved raising the GPU experiment/inference stop threshold from 58 to 65°C.

- GPU core and GPU memory hwmon sensors: stop this model task at >=65°C.
- NVML GPU core sensor: stop at >=65°C.
- Large-prefill pacing: pause at >=59°C, resume at <=54°C; poll every 200ms and fail at >=65°C or 120s cooldown timeout.
- CPU/DIMM/VDD/ambient/PCIe: existing 58°C stop unchanged.
- Low tach, missing/stale sensor, 256MiB memory floor and persisted fault latch unchanged.
- No BMC, fan targets/masks, hardware shutdown thresholds, driver, power or clock settings changed.

Scoped files are combo_temperature_policy.py, combo-guard.py, combo_safety_monitor.py and thermal-pacing.h. C8 remains historical; launch-combo-c9.py selects the new guard. Re-link the host worker with the updated header before launching. The installed R299 fallback release remains frozen and its services are currently disabled/stopped.

Verification: 9 hardware-free monitor tests passed both on the Mac and on AC922, including 64°C admitted / 65°C rejected and CPU/DIMM/VDD at 58°C still rejected. No intentional thermal stress test.
