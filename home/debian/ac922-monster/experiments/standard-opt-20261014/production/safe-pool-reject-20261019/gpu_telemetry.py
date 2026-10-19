"""Bounded read-only NVML trace; no CUDA initialization or hardware writes."""
import ctypes as C
import json
from pathlib import Path
import sys
import time

lib = C.CDLL("libnvidia-ml.so.1")


class Memory(C.Structure):
    _fields_ = [("total", C.c_ulonglong), ("free", C.c_ulonglong), ("used", C.c_ulonglong)]


lib.nvmlDeviceGetHandleByIndex_v2.argtypes = [C.c_uint, C.POINTER(C.c_void_p)]
lib.nvmlDeviceGetMemoryInfo.argtypes = [C.c_void_p, C.POINTER(Memory)]
lib.nvmlDeviceGetTemperature.argtypes = [C.c_void_p, C.c_uint, C.POINTER(C.c_uint)]
if lib.nvmlInit_v2() != 0:
    raise RuntimeError("NVML read-only telemetry initialization failed")
handles = []
for index in range(6):
    handle = C.c_void_p()
    if lib.nvmlDeviceGetHandleByIndex_v2(index, C.byref(handle)) != 0:
        raise RuntimeError("NVML GPU inventory mismatch")
    handles.append(handle)
output = Path(sys.argv[1])
duration = min(1800, max(1, int(sys.argv[2]) if len(sys.argv)>2 else 1200))
until = time.monotonic()+duration
output.parent.mkdir(parents=True, exist_ok=True)
try:
    with output.open("w", buffering=1) as stream:
        while time.monotonic() < until:
            gpu = []
            for handle in handles:
                memory, temp = Memory(), C.c_uint()
                if lib.nvmlDeviceGetMemoryInfo(handle, C.byref(memory)) != 0 or \
                        lib.nvmlDeviceGetTemperature(handle, 0, C.byref(temp)) != 0:
                    raise RuntimeError("NVML telemetry read failed")
                gpu.append([round(memory.free/(1024*1024), 3), temp.value])
            stream.write(json.dumps({"unix":time.time(),"gpu":gpu})+"\n")
            time.sleep(0.2)
finally:
    lib.nvmlShutdown()
