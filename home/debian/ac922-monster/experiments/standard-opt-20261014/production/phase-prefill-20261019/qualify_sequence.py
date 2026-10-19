"""Run one finite qualification sequence, stopping on any failed gate."""
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
for name in ['run_qualification.py', 'multi_long_gate.py', 'aligned_generation_gate.py']:
    subprocess.run([sys.executable, '-u', str(HERE / name)], check=True)
