"""Add host-delta identity checks to the unchanged C12 native semantic gate."""
import subprocess
import sys
from pathlib import Path

from verify_candidate import verify

if __name__ == "__main__":
    verify()
    raise SystemExit(subprocess.run([sys.executable, str(Path(__file__).with_name("verify_combo_native.py")), *sys.argv[1:]]).returncode)
