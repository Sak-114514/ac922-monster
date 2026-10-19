"""Verify the exact private build, then all four unchanged native canaries."""
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path
from verify_hardware import main
if __name__ == '__main__':
    with redirect_stdout(sys.stderr):
        main()
    raise SystemExit(subprocess.run([sys.executable, str(Path(__file__).with_name('verify_combo_native.py')), *sys.argv[1:]]).returncode)
