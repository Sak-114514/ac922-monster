"""Check actual private artifacts before the unchanged frozen native answers."""
from contextlib import redirect_stdout
from pathlib import Path
import subprocess
import sys
from verify_phase import main

if __name__ == '__main__':
    with redirect_stdout(sys.stderr):
        main()
    raise SystemExit(subprocess.run([sys.executable,str(Path(__file__).with_name('verify_combo_native.py')),
                                    *sys.argv[1:]]).returncode)
