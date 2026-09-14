"""Runs the full PCC_control validation suite.

  python run_validation.py
"""
import subprocess
import sys
from pathlib import Path

SUITE = ["test_arc_kinematics.py", "test_cable_lengths.py", "test_motor_positions.py"]


def main():
    here = Path(__file__).parent
    codes = []
    for script in SUITE:
        print(f"\n{'='*70}\n{script}\n{'='*70}")
        codes.append(subprocess.call([sys.executable, str(here / script)], cwd=here))
    print(f"\n{'='*70}")
    bad = sum(1 for c in codes if c != 0)
    if bad:
        print(f"SUITE RESULT: {bad}/{len(SUITE)} test scripts reported failures")
    else:
        print(f"SUITE RESULT: all {len(SUITE)} test scripts passed")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
