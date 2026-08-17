#!/usr/bin/env python
"""QConfLib CLI — numerical validation of all shot-based estimators against exact values.

Examples:
  python cli/validate.py                    # 1D (with potential) + asymmetric 2D
  python cli/validate.py --shots 5000000    # tighter check
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qconflib import problem_1d, problem_2d, validate_problem  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shots', type=int, default=2_000_000)
    ap.add_argument('--device', default='auto', choices=['auto', 'GPU', 'CPU'])
    a = ap.parse_args()
    ok = True
    print("[validate] 1D box n=5 with potential u0=1.0 eV:")
    ok &= validate_problem(problem_1d(5, u0=1.0), shots=a.shots, device=a.device)
    print("[validate] 2D anisotropic box nx=2, ny=3 (asymmetric sub-registers):")
    ok &= validate_problem(problem_2d(2, 3), shots=a.shots, device=a.device)
    print("[validate] 2D anisotropic box nx=3, ny=2:")
    ok &= validate_problem(problem_2d(3, 2), shots=a.shots, device=a.device)
    print("ALL_VALID" if ok else "VALIDATION_FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
