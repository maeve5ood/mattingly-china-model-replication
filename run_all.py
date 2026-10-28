"""Run the analysis pipeline; optionally check the manuscript with --check-paper."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    "src/clean.py",
    "src/replicate.py",
    "src/message_effects.py",
    "src/mechanisms.py",
    "src/diagnostics.py",
    "src/figures.py",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-paper", action="store_true",
                        help="also check reported numbers against paper/main.tex")
    args = parser.parse_args()
    steps = STEPS + (["checks/check_paper.py"] if args.check_paper else [])
    for step in steps:
        print(f"=== Running {step} ===", flush=True)
        subprocess.run([sys.executable, str(ROOT / step)], cwd=ROOT, check=True)
    print("=== All steps finished ===")


if __name__ == "__main__":
    main()
