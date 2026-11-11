"""Run steps 1-7: clean, estimate, draw figures, write paper/numbers.tex."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    "src/01_clean.py",
    "src/02_replicate.py",
    "src/03_message_effects.py",
    "src/04_mechanisms.py",
    "src/05_diagnostics.py",
    "src/06_figures.py",
    "src/07_paper_numbers.py",
]


def main():
    for step in STEPS:
        print(f"=== Running {step} ===", flush=True)
        subprocess.run([sys.executable, str(ROOT / step)], cwd=ROOT, check=True)
    print("=== All steps finished ===")


if __name__ == "__main__":
    main()
