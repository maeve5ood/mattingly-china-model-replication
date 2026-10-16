"""Run the full replication and extension pipeline from raw data to outputs.

The last step checks that every reported number in paper/main.tex matches the
outputs.

Usage:
    python run_all.py
"""

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS = [
    "src/01_clean.py",
    "src/02_replicate.py",
    "src/03_extension.py",
    "src/04_figures.py",
    "src/05_check_paper.py",
]


def main():
    """Execute each pipeline script in order, stopping at the first error."""
    for step in STEPS:
        print(f"=== Running {step} ===")
        runpy.run_path(str(ROOT / step), run_name="__main__")
    print("=== All steps finished ===")


if __name__ == "__main__":
    main()
