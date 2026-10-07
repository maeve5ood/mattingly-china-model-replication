"""Replicate the main treatment effects (Table A3 of Mattingly et al. 2025).

Input:  data/processed/clean.parquet
Output: output/tables/
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "output" / "tables"


def main():
    """Placeholder: to be implemented."""
    pass


if __name__ == "__main__":
    main()
