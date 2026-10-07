"""Load the authors' cleaned data and save an analysis-ready file.

Input:  data/raw/clean.RData (object `clean`)
Output: data/processed/clean.parquet
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


def main():
    """Placeholder: to be implemented."""
    pass


if __name__ == "__main__":
    main()
