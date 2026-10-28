"""Load the authors' cleaned data and save an analysis-ready file.

The authors' `1. clean.R` builds `clean.RData` from the raw Qualtrics exports,
including multiple imputation of covariates with `mice`. That imputation cannot
be reproduced exactly in Python, so this script starts from `clean.RData`
rather than the raw CSVs.

Steps:
    1. Load `clean.RData`.
    2. Document why respondents without a treatment assignment were screened
       out (sample flow table).
    3. Check the data against figures reported in the paper.
    4. Keep randomized respondents only, fix the treatment reference category,
       and save.

Input:  data/raw/clean.RData (object `clean`)
Output: data/processed/clean.parquet
        output/tables/sample_flow.csv
"""

import numpy as np
import pandas as pd
import pyreadr

from analysis_utils import CSV_FLOAT_FORMAT, PROCESSED, ROOT, TABLES, check

RAW = ROOT / "data" / "raw"

# Control first so that it is the reference category, as in the R code.
TREATMENT_LEVELS = ["Control", "China", "USA", "Competition"]

# Figures from Mattingly et al. (text and Table A3) used to validate the loaded data.
EXPECTED_GROUP_SIZES = {"Control": 1558, "China": 1622, "USA": 1565, "Competition": 1531}
EXPECTED_N_OUTCOME = {"political_model": 5952, "econ_model": 5946}
EXPECTED_N_COUNTRIES = 19


def load_raw():
    """Read the `clean` data frame from the authors' RData file."""
    path = RAW / "clean.RData"
    if not path.exists():
        raise FileNotFoundError(
            f"{path.relative_to(ROOT)} not found. Download the replication package from "
            "https://doi.org/10.7910/DVN/CQ4FZR and place the files from its Data/ folder "
            "in data/raw/ (see README)."
        )
    return pyreadr.read_r(path)["clean"]


def classify_screening(df):
    """Label each respondent by the survey stage at which they left the sample.

    Categories are mutually exclusive and follow the order of the survey:
    non-citizens are screened out first, then respondents who never answered
    the pre-treatment attention check, then those who failed it, then those
    who passed but quit before being randomized.
    """
    assigned = df["treatment"].notna()
    no_check = df["attention"].isna()

    conditions = [
        assigned,
        no_check & (df["citizen"] == 0),
        no_check,
        df["attention"] == 0,
        df["attention"] == 1,
    ]
    labels = [
        "Randomized",
        "Screened out: not a citizen",
        "Left before attention check",
        "Failed attention check",
        "Passed check, left before randomization",
    ]
    # Like R's case_when: the first true condition determines the label.
    stage = np.select(conditions, labels, default="Unclassified")
    return pd.Categorical(stage, categories=labels + ["Unclassified"])


def sample_flow(df):
    """Count respondents at each screening stage."""
    stage = classify_screening(df)
    flow = pd.Series(stage).value_counts(sort=False).rename_axis("stage").reset_index(name="n")
    total = pd.DataFrame({"stage": ["Total"], "n": [len(df)]})
    return pd.concat([flow, total], ignore_index=True)


def validate(df):
    """Check randomized respondents against figures reported by the authors."""
    check(df["country"].nunique() == EXPECTED_N_COUNTRIES,
          f"Expected {EXPECTED_N_COUNTRIES} countries, found {df['country'].nunique()}")
    check("22" not in set(df["country"]), "Respondents without a country (code 22) present")

    sizes = df["treatment"].value_counts().to_dict()
    check(sizes == EXPECTED_GROUP_SIZES, f"Group sizes {sizes} differ from paper")

    for outcome, n in EXPECTED_N_OUTCOME.items():
        found = df[outcome].notna().sum()
        check(found == n, f"{outcome}: expected {n} non-missing, found {found}")


def prepare(raw):
    """Keep randomized respondents and set Control as the reference category."""
    df = raw[raw["treatment"].notna()].copy()
    df["treatment"] = pd.Categorical(df["treatment"].astype(str), categories=TREATMENT_LEVELS)
    return df.reset_index(drop=True)


def main():
    """Load, document, validate, and save the analysis sample."""
    raw = load_raw()

    flow = sample_flow(raw)
    print(flow.to_string(index=False))
    TABLES.mkdir(parents=True, exist_ok=True)
    flow.to_csv(TABLES / "sample_flow.csv", index=False, float_format=CSV_FLOAT_FORMAT)

    df = prepare(raw)
    validate(df)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED / "clean.parquet", index=False)
    print(f"Saved {len(df)} randomized respondents to data/processed/clean.parquet")


if __name__ == "__main__":
    main()
