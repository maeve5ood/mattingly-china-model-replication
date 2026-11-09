# Step 1. Starts from the authors' clean.RData (their mice imputation
# can't be reproduced exactly in Python).

import numpy as np
import pandas as pd
import pyreadr

from analysis_utils import CSV_FLOAT_FORMAT, PROCESSED, ROOT, TABLES, check

RAW = ROOT / "data" / "raw"

# Control = reference
TREATMENT_LEVELS = ["Control", "China", "USA", "Competition"]

# from the paper
EXPECTED_GROUP_SIZES = {"Control": 1558, "China": 1622, "USA": 1565, "Competition": 1531}
EXPECTED_N_OUTCOME = {"political_model": 5952, "econ_model": 5946}
EXPECTED_N_COUNTRIES = 19


def load_raw():
    path = RAW / "clean.RData"
    if not path.exists():
        raise FileNotFoundError(
            f"{path.relative_to(ROOT)} not found. Download the replication package from "
            "https://doi.org/10.7910/DVN/CQ4FZR and place the files from its Data/ folder "
            "in data/raw/ (see README)."
        )
    return pyreadr.read_r(path)["clean"]


def classify_screening(df):
    # first match wins (like case_when), so order matters
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
    stage = np.select(conditions, labels, default="Unclassified")
    return pd.Categorical(stage, categories=labels + ["Unclassified"])


def sample_flow(df):
    stage = classify_screening(df)
    flow = pd.Series(stage).value_counts(sort=False).rename_axis("stage").reset_index(name="n")
    total = pd.DataFrame({"stage": ["Total"], "n": [len(df)]})
    return pd.concat([flow, total], ignore_index=True)


def validate(df):
    check(df["country"].nunique() == EXPECTED_N_COUNTRIES,
          f"Expected {EXPECTED_N_COUNTRIES} countries, found {df['country'].nunique()}")
    check("22" not in set(df["country"]), "Respondents without a country (code 22) present")

    sizes = df["treatment"].value_counts().to_dict()
    check(sizes == EXPECTED_GROUP_SIZES, f"Group sizes {sizes} differ from paper")

    for outcome, n in EXPECTED_N_OUTCOME.items():
        found = df[outcome].notna().sum()
        check(found == n, f"{outcome}: expected {n} non-missing, found {found}")


def prepare(raw):
    df = raw[raw["treatment"].notna()].copy()
    df["treatment"] = pd.Categorical(df["treatment"].astype(str), categories=TREATMENT_LEVELS)
    return df.reset_index(drop=True)


def main():
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
