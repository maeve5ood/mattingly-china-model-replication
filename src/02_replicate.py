"""Replicate the main treatment effects (Table A3 of Mattingly et al. 2025).

Python translation of the estimation part of the authors' `6. primary_analysis.R`:
OLS of each primary outcome on treatment indicators (Control = reference),
with and without covariate adjustment, using HC2 robust standard errors
(the default of `estimatr::lm_robust` in R).

Input:  data/processed/clean.parquet
Output: output/tables/primary_coefficients.csv  (all treatment coefficients)
        output/tables/table_a3_check.csv         (comparison with the paper)
        output/tables/standardized_effects.csv   (effects in control-group SDs)
        output/tables/share_prefer_china.csv     (raw shares preferring China, by arm)
"""

import pandas as pd

from analysis_utils import COVARIATES, extract, fit, load_data, save_tables

OUTCOMES = ["political_model", "econ_model", "world_leader"]
TREATMENT = 'C(treatment, Treatment("Control"))'

# Readable names for statsmodels' coefficient labels.
TERM_NAMES = {
    "Intercept": "Constant",
    f"{TREATMENT}[T.China]": "China",
    f"{TREATMENT}[T.USA]": "USA",
    f"{TREATMENT}[T.Competition]": "Competition",
}

# Mattingly et al.'s Table A3: (estimate, standard error), rounded to 2 decimals.
TABLE_A3 = {
    ("political_model", True): {"Constant": (3.32, 0.20), "China": (1.04, 0.05),
                                "USA": (-0.43, 0.04), "Competition": (0.36, 0.05)},
    ("econ_model", True): {"Constant": (3.76, 0.23), "China": (0.87, 0.05),
                           "USA": (-0.57, 0.05), "Competition": (0.28, 0.06)},
    ("world_leader", True): {"Constant": (2.74, 0.20), "China": (0.72, 0.05),
                             "USA": (-0.31, 0.04), "Competition": (0.24, 0.05)},
    ("political_model", False): {"Constant": (2.56, 0.03), "China": (1.05, 0.05),
                                 "USA": (-0.44, 0.04), "Competition": (0.37, 0.05)},
    ("econ_model", False): {"Constant": (2.88, 0.04), "China": (0.87, 0.06),
                            "USA": (-0.58, 0.05), "Competition": (0.28, 0.06)},
    ("world_leader", False): {"Constant": (2.75, 0.03), "China": (0.72, 0.05),
                              "USA": (-0.33, 0.05), "Competition": (0.23, 0.05)},
}
TABLE_A3_N = {"political_model": 5952, "econ_model": 5946, "world_leader": 5943}

# A value matches the paper if it rounds to the reported 2-decimal number.
TOLERANCE = 0.005 + 1e-9


def fit_primary(df, outcome, adjusted):
    """Fit `outcome` on treatment, with or without covariates.

    Rows with a missing outcome are dropped automatically, as in R's lm_robust.
    """
    rhs = f"{TREATMENT} + {COVARIATES}" if adjusted else TREATMENT
    return fit(df, f"{outcome} ~ {rhs}")


def estimate_all(df):
    """Fit every outcome with and without covariates and stack the results."""
    tables = []
    for outcome in OUTCOMES:
        for adjusted in (True, False):
            table = extract(fit_primary(df, outcome, adjusted), TERM_NAMES)
            table.insert(0, "outcome", outcome)
            table.insert(1, "adjusted", adjusted)
            tables.append(table)
    return pd.concat(tables, ignore_index=True)


def compare_with_paper(coefs):
    """Line up our estimates with the published Table A3 and flag any mismatch."""
    rows = []
    for (outcome, adjusted), terms in TABLE_A3.items():
        for term, (paper_est, paper_se) in terms.items():
            ours = coefs.query("outcome == @outcome and adjusted == @adjusted and term == @term").iloc[0]
            rows.append({
                "outcome": outcome,
                "adjusted": adjusted,
                "term": term,
                "paper_estimate": paper_est,
                "our_estimate": round(ours["estimate"], 4),
                "paper_se": paper_se,
                "our_se": round(ours["std_error"], 4),
                "paper_n": TABLE_A3_N[outcome],
                "our_n": ours["n"],
                "match": (abs(ours["estimate"] - paper_est) <= TOLERANCE
                          and abs(ours["std_error"] - paper_se) <= TOLERANCE
                          and ours["n"] == TABLE_A3_N[outcome]),
            })
    return pd.DataFrame(rows)


def standardized_effects(df, coefs):
    """Divide unadjusted effects by the control-group SD of each outcome.

    Mirrors lines 97-99 of `6. primary_analysis.R`. pandas' `.std()` uses the
    sample SD (n - 1), the same as R's `sd()`.
    """
    control = df[df["treatment"] == "Control"]
    unadjusted = coefs[~coefs["adjusted"] & (coefs["term"] != "Constant")].copy()
    sd = unadjusted["outcome"].map(lambda outcome: control[outcome].std())
    unadjusted["control_sd"] = sd
    unadjusted["standardized_effect"] = unadjusted["estimate"] / sd
    return unadjusted[["outcome", "term", "estimate", "control_sd", "standardized_effect"]]


def share_prefer_china(df):
    """Share of respondents who prefer China's model (4-6 on the 1-6 scale), by arm.

    Reported in the text to convey magnitude in raw terms.
    """
    rows = []
    for outcome in ["political_model", "econ_model"]:
        answered = df[df[outcome].notna()]
        shares = (answered[outcome] >= 4).groupby(answered["treatment"], observed=True).mean()
        for arm, share in shares.items():
            rows.append({"outcome": outcome, "treatment": arm, "share_prefer_china": share,
                         "n": int((answered["treatment"] == arm).sum())})
    return pd.DataFrame(rows)


def main():
    """Estimate the primary models, check them against the paper, and save tables."""
    df = load_data()
    coefs = estimate_all(df)
    comparison = compare_with_paper(coefs)
    standardized = standardized_effects(df, coefs)
    shares = share_prefer_china(df)
    save_tables({
        "primary_coefficients.csv": coefs,
        "table_a3_check.csv": comparison,
        "standardized_effects.csv": standardized,
        "share_prefer_china.csv": shares,
    })

    print(comparison.to_string(index=False))
    print()
    print(standardized.round(3).to_string(index=False))
    print()
    print(shares.round(3).to_string(index=False))

    if not comparison["match"].all():
        raise ValueError("Some estimates do not match Table A3; see table_a3_check.csv")
    print(f"\nAll {len(comparison)} Table A3 coefficients match the paper.")


if __name__ == "__main__":
    main()
