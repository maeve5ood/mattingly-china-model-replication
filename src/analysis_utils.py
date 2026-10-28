"""Shared paths, model settings, and result-table helpers."""

from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "output" / "tables"
CSV_FLOAT_FORMAT = "%.10g"
OUTCOMES = ["political_model", "econ_model"]
COVARIATES = "C(country) + gender + age + education + national_pride + leftright"


def load_data():
    """Read the analysis sample produced by 01_clean.py."""
    return pd.read_parquet(PROCESSED / "clean.parquet")


def check(condition, message):
    """Raise an error with `message` if `condition` is false."""
    if not condition:
        raise ValueError(message)


def fit(df, formula):
    """Fit OLS with HC2 robust standard errors."""
    return smf.ols(formula, data=df).fit(cov_type="HC2")


def extract(result, terms):
    """Return estimate, SE, p-value, and 95% CI for the given coefficient names.

    `terms` maps statsmodels' coefficient names to readable labels.
    """
    ci = result.conf_int()
    rows = []
    for raw_name, label in terms.items():
        rows.append({
            "term": label,
            "estimate": result.params[raw_name],
            "std_error": result.bse[raw_name],
            "p_value": result.pvalues[raw_name],
            "ci_low": ci.loc[raw_name, 0],
            "ci_high": ci.loc[raw_name, 1],
            "n": int(result.nobs),
        })
    return pd.DataFrame(rows)


def contrast(result, weights, label):
    """Estimate a linear combination of coefficients, e.g. 0.5*a + 0.5*b - c.

    `weights` maps statsmodels' coefficient names to weights.
    """
    expr = " + ".join(f"{w} * {name}" for name, w in weights.items())
    test = result.t_test(f"{expr} = 0")
    low, high = test.conf_int()[0]
    return {
        "term": label,
        "estimate": float(test.effect[0]),
        "std_error": float(test.sd[0, 0]),
        "p_value": float(test.pvalue),
        "ci_low": low,
        "ci_high": high,
        "n": int(result.nobs),
    }


def adjust_pvalues(table):
    """Add Holm and Benjamini-Hochberg adjusted p-values to a family of tests."""
    table = table.copy()
    table["p_holm"] = multipletests(table["p_value"], method="holm")[1]
    table["p_bh"] = multipletests(table["p_value"], method="fdr_bh")[1]
    return table


def save_tables(tables):
    """Write stable CSVs and print one progress line per table."""
    TABLES.mkdir(parents=True, exist_ok=True)
    for filename, table in tables.items():
        table.to_csv(TABLES / filename, index=False, float_format=CSV_FLOAT_FORMAT)
        print(f"Saved {filename}: {len(table)} rows")
