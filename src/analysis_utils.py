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
SINGLE_ARMS = {"China": "china", "USA": "us"}


def load_data():
    return pd.read_parquet(PROCESSED / "clean.parquet")


def check(condition, message):
    if not condition:
        raise ValueError(message)


def fit(df, formula):
    # HC2 = lm_robust default
    return smf.ols(formula, data=df).fit(cov_type="HC2")


def extract(result, terms):
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
    table = table.copy()
    table["p_holm"] = multipletests(table["p_value"], method="holm")[1]
    table["p_bh"] = multipletests(table["p_value"], method="fdr_bh")[1]
    return table


def save_tables(tables):
    TABLES.mkdir(parents=True, exist_ok=True)
    for filename, table in tables.items():
        table.to_csv(TABLES / filename, index=False, float_format=CSV_FLOAT_FORMAT)
        print(f"Saved {filename}: {len(table)} rows")
