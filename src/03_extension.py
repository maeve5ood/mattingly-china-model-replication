"""Extension: unbundle the treatments into video-level effects.

The paper estimates the effect of each treatment arm as a bundle of two videos.
Within arms, the specific videos were randomized, which allows two causal
comparisons the paper does not report:

A. China and USA arms. Every respondent saw the country's `pol2` video; the
   companion video was randomized between `pol1` (about 50%), `econ1` and
   `econ2` (about 25% each). Comparing respondents whose companion was an
   economic vs. a political video estimates the added value of economic
   content, holding `pol2` fixed.

B. Competition arm. One US and one Chinese video were drawn independently,
   each from all four of that country's videos (a 4x4 factorial). This
   identifies the effect of each Chinese video relative to the others while
   the US video is balanced by randomization, and vice versa.

All models use the same covariates and HC2 standard errors as the primary
analysis. p-values are adjusted for multiple comparisons within each family
of tests using Holm and Benjamini-Hochberg, as in the authors' appendix.

Input:  data/processed/clean.parquet
Output: output/tables/ext_companion.csv
        output/tables/ext_competition_independence.csv
        output/tables/ext_competition_types.csv
        output/tables/ext_competition_videos.csv
"""

from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency, norm
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "output" / "tables"

OUTCOMES = ["political_model", "econ_model"]
# Same adjustment set as the primary analysis (02_replicate.py).
COVARIATES = "C(country) + gender + age + education + national_pride + leftright"

# Minimum detectable effect multiplier: two-sided alpha = 0.05, power = 0.80.
MDE_MULTIPLIER = norm.ppf(0.975) + norm.ppf(0.80)


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


def adjust_pvalues(table):
    """Add Holm and Benjamini-Hochberg adjusted p-values to a family of tests."""
    table = table.copy()
    table["p_holm"] = multipletests(table["p_value"], method="holm")[1]
    table["p_bh"] = multipletests(table["p_value"], method="fdr_bh")[1]
    return table


# Part A: companion video in the China and USA arms ---------------------------

def add_companion(arm_df, prefix):
    """Identify the randomized companion of the always-shown `pol2` video.

    `prefix` is "china" or "us". Fails if any respondent did not see `pol2`,
    since the comparison relies on `pol2` being held fixed.
    """
    pol2 = f"{prefix}_pol2"
    saw_pol2 = (arm_df["video1"] == pol2) | (arm_df["video2"] == pol2)
    check(saw_pol2.all(), f"Not every respondent in the {prefix} arm saw {pol2}")

    arm_df = arm_df.copy()
    arm_df["companion"] = arm_df["video1"].where(arm_df["video1"] != pol2, arm_df["video2"])
    arm_df["companion_econ"] = arm_df["companion"].str.contains("econ").astype(int)
    return arm_df


def companion_effects(df):
    """Estimate the effect of an economic (vs. political) companion video.

    Run separately for each single-country arm and outcome. The MDE is the
    smallest true effect detectable with 80% power given the estimated SE.
    """
    tables = []
    for arm, prefix in [("China", "china"), ("USA", "us")]:
        arm_df = add_companion(df[df["treatment"] == arm], prefix)
        for outcome in OUTCOMES:
            result = fit(arm_df, f"{outcome} ~ companion_econ + {COVARIATES}")
            row = extract(result, {"companion_econ": "Economic companion"})
            used = arm_df.dropna(subset=[outcome])
            row.insert(0, "arm", arm)
            row.insert(1, "outcome", outcome)
            row["n_econ"] = int(used["companion_econ"].sum())
            row["n_pol"] = int((1 - used["companion_econ"]).sum())
            row["mde"] = MDE_MULTIPLIER * row["std_error"]
            tables.append(row)
    return adjust_pvalues(pd.concat(tables, ignore_index=True))


# Part B: 4x4 factorial in the Competition arm --------------------------------

def competition_sample(df):
    """Return the Competition arm with video type indicators.

    In this arm `video1` is always the US video and `video2` the Chinese one.
    """
    comp = df[df["treatment"] == "Competition"].copy()
    check(comp["video1"].str.startswith("us_").all(), "video1 is not always a US video")
    check(comp["video2"].str.startswith("china_").all(), "video2 is not always a Chinese video")
    comp["china_econ"] = comp["video2"].str.contains("econ").astype(int)
    comp["us_econ"] = comp["video1"].str.contains("econ").astype(int)
    return comp


def independence_test(comp):
    """Chi-squared test that the US and Chinese videos were assigned independently."""
    chi2, p, dof, _ = chi2_contingency(pd.crosstab(comp["video1"], comp["video2"]))
    return pd.DataFrame([{"chi2": chi2, "df": dof, "p_value": p, "n": len(comp)}])


def type_effects(comp):
    """Estimate effects of economic (vs. political) Chinese and US videos.

    The main model is additive. A second model adds the China x US interaction;
    with about 380 respondents per type combination it is exploratory.
    """
    main_terms = {"china_econ": "China economic video", "us_econ": "US economic video"}
    tables = []
    for outcome in OUTCOMES:
        additive = fit(comp, f"{outcome} ~ china_econ + us_econ + {COVARIATES}")
        interacted = fit(comp, f"{outcome} ~ china_econ * us_econ + {COVARIATES}")
        rows = pd.concat([
            extract(additive, main_terms).assign(model="additive"),
            extract(interacted, {"china_econ:us_econ": "China econ x US econ"}).assign(model="interaction"),
        ])
        rows.insert(0, "outcome", outcome)
        tables.append(rows)
    table = pd.concat(tables, ignore_index=True)

    # Adjust the main effects as one family; the interaction is reported unadjusted.
    main = adjust_pvalues(table[table["model"] == "additive"])
    return pd.concat([main, table[table["model"] == "interaction"]], ignore_index=True)


def video_effects(comp):
    """Estimate each video's effect relative to its country's `pol2` video.

    `pol2` is the reference because it is the video every respondent in the
    single-country arms saw, which links these estimates to Part A.
    """
    china = 'C(video2, Treatment("china_pol2"))'
    us = 'C(video1, Treatment("us_pol2"))'
    terms = {f"{china}[T.china_{v}]": f"china_{v}" for v in ["econ1", "econ2", "pol1"]}
    terms.update({f"{us}[T.us_{v}]": f"us_{v}" for v in ["econ1", "econ2", "pol1"]})

    tables = []
    for outcome in OUTCOMES:
        result = fit(comp, f"{outcome} ~ {china} + {us} + {COVARIATES}")
        rows = extract(result, terms)
        rows.insert(0, "outcome", outcome)
        tables.append(rows)
    return adjust_pvalues(pd.concat(tables, ignore_index=True))


def main():
    """Run both parts of the extension and save the tables."""
    df = load_data()
    comp = competition_sample(df)

    results = {
        "ext_companion.csv": companion_effects(df),
        "ext_competition_independence.csv": independence_test(comp),
        "ext_competition_types.csv": type_effects(comp),
        "ext_competition_videos.csv": video_effects(comp),
    }

    TABLES.mkdir(parents=True, exist_ok=True)
    for filename, table in results.items():
        table.to_csv(TABLES / filename, index=False)
        print(f"--- {filename}")
        print(table.round(3).to_string(index=False))
        print()


if __name__ == "__main__":
    main()
