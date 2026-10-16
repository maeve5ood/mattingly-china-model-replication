"""Extension: unbundle the treatments into video-level effects.

The paper estimates the effect of each treatment arm as a bundle of two videos.
Within arms, the specific videos were randomized, which allows two causal
comparisons the paper does not report:

A. China and USA arms. `video1` was drawn uniformly from the country's four
   videos; `video2` was then `pol2`, unless `video1` was `pol2`, in which case
   it was `pol1`. Every respondent therefore saw `pol2`, and the companion
   video was `pol1` (about 50%), `econ1` or `econ2` (about 25% each).
   Comparing respondents whose companion was an economic vs. a political
   video estimates the added value of economic content, holding `pol2` fixed.
   The authors' replication code (`7. mechanism_analysis.R`) contains an
   unpublished version of this comparison, which is reproduced here.

B. Competition arm. One US and one Chinese video were drawn independently,
   each from all four of that country's videos (a 4x4 factorial). This
   identifies the effect of each Chinese video relative to the others while
   the US video is balanced by randomization, and vice versa.

The codebook labels `video1_clean` as the first video watched, while the
paper states that the order of countries was randomized in the Competition
arm, where `video1` is always the US video. The public data cannot resolve
this, so Part A is also estimated with `pol2` held in the second slot for
everyone, which removes any confound with position if `video1` is the first
video shown.

All models use the same covariates and HC2 standard errors as the primary
analysis. p-values are adjusted for multiple comparisons within each family
of tests using Holm and Benjamini-Hochberg, as in the authors' appendix.

Input:  data/processed/clean.parquet
Output: output/tables/ext_assignment.csv
        output/tables/ext_companion.csv
        output/tables/ext_companion_position.csv
        output/tables/ext_companion_mechanisms.csv
        output/tables/ext_authors_video_type.csv
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

# Single-country arms and the prefix of their video identifiers.
ARMS = [("China", "china"), ("USA", "us")]

# Mechanism items (1-7 agreement scales), grouped into the performance and
# democracy dimensions identified by the authors' clustering
# (`7. mechanism_analysis.R`). `{c}` is "china" or "us"; the China free speech
# variable is misspelled in the data.
MECHANISMS = {
    "Selects competent leaders": ("performance", "mech_competent_{c}"),
    "Delivers economic growth": ("performance", "mech_growth_{c}"),
    "Political stability": ("performance", "mech_stability_{c}"),
    "Responsive to the people": ("performance", "mech_responsive_{c}"),
    "Free speech": ("democracy", "mech_freespeech_{c}"),
    "Ruling party sometimes loses": ("democracy", "mech_turnover_{c}"),
    "Democratic system": ("democracy", "mech_dem_{c}"),
    "All adults may vote": ("democracy", "mech_vote_{c}"),
}


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


def contrast(result, plus, minus, label):
    """Estimate the difference between two coefficients (a Wald test).

    Returns one row in the same format as `extract`.
    """
    weights = pd.Series(0.0, index=result.params.index)
    weights[plus] = 1.0
    weights[minus] = -1.0
    test = result.t_test(weights.to_numpy())
    ci = test.conf_int()[0]
    return pd.DataFrame([{
        "term": label,
        "estimate": float(test.effect[0]),
        "std_error": float(test.sd[0][0]),
        "p_value": float(test.pvalue),
        "ci_low": ci[0],
        "ci_high": ci[1],
        "n": int(result.nobs),
    }])


# Assignment structure ---------------------------------------------------------

def assignment_table(df):
    """Count each (video1, video2) pair in the China and USA arms.

    Documents the assignment rule: video1 is roughly uniform over the four
    videos and video2 is pol2 unless video1 is pol2.
    """
    tables = []
    for arm, _ in ARMS:
        arm_df = df[df["treatment"] == arm]
        counts = arm_df.groupby(["video1", "video2"]).size().reset_index(name="n")
        counts.insert(0, "arm", arm)
        tables.append(counts)
    return pd.concat(tables, ignore_index=True)


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
    for arm, prefix in ARMS:
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


def companion_position(df):
    """Check whether the companion comparison is confounded with video position.

    If `video1` is the first video shown (as the codebook states), economic
    companions always play before `pol2`, while political companions play
    before `pol2` only half the time. Two checks:

    1. Re-estimate the companion effect among respondents with `pol2` in the
       second slot, so position is identical in both groups.
    2. Among political-political pairs, compare `pol2` in the first vs. the
       second slot, which isolates any effect of position itself.
    """
    tables = []
    for arm, prefix in ARMS:
        arm_df = add_companion(df[df["treatment"] == arm], prefix)
        pol2 = f"{prefix}_pol2"
        same_position = arm_df[arm_df["video2"] == pol2]
        political_pairs = arm_df[arm_df["companion_econ"] == 0].copy()
        political_pairs["pol2_first"] = (political_pairs["video1"] == pol2).astype(int)

        for outcome in OUTCOMES:
            held = fit(same_position, f"{outcome} ~ companion_econ + {COVARIATES}")
            order = fit(political_pairs, f"{outcome} ~ pol2_first + {COVARIATES}")
            rows = pd.concat([
                extract(held, {"companion_econ": "Economic companion, pol2 second for all"}),
                extract(order, {"pol2_first": "pol2 first vs. second (political pairs)"}),
            ])
            rows.insert(0, "arm", arm)
            rows.insert(1, "outcome", outcome)
            tables.append(rows)
    return pd.concat(tables, ignore_index=True)


def companion_mechanisms(df):
    """Estimate the companion effect on each perception of the arm's country.

    China-arm respondents are asked about China, USA-arm respondents about the
    US. p-values are adjusted across the eight items within each arm.
    """
    tables = []
    for arm, prefix in ARMS:
        arm_df = add_companion(df[df["treatment"] == arm], prefix)
        rows = []
        for label, (dimension, template) in MECHANISMS.items():
            variable = template.format(c=prefix)
            if variable == "mech_freespeech_china":
                variable = "mech_freespech_china"
            result = fit(arm_df, f"{variable} ~ companion_econ + {COVARIATES}")
            row = extract(result, {"companion_econ": label})
            row.insert(0, "dimension", dimension)
            rows.append(row)
        table = adjust_pvalues(pd.concat(rows, ignore_index=True))
        table.insert(0, "arm", arm)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def authors_video_type(df):
    """Reproduce the authors' unpublished video-type models, then split by arm.

    The authors regress outcomes on `video1_clean` (`7. mechanism_analysis.R`,
    lines 219-244). In the Competition arm `video1` is always the US video,
    so those respondents are coded as US economic or US political and pooled
    with the USA arm. The split model gives USA-arm and Competition-arm
    respondents separate categories, as in the paper's own description of
    the design as six groups (footnote 10).
    """
    pooled_var = 'C(video1_clean, Treatment("control"))'
    split_var = 'C(video_group, Treatment("control"))'

    data = df.copy()
    video_type = data["video1_clean"].astype(str)
    data["video_group"] = video_type.where(
        data["treatment"] != "Competition", "competition_" + video_type
    )

    tables = []
    for outcome in OUTCOMES:
        pooled = fit(data, f"{outcome} ~ {pooled_var} + {COVARIATES}")
        pooled_terms = {f"{pooled_var}[T.{v}]": v for v in
                        ["china_econ", "china_pol", "us_econ", "us_pol"]}
        pooled_rows = pd.concat([
            extract(pooled, pooled_terms),
            contrast(pooled, f"{pooled_var}[T.china_econ]", f"{pooled_var}[T.china_pol]",
                     "china_econ - china_pol"),
        ]).assign(model="authors (pooled)")

        split = fit(data, f"{outcome} ~ {split_var} + {COVARIATES}")
        split_terms = {f"{split_var}[T.{v}]": v for v in
                       ["us_econ", "us_pol", "competition_us_econ", "competition_us_pol"]}
        split_rows = extract(split, split_terms).assign(model="split by arm")

        rows = pd.concat([pooled_rows, split_rows], ignore_index=True)
        rows.insert(0, "outcome", outcome)
        tables.append(rows)
    return pd.concat(tables, ignore_index=True)


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
        "ext_assignment.csv": assignment_table(df),
        "ext_companion.csv": companion_effects(df),
        "ext_companion_position.csv": companion_position(df),
        "ext_companion_mechanisms.csv": companion_mechanisms(df),
        "ext_authors_video_type.csv": authors_video_type(df),
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
