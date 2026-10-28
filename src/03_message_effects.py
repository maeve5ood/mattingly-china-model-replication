"""Message-level extension: assignment, video effects, and content contrasts.

Reproduces Sections 4.3, 4.5, and 4.6, plus their supporting checks.
OLS uses the original adjustment set and HC2 standard errors. Each table's
p-values are adjusted as one family unless its function documents otherwise.
Run directly after 01_clean.py, or through run_all.py.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, norm

from analysis_utils import (
    COVARIATES, OUTCOMES, adjust_pvalues, check, contrast, extract, fit,
    load_data, save_tables,
)

CHINA_VIDEO = 'C(video2, Treatment("china_pol2"))'
US_VIDEO = 'C(video1, Treatment("us_pol2"))'
MDE_MULTIPLIER = norm.ppf(0.975) + norm.ppf(0.80)


def assignment_table(df):
    """Count video pairs in the China and USA arms and verify the assignment rule.

    The rule: `video2` is `pol2` unless `video1` is `pol2`, in which case it is
    `pol1`. It implements the paper's requirement (footnote 9) that every
    respondent see at least one political video, but fixes which one.
    """
    tables = []
    for arm, prefix in [("China", "china"), ("USA", "us")]:
        arm_df = df[df["treatment"] == arm]
        expected = arm_df["video1"].map(lambda v: f"{prefix}_pol1" if v == f"{prefix}_pol2"
                                        else f"{prefix}_pol2")
        check((arm_df["video2"] == expected).all(),
              f"{arm} arm: video2 does not follow the pol2-unless-drawn rule")
        counts = arm_df.groupby(["video1", "video2"]).size().reset_index(name="n")
        counts.insert(0, "arm", arm)
        counts["share"] = counts["n"] / counts["n"].sum()
        tables.append(counts)
    return pd.concat(tables, ignore_index=True)


def add_companion(arm_df, prefix):
    """Identify the randomized companion of the always-shown `pol2` video.

    `prefix` is "china" or "us". Fails if any respondent did not see `pol2`,
    since the comparison relies on `pol2` being held fixed.
    """
    pol2 = f"{prefix}_pol2"
    check(prefix in {"china", "us"}, f"Unknown video prefix: {prefix}")
    valid = {f"{prefix}_{v}" for v in ("econ1", "econ2", "pol1", "pol2")}
    check(not arm_df.empty, f"Empty {prefix} arm")
    check(arm_df[["video1", "video2"]].isin(valid).all().all(),
          f"Invalid or missing video in the {prefix} arm")
    saw_pol2 = (arm_df["video1"] == pol2).astype(int) + (arm_df["video2"] == pol2).astype(int)
    check((saw_pol2 == 1).all(), f"Expected exactly one {pol2} per respondent")

    arm_df = arm_df.copy()
    arm_df["companion"] = arm_df["video1"].where(arm_df["video1"] != pol2, arm_df["video2"])
    arm_df["companion_econ"] = arm_df["companion"].str.contains("econ").astype(int)
    return arm_df


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
    """Chi-squared test that the US and Chinese videos were assigned independently.

    Also reports the smallest and largest of the 16 cell counts.
    """
    cells = pd.crosstab(comp["video1"], comp["video2"])
    chi2, p, dof, _ = chi2_contingency(cells)
    return pd.DataFrame([{"chi2": chi2, "df": dof, "p_value": p, "n": len(comp),
                          "min_cell": int(cells.values.min()), "max_cell": int(cells.values.max())}])


def video_terms(prefix):
    """statsmodels names of the three non-reference videos of one country."""
    factor = CHINA_VIDEO if prefix == "china" else US_VIDEO
    return {f"{factor}[T.{prefix}_{v}]": f"{prefix}_{v}" for v in ["econ1", "econ2", "pol1"]}


def fit_video_model(comp, outcome):
    """Regress `outcome` on Chinese and US video indicators (reference: `pol2`)."""
    return fit(comp, f"{outcome} ~ {CHINA_VIDEO} + {US_VIDEO} + {COVARIATES}")


def video_effects(models):
    """Estimate each video's effect relative to its country's `pol2` video.

    `pol2` is the reference because it is the video every respondent in the
    single-country arms saw, linking these estimates to the content contrasts.
    """
    terms = {**video_terms("china"), **video_terms("us")}
    tables = []
    for outcome in OUTCOMES:
        rows = extract(models[outcome], terms)
        rows.insert(0, "outcome", outcome)
        tables.append(rows)
    return adjust_pvalues(pd.concat(tables, ignore_index=True))


def joint_tests(models):
    """Wald test that a country's four videos have equal effects.

    Equivalent to testing that the three coefficients relative to `pol2` are
    all zero. Also reports the range of the four video effects.
    """
    rows = []
    for outcome in OUTCOMES:
        result = models[outcome]
        for prefix in ["china", "us"]:
            names = list(video_terms(prefix))
            wald = result.wald_test(", ".join(f"{n} = 0" for n in names), scalar=True)
            effects = list(result.params[names]) + [0.0]  # pol2 is the reference
            rows.append({
                "outcome": outcome,
                "videos": prefix,
                "chi2": float(wald.statistic),
                "df": len(names),
                "p_value": float(wald.pvalue),
                "range": max(effects) - min(effects),
                "n": int(result.nobs),
            })
    return adjust_pvalues(pd.DataFrame(rows))


def with_reference(effects):
    """Append the reference video's effect (zero) along the last axis."""
    return np.concatenate([effects, np.zeros(effects.shape[:-1] + (1,))], axis=-1)


def simulate_coefficients(result, names, draws, rng):
    """Point estimates of `names` and parametric bootstrap draws around them."""
    theta = result.params[names].to_numpy()
    covariance = result.cov_params().loc[names, names].to_numpy()
    # Cholesky rather than the default SVD: the SVD's signs and rotations
    # depend on the linear-algebra library, so the same seed would give
    # different draws on different machines.
    return theta, rng.multivariate_normal(theta, covariance, size=draws, method="cholesky")


def range_uncertainty(models, draws=20000, seed=353):
    """Bias and basic bootstrap interval for the range of the four Chinese videos.

    The range of noisy estimates is biased upward, so a parametric bootstrap
    redraws the three coefficients relative to `china_pol2` (Section 4.5).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for outcome in OUTCOMES:
        theta, simulated = simulate_coefficients(models[outcome], list(video_terms("china")),
                                                 draws, rng)
        observed = np.ptp(with_reference(theta))
        ranges = np.ptp(with_reference(simulated), axis=1)
        low, high = np.percentile(2 * observed - ranges, [2.5, 97.5])
        rows.append({
            "outcome": outcome,
            "observed_range": observed,
            "bootstrap_mean_range": ranges.mean(),
            "bias": ranges.mean() - observed,
            "bias_corrected_range": 2 * observed - ranges.mean(),
            "ci_low": low,
            "ci_high": high,
            "draws": draws,
        })
    return pd.DataFrame(rows)


def dispersion_comparison(models, draws=20000, seed=354):
    """Test whether the CGTN videos' effects are more dispersed than the US videos'.

    Dispersion is the range or SD of a country's four effects (`pol2` at zero).
    The CGTN-minus-US difference gets a basic interval from a parametric
    bootstrap of all six coefficients, which share one regression (Section 4.5).
    """
    rng = np.random.default_rng(seed)
    names = list(video_terms("china")) + list(video_terms("us"))
    measures = {"range": lambda x: np.ptp(x, axis=-1), "sd": lambda x: np.std(x, axis=-1, ddof=1)}

    def china_minus_us(effects, statistic):
        china, us = with_reference(effects[..., :3]), with_reference(effects[..., 3:])
        return statistic(china), statistic(us), statistic(china) - statistic(us)

    rows = []
    for outcome in OUTCOMES:
        theta, simulated = simulate_coefficients(models[outcome], names, draws, rng)
        for measure, statistic in measures.items():
            china, us, observed = china_minus_us(theta, statistic)
            differences = china_minus_us(simulated, statistic)[2]
            low, high = np.percentile(2 * observed - differences, [2.5, 97.5])
            rows.append({
                "outcome": outcome,
                "measure": measure,
                "china": china,
                "us": us,
                "difference": observed,
                "ci_low": low,
                "ci_high": high,
                "draws": draws,
            })
    return pd.DataFrame(rows)


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


def econ_vs_pol1_weights(prefix):
    """Weights for (average economic video) - pol1 in the Competition video model."""
    econ1, econ2, pol1 = list(video_terms(prefix))
    return {econ1: 0.5, econ2: 0.5, pol1: -1}


def content_contrasts(df, models, outcomes):
    """Economic content vs. `pol1`, estimated in both designs for each outcome.

    Single-country arms: economic vs. political companion of `pol2`.
    Competition arm: average of the two economic videos minus `pol1`.
    Positive values mean more favorable to China in both designs.
    """
    rows = []
    for outcome in outcomes:
        comp_result = models[outcome]
        for arm, prefix, source in [("China", "china", "CGTN"), ("USA", "us", "State Department")]:
            arm_df = add_companion(df[df["treatment"] == arm], prefix)
            result = fit(arm_df, f"{outcome} ~ companion_econ + {COVARIATES}")
            single = extract(result, {"companion_econ": "Economic vs. pol1"}).iloc[0].to_dict()
            single.update({"design": f"{arm} arm", "mde": MDE_MULTIPLIER * single["std_error"]})

            competition = contrast(comp_result, econ_vs_pol1_weights(prefix), "Economic vs. pol1")
            competition.update({"design": "Competition arm",
                                "mde": MDE_MULTIPLIER * competition["std_error"]})

            for row in (single, competition):
                row.update({"outcome": outcome, "source": source})
                rows.append(row)
    columns = ["outcome", "source", "design", "term", "estimate", "std_error", "p_value",
               "ci_low", "ci_high", "mde", "n"]
    return adjust_pvalues(pd.DataFrame(rows)[columns])


def companion_position(df):
    """Check that the companion contrast is not driven by video order (Table A4).

    If `video1` played first, as the codebook says, economic companions always
    preceded `pol2`. So the contrast is re-estimated with `pol2` second for
    everyone, and `pol2`'s slot is estimated among political-political pairs.
    """
    tables = []
    for arm, prefix in [("China", "china"), ("USA", "us")]:
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


def main():
    """Fit each primary video model once, then derive related summaries."""
    df = load_data()
    comp = competition_sample(df)
    models = {outcome: fit_video_model(comp, outcome) for outcome in OUTCOMES}
    save_tables({
        "ext_assignment.csv": assignment_table(df),
        "ext_competition_independence.csv": independence_test(comp),
        "ext_competition_videos.csv": video_effects(models),
        "ext_competition_joint_tests.csv": joint_tests(models),
        "ext_competition_range.csv": range_uncertainty(models),
        "ext_competition_dispersion.csv": dispersion_comparison(models),
        "ext_competition_types.csv": type_effects(comp),
        "ext_content_type.csv": content_contrasts(df, models, OUTCOMES),
        "ext_companion_position.csv": companion_position(df),
    })


if __name__ == "__main__":
    main()
