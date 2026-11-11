# Step 3. Video-level results (Secs 4.3, 4.5, 4.6).

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, norm

from analysis_utils import (
    COVARIATES, OUTCOMES, SINGLE_ARMS, adjust_pvalues, check, contrast, extract, fit,
    load_data, save_tables,
)

CHINA_VIDEO = 'C(video2, Treatment("china_pol2"))'
US_VIDEO = 'C(video1, Treatment("us_pol2"))'
# MDE: alpha .05, power .8
MDE_MULTIPLIER = norm.ppf(0.975) + norm.ppf(0.80)


def assignment_table(df):
    # video2 = pol2 unless video1 is pol2, then pol1 (cf. fn. 9)
    tables = []
    for arm, prefix in SINGLE_ARMS.items():
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
    # companion = the other video shown with pol2
    check(prefix in SINGLE_ARMS.values(), f"Unknown video prefix: {prefix}")
    pol2 = f"{prefix}_pol2"
    valid = {f"{prefix}_{v}" for v in ("econ1", "econ2", "pol1", "pol2")}
    check(not arm_df.empty, f"Empty {prefix} arm")
    check(arm_df[["video1", "video2"]].isin(valid).all().all(),
          f"Invalid or missing video in the {prefix} arm")
    saw_pol2 = (arm_df[["video1", "video2"]] == pol2).sum(axis=1)
    check((saw_pol2 == 1).all(), f"Expected exactly one {pol2} per respondent")

    arm_df = arm_df.copy()
    arm_df["companion"] = arm_df["video1"].where(arm_df["video1"] != pol2, arm_df["video2"])
    arm_df["companion_econ"] = arm_df["companion"].str.contains("econ").astype(int)
    return arm_df


def competition_sample(df):
    comp = df[df["treatment"] == "Competition"].copy()
    check(comp["video1"].str.startswith("us_").all(), "video1 is not always a US video")
    check(comp["video2"].str.startswith("china_").all(), "video2 is not always a Chinese video")
    comp["china_econ"] = comp["video2"].str.contains("econ").astype(int)
    comp["us_econ"] = comp["video1"].str.contains("econ").astype(int)
    return comp


def independence_test(comp):
    cells = pd.crosstab(comp["video1"], comp["video2"])
    chi2, p, dof, _ = chi2_contingency(cells)
    return pd.DataFrame([{"chi2": chi2, "df": dof, "p_value": p, "n": len(comp),
                          "min_cell": int(cells.values.min()), "max_cell": int(cells.values.max())}])


def video_term(prefix, video):
    factor = CHINA_VIDEO if prefix == "china" else US_VIDEO
    return f"{factor}[T.{prefix}_{video}]"


def video_terms(prefix):
    return {video_term(prefix, v): f"{prefix}_{v}" for v in ["econ1", "econ2", "pol1"]}


def fit_video_model(comp, outcome):
    return fit(comp, f"{outcome} ~ {CHINA_VIDEO} + {US_VIDEO} + {COVARIATES}")


def video_effects(models):
    terms = {**video_terms("china"), **video_terms("us")}
    tables = []
    for outcome in OUTCOMES:
        rows = extract(models[outcome], terms)
        rows.insert(0, "outcome", outcome)
        tables.append(rows)
    return adjust_pvalues(pd.concat(tables, ignore_index=True))


def joint_tests(models):
    # H0: all four videos equal
    rows = []
    for outcome in OUTCOMES:
        result = models[outcome]
        for prefix in ["china", "us"]:
            names = list(video_terms(prefix))
            wald = result.wald_test(", ".join(f"{n} = 0" for n in names), scalar=True)
            effects = list(result.params[names]) + [0.0]  # pol2 = 0
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
    return np.concatenate([effects, np.zeros(effects.shape[:-1] + (1,))], axis=-1)


def simulate_coefficients(result, names, draws, rng):
    theta = result.params[names].to_numpy()
    covariance = result.cov_params().loc[names, names].to_numpy()
    # cholesky: svd draws differ across BLAS builds
    return theta, rng.multivariate_normal(theta, covariance, size=draws, method="cholesky")


def range_uncertainty(models, draws=20000, seed=353):
    # range is biased up -> parametric bootstrap (Sec 4.5)
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
    # CGTN vs State Dept dispersion; all 6 coefs drawn jointly
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
    # interaction is exploratory only (~380 per cell)
    main_terms = {"china_econ": "China economic video", "us_econ": "US economic video"}
    main, interaction = [], []
    for outcome in OUTCOMES:
        additive = fit(comp, f"{outcome} ~ china_econ + us_econ + {COVARIATES}")
        interacted = fit(comp, f"{outcome} ~ china_econ * us_econ + {COVARIATES}")
        main.append(extract(additive, main_terms).assign(outcome=outcome, model="additive"))
        interaction.append(extract(interacted, {"china_econ:us_econ": "China econ x US econ"})
                           .assign(outcome=outcome, model="interaction"))
    main = adjust_pvalues(pd.concat(main, ignore_index=True))
    table = pd.concat([main, *interaction], ignore_index=True)
    return table[["outcome"] + [c for c in table.columns if c != "outcome"]]


def econ_vs_pol1_weights(prefix):
    return {video_term(prefix, "econ1"): 0.5, video_term(prefix, "econ2"): 0.5,
            video_term(prefix, "pol1"): -1}


def content_contrasts(df, models, outcomes, arms):
    # Single-country arms pool econ1/econ2. Competition weights them equally.
    sources = {"China": "CGTN", "USA": "State Department"}
    rows = []
    for outcome in outcomes:
        comp_result = models[outcome]
        for arm in arms:
            prefix = SINGLE_ARMS[arm]
            arm_df = add_companion(df[df["treatment"] == arm], prefix)
            result = fit(arm_df, f"{outcome} ~ companion_econ + {COVARIATES}")
            single = extract(result, {"companion_econ": "Economic vs. pol1"}).iloc[0].to_dict()
            single.update({"design": f"{arm} arm", "mde": MDE_MULTIPLIER * single["std_error"]})

            competition = contrast(comp_result, econ_vs_pol1_weights(prefix), "Economic vs. pol1")
            competition.update({"design": "Competition arm",
                                "mde": MDE_MULTIPLIER * competition["std_error"]})

            for row in (single, competition):
                row.update({"outcome": outcome, "source": sources[arm]})
                rows.append(row)
    columns = ["outcome", "source", "design", "term", "estimate", "std_error", "p_value",
               "ci_low", "ci_high", "mde", "n"]
    return pd.DataFrame(rows)[columns]


def companion_position(df):
    # Table A4: econ companions always played before pol2, so check order
    tables = []
    for arm, prefix in SINGLE_ARMS.items():
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
        "ext_content_type.csv": adjust_pvalues(content_contrasts(df, models, OUTCOMES, SINGLE_ARMS)),
        "ext_companion_position.csv": companion_position(df),
    })


if __name__ == "__main__":
    main()
