"""Exploratory perception outcomes, summary indices, and sensitivity checks.

Produces Section 4.7, Figure 4, and this paper's Table A3 (not Mattingly et
al.'s Table A3, which 02_replicate.py reproduces). Post-treatment perceptions
are outcomes here; these regressions do not identify causal mediation.
"""

from importlib import import_module

import numpy as np
import pandas as pd

from analysis_utils import (
    COVARIATES, adjust_pvalues, check, contrast, extract, fit, load_data, save_tables,
)

# Module names that start with a digit need import_module.
effects = import_module("03_message_effects")

# Perceptions measured after treatment (1-7 agreement scale), asked about both
# China and the US; `{c}` is "china" or "us". Labels abbreviate the item
# wording in Appendix D.2 of the paper.
PERCEPTION_ITEMS = {
    "mech_competent_{c}": "Selects competent leaders",
    "mech_growth_{c}": "Delivers economic growth",
    "mech_stability_{c}": "Political stability",
    "mech_responsive_{c}": "Responsive to people's needs",
    "mech_turnover_{c}": "Ruling party sometimes loses",
    "mech_freespeech_{c}": "Right to free speech",
    "mech_dem_{c}": "Democratic system",
    "mech_vote_{c}": "Adults vote for national leader",
}
# The authors' data misspell one variable name.
MISSPELLED = {"mech_freespeech_china": "mech_freespech_china"}


def perception_variable(template, country):
    """Name of a perception item's variable for `country` ("china" or "us")."""
    variable = template.format(c=country)
    return MISSPELLED.get(variable, variable)


# Perceptions of China used as outcomes in the exploratory analysis.
MECHANISMS = {perception_variable(template, "china"): label
              for template, label in PERCEPTION_ITEMS.items()}


def companion_mechanisms(df):
    """Single-country companion contrast on perceptions of the arm's own country.

    China-arm respondents' perceptions of China and USA-arm respondents'
    perceptions of the US. p-values are adjusted across the eight items
    within each arm. Also reports each item's control-group mean.
    """
    control = df[df["treatment"] == "Control"]
    tables = []
    for arm, prefix in [("China", "china"), ("USA", "us")]:
        arm_df = effects.add_companion(df[df["treatment"] == arm], prefix)
        rows = []
        for template, label in PERCEPTION_ITEMS.items():
            variable = perception_variable(template, prefix)
            row = extract(fit(arm_df, f"{variable} ~ companion_econ + {COVARIATES}"),
                          {"companion_econ": label})
            row.insert(0, "outcome", variable)
            row["control_mean"] = control[variable].mean()
            rows.append(row)
        table = adjust_pvalues(pd.concat(rows, ignore_index=True))
        table.insert(0, "arm", arm)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def mechanism_contrasts(df, models):
    """Economic content vs. `pol1` on each perception of China, in both designs."""
    table = effects.content_contrasts(df, models, list(MECHANISMS))
    table = table[table["source"] == "CGTN"].drop(columns=["p_holm", "p_bh"])
    table.insert(1, "label", table["outcome"].map(MECHANISMS))
    return adjust_pvalues(table.reset_index(drop=True))


def mechanism_baseline(df):
    """Control-group level of each perception of China: mean and share agreeing (6-7)."""
    control = df[df["treatment"] == "Control"]
    rows = []
    for outcome, label in MECHANISMS.items():
        answered = control[outcome].dropna()
        rows.append({"outcome": outcome, "label": label, "mean": answered.mean(),
                     "share_agree": (answered >= 6).mean(), "n": len(answered)})
    return pd.DataFrame(rows)


def mechanism_videos(models):
    """Each Chinese video's effect on perceptions of China relative to `china_pol2`."""
    tables = []
    for outcome, label in MECHANISMS.items():
        rows = extract(models[outcome], effects.video_terms("china"))
        rows.insert(0, "outcome", outcome)
        rows.insert(1, "label", label)
        tables.append(rows)
    return adjust_pvalues(pd.concat(tables, ignore_index=True))


def mechanism_index(df, comp, items=None):
    """Effect of each Chinese video on a summary index of the perceptions (our Table A3).

    Items are standardized and weighted using the control group, then averaged
    equally (Kling, Liebman, and Katz 2007) or by inverse covariance (Anderson
    2008). Respondents need all items. `items` defaults to all eight.
    """
    items = list(MECHANISMS) if items is None else list(items)
    control = df[df["treatment"] == "Control"]
    data = build_indices(control, comp, items)
    correlation = control[items].dropna().corr().to_numpy()
    item_correlation = correlation[np.triu_indices(len(items), 1)].mean()

    terms = effects.video_terms("china")
    tables = []
    for column, label in [("index_equal", "Equally weighted"),
                          ("index_inverse_covariance", "Inverse-covariance weighted")]:
        result = effects.fit_video_model(data, column)
        rows = extract(result, terms)
        others = contrast(result, {name: 1 / 3 for name in terms}, "average of other three")
        rows = pd.concat([rows, pd.DataFrame([others])], ignore_index=True)
        wald = result.wald_test(", ".join(f"{n} = 0" for n in terms), scalar=True)
        rows.insert(0, "index", label)
        rows["joint_chi2"] = float(wald.statistic)
        rows["joint_p"] = float(wald.pvalue)
        rows["mean_item_correlation"] = item_correlation
        tables.append(rows)
    return pd.concat(tables, ignore_index=True)


def mechanism_index_leave_one_out(df, comp):
    """The summary index re-estimated without each of the eight items in turn."""
    tables = []
    for dropped in MECHANISMS:
        table = mechanism_index(df, comp, [item for item in MECHANISMS if item != dropped])
        table.insert(0, "dropped_item", MECHANISMS[dropped])
        tables.append(table.drop(columns=["mean_item_correlation"]))
    return pd.concat(tables, ignore_index=True)


def build_indices(control, sample, items):
    """Construct complete-case indices using control-group moments and weights."""
    center, scale = control[items].mean(), control[items].std()
    check((scale > 0).all(), "Index items must vary in the control group")
    covariance = ((control[items] - center) / scale).dropna().cov().to_numpy()
    check(np.linalg.cond(covariance) < 1e12, "Index covariance matrix is singular")
    weights = np.linalg.solve(covariance, np.ones(len(items)))
    z = (sample[items] - center) / scale
    complete = z.notna().all(axis=1)
    check(complete.any(), "No complete respondents for the index")
    data = sample.copy()
    data["index_equal"] = z.mean(axis=1).where(complete)
    data["index_inverse_covariance"] = pd.Series(
        z.to_numpy() @ weights / weights.sum(), index=data.index,
    ).where(complete)
    return data


def main():
    """Reuse the perception video models across contrasts and video effects."""
    df = load_data()
    comp = effects.competition_sample(df)
    models = {outcome: effects.fit_video_model(comp, outcome) for outcome in MECHANISMS}
    save_tables({
        "ext_companion_mechanisms.csv": companion_mechanisms(df),
        "ext_mechanisms_content.csv": mechanism_contrasts(df, models),
        "ext_mechanisms_baseline.csv": mechanism_baseline(df),
        "ext_mechanisms_videos.csv": mechanism_videos(models),
        "ext_mechanisms_index.csv": mechanism_index(df, comp),
        "ext_mechanisms_index_loo.csv": mechanism_index_leave_one_out(df, comp),
    })


if __name__ == "__main__":
    main()
