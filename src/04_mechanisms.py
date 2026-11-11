# Step 4. Perceptions (Sec 4.7, our Table A3). Post-treatment outcomes, not mediation.

from importlib import import_module

import numpy as np
import pandas as pd

from analysis_utils import (
    COVARIATES, SINGLE_ARMS, adjust_pvalues, check, contrast, extract, fit, load_data,
    save_tables,
)

effects = import_module("03_message_effects")

# 1-7 scale, App. D.2
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
# typo in the original data
MISSPELLED = {"mech_freespeech_china": "mech_freespech_china"}


def perception_variable(template, country):
    variable = template.format(c=country)
    return MISSPELLED.get(variable, variable)


MECHANISMS = {perception_variable(template, "china"): label
              for template, label in PERCEPTION_ITEMS.items()}


def companion_mechanisms(df):
    control = df[df["treatment"] == "Control"]
    tables = []
    for arm, prefix in SINGLE_ARMS.items():
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
    table = effects.content_contrasts(df, models, MECHANISMS, ("China",))
    table.insert(1, "label", table["outcome"].map(MECHANISMS))
    return adjust_pvalues(table)


def mechanism_baseline(df):
    control = df[df["treatment"] == "Control"]
    rows = []
    for outcome, label in MECHANISMS.items():
        answered = control[outcome].dropna()
        rows.append({"outcome": outcome, "label": label, "mean": answered.mean(),
                     "share_agree": (answered >= 6).mean(), "n": len(answered)})
    return pd.DataFrame(rows)


def mechanism_videos(models):
    tables = []
    for outcome, label in MECHANISMS.items():
        rows = extract(models[outcome], effects.video_terms("china"))
        rows.insert(0, "outcome", outcome)
        rows.insert(1, "label", label)
        tables.append(rows)
    return adjust_pvalues(pd.concat(tables, ignore_index=True))


def build_indices(control, sample, items):
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


def mechanism_index(df, comp, items=None):
    # Kling, Liebman & Katz (2007) and Anderson (2008) indices
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
    tables = []
    for dropped in MECHANISMS:
        table = mechanism_index(df, comp, [item for item in MECHANISMS if item != dropped])
        table.insert(0, "dropped_item", MECHANISMS[dropped])
        tables.append(table.drop(columns=["mean_item_correlation"]))
    return pd.concat(tables, ignore_index=True)


def main():
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
