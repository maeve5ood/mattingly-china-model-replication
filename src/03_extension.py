"""Extension: unbundle the treatments into message-level effects.

The paper estimates the effect of each treatment arm as a bundle of two videos.
Within arms, the specific videos were also assigned at random, which allows
causal comparisons across individual messages that the published paper and
appendix do not report.

A. Assignment structure. In the China and USA arms, `video1` is drawn from all
   four of the country's videos and `video2` is a political video: `pol2`,
   unless `video1` is `pol2`, in which case `pol1`. Every respondent therefore
   saw `pol2`, and the randomized "companion" is `pol1` (about 50%), `econ1`
   or `econ2` (about 25% each). In the Competition arm, one US and one Chinese
   video were drawn independently from all four (a 4x4 factorial).

B. Message heterogeneity (Competition arm). The effect of each video relative
   to its country's `pol2`, with joint Wald tests that a country's four videos
   are equally persuasive. The range of the four Chinese videos' effects is an
   extreme of noisy estimates, so its bias and a basic 95% interval are
   computed by a parametric bootstrap.

C. Economic vs. political content in two designs. Within the China and USA
   arms, an economic vs. a political companion (holding `pol2` fixed); within
   the Competition arm, the average economic video vs. `pol1`. Both contrasts
   compare economic content with `pol1`, in independent samples.

   Position check: the codebook labels `video1` as the first video watched.
   If that is literal, economic companions always played before `pol2`, while
   `pol1` played before `pol2` only half the time. The contrast is therefore
   re-estimated among respondents with `pol2` in the second slot, and the
   effect of `pol2`'s slot is estimated among political-political pairs.

D. Mechanisms (exploratory). The same contrasts on the eight perceptions of
   China that the authors measure after treatment, plus each Chinese video's
   effect on those perceptions relative to `china_pol2`, both item by item and
   on a standardized summary index of the eight items (equally weighted, as in
   Kling, Liebman, and Katz 2007, and inverse-covariance weighted, as in
   Anderson 2008). The index is re-estimated dropping each item in turn, to
   check that no single item drives it. For comparison, the
   single-country contrast is also estimated on perceptions of each arm's own
   country (China arm: China; USA arm: United States).

E. Balance and missing outcomes. Pre-treatment covariate means and missing-
   outcome rates by video, for each way respondents were divided by video,
   with tests that they do not differ across videos. For the test that video is
   independent of country, the lower-tail p-value is also reported, which is
   small when videos are spread across countries more evenly than independent
   draws would usually produce.

F. The replication code's video-type analysis. `7. mechanism_analysis.R`
   estimates effects by `video1_clean`, which the published paper does not
   report. In the Competition arm `video1` is always the US video, so that
   specification pools Competition respondents with the USA arm. This part
   reproduces it and re-estimates it with the arms kept separate.

All models use the covariates and HC2 standard errors of the primary analysis.
p-values are adjusted within each family of tests using Holm and
Benjamini-Hochberg, as in the authors' appendix.

Input:  data/processed/clean.parquet
Output: output/tables/ext_assignment.csv
        output/tables/ext_competition_independence.csv
        output/tables/ext_competition_videos.csv
        output/tables/ext_competition_joint_tests.csv
        output/tables/ext_competition_range.csv
        output/tables/ext_competition_types.csv
        output/tables/ext_content_type.csv
        output/tables/ext_companion_position.csv
        output/tables/ext_companion_mechanisms.csv
        output/tables/ext_mechanisms_content.csv
        output/tables/ext_mechanisms_baseline.csv
        output/tables/ext_mechanisms_videos.csv
        output/tables/ext_mechanisms_index.csv
        output/tables/ext_mechanisms_index_loo.csv
        output/tables/ext_balance_means.csv
        output/tables/ext_balance_tests.csv
        output/tables/ext_authors_video_type.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2 as chi2_distribution
from scipy.stats import chi2_contingency, norm
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "output" / "tables"

OUTCOMES = ["political_model", "econ_model"]
# Same adjustment set as the primary analysis (02_replicate.py).
COVARIATES = "C(country) + gender + age + education + national_pride + leftright"

# Perceptions of China measured after treatment (1-7 agreement scale).
# Labels abbreviate the item wording in Appendix D.2 of the paper. Variable
# names follow the authors' data, including the `freespech` typo.
MECHANISMS = {
    "mech_competent_china": "Selects competent leaders",
    "mech_growth_china": "Delivers economic growth",
    "mech_stability_china": "Political stability",
    "mech_responsive_china": "Responsive to people's needs",
    "mech_turnover_china": "Ruling party sometimes loses",
    "mech_freespech_china": "Right to free speech",
    "mech_dem_china": "Democratic system",
    "mech_vote_china": "Adults vote for national leader",
}

# Video indicators in the Competition arm, with `pol2` as each country's reference.
CHINA_VIDEO = 'C(video2, Treatment("china_pol2"))'
US_VIDEO = 'C(video1, Treatment("us_pol2"))'

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


# Part A: assignment structure -------------------------------------------------

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
    saw_pol2 = (arm_df["video1"] == pol2) | (arm_df["video2"] == pol2)
    check(saw_pol2.all(), f"Not every respondent in the {prefix} arm saw {pol2}")

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


# Part B: message heterogeneity in the Competition arm -------------------------

def video_terms(prefix):
    """statsmodels names of the three non-reference videos of one country."""
    factor = CHINA_VIDEO if prefix == "china" else US_VIDEO
    return {f"{factor}[T.{prefix}_{v}]": f"{prefix}_{v}" for v in ["econ1", "econ2", "pol1"]}


def fit_video_model(comp, outcome):
    """Regress `outcome` on Chinese and US video indicators (reference: `pol2`)."""
    return fit(comp, f"{outcome} ~ {CHINA_VIDEO} + {US_VIDEO} + {COVARIATES}")


def video_effects(comp):
    """Estimate each video's effect relative to its country's `pol2` video.

    `pol2` is the reference because it is the video every respondent in the
    single-country arms saw, which links these estimates to Part C.
    """
    terms = {**video_terms("china"), **video_terms("us")}
    tables = []
    for outcome in OUTCOMES:
        rows = extract(fit_video_model(comp, outcome), terms)
        rows.insert(0, "outcome", outcome)
        tables.append(rows)
    return adjust_pvalues(pd.concat(tables, ignore_index=True))


def joint_tests(comp):
    """Wald test that a country's four videos have equal effects.

    Equivalent to testing that the three coefficients relative to `pol2` are
    all zero. Also reports the range of the four video effects.
    """
    rows = []
    for outcome in OUTCOMES:
        result = fit_video_model(comp, outcome)
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


def range_uncertainty(comp, draws=20000, seed=353):
    """Bias and basic bootstrap interval for the range of the four Chinese videos.

    The range (largest minus smallest effect, with `china_pol2` at zero) picks
    the extremes of noisy estimates, so it is biased upward. A parametric
    bootstrap draws the three coefficients from their estimated sampling
    distribution, recomputes the range, and compares it with the observed one.
    """
    rng = np.random.default_rng(seed)
    names = list(video_terms("china"))
    rows = []
    for outcome in OUTCOMES:
        result = fit_video_model(comp, outcome)
        theta = result.params[names].to_numpy()
        covariance = result.cov_params().loc[names, names].to_numpy()
        observed = np.ptp(np.append(theta, 0.0))
        simulated = rng.multivariate_normal(theta, covariance, size=draws)
        ranges = np.ptp(np.column_stack([simulated, np.zeros(draws)]), axis=1)
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


# Part C: economic vs. political content in two designs ------------------------

def econ_vs_pol1_weights(prefix):
    """Weights for (average economic video) - pol1 in the Competition video model."""
    econ1, econ2, pol1 = list(video_terms(prefix))
    return {econ1: 0.5, econ2: 0.5, pol1: -1}


def content_contrasts(df, comp, outcomes):
    """Economic content vs. `pol1`, estimated in both designs for each outcome.

    Single-country arms: economic vs. political companion of `pol2`.
    Competition arm: average of the two economic videos minus `pol1`.
    Positive values mean more favorable to China in both designs.
    """
    rows = []
    for outcome in outcomes:
        comp_result = fit_video_model(comp, outcome)
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
    """Check whether the companion contrast is confounded with video position.

    If `video1` is the first video shown, as the codebook states, economic
    companions always played before `pol2`, while `pol1` played before `pol2`
    only half the time. Two checks for each single-country arm:

    1. Re-estimate the companion contrast among respondents with `pol2` in the
       second slot, so position is the same in both groups.
    2. Among political-political pairs, compare `pol2` in the first vs. the
       second slot, which isolates any effect of position itself.
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


# Part D: mechanisms -------------------------------------------------------------

# Perception items asked about both countries; `{c}` is "china" or "us".
# The China free-speech variable is misspelled in the authors' data.
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


def companion_mechanisms(df):
    """Single-country companion contrast on perceptions of the arm's own country.

    China-arm respondents' perceptions of China and USA-arm respondents'
    perceptions of the US. p-values are adjusted across the eight items
    within each arm. Also reports each item's control-group mean.
    """
    control = df[df["treatment"] == "Control"]
    tables = []
    for arm, prefix in [("China", "china"), ("USA", "us")]:
        arm_df = add_companion(df[df["treatment"] == arm], prefix)
        rows = []
        for template, label in PERCEPTION_ITEMS.items():
            variable = template.format(c=prefix)
            if variable == "mech_freespeech_china":
                variable = "mech_freespech_china"
            row = extract(fit(arm_df, f"{variable} ~ companion_econ + {COVARIATES}"),
                          {"companion_econ": label})
            row.insert(0, "outcome", variable)
            row["control_mean"] = control[variable].mean()
            rows.append(row)
        table = adjust_pvalues(pd.concat(rows, ignore_index=True))
        table.insert(0, "arm", arm)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)

def mechanism_contrasts(df, comp):
    """Economic content vs. `pol1` on each perception of China, in both designs."""
    table = content_contrasts(df, comp, list(MECHANISMS))
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


def mechanism_videos(comp):
    """Each Chinese video's effect on perceptions of China relative to `china_pol2`."""
    tables = []
    for outcome, label in MECHANISMS.items():
        rows = extract(fit_video_model(comp, outcome), video_terms("china"))
        rows.insert(0, "outcome", outcome)
        rows.insert(1, "label", label)
        tables.append(rows)
    return adjust_pvalues(pd.concat(tables, ignore_index=True))


def mechanism_index(df, comp, items=None):
    """Effect of each Chinese video on a summary index of the eight perceptions.

    The items are correlated and every comparison shares `china_pol2` as the
    reference, so item-by-item results are not independent evidence. Items are
    standardized by the control group's mean and SD and combined two ways:
    an equally weighted mean (Kling, Liebman, and Katz 2007) and an
    inverse-covariance weighted mean (Anderson 2008), with weights from the
    control group. Respondents need all items. `items` defaults to all eight.
    """
    items = list(MECHANISMS) if items is None else list(items)
    control = df[df["treatment"] == "Control"]
    center, scale = control[items].mean(), control[items].std()
    control_z = ((control[items] - center) / scale).dropna()
    weights = np.linalg.inv(control_z.cov().to_numpy()).sum(axis=1)
    item_correlation = control_z.corr().to_numpy()[np.triu_indices(len(items), 1)].mean()

    data = comp.copy()
    z = (data[items] - center) / scale
    complete = z.notna().all(axis=1)
    data["index_equal"] = z.mean(axis=1).where(complete)
    data["index_inverse_covariance"] = pd.Series(z.to_numpy() @ weights / weights.sum(),
                                                 index=data.index).where(complete)

    terms = video_terms("china")
    tables = []
    for column, label in [("index_equal", "Equally weighted"),
                          ("index_inverse_covariance", "Inverse-covariance weighted")]:
        result = fit_video_model(data, column)
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


# Part E: balance and missing outcomes -------------------------------------------

BALANCE_COVARIATES = ["gender", "age", "education", "national_pride", "leftright"]


def video_groupings(df):
    """The four ways respondents were divided by video: (label, data, column)."""
    comp = df[df["treatment"] == "Competition"]
    return [
        ("China arm", df[df["treatment"] == "China"], "video1"),
        ("USA arm", df[df["treatment"] == "USA"], "video1"),
        ("Competition arm, Chinese video", comp, "video2"),
        ("Competition arm, US video", comp, "video1"),
    ]


def multinomial_omnibus(sub, column):
    """Likelihood-ratio test that covariates and country jointly predict the video.

    Multinomial logit of the video on the five covariates and country dummies,
    against an intercept-only model.
    """
    predictors = sub[BALANCE_COVARIATES].assign(country=sub["country"].astype(str))
    X = sm.add_constant(pd.get_dummies(predictors, columns=["country"], drop_first=True).astype(float))
    y = pd.Categorical(sub[column]).codes
    full = sm.MNLogit(y, X).fit(disp=0)
    null = sm.MNLogit(y, np.ones((len(y), 1))).fit(disp=0)
    statistic = 2 * (full.llf - null.llf)
    dof = (X.shape[1] - 1) * (len(np.unique(y)) - 1)
    return statistic, dof, chi2_distribution.sf(statistic, dof)


def balance(df):
    """Covariate means and missing-outcome rates by video, with tests.

    Covariates are the authors' imputed values. Tests, for each grouping:
    equal means of each covariate (Wald, HC2), country independent of video
    (chi-squared, with the lower-tail p-value as a check for a more-even-than-
    chance spread), a multinomial-logit omnibus test, and missing outcomes
    independent of video (chi-squared).
    """
    mean_rows, test_rows = [], []
    for label, sub, column in video_groupings(df):
        for video, group in sub.groupby(column):
            row = {"grouping": label, "video": video, "n": len(group)}
            row.update(group[BALANCE_COVARIATES].mean().to_dict())
            for outcome in OUTCOMES:
                row[f"missing_{outcome}"] = group[outcome].isna().mean()
            mean_rows.append(row)

        def add(test, statistic, dof, p_value):
            test_rows.append({"grouping": label, "test": test, "statistic": statistic,
                              "df": dof, "p_value": p_value})

        for covariate in BALANCE_COVARIATES:
            result = fit(sub, f"{covariate} ~ C({column})")
            names = [n for n in result.params.index if n.startswith(f"C({column})")]
            wald = result.wald_test(", ".join(f"{n} = 0" for n in names), scalar=True)
            add(f"equal means: {covariate}", float(wald.statistic), len(names), float(wald.pvalue))
        chi2, p, dof, _ = chi2_contingency(pd.crosstab(sub["country"], sub[column]))
        add("country independent of video", chi2, dof, p)
        add("country independent of video, lower tail", chi2, dof, chi2_distribution.cdf(chi2, dof))
        add("multinomial logit omnibus", *multinomial_omnibus(sub, column))
        for outcome in OUTCOMES:
            chi2, p, dof, _ = chi2_contingency(pd.crosstab(sub[column], sub[outcome].isna()))
            add(f"missing {outcome} independent of video", chi2, dof, p)
    return pd.DataFrame(mean_rows), pd.DataFrame(test_rows)


# Part F: the replication code's video-type analysis ----------------------------

def authors_video_type(df):
    """Reproduce the unreported `video1_clean` regression and re-estimate it by arm.

    `pooled`: as in `7. mechanism_analysis.R`, where Competition respondents
    fall into the US categories because `video1` is their US video.
    `by_arm`: Competition respondents get their own categories.
    """
    data = df.copy()
    data["video_type"] = data["video1_clean"].astype(str)
    is_comp = data["treatment"] == "Competition"
    data["video_type_by_arm"] = data["video_type"].where(~is_comp, "competition_" + data["video_type"])

    rows = []
    for outcome in OUTCOMES:
        for spec, column in [("pooled", "video_type"), ("by_arm", "video_type_by_arm")]:
            factor = f'C({column}, Treatment("control"))'
            result = fit(data, f"{outcome} ~ {factor} + {COVARIATES}")
            levels = [name for name in result.params.index if name.startswith(factor)]
            table = extract(result, {name: name.split("[T.")[1].rstrip("]") for name in levels})
            for a, b in [("china_econ", "china_pol"), ("us_econ", "us_pol")]:
                diff = contrast(result, {f"{factor}[T.{a}]": 1, f"{factor}[T.{b}]": -1}, f"{a} - {b}")
                table = pd.concat([table, pd.DataFrame([diff])], ignore_index=True)
            table.insert(0, "outcome", outcome)
            table.insert(1, "specification", spec)
            rows.append(table)
    return pd.concat(rows, ignore_index=True)


def main():
    """Run all parts of the extension and save the tables."""
    df = load_data()
    comp = competition_sample(df)
    balance_means, balance_tests = balance(df)

    results = {
        "ext_assignment.csv": assignment_table(df),
        "ext_competition_independence.csv": independence_test(comp),
        "ext_competition_videos.csv": video_effects(comp),
        "ext_competition_joint_tests.csv": joint_tests(comp),
        "ext_competition_range.csv": range_uncertainty(comp),
        "ext_competition_types.csv": type_effects(comp),
        "ext_content_type.csv": content_contrasts(df, comp, OUTCOMES),
        "ext_companion_position.csv": companion_position(df),
        "ext_companion_mechanisms.csv": companion_mechanisms(df),
        "ext_mechanisms_content.csv": mechanism_contrasts(df, comp),
        "ext_mechanisms_baseline.csv": mechanism_baseline(df),
        "ext_mechanisms_videos.csv": mechanism_videos(comp),
        "ext_mechanisms_index.csv": mechanism_index(df, comp),
        "ext_mechanisms_index_loo.csv": mechanism_index_leave_one_out(df, comp),
        "ext_balance_means.csv": balance_means,
        "ext_balance_tests.csv": balance_tests,
        "ext_authors_video_type.csv": authors_video_type(df),
    }

    TABLES.mkdir(parents=True, exist_ok=True)
    for filename, table in results.items():
        table.to_csv(TABLES / filename, index=False)
        print(f"--- {filename}")
        print(table.round(3).to_string(index=False))
        print()


if __name__ == "__main__":
    main()
