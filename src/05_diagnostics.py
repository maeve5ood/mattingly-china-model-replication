"""Balance, missingness, and the original code's video-type specification.

Produces Table A5 and Appendix B / Table A6. Balance checks describe the
realized sample; they do not by themselves establish random assignment.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2 as chi2_distribution
from scipy.stats import chi2_contingency

from analysis_utils import (
    COVARIATES, OUTCOMES, check, contrast, extract, fit, load_data, save_tables,
)

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


def validate_multinomial(result, label):
    """Reject non-converged or non-finite fits before reporting an LR test."""
    check(result.mle_retvals.get("converged", False),
          f"Multinomial {label} model did not converge")
    check(np.isfinite(result.llf) and np.isfinite(np.asarray(result.params)).all(),
          f"Multinomial {label} model has non-finite estimates")


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
    for label, result in (("full", full), ("null", null)):
        validate_multinomial(result, label)
    statistic = 2 * (full.llf - null.llf)
    dof = (X.shape[1] - 1) * (len(np.unique(y)) - 1)
    return statistic, dof, chi2_distribution.sf(statistic, dof)


def balance(df):
    """Covariate means and missing-outcome rates by video, with tests (Table A5).

    Covariates are the authors' imputed values. Per grouping: equal covariate
    means, country independent of video (lower tail too, which flags a
    more-even-than-chance spread), a multinomial-logit omnibus, and missingness.
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
    df = load_data()
    means, tests = balance(df)
    save_tables({
        "ext_balance_means.csv": means,
        "ext_balance_tests.csv": tests,
        "ext_authors_video_type.csv": authors_video_type(df),
    })


if __name__ == "__main__":
    main()
