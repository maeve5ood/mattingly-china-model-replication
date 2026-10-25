"""Check that every reported number in the paper matches the pipeline's output.

Each number-bearing passage of `paper/main.tex` (every table row and every
sentence that reports a result) is rebuilt from the CSV files in
`output/tables/`, rounded as in the paper, and looked up verbatim in the LaTeX
source. Qualitative claims that rest on numbers (e.g. "one of 20 below 0.10",
"none of the other contrasts survives Holm correction") are checked directly.

The check fails, and lists the passages that were not found, if a number in
the paper no longer matches the code. If the paper's wording changes, update
the corresponding passage below.

Input:  output/tables/*.csv, data/processed/clean.parquet, paper/main.tex
Output: a report on standard output; exits with an error if any check fails
"""

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "output" / "tables"
PROCESSED = ROOT / "data" / "processed"
PAPER = ROOT / "paper" / "main.tex"

PM, EM = "political_model", "econ_model"
CHINA_TERMS = ["china_econ1", "china_econ2", "china_pol1"]
US_TERMS = ["us_econ1", "us_econ2", "us_pol1"]


def squash(text):
    """Collapse runs of whitespace so that line breaks and alignment do not matter."""
    return re.sub(r"\s+", " ", text)


TEX = squash(PAPER.read_text())
RESULTS = []  # (section, passage or claim, passed)


def passage(section, text):
    """Record whether `text`, built from the outputs, appears verbatim in the paper."""
    RESULTS.append((section, text, squash(text) in TEX))


def claim(section, description, condition):
    """Record a check of a claim that is not a single verbatim passage."""
    RESULTS.append((section, description, bool(condition)))


# Number formats used in the paper ----------------------------------------------

def num(x, digits=2):
    """A number in a table cell, with a LaTeX minus sign."""
    text = f"{x:.{digits}f}"
    return "$-$" + text[1:] if text.startswith("-") else text


def est(x, se, digits=3):
    """Estimate with standard error in parentheses."""
    return f"{num(x, digits)} ({se:.{digits}f})"


def pcell(p):
    """Adjusted p-value in a table cell."""
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def count(n):
    """A count with thousands separators, e.g. 6,276."""
    return f"{int(n):,}"


def vid(name):
    """A video name in the paper's `\\vid{}` macro, with underscores escaped."""
    return "\\vid{" + name.replace("_", "\\_") + "}"


def read(name):
    """Read one of the output tables by its name, without the .csv extension."""
    return pd.read_csv(TABLES / f"{name}.csv")


def main():
    """Rebuild each reported number from the outputs and check it against the paper."""
    flow = read("sample_flow").set_index("stage")["n"]
    coef = read("primary_coefficients")
    table_a3 = read("table_a3_check")
    standardized = (read("standardized_effects")
                    .set_index(["outcome", "term"])["standardized_effect"])
    share = (read("share_prefer_china")
             .set_index(["outcome", "treatment"])["share_prefer_china"])
    assignment = read("ext_assignment")
    independence = read("ext_competition_independence").iloc[0]
    videos = read("ext_competition_videos")
    joint = read("ext_competition_joint_tests")
    ranges = read("ext_competition_range").set_index("outcome")
    dispersion = read("ext_competition_dispersion").set_index(["outcome", "measure"])
    types = read("ext_competition_types")
    content = read("ext_content_type")
    position = read("ext_companion_position")
    companion_mech = read("ext_companion_mechanisms")
    mech_content = read("ext_mechanisms_content")
    baseline = read("ext_mechanisms_baseline").set_index("outcome")
    mech_videos = read("ext_mechanisms_videos")
    index = read("ext_mechanisms_index")
    index_loo = read("ext_mechanisms_index_loo")
    balance_means = read("ext_balance_means")
    balance_tests = read("ext_balance_tests")
    authors = read("ext_authors_video_type")
    df = pd.read_parquet(PROCESSED / "clean.parquet")
    arms = df["treatment"].value_counts()

    def c(outcome, adjusted, term, col="estimate"):
        row = coef[(coef.outcome == outcome) & (coef.adjusted == adjusted)
                   & (coef.term == term)]
        return row.iloc[0][col]

    def v(outcome, term, col="estimate"):
        return videos[(videos.outcome == outcome) & (videos.term == term)].iloc[0][col]

    def jt(outcome, country, col):
        return joint[(joint.outcome == outcome) & (joint.videos == country)].iloc[0][col]

    def ct(outcome, source, design, col="estimate"):
        row = content[(content.outcome == outcome) & (content.source == source)
                      & (content.design == design)]
        return row.iloc[0][col]

    # Abstract and introduction --------------------------------------------------
    s = "Abstract/Intro"
    claim(s, "all 24 Table A3 cells match the published values",
          len(table_a3) == 24 and table_a3["match"].all())
    passage(s, f"differ by {jt(PM, 'china', 'range'):.1f} points on preference for China's "
               "political model and by nearly one point")
    claim(s, "'nearly one point' (economic-model range)", 0.9 <= jt(EM, "china", "range") < 1.0)
    passage(s, f"showed {count(flow['Randomized'])} respondents in "
               f"{df['country'].nunique()} countries")
    passage(s, f"from {share[(PM, 'Control')] * 100:.0f} to "
               f"{share[(PM, 'China')] * 100:.0f} percent")
    claim(s, "US effects smaller than China effects",
          all(abs(c(o, True, "USA")) < c(o, True, "China") for o in (PM, EM)))
    claim(s, "Competition effect favors China, p < 0.05",
          all(c(o, True, "Competition") > 0 and c(o, True, "Competition", "p_value") < 0.05
              for o in (PM, EM)))

    # Section 2 ------------------------------------------------------------------
    s = "Sec 2"
    passage(s, f"political model by {c(PM, True, 'China'):.2f} points and for its economic "
               f"model by {c(EM, True, 'China'):.2f} points. The US treatment lowers them by "
               f"{-c(PM, True, 'USA'):.2f} and {-c(EM, True, 'USA'):.2f} points. When the two "
               f"compete, the net effect still favors China ({c(PM, True, 'Competition'):.2f} "
               f"and {c(EM, True, 'Competition'):.2f} points)")

    # Section 3 and Tables 1-2 ---------------------------------------------------
    s = "Sec 3"
    passage(s, f"arm sizes ({count(arms['Control'])}, {count(arms['China'])}, "
               f"{count(arms['USA'])}, and {count(arms['Competition'])})")
    passage(s, f"contains {count(flow['Total'])} respondents, of whom "
               f"{count(flow['Randomized'])} were randomized")
    reached = (flow["Failed attention check"] + flow["Passed check, left before randomization"]
               + flow["Randomized"])
    claim(s, "more than two in five who completed the checks failed them",
          flow["Failed attention check"] / reached > 0.4)
    claim(s, "no unclassified respondents; flow sums to total",
          flow["Unclassified"] == 0 and flow.drop("Total").sum() == flow["Total"])
    stages = [
        ("Screened out: not a citizen of the survey country", "Screened out: not a citizen"),
        ("Left before the attention checks", "Left before attention check"),
        ("Failed an attention check", "Failed attention check"),
        ("Passed the checks but left before randomization",
         "Passed check, left before randomization"),
    ]
    for label, key in stages:
        passage("Table 1", f"{label} & {count(flow[key])} \\\\")
    passage("Table 1", f"\\textbf{{Randomized}} & \\textbf{{{count(flow['Randomized'])}}}")
    passage("Table 1", f"Total in the authors' cleaned dataset & {count(flow['Total'])}")
    for term in ["China", "USA", "Competition", "Constant"]:
        cells = [f"{num(c(o, adj, term))} ({c(o, adj, term, 'std_error'):.2f})"
                 for adj in (True, False) for o in (PM, EM)]
        passage("Table 2", f"{term} & " + " & ".join(cells) + " \\\\")
    sizes = [count(c(o, adj, "China", "n")) for adj in (True, False) for o in (PM, EM)]
    passage("Table 2", "$N$ & " + " & ".join(sizes))
    passage(s, f"from {c(PM, False, 'China'):.3f} to {c(PM, True, 'China'):.3f}")
    claim(s, "covariates shrink every standard error",
          all(c(o, True, t, "std_error") < c(o, False, t, "std_error")
              for o in (PM, EM, "world_leader") for t in ("China", "USA", "Competition")))
    passage(s, f"by {standardized[(PM, 'China')]:.2f} standard deviations and for its "
               f"economic model by {standardized[(EM, 'China')]:.2f}")
    passage(s, f"rises from {share[(PM, 'Control')] * 100:.1f} percent in the control group to "
               f"{share[(PM, 'China')] * 100:.1f} percent in the China arm")

    # Table 3: effects of each video -------------------------------------------
    for term in CHINA_TERMS + US_TERMS:
        passage("Table 3", f"{num(v(PM, term))} / {num(v(EM, term))} \\\\")
    claim("Sec 4.2", "china_econ2 weaker than china_econ1 on both outcomes",
          all(v(o, "china_econ2") < v(o, "china_econ1") for o in (PM, EM)))

    # Section 4.3 and Table 4 ----------------------------------------------------
    s = "Sec 4.3"
    pairs = assignment.set_index(["arm", "video1"])
    china_rows = [("china_econ1", "china_pol2"), ("china_econ2", "china_pol2"),
                  ("china_pol1", "china_pol2"), ("china_pol2", "china_pol1")]
    us_rows = [("us_econ1", "us_pol2"), ("us_econ2", "us_pol2"),
               ("us_pol1", "us_pol2"), ("us_pol2", "us_pol1")]
    for (cn1, cn2), (us1, us2) in zip(china_rows, us_rows):
        rc, ru = pairs.loc[("China", cn1)], pairs.loc[("USA", us1)]
        claim("Table 4", f"{cn1} is paired with {cn2}; {us1} with {us2}",
              rc.video2 == cn2 and ru.video2 == us2)
        passage("Table 4", f"{vid(cn1)} & {vid(cn2)} & {rc.n} & & "
                           f"{vid(us1)} & {vid(us2)} & {ru.n} \\\\")
    claim("Table 4", "no other pairs occur", len(assignment) == 8)
    passage(s, f"each with {int(independence.min_cell)} to {int(independence.max_cell)} "
               "respondents")
    passage(s, f"($\\chi^2 = {independence.chi2:.2f}$, $df = {int(independence.df)}$, "
               f"$p = {independence.p_value:.2f}$)")
    means_tests = balance_tests[balance_tests.test.str.startswith("equal means")]
    passage(s, f"$p$-values between {means_tests.p_value.min():.2f} and "
               f"{means_tests.p_value.max():.2f}, with one of {len(means_tests)} below 0.10")
    claim(s, "exactly one balance p-value below 0.10", (means_tests.p_value < 0.10).sum() == 1)
    country_tests = balance_tests[balance_tests.test == "country independent of video"]
    passage(s, f"$p$-values of {country_tests.p_value.min():.2f} or higher")
    omnibus = balance_tests[balance_tests.test == "multinomial logit omnibus"]
    passage(s, f"omnibus $p$-values between {omnibus.p_value.min():.2f} and "
               f"{omnibus.p_value.max():.2f}")
    missing = balance_means[["missing_political_model", "missing_econ_model"]].to_numpy()
    passage(s, f"missing for {missing.min() * 100:.1f} to {missing.max() * 100:.1f} percent")
    missing_tests = balance_tests[balance_tests.test.str.startswith("missing")]
    passage(s, f"($p$-values between {missing_tests.p_value.min():.2f} and "
               f"{missing_tests.p_value.max():.2f})")
    passage("Intro", "the observed assignment frequencies and balance tests are consistent with "
                     "this (Section~\\ref{sec:assignment})")

    # Section 4.5 ----------------------------------------------------------------
    s = "Sec 4.5"
    china_pm = [v(PM, t) for t in CHINA_TERMS]
    china_em = [v(EM, t) for t in CHINA_TERMS]
    passage(s, f"by between {min(china_pm):.2f} and {max(china_pm):.2f} points, and for its "
               f"economic model by between {min(china_em):.2f} and {max(china_em):.2f} points")
    passage(s, f"political model ($\\chi^2_3 = {jt(PM, 'china', 'chi2'):.1f}$) and the economic "
               f"model ($\\chi^2_3 = {jt(EM, 'china', 'chi2'):.1f}$; both $p < 0.001$)")
    claim(s, "both CGTN joint tests p < 0.001",
          jt(PM, "china", "p_value") < 0.001 and jt(EM, "china", "p_value") < 0.001)
    claim(s, "china_pol2 is the weakest CGTN video",
          all(v(o, t) > 0 for o in (PM, EM) for t in CHINA_TERMS))
    passage(s, f"The ranges, {jt(PM, 'china', 'range'):.2f} and {jt(EM, 'china', 'range'):.2f} "
               "points, are larger than the Competition arm's overall effects on the two "
               f"outcomes ({c(PM, True, 'Competition'):.2f} and "
               f"{c(EM, True, 'Competition'):.2f})")
    passage(s, f"the range is as large as the China arm's effect ({c(EM, True, 'China'):.2f})")
    claim(s, "range bias about 0.01 points or less", ranges.bias.abs().max() <= 0.0105)
    passage(s, f"intervals of {ranges.loc[PM, 'ci_low']:.2f} to {ranges.loc[PM, 'ci_high']:.2f} "
               f"and {ranges.loc[EM, 'ci_low']:.2f} to {ranges.loc[EM, 'ci_high']:.2f}")
    passage(s, f"{vid('china_pol1')} ({v(PM, 'china_pol1'):.2f}), is about as persuasive about "
               f"the political model as the economic video {vid('china_econ2')} "
               f"({v(PM, 'china_econ2'):.2f})")
    passage(s, f"(joint $p = {jt(PM, 'us', 'p_value'):.2f}$; "
               f"range {jt(PM, 'us', 'range'):.2f} points)")
    passage(s, f"($p = {jt(EM, 'us', 'p_value'):.3f}$; "
               f"Holm-adjusted $p = {jt(EM, 'us', 'p_holm'):.3f}$)")
    claim(s, "both US economic videos below us_pol2 on the economic model",
          v(EM, "us_econ1") < 0 and v(EM, "us_econ2") < 0)
    pr, er = dispersion.loc[(PM, "range")], dispersion.loc[(EM, "range")]
    passage(s, f"exceeds that of the four State Department videos by {pr.difference:.2f} points "
               f"on the political model (basic 95 percent bootstrap interval {pr.ci_low:.2f} to "
               f"{pr.ci_high:.2f}) and by {er.difference:.2f} points on the economic model "
               f"({er.ci_low:.2f} to {er.ci_high:.2f})")
    claim(s, "CGTN effects more dispersed than US effects: range and SD intervals exclude zero",
          (dispersion.ci_low > 0).all())
    claim(s, "range of each country's effects matches the joint-test table",
          all(abs(dispersion.loc[(o, "range"), k] - jt(o, k, "range")) < 1e-9
              for o in (PM, EM) for k in ("china", "us")))

    # Section 4.6, Table 5, Table A4 --------------------------------------------
    s = "Sec 4.6"
    passage(s, f"by {ct(EM, 'CGTN', 'China arm'):.2f} points in the China arm "
               f"(SE {ct(EM, 'CGTN', 'China arm', 'std_error'):.2f}) "
               f"and by {ct(EM, 'CGTN', 'Competition arm'):.2f} points in the Competition arm "
               f"(SE {ct(EM, 'CGTN', 'Competition arm', 'std_error'):.2f})")
    claim(s, "both CGTN economic-model contrasts survive Holm",
          all(ct(EM, "CGTN", d, "p_holm") < 0.05 for d in ("China arm", "Competition arm")))
    passage(s, f"the differences are {ct(PM, 'CGTN', 'China arm'):.2f} and "
               f"{ct(PM, 'CGTN', 'Competition arm'):.2f} points")
    half = [1.959964 * ct(PM, "CGTN", d, "std_error") for d in ("China arm", "Competition arm")]
    passage(s, f"roughly $\\pm {half[0]:.2f}$ and $\\pm {half[1]:.2f}$")
    passage(s, f"Given MDEs of {ct(PM, 'CGTN', 'China arm', 'mde'):.2f} and "
               f"{ct(PM, 'CGTN', 'Competition arm', 'mde'):.2f}")
    lows = [-ct(PM, "CGTN", d, "ci_low") for d in ("China arm", "Competition arm")]
    passage(s, f"disadvantage larger than about {max(lows):.2f} points in either design")
    designs = [("CGTN", "China arm", "CGTN & China arm"),
               ("CGTN", "Competition arm", "CGTN & Competition arm"),
               ("State Department", "USA arm", "State Dept. & USA arm"),
               ("State Department", "Competition arm", "State Dept. & Competition arm")]
    for source, design, label in designs:
        cells = [f"{est(ct(o, source, design), ct(o, source, design, 'std_error'))} & "
                 f"{pcell(ct(o, source, design, 'p_holm'))}"
                 for o in (PM, EM)]
        passage("Table 5", f"{label} & " + " & ".join(cells) + " \\\\")
    single_n = content[content.design.isin(["China arm", "USA arm"])].n
    passage("Table 5", f"$N \\approx {round(single_n.min(), -1):,.0f}$--"
                       f"${round(single_n.max(), -1):,.0f}$".replace(",", "{,}"))
    competition_n = content[content.design == "Competition arm"].n
    passage("Table 5",
            f"($N \\approx {round(competition_n.mean(), -1):,.0f}$)".replace(",", "{,}"))
    passage(s, f"{v(PM, 'china_econ2'):.2f} and {v(PM, 'china_econ1'):.2f} points "
               "(Table~\\ref{tab:videos})")
    passage(s, f"({ct(EM, 'State Department', 'USA arm'):.2f} in the USA arm and "
               f"${ct(EM, 'State Department', 'Competition arm'):.2f}$ in the Competition arm)")
    claim(s, "no State Department contrast distinguishable from zero",
          (content[content.source == "State Department"].p_value > 0.05).all())
    state_se = [ct(EM, "State Department", d, "std_error") for d in ("USA arm", "Competition arm")]
    passage(s, f"standard errors of {min(state_se):.2f} to {max(state_se):.2f}")

    def pz(arm, outcome, start, col="estimate"):
        row = position[(position.arm == arm) & (position.outcome == outcome)
                       & position.term.str.startswith(start)]
        return row.iloc[0][col]

    econ, order = "Economic companion", "pol2 first"
    passage(s, f"is ${pz('China', PM, econ):.2f}$ (SE {pz('China', PM, econ, 'std_error'):.2f}) "
               f"on the political model and {pz('China', EM, econ):.2f} "
               f"(SE {pz('China', EM, econ, 'std_error'):.2f}) on the economic model")
    passage(s, f"(${pz('China', PM, order):.2f}$, SE {pz('China', PM, order, 'std_error'):.2f}, "
               f"and ${pz('China', EM, order):.2f}$, "
               f"SE {pz('China', EM, order, 'std_error'):.2f})")
    claim(s, "no order effect distinguishable from zero",
          (position[position.term.str.startswith(order)].p_value > 0.05).all())
    order_se = position[(position.arm == "China") & position.term.str.startswith(order)].std_error
    passage(s, f"an order effect of about {2.8016 * order_se.max():.1f} points could go undetected")
    for arm in ("China", "USA"):
        for start, label in [(econ, "Economic vs.\\ \\vid{pol1}, \\vid{pol2} second for all"),
                             (order, "\\vid{pol2} first vs.\\ second, political pairs")]:
            passage("Table A4", f"{arm} & {label} & "
                                f"{est(pz(arm, PM, start), pz(arm, PM, start, 'std_error'))} & "
                                f"{est(pz(arm, EM, start), pz(arm, EM, start, 'std_error'))} & "
                                f"{count(pz(arm, PM, start, 'n'))} \\\\")

    # Section 4.7 ----------------------------------------------------------------
    s = "Sec 4.7"

    def mc(outcome, design, col="estimate"):
        row = mech_content[(mech_content.outcome == outcome) & (mech_content.design == design)]
        return row.iloc[0][col]

    growth, competent = "mech_growth_china", "mech_competent_china"
    passage(s, f"the differences are ${mc(growth, 'China arm'):.2f}$ "
               f"(SE {mc(growth, 'China arm', 'std_error'):.2f}) and "
               f"${mc(growth, 'Competition arm'):.2f}$ "
               f"(SE {mc(growth, 'Competition arm', 'std_error'):.2f})")
    claim(s, "growth is the most widely held belief in the control group",
          baseline["mean"].idxmax() == growth and baseline["share_agree"].idxmax() == growth)
    passage(s, f"(mean {baseline.loc[growth, 'mean']:.2f}; "
               f"{baseline.loc[growth, 'share_agree'] * 100:.0f} percent agree or strongly agree)")
    us_growth = companion_mech[(companion_mech.arm == "USA")
                               & (companion_mech.outcome == "mech_growth_us")].iloc[0]
    passage(s, f"(mean {us_growth.control_mean:.2f})")
    passage(s, f"raises them by {us_growth.estimate:.2f} points (SE {us_growth.std_error:.2f}; "
               f"$p = {us_growth.p_value:.3f}$, or {us_growth.p_holm:.2f} after Holm correction "
               "across the eight items)")
    passage(s, f"by {-mc(competent, 'China arm'):.2f} "
               f"(SE {mc(competent, 'China arm', 'std_error'):.2f}) and "
               f"{-mc(competent, 'Competition arm'):.2f} "
               f"(SE {mc(competent, 'Competition arm', 'std_error'):.2f}) points")
    claim(s, "only the competence contrasts survive Holm across 16 tests",
          len(mech_content) == 16
          and (mech_content[mech_content.outcome == competent].p_holm < 0.05).all()
          and (mech_content[mech_content.outcome != competent].p_holm >= 0.05).all())
    growth_videos = mech_videos[mech_videos.outcome == growth]
    passage(s, f"by {growth_videos.estimate.min():.2f} to "
               f"{growth_videos.estimate.max():.2f} points")
    claim(s, "all three videos raise growth beliefs relative to china_pol2",
          (growth_videos.estimate > 0).all() and (growth_videos.p_value < 0.05).all())

    equal = index[index["index"] == "Equally weighted"].set_index("term")
    inverse = index[index["index"] == "Inverse-covariance weighted"].set_index("term")
    average = "average of other three"
    passage(s, f"(an average correlation of {equal.mean_item_correlation.iloc[0]:.2f} "
               "in the control group)")
    passage(s, f"score {equal.loc[average, 'estimate']:.2f} standard deviations higher than "
               f"{vid('china_pol2')} on average (SE {equal.loc[average, 'std_error']:.2f}; "
               f"{inverse.loc[average, 'estimate']:.2f} with inverse-covariance weights)")
    largest_p = max(equal.loc[CHINA_TERMS, "p_value"].max(),
                    inverse.loc[CHINA_TERMS, "p_value"].max())
    claim(s, "each video individually higher at p < 0.01", largest_p < 0.01)
    passage(s, f"at $p < 0.01$ (largest $p = {largest_p:.3f}$;")
    loo = index_loo[index_loo["index"] == "Equally weighted"]
    loo_average = loo[loo.term == average]
    loo_videos = loo[loo.term != average]
    passage(s, f"leaves this gap between {loo_average.estimate.min():.2f} and "
               f"{loo_average.estimate.max():.2f} standard deviations")
    claim(s, "leave-one-out: all gaps p < 0.001, each video p < 0.05",
          (loo_average.p_value < 0.001).all() and (loo_videos.estimate > 0).all()
          and (loo_videos.p_value < 0.05).all())
    claim(s, "china_pol1 far more persuasive than china_pol2",
          all(v(o, "china_pol1", "p_value") < 0.001 for o in (PM, EM)))

    # Discussion -----------------------------------------------------------------
    s = "Discussion"
    passage(s, f"differ by about {jt(PM, 'china', 'range'):.1f} points on the political model "
               "and nearly 1 point")
    passage(s, f"MDEs of {content.mde.min():.2f} to {content.mde.max():.2f} points")

    # Table A1 -------------------------------------------------------------------
    for term in CHINA_TERMS + US_TERMS:
        cells = []
        for o in (PM, EM):
            row = videos[(videos.outcome == o) & (videos.term == term)].iloc[0]
            cells.append(f"{est(row.estimate, row.std_error)} & {pcell(row.p_holm)} & "
                         f"{pcell(row.p_bh)}")
        passage("Table A1", f"{vid(term)} & " + " & ".join(cells) + " \\\\")

    def joint_cell(outcome, country, chi_digits, p_digits=2):
        p = jt(outcome, country, "p_value")
        p_text = "$p < 0.001$" if p < 0.001 else f"$p = {p:.{p_digits}f}$"
        return (f"\\multicolumn{{3}}{{c}}{{$\\chi^2_3 = "
                f"{jt(outcome, country, 'chi2'):.{chi_digits}f}$, {p_text}}}")

    passage("Table A1", f"Joint test, CGTN & {joint_cell(PM, 'china', 1)} & "
                        f"{joint_cell(EM, 'china', 1)}")
    passage("Table A1", f"Joint test, US & {joint_cell(PM, 'us', 2)} & "
                        f"{joint_cell(EM, 'us', 1, 3)}")
    passage("Table A1", f"$N$ & \\multicolumn{{3}}{{c}}{{{count(v(PM, 'china_econ1', 'n'))}}} & "
                        f"\\multicolumn{{3}}{{c}}{{{count(v(EM, 'china_econ1', 'n'))}}}")

    # Table A2 -------------------------------------------------------------------
    def ty(outcome, term, col="estimate"):
        return types[(types.outcome == outcome) & (types.term == term)].iloc[0][col]

    type_rows = [("China economic video", "Chinese video is economic"),
                 ("US economic video", "US video is economic")]
    for term, label in type_rows:
        passage("Table A2", f"{label} & {est(ty(PM, term), ty(PM, term, 'std_error'))} & "
                            f"{pcell(ty(PM, term, 'p_holm'))} & "
                            f"{est(ty(EM, term), ty(EM, term, 'std_error'))} & "
                            f"{pcell(ty(EM, term, 'p_holm'))} \\\\")
    inter = "China econ x US econ"
    passage("Table A2", f"Chinese econ.\\ $\\times$ US econ. & "
                        f"{est(ty(PM, inter), ty(PM, inter, 'std_error'))} & --- & "
                        f"{est(ty(EM, inter), ty(EM, inter, 'std_error'))} & --- \\\\")
    passage("Table A2", f"are {ty(PM, inter, 'p_value'):.2f} and {ty(EM, inter, 'p_value'):.2f}")
    type_n = [count(ty(o, "China economic video", "n")) for o in (PM, EM)]
    passage("Table A2", f"$N$ & \\multicolumn{{2}}{{c}}{{{type_n[0]}}} & "
                        f"\\multicolumn{{2}}{{c}}{{{type_n[1]}}}")

    # Table A3 -------------------------------------------------------------------
    # Item labels and their order come from 03_extension.py, via the baseline table.
    for outcome, label in baseline["label"].items():
        rows = mech_videos[mech_videos.outcome == outcome].set_index("term")
        cells = [est(rows.loc[t, "estimate"], rows.loc[t, "std_error"], 2) for t in CHINA_TERMS]
        passage("Table A3", f"{label} & " + " & ".join(cells) + " \\\\")
    for table, label in [(equal, "Index, equally weighted"),
                         (inverse, "Index, inverse-covariance weighted")]:
        cells = [est(table.loc[t, "estimate"], table.loc[t, "std_error"], 2) for t in CHINA_TERMS]
        passage("Table A3", f"{label} & " + " & ".join(cells) + " \\\\")
    significant = (mech_videos.p_holm < 0.05).sum()
    words = {5: "Five", 6: "Six", 7: "Seven"}
    passage("Table A3", f"{words.get(significant, significant)} of the {len(mech_videos)} "
                        "item-level coefficients have Holm-adjusted $p < 0.05$ across all "
                        f"{len(mech_videos)}")
    passage("Table A3", "($N \\approx 1{,}455$)")
    claim("Table A3", "item-level N between 1,450 and 1,460",
          mech_videos.n.between(1450, 1460).all())
    passage("Table A3", f"use the {count(equal.n.iloc[0])} respondents with all eight items")
    passage("Table A3", f"give $\\chi^2_3 = {equal.joint_chi2.iloc[0]:.1f}$ and "
                        f"{inverse.joint_chi2.iloc[0]:.1f} (both $p < 0.001$)")
    claim("Table A3", "both index joint tests p < 0.001",
          equal.joint_p.iloc[0] < 0.001 and inverse.joint_p.iloc[0] < 0.001)

    # Table A5 -------------------------------------------------------------------
    groupings = ["China arm", "USA arm", "Competition arm, Chinese video",
                 "Competition arm, US video"]
    covariates = ["gender", "age", "education", "national_pride", "leftright"]
    for grouping in groupings:
        for _, row in balance_means[balance_means.grouping == grouping].iterrows():
            passage("Table A5", f"{vid(row.video)} & {row.n} & {row.gender:.2f} & "
                                f"{row.age:.1f} & {row.education:.2f} & "
                                f"{row.national_pride:.2f} & {row.leftright:.2f} & "
                                f"{row.missing_political_model * 100:.1f} \\\\")
        tests = balance_tests[balance_tests.grouping == grouping].set_index("test").p_value
        cells = ([tests[f"equal means: {cov}"] for cov in covariates]
                 + [tests["missing political_model independent of video"]])
        passage("Table A5", "\\quad $p$, equal across videos & & "
                            + " & ".join(f"{p:.2f}" for p in cells) + " \\\\")

    def p_list(test):
        """The p-values of one balance test, for each grouping in order."""
        tests = balance_tests[balance_tests.test == test].set_index("grouping").p_value
        return [tests[g] for g in groupings]

    def math_p(p):
        return "$>0.99$" if p > 0.99 else f"${p:.2f}$"

    country = p_list("country independent of video")
    passage("Table A5", f"give $p = {country[0]:.2f}$, {math_p(country[1])}, "
                        f"{math_p(country[2])}, and {math_p(country[3])} "
                        "for the four groupings in order")
    lower = p_list("country independent of video, lower tail")
    passage("Table A5", "In the USA arm, the videos are spread across countries more evenly "
                        "than independent draws would usually produce "
                        f"(lower-tail $p = {lower[1]:.3f}$)")
    claim("Table A5", "only the USA arm has an unusually even spread",
          lower[1] < 0.01 and min(lower[0], lower[2], lower[3]) > 0.1)
    passage("Table A5", "omnibus tests of all covariates and country dummies give $p = "
                        + ", ".join(f"{p:.2f}" for p in p_list("multinomial logit omnibus"))
                        + "$")
    passage("Table A5", "Missingness of the economic-model outcome gives $p = "
                        + ", ".join(f"{p:.2f}"
                                    for p in p_list("missing econ_model independent of video"))
                        + "$")
    claim("Table A5", "video group sizes sum to arm sizes",
          balance_means[balance_means.grouping == "China arm"].n.sum() == arms["China"]
          and balance_means[balance_means.grouping == "USA arm"].n.sum() == arms["USA"]
          and all(balance_means[balance_means.grouping == g].n.sum() == arms["Competition"]
                  for g in groupings[2:]))

    # Appendix B and Table A6 ----------------------------------------------------
    def au(outcome, spec, term, col="estimate"):
        row = authors[(authors.outcome == outcome) & (authors.specification == spec)
                      & (authors.term == term)]
        return row.iloc[0][col]

    s = "App. B"
    passage(s, f"({au(EM, 'pooled', 'china_econ - china_pol'):.3f} on the economic model) is "
               "close to the China-arm estimate in Table~\\ref{tab:content} "
               f"({ct(EM, 'CGTN', 'China arm'):.3f})")
    passage(s, f"(${au(PM, 'pooled', 'us_econ'):.3f}$ and ${au(PM, 'pooled', 'us_pol'):.3f}$), "
               "whereas the USA arm alone shows effects of "
               f"${au(PM, 'by_arm', 'us_econ'):.2f}$ and ${au(PM, 'by_arm', 'us_pol'):.2f}$")
    claim(s, "US economic vs. political not significant in either specification",
          all(au(o, spec, "us_econ - us_pol", "p_value") > 0.05
              for o in (PM, EM) for spec in ("pooled", "by_arm")))
    rows = [("china_econ", "China economic"), ("china_pol", "China political"),
            ("us_econ", "US economic"), ("us_pol", "US political"),
            ("competition_us_econ", "Competition: US econ."),
            ("competition_us_pol", "Competition: US pol."),
            ("china_econ - china_pol", "China econ.\\ $-$ China pol."),
            ("us_econ - us_pol", "US econ.\\ $-$ US pol.")]
    for term, label in rows:
        cells = []
        for o in (PM, EM):
            for spec in ("pooled", "by_arm"):
                row = authors[(authors.outcome == o) & (authors.specification == spec)
                              & (authors.term == term)]
                cells.append("---" if row.empty
                             else est(row.iloc[0].estimate, row.iloc[0].std_error))
        passage("Table A6", f"{label} & " + " & ".join(cells) + " \\\\")
    authors_n = [count(au(o, "pooled", "china_econ", "n")) for o in (PM, EM)]
    passage("Table A6", f"$N$ & \\multicolumn{{2}}{{c}}{{{authors_n[0]}}} & "
                        f"\\multicolumn{{2}}{{c}}{{{authors_n[1]}}}")

    # Report ---------------------------------------------------------------------
    failed = [r for r in RESULTS if not r[2]]
    print(f"Checked {len(RESULTS)} passages and claims in paper/main.tex: "
          f"{len(RESULTS) - len(failed)} passed, {len(failed)} failed.")
    for section, text, _ in failed:
        print(f"  NOT FOUND [{section}]: {text}")
    if failed:
        sys.exit("The paper does not match the pipeline output; see the passages above.")


if __name__ == "__main__":
    main()
