# Generate LaTeX values from saved tables and check selected manuscript claims.

import pandas as pd

from analysis_utils import ROOT, SINGLE_ARMS, TABLES, check

OUT = ROOT / "paper" / "numbers.tex"
PM, EM = "political_model", "econ_model"
CHINA_TERMS = ["china_econ1", "china_econ2", "china_pol1"]
US_TERMS = ["us_econ1", "us_econ2", "us_pol1"]
GROUPINGS = ["China arm", "USA arm", "Competition arm, Chinese video",
             "Competition arm, US video"]
BALANCE = ["gender", "age", "education", "national_pride", "leftright"]


def num(x, digits=2):
    text = f"{x:.{digits}f}"
    return "$-$" + text[1:] if text.startswith("-") else text


def est(x, se, digits=3):
    return f"{num(x, digits)} ({se:.{digits}f})"


def pcell(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def count(n):
    return f"{int(n):,}"


def vid(name):
    return "\\vid{" + name.replace("_", "\\_") + "}"


def rows(lines):
    return "\n".join(" & ".join(cells) + " \\\\" for cells in lines)


def read(name):
    return pd.read_csv(TABLES / f"{name}.csv")


def pick(table, **conditions):
    match = table
    for column, value in conditions.items():
        match = match[match[column] == value]
    check(len(match) == 1, f"Expected one row for {conditions}, found {len(match)}")
    return match.iloc[0]


def values():
    V = {}
    flow = read("sample_flow").set_index("stage")["n"]
    coef = read("primary_coefficients")
    std = read("standardized_effects").set_index(["outcome", "term"])["standardized_effect"]
    share = read("share_prefer_china").set_index(["outcome", "treatment"])["share_prefer_china"]
    assignment = read("ext_assignment")
    indep = read("ext_competition_independence").iloc[0]
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

    c = lambda o, adj, term: pick(coef, outcome=o, adjusted=adj, term=term)
    v = lambda o, term: pick(videos, outcome=o, term=term)
    jt = lambda o, country: pick(joint, outcome=o, videos=country)
    ct = lambda o, source, design: pick(content, outcome=o, source=source, design=design)

    # Sample and replication
    check(flow["Unclassified"] == 0 and flow.drop("Total").sum() == flow["Total"],
          "Sample flow does not add up")
    reached = (flow["Failed attention check"] + flow["Passed check, left before randomization"]
               + flow["Randomized"])
    check(flow["Failed attention check"] / reached > 0.4, "'more than two in five' failed")
    V["n.total"] = count(flow["Total"])
    V["n.randomized"] = count(flow["Randomized"])
    V["flow.citizen"] = count(flow["Screened out: not a citizen"])
    V["flow.left"] = count(flow["Left before attention check"])
    V["flow.failed"] = count(flow["Failed attention check"])
    V["flow.passedleft"] = count(flow["Passed check, left before randomization"])

    check(read("table_a3_check")["match"].all(), "Table A3 mismatch")
    check(all(c(o, True, t).std_error < c(o, False, t).std_error
              for o in (PM, EM, "world_leader") for t in ("China", "USA", "Competition")),
          "Covariates do not shrink every standard error")
    check(all(abs(c(o, True, "USA").estimate) < c(o, True, "China").estimate for o in (PM, EM)),
          "US effects not smaller than China effects")
    for o, key in ((PM, "pm"), (EM, "em")):
        V[f"adj.china.{key}"] = num(c(o, True, "China").estimate)
        V[f"adj.usa.{key}.neg"] = num(-c(o, True, "USA").estimate)
        V[f"adj.comp.{key}"] = num(c(o, True, "Competition").estimate)
        check(c(o, True, "Competition").p_value < 0.05, "Competition effect not significant")
        V[f"std.china.{key}"] = num(std[(o, "China")])
    V["adj.china.pm.3"] = num(c(PM, True, "China").estimate, 3)
    V["unadj.china.pm.3"] = num(c(PM, False, "China").estimate, 3)
    V["share.pm.control"] = f"{share[(PM, 'Control')] * 100:.1f}"
    V["share.pm.china"] = f"{share[(PM, 'China')] * 100:.1f}"
    V["share.pm.control.0"] = f"{share[(PM, 'Control')] * 100:.0f}"
    V["share.pm.china.0"] = f"{share[(PM, 'China')] * 100:.0f}"

    V["tab.replication"] = rows(
        [[term] + [f"{num(c(o, adj, term).estimate)} ({c(o, adj, term).std_error:.2f})"
                   for adj in (True, False) for o in (PM, EM)]
         for term in ("China", "USA", "Competition", "Constant")])
    V["tab.replication.n"] = " & ".join(count(c(o, adj, "China").n)
                                        for adj in (True, False) for o in (PM, EM))

    # Assignment and balance
    pairs = [("econ1", "pol2"), ("econ2", "pol2"), ("pol1", "pol2"), ("pol2", "pol1")]
    check(len(assignment) == 8, "Unexpected video pairs in the single-country arms")
    lines = []
    for first, second in pairs:
        cells = []
        for arm, prefix in SINGLE_ARMS.items():
            row = pick(assignment, arm=arm, video1=f"{prefix}_{first}")
            check(row.video2 == f"{prefix}_{second}", f"{row.video1} not paired with {second}")
            cells += [vid(row.video1), vid(row.video2), str(row.n)]
        lines.append(cells[:3] + [""] + cells[3:])
    V["tab.assignment"] = rows(lines)
    V["indep.min"], V["indep.max"] = int(indep.min_cell), int(indep.max_cell)
    V["indep.chi2"], V["indep.df"] = f"{indep.chi2:.2f}", int(indep.df)
    V["indep.p"] = f"{indep.p_value:.2f}"

    means_tests = balance_tests[balance_tests.test.str.startswith("equal means")]
    check(len(means_tests) == 20 and (means_tests.p_value < 0.10).sum() == 1,
          "'one of 20 below 0.10' no longer holds")
    V["bal.means.pmin"] = f"{means_tests.p_value.min():.2f}"
    V["bal.means.pmax"] = f"{means_tests.p_value.max():.2f}"

    def p_list(test):
        tests = balance_tests[balance_tests.test == test].set_index("grouping").p_value
        return [tests[g] for g in GROUPINGS]

    country = p_list("country independent of video")
    V["bal.country.pmin"] = f"{min(country):.2f}"
    cells = [f"{p:.2f}" if p <= 0.99 else ">0.99" for p in country]
    V["bal.country.list"] = (f"$p = {cells[0]}$, ${cells[1]}$, ${cells[2]}$, "
                             f"and ${cells[3]}$")
    lower = p_list("country independent of video, lower tail")
    check(lower[1] < 0.01 and min(lower[0], lower[2], lower[3]) > 0.1,
          "USA-arm country spread is no longer the only unusual one")
    V["bal.lower.usa"] = f"{lower[1]:.3f}"
    omnibus = p_list("multinomial logit omnibus")
    V["bal.omnibus.pmin"], V["bal.omnibus.pmax"] = f"{min(omnibus):.2f}", f"{max(omnibus):.2f}"
    V["bal.omnibus.list"] = ", ".join(f"{p:.2f}" for p in omnibus)
    V["bal.missing.em.list"] = ", ".join(
        f"{p:.2f}" for p in p_list("missing econ_model independent of video"))
    missing = balance_means[["missing_political_model", "missing_econ_model"]].to_numpy()
    V["missing.min"], V["missing.max"] = f"{missing.min() * 100:.1f}", f"{missing.max() * 100:.1f}"
    missing_tests = balance_tests[balance_tests.test.str.startswith("missing")]
    V["missing.pmin"] = f"{missing_tests.p_value.min():.2f}"
    V["missing.pmax"] = f"{missing_tests.p_value.max():.2f}"

    for grouping, key in zip(GROUPINGS, ["china", "usa", "compchina", "compus"]):
        group = balance_means[balance_means.grouping == grouping]
        lines = [[vid(r.video), str(r.n), f"{r.gender:.2f}", f"{r.age:.1f}",
                  f"{r.education:.2f}", f"{r.national_pride:.2f}", f"{r.leftright:.2f}",
                  f"{r.missing_political_model * 100:.1f}"] for r in group.itertuples()]
        tests = balance_tests[balance_tests.grouping == grouping].set_index("test").p_value
        cells = [tests[f"equal means: {cov}"] for cov in BALANCE]
        cells.append(tests["missing political_model independent of video"])
        lines.append(["\\quad $p$, equal across videos", ""] + [f"{p:.2f}" for p in cells])
        V[f"tab.balance.{key}"] = rows(lines)

    # Individual videos
    for o, key in ((PM, "pm"), (EM, "em")):
        china = [v(o, t).estimate for t in CHINA_TERMS]
        check(min(china) > 0, "china_pol2 is no longer the weakest CGTN video")
        check(v(o, "china_pol1").p_value < 0.001, "china_pol1 not clearly above china_pol2")
        check(jt(o, "china").p_value < 0.001, "CGTN joint test not p < 0.001")
        V[f"china.{key}.min"], V[f"china.{key}.max"] = num(min(china)), num(max(china))
        V[f"joint.china.{key}.chi2"] = f"{jt(o, 'china').chi2:.1f}"
        V[f"range.{key}"] = num(jt(o, "china")["range"])
        V[f"rangeci.{key}.low"] = num(ranges.loc[o, "ci_low"])
        V[f"rangeci.{key}.high"] = num(ranges.loc[o, "ci_high"])
        d = dispersion.loc[(o, "range")]
        V[f"disp.{key}"], V[f"disp.{key}.low"], V[f"disp.{key}.high"] = (
            num(d.difference), num(d.ci_low), num(d.ci_high))
        for k in ("china", "us"):
            check(abs(dispersion.loc[(o, "range"), k] - jt(o, k)["range"]) < 1e-9,
                  "Dispersion and joint-test ranges disagree")
        if o == PM:
            for term in CHINA_TERMS:
                V[f"v.pm.{term}"] = num(v(PM, term).estimate)
    check(ranges.bias.abs().max() <= 0.0105, "Range bias no longer 'about 0.01 or less'")
    check((dispersion.ci_low > 0).all(), "Dispersion intervals include zero")
    check(0.9 <= jt(EM, "china")["range"] < 1.0, "'nearly one point' no longer holds")
    check(all(v(o, "china_econ2").estimate < v(o, "china_econ1").estimate for o in (PM, EM)),
          "china_econ2 no longer weaker than china_econ1")
    check(v(EM, "us_econ1").estimate < 0 and v(EM, "us_econ2").estimate < 0,
          "US economic videos no longer below us_pol2")
    V["range.pm.1"] = f"{jt(PM, 'china')['range']:.1f}"
    V["range.us.pm"] = num(jt(PM, "us")["range"])
    V["joint.us.pm.p"] = f"{jt(PM, 'us').p_value:.2f}"
    V["joint.us.em.p"] = f"{jt(EM, 'us').p_value:.3f}"
    V["joint.us.em.pholm"] = f"{jt(EM, 'us').p_holm:.3f}"
    for term in CHINA_TERMS + US_TERMS:
        V[f"effpair.{term}"] = f"{num(v(PM, term).estimate)} / {num(v(EM, term).estimate)}"

    V["tab.videos"] = rows(
        [[vid(term)] + [f"{est(r.estimate, r.std_error)} & {pcell(r.p_holm)} & {pcell(r.p_bh)}"
                        for r in (v(PM, term), v(EM, term))]
         for term in CHINA_TERMS + US_TERMS])

    def joint_cell(o, country, chi_digits, p_digits=2):
        r = jt(o, country)
        p = "$p < 0.001$" if r.p_value < 0.001 else f"$p = {r.p_value:.{p_digits}f}$"
        return f"\\multicolumn{{3}}{{c}}{{$\\chi^2_3 = {r.chi2:.{chi_digits}f}$, {p}}}"

    V["tab.videos.tests"] = rows([
        ["Joint test, CGTN", joint_cell(PM, "china", 1), joint_cell(EM, "china", 1)],
        ["Joint test, US", joint_cell(PM, "us", 2), joint_cell(EM, "us", 1, 3)],
        ["$N$"] + [f"\\multicolumn{{3}}{{c}}{{{count(v(o, 'china_econ1').n)}}}"
                   for o in (PM, EM)],
    ])

    def ty(o, term):
        return pick(types, outcome=o, term=term)

    V["tab.types.additive"] = rows(
        [[label] + [f"{est(ty(o, term).estimate, ty(o, term).std_error)} & "
                    f"{pcell(ty(o, term).p_holm)}" for o in (PM, EM)]
         for term, label in [("China economic video", "Chinese video is economic"),
                             ("US economic video", "US video is economic")]])
    inter = "China econ x US econ"
    V["tab.types.interaction"] = rows(
        [["Chinese econ.\\ $\\times$ US econ."]
         + [f"{est(ty(o, inter).estimate, ty(o, inter).std_error)} & ---" for o in (PM, EM)]])
    V["types.n.pm"], V["types.n.em"] = (count(ty(o, "China economic video").n) for o in (PM, EM))
    V["types.inter.p.pm"] = f"{ty(PM, inter).p_value:.2f}"
    V["types.inter.p.em"] = f"{ty(EM, inter).p_value:.2f}"

    # Economic versus pol1
    designs = ("China arm", "Competition arm")
    for d, key in zip(designs, ("china", "comp")):
        political, economic = ct(PM, "CGTN", d), ct(EM, "CGTN", d)
        V[f"ct.{key}.pm"], V[f"ct.{key}.em"] = num(political.estimate), num(economic.estimate)
        V[f"ct.{key}.em.se"] = num(economic.std_error)
        V[f"ct.{key}.pm.half"] = num(1.959964 * political.std_error)
        V[f"ct.{key}.pm.mde"] = num(political.mde)
        check(economic.p_holm < 0.05, "CGTN economic contrast does not survive Holm")
    V["ct.china.em.3"] = num(ct(EM, "CGTN", "China arm").estimate, 3)
    V["ct.maxdisadv"] = num(max(-ct(PM, "CGTN", d).ci_low for d in designs))
    check((content[content.source == "State Department"].p_value > 0.05).all(),
          "A State Department contrast is now significant")
    V["ct.state.usa.em"] = num(ct(EM, "State Department", "USA arm").estimate)
    V["ct.state.comp.em"] = num(ct(EM, "State Department", "Competition arm").estimate)
    state_se = [ct(EM, "State Department", d).std_error for d in ("USA arm", "Competition arm")]
    V["ct.state.se.min"], V["ct.state.se.max"] = num(min(state_se)), num(max(state_se))
    V["mde.min"], V["mde.max"] = num(content.mde.min()), num(content.mde.max())

    V["tab.content"] = rows(
        [[label] + [f"{est(ct(o, source, d).estimate, ct(o, source, d).std_error)} & "
                    f"{pcell(ct(o, source, d).p_holm)}" for o in (PM, EM)]
         for source, d, label in [
             ("CGTN", "China arm", "CGTN & China arm"),
             ("CGTN", "Competition arm", "CGTN & Competition arm"),
             ("State Department", "USA arm", "State Dept. & USA arm"),
             ("State Department", "Competition arm", "State Dept. & Competition arm")]])
    single_n = content[content.design.isin(["China arm", "USA arm"])].n
    V["content.n.low"] = f"{round(single_n.min(), -1):,.0f}".replace(",", "{,}")
    V["content.n.high"] = f"{round(single_n.max(), -1):,.0f}".replace(",", "{,}")
    comp_n = content[content.design == "Competition arm"].n
    V["content.n.comp"] = f"{round(comp_n.mean(), -1):,.0f}".replace(",", "{,}")

    def pz(arm, o, start):
        return position[(position.arm == arm) & (position.outcome == o)
                        & position.term.str.startswith(start)].iloc[0]

    econ, order = "Economic companion", "pol2 first"
    for start, key in ((econ, "econ"), (order, "order")):
        for o, okey in ((PM, "pm"), (EM, "em")):
            r = pz("China", o, start)
            V[f"pos.{key}.{okey}"], V[f"pos.{key}.{okey}.se"] = num(r.estimate), num(r.std_error)
    check((position[position.term.str.startswith(order)].p_value > 0.05).all(),
          "An order effect is now significant")
    order_se = position[(position.arm == "China") & position.term.str.startswith(order)].std_error
    V["pos.order.mde"] = f"{2.8016 * order_se.max():.1f}"
    V["tab.position"] = rows(
        [[arm, label] + [est(pz(arm, o, start).estimate, pz(arm, o, start).std_error)
                         for o in (PM, EM)] + [count(pz(arm, PM, start).n)]
         for arm in ("China", "USA")
         for start, label in [(econ, "Economic vs.\\ \\vid{pol1}, \\vid{pol2} second for all"),
                              (order, "\\vid{pol2} first vs.\\ second, political pairs")]])

    # Perceptions
    growth, competent = "mech_growth_china", "mech_competent_china"
    for d, key in zip(designs, ("china", "comp")):
        g, k = pick(mech_content, outcome=growth, design=d), pick(mech_content, outcome=competent, design=d)
        V[f"mech.growth.{key}"], V[f"mech.growth.{key}.se"] = num(g.estimate), num(g.std_error)
        V[f"mech.competent.{key}"] = num(-k.estimate)
        V[f"mech.competent.{key}.se"] = num(k.std_error)
    check(len(mech_content) == 16
          and (mech_content[mech_content.outcome == competent].p_holm < 0.05).all()
          and (mech_content[mech_content.outcome != competent].p_holm >= 0.05).all(),
          "Only the competence contrasts should survive Holm across 16 tests")
    check(baseline["mean"].idxmax() == growth and baseline["share_agree"].idxmax() == growth,
          "Growth is no longer the most widely held belief")
    V["base.growth.mean"] = num(baseline.loc[growth, "mean"])
    V["base.growth.agree"] = f"{baseline.loc[growth, 'share_agree'] * 100:.0f}"
    us = pick(companion_mech, arm="USA", outcome="mech_growth_us")
    V["us.growth.mean"], V["us.growth"], V["us.growth.se"] = (
        num(us.control_mean), num(us.estimate), num(us.std_error))
    V["us.growth.p"], V["us.growth.pholm"] = f"{us.p_value:.3f}", f"{us.p_holm:.2f}"
    growth_videos = mech_videos[mech_videos.outcome == growth]
    check((growth_videos.estimate > 0).all() and (growth_videos.p_value < 0.05).all(),
          "Not all videos raise growth beliefs relative to china_pol2")
    V["growth.videos.min"] = num(growth_videos.estimate.min())
    V["growth.videos.max"] = num(growth_videos.estimate.max())

    equal = index[index["index"] == "Equally weighted"].set_index("term")
    inverse = index[index["index"] == "Inverse-covariance weighted"].set_index("term")
    average = "average of other three"
    V["index.corr"] = num(equal.mean_item_correlation.iloc[0])
    V["index.avg"], V["index.avg.se"] = (num(equal.loc[average, "estimate"]),
                                         num(equal.loc[average, "std_error"]))
    V["index.avg.inverse"] = num(inverse.loc[average, "estimate"])
    largest_p = max(equal.loc[CHINA_TERMS, "p_value"].max(),
                    inverse.loc[CHINA_TERMS, "p_value"].max())
    check(largest_p < 0.01, "A video is no longer individually higher at p < 0.01")
    V["index.largest.p"] = f"{largest_p:.3f}"
    loo = index_loo[index_loo["index"] == "Equally weighted"]
    loo_average, loo_videos = loo[loo.term == average], loo[loo.term != average]
    check((loo_average.p_value < 0.001).all() and (loo_videos.estimate > 0).all()
          and (loo_videos.p_value < 0.05).all(), "Leave-one-out results changed")
    V["loo.min"], V["loo.max"] = num(loo_average.estimate.min()), num(loo_average.estimate.max())
    check(equal.joint_p.iloc[0] < 0.001 and inverse.joint_p.iloc[0] < 0.001,
          "Index joint tests no longer p < 0.001")
    V["index.chi2.equal"] = f"{equal.joint_chi2.iloc[0]:.1f}"
    V["index.chi2.inverse"] = f"{inverse.joint_chi2.iloc[0]:.1f}"
    V["index.n"] = count(equal.n.iloc[0])

    lines = []
    for outcome, label in baseline["label"].items():
        items = mech_videos[mech_videos.outcome == outcome].set_index("term")
        lines.append([label] + [est(items.loc[t, "estimate"], items.loc[t, "std_error"], 2)
                                for t in CHINA_TERMS])
    V["tab.mechvideos"] = rows(lines)
    V["tab.mechvideos.index"] = rows(
        [[label] + [est(table.loc[t, "estimate"], table.loc[t, "std_error"], 2)
                    for t in CHINA_TERMS]
         for table, label in [(equal, "Index, equally weighted"),
                              (inverse, "Index, inverse-covariance weighted")]])
    check(mech_videos.n.between(1450, 1460).all(), "Item-level N no longer about 1,455")
    words = {5: "Five", 6: "Six", 7: "Seven"}
    significant = int((mech_videos.p_holm < 0.05).sum())
    V["mech.nsig"] = words.get(significant, str(significant))
    V["mech.ncoef"] = len(mech_videos)

    # Appendix B
    def au(o, spec, term):
        return pick(authors, outcome=o, specification=spec, term=term)

    check(all(au(o, spec, "us_econ - us_pol").p_value > 0.05
              for o in (PM, EM) for spec in ("pooled", "by_arm")),
          "US economic vs. political now significant")
    V["au.chinadiff.em"] = num(au(EM, "pooled", "china_econ - china_pol").estimate, 3)
    V["au.pooled.usecon"] = num(au(PM, "pooled", "us_econ").estimate, 3)
    V["au.pooled.uspol"] = num(au(PM, "pooled", "us_pol").estimate, 3)
    V["au.byarm.usecon"] = num(au(PM, "by_arm", "us_econ").estimate)
    V["au.byarm.uspol"] = num(au(PM, "by_arm", "us_pol").estimate)

    def au_row(term, label):
        cells = []
        for o in (PM, EM):
            for spec in ("pooled", "by_arm"):
                match = authors[(authors.outcome == o) & (authors.specification == spec)
                                & (authors.term == term)]
                cells.append("---" if match.empty
                             else est(match.iloc[0].estimate, match.iloc[0].std_error))
        return [label] + cells

    V["tab.authors"] = rows([au_row(*r) for r in [
        ("china_econ", "China economic"), ("china_pol", "China political"),
        ("us_econ", "US economic"), ("us_pol", "US political"),
        ("competition_us_econ", "Competition: US econ."),
        ("competition_us_pol", "Competition: US pol.")]])
    V["tab.authors.diffs"] = rows([au_row(*r) for r in [
        ("china_econ - china_pol", "China econ.\\ $-$ China pol."),
        ("us_econ - us_pol", "US econ.\\ $-$ US pol.")]])
    V["au.n.pm"], V["au.n.em"] = (count(au(o, "pooled", "china_econ").n) for o in (PM, EM))
    return V


def main():
    V = values()
    lines = ["% Generated by src/07_paper_numbers.py from output/tables. Do not edit.",
             "\\newcommand{\\defval}[2]{\\expandafter\\def\\csname val:#1\\endcsname{#2}}"]
    lines += [f"\\defval{{{key}}}{{{value}}}" for key, value in V.items()]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"Saved {len(V)} values to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
