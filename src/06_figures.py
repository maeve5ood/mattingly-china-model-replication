# Step 6. Figures 1-4, from the csv tables.

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from analysis_utils import ROOT, TABLES

FIGURES = ROOT / "output" / "figures"

OUTCOME_STYLE = {
    "political_model": {"label": "Political model", "color": "#2a78d6", "marker": "o"},
    "econ_model": {"label": "Economic model", "color": "#eb6834", "marker": "s"},
}
DESIGN_STYLE = {
    "China arm": {"label": "China arm (companion of pol2)", "color": "#1b7f6b", "marker": "D"},
    "Competition arm": {"label": "Competition arm (single video)", "color": "#6a51a3", "marker": "^"},
}
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"

DODGE = 0.18

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": TEXT,
    "xtick.color": MUTED,
    "ytick.color": TEXT,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def load_table(name):
    return pd.read_csv(TABLES / name)


def coef_plot(ax, table, row_labels, xlabel, series="outcome", styles=OUTCOME_STYLE):
    rows = list(row_labels)
    for i, key in enumerate(styles):
        style = styles[key]
        sub = table[table[series] == key].set_index("row").loc[rows]
        y = [len(rows) - 1 - k + (DODGE / 2 - i * DODGE) for k in range(len(rows))]
        ax.hlines(y, sub["ci_low"], sub["ci_high"], color=style["color"], linewidth=1.5)
        ax.plot(sub["estimate"], y, linestyle="none", marker=style["marker"], markersize=6,
                color=style["color"], markeredgecolor="white", markeredgewidth=0.8,
                label=style["label"])

    ax.axvline(0, color=MUTED, linestyle="--", linewidth=0.8, zorder=0)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([row_labels[r] for r in reversed(rows)])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)


def add_legend(fig):
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.98))


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    # no CreationDate -> reproducible pdf
    fig.savefig(FIGURES / f"{name}.pdf", metadata={"CreationDate": None})
    fig.savefig(FIGURES / f"{name}.png")
    plt.close(fig)


def fig_main_effects():
    coefs = load_table("primary_coefficients.csv")
    table = coefs[coefs["adjusted"] & (coefs["term"] != "Constant")].rename(columns={"term": "row"})
    table = table[table["outcome"].isin(OUTCOME_STYLE)]
    rows = {"China": "China", "USA": "USA", "Competition": "Competition"}

    fig, ax = plt.subplots(figsize=(5.5, 2.6))
    coef_plot(ax, table, rows, "Effect on preference for China (1–6 scale), vs. Control")
    add_legend(fig)
    save(fig, "fig1_main_effects")


def fig_competition_videos():
    table = load_table("ext_competition_videos.csv").rename(columns={"term": "row"})
    panels = [
        ("Chinese videos (vs. china_pol2)",
         {"china_econ1": "Economic 1", "china_econ2": "Economic 2", "china_pol1": "Political 1"}),
        ("US videos (vs. us_pol2)",
         {"us_econ1": "Economic 1", "us_econ2": "Economic 2", "us_pol1": "Political 1"}),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7, 2.6), sharex=True, sharey=True)
    for ax, (title, rows) in zip(axes, panels):
        coef_plot(ax, table, rows, xlabel="")
        ax.set_title(title, fontsize=9, color=TEXT, loc="left")
    axes[1].tick_params(axis="y", labelleft=False)
    fig.supxlabel("Effect on preference for China (1–6 scale), vs. the country's pol2 video",
                  fontsize=9, y=-0.08)
    add_legend(fig)
    save(fig, "fig2_competition_videos")


def fig_content_type():
    table = load_table("ext_content_type.csv")
    table["row"] = table["source"] + " | " + table["design"]
    rows = {
        "CGTN | China arm": "CGTN, China arm\n(companion of pol2)",
        "CGTN | Competition arm": "CGTN, Competition arm\n(single video)",
        "State Department | USA arm": "State Dept., USA arm\n(companion of pol2)",
        "State Department | Competition arm": "State Dept., Competition arm\n(single video)",
    }

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    coef_plot(ax, table, rows, "Economic video vs. pol1 (difference, 1–6 scale; + = toward China)")
    add_legend(fig)
    save(fig, "fig3_content_type")


def fig_mechanisms():
    table = load_table("ext_mechanisms_content.csv").rename(columns={"outcome": "row"})
    order = (table[table["design"] == "Competition arm"]
             .sort_values("estimate")[["row", "label"]].itertuples(index=False))
    rows = {row: label for row, label in order}

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    coef_plot(ax, table, rows,
              "Economic video vs. pol1 (difference in agreement, 1–7 scale)",
              series="design", styles=DESIGN_STYLE)
    add_legend(fig)
    save(fig, "fig4_mechanisms")


def main():
    fig_main_effects()
    fig_competition_videos()
    fig_content_type()
    fig_mechanisms()
    print(f"Saved figures to {FIGURES.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
