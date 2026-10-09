"""Produce figures for the replication and extension.

All figures are coefficient plots: a point estimate with its 95% confidence
interval, one color per outcome. They read only the CSV tables written by
02_replicate.py and 03_extension.py, so no model is re-estimated here.

Input:  output/tables/primary_coefficients.csv
        output/tables/ext_companion.csv
        output/tables/ext_competition_videos.csv
Output: output/figures/fig1_main_effects.{pdf,png}
        output/figures/fig2_companion.{pdf,png}
        output/figures/fig3_competition_videos.{pdf,png}
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files without opening a window
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "output" / "tables"
FIGURES = ROOT / "output" / "figures"

# One color and marker per outcome. The colors were checked for
# colorblind-safe contrast; the marker shape keeps the series distinct
# in grayscale print.
OUTCOME_STYLE = {
    "political_model": {"label": "Political model", "color": "#2a78d6", "marker": "o"},
    "econ_model": {"label": "Economic model", "color": "#eb6834", "marker": "s"},
}
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"

# Vertical distance between the two outcomes plotted for the same row.
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
    """Read one of the CSV tables written by the analysis scripts."""
    return pd.read_csv(TABLES / name)


def coef_plot(ax, table, row_labels, xlabel):
    """Draw estimates and 95% CIs on `ax`, one row per entry of `row_labels`.

    `table` needs columns `row`, `outcome`, `estimate`, `ci_low`, `ci_high`.
    `row_labels` maps each value of `row` to its axis label, top to bottom.
    """
    rows = list(row_labels)
    for i, outcome in enumerate(OUTCOME_STYLE):
        style = OUTCOME_STYLE[outcome]
        sub = table[table["outcome"] == outcome].set_index("row").loc[rows]
        # Rows run top to bottom, so y decreases; outcomes are offset around each row.
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
    """Place a single legend for both outcomes above the figure."""
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.98))


def save(fig, name):
    """Save a figure as PDF (for the paper) and PNG (for previews)."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{ext}")
    plt.close(fig)


def fig_main_effects():
    """Figure 1: covariate-adjusted treatment effects (replication of Table A3)."""
    coefs = load_table("primary_coefficients.csv")
    table = coefs[coefs["adjusted"] & (coefs["term"] != "Constant")].rename(columns={"term": "row"})
    rows = {"China": "China", "USA": "USA", "Competition": "Competition"}

    fig, ax = plt.subplots(figsize=(5.5, 2.6))
    coef_plot(ax, table, rows, "Effect on preference for China (1–6 scale), vs. Control")
    add_legend(fig)
    save(fig, "fig1_main_effects")


def fig_companion():
    """Figure 2: effect of an economic (vs. political) companion video, by arm."""
    table = load_table("ext_companion.csv").rename(columns={"arm": "row"})
    rows = {"China": "China arm\n(CGTN videos)", "USA": "USA arm\n(State Dept. videos)"}

    fig, ax = plt.subplots(figsize=(5.5, 2.2))
    coef_plot(ax, table, rows, "Economic vs. political companion video (difference, 1–6 scale)")
    add_legend(fig)
    save(fig, "fig2_companion")


def fig_competition_videos():
    """Figure 3: each video's effect relative to its country's pol2, Competition arm."""
    table = load_table("ext_competition_videos.csv").rename(columns={"term": "row"})
    panels = [
        ("Chinese videos (vs. china_pol2)",
         {"china_econ1": "Economic 1", "china_econ2": "Economic 2", "china_pol1": "Political 1"}),
        ("US videos (vs. us_pol2)",
         {"us_econ1": "Economic 1", "us_econ2": "Economic 2", "us_pol1": "Political 1"}),
    ]

    # Both panels have the same row labels, so only the left panel shows them.
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.6), sharex=True, sharey=True)
    for ax, (title, rows) in zip(axes, panels):
        coef_plot(ax, table[table["row"].isin(rows)], rows, xlabel="")
        ax.set_title(title, fontsize=9, color=TEXT, loc="left")
    axes[1].tick_params(axis="y", labelleft=False)
    fig.supxlabel("Effect on preference for China (1–6 scale), vs. the country's pol2 video",
                  fontsize=9, y=-0.08)
    add_legend(fig)
    save(fig, "fig3_competition_videos")


def main():
    """Draw all figures."""
    fig_main_effects()
    fig_companion()
    fig_competition_videos()
    print(f"Saved figures to {FIGURES.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
