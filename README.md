# Chinese State Media and the "China Model": A Replication and Extension

This repository contains a **replication and extension** of:

> Mattingly, Daniel C., Trevor Incerti, Changwook Ju, Colin Moreshead, Seiki Tanaka, and Hikaru Yamagishi. 2025. "Chinese State Media Persuades a Global Audience That the 'China Model' Is Superior: Evidence from a 19-Country Experiment." *American Journal of Political Science* 69(3): 1029–46. https://doi.org/10.1111/ajps.12887

The original analysis was written in R. This project reproduces the main results in Python and extends them by *unbundling the treatments*: using the random assignment of specific videos within treatment arms to estimate how much individual messages differ in persuasiveness, whether economic and political content differ, and what each kind of video changes in perceptions of China.

## Data

The raw data are not included in this repository. Download the authors' replication package from Harvard Dataverse (https://doi.org/10.7910/DVN/CQ4FZR) and place the files from its `Data/` folder directly in `data/raw/` (e.g. `data/raw/clean.RData`).

## Environment

Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
python run_all.py
```

The last step, `src/05_check_paper.py`, rebuilds every reported number in `paper/main.tex` (each table row and each sentence that reports a result) from the output tables and confirms that it appears in the paper; the run fails if any number no longer matches.

To rebuild the paper after running the pipeline (requires a LaTeX distribution):

```bash
cd paper && latexmk -pdf main.tex
```

## Repository structure

```
├── run_all.py            # runs the full pipeline
├── src/
│   ├── 01_clean.py       # load and prepare data
│   ├── 02_replicate.py   # main results (Table A3)
│   ├── 03_extension.py   # within-arm assignment, video-level effects, mechanisms
│   ├── 04_figures.py     # figures
│   └── 05_check_paper.py # checks every number in the paper against the outputs
├── paper/
│   ├── main.tex          # replication and extension paper (LaTeX)
│   └── main.pdf          # compiled paper
├── data/                 # raw and processed data (not tracked)
└── output/
    ├── tables/
    └── figures/
```

## Outputs

| File | Corresponds to |
|---|---|
| `output/tables/sample_flow.csv` | Sample construction: respondents screened out before randomization (N = 6,276 randomized) |
| `output/tables/primary_coefficients.csv` | Table A3: effects on preference for China's political model, economic model, and world leadership, with and without covariates |
| `output/tables/table_a3_check.csv` | Side-by-side comparison of all 24 replicated and published Table A3 estimates |
| `output/tables/standardized_effects.csv` | Unadjusted effects in control-group standard deviations (reported in the main text) |
| `output/tables/share_prefer_china.csv` | Share of respondents preferring China's model, by arm (reported in the main text) |
| `output/tables/ext_assignment.csv` | Video pairs in the China and USA arms; verifies that `pol2` was always shown (paper Table 4) |
| `output/tables/ext_competition_independence.csv` | Test that US and Chinese videos were assigned independently in the Competition arm |
| `output/tables/ext_competition_videos.csv` | Effect of each video relative to its country's `pol2`, Competition arm (paper Figure 2, Table A1) |
| `output/tables/ext_competition_joint_tests.csv` | Wald tests that a country's four videos are equally persuasive (Table A1) |
| `output/tables/ext_competition_range.csv` | Bias and basic bootstrap interval for the range of the four CGTN videos' effects (Section 4.5) |
| `output/tables/ext_competition_types.csv` | Economic vs. political videos with pooled political reference, Competition arm (Table A2) |
| `output/tables/ext_content_type.csv` | Economic video vs. `pol1` in the single-country and Competition arms, with MDEs (paper Table 5, Figure 3) |
| `output/tables/ext_companion_position.csv` | Position check: companion contrast with `pol2` held in the second slot, and the effect of `pol2`'s slot (Table A4) |
| `output/tables/ext_companion_mechanisms.csv` | Companion contrast on perceptions of each arm's own country, with control-group means (Section 4.7) |
| `output/tables/ext_mechanisms_content.csv` | Economic video vs. `china_pol1` on eight perceptions of China, both designs (paper Figure 4) |
| `output/tables/ext_mechanisms_baseline.csv` | Control-group level of each perception of China |
| `output/tables/ext_mechanisms_videos.csv` | Each CGTN video's effect on perceptions relative to `china_pol2` (Table A3) |
| `output/tables/ext_mechanisms_index.csv` | Summary index of the eight perceptions (equally and inverse-covariance weighted): each CGTN video vs. `china_pol2` (Table A3) |
| `output/tables/ext_mechanisms_index_loo.csv` | The summary index re-estimated without each of the eight items in turn (Section 4.7) |
| `output/tables/ext_balance_means.csv` | Covariate means and missing-outcome rates by video (Table A5) |
| `output/tables/ext_balance_tests.csv` | Balance, country (including the lower-tail p-value), omnibus, and missingness tests by video (Table A5) |
| `output/tables/ext_authors_video_type.csv` | The replication code's unreported video-type regression, pooled and by arm (Table A6) |
| `output/figures/fig1_main_effects.{pdf,png}` | Covariate-adjusted treatment effects (replicates the paper's main results figure) |
| `output/figures/fig2_competition_videos.{pdf,png}` | Video-level effects in the Competition arm |
| `output/figures/fig3_content_type.{pdf,png}` | Economic video vs. `pol1` in two designs |
| `output/figures/fig4_mechanisms.{pdf,png}` | Economic video vs. `china_pol1` on perceptions of China |
