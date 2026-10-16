# Chinese State Media and the "China Model": A Replication and Extension

This repository contains a **replication and extension** of:

> Mattingly, Daniel C., Trevor Incerti, Changwook Ju, Colin Moreshead, Seiki Tanaka, and Hikaru Yamagishi. 2025. "Chinese State Media Persuades a Global Audience That the 'China Model' Is Superior: Evidence from a 19-Country Experiment." *American Journal of Political Science* 69(3): 1029–46. https://doi.org/10.1111/ajps.12887

The original analysis was written in R. This project reproduces the main results in Python and extends them by *unbundling the treatments*: estimating video-level effects and order effects within the Competition arm.

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
│   ├── 03_extension.py   # video-level and order effects
│   └── 04_figures.py     # figures
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
| `output/tables/table_a3_check.csv` | Side-by-side comparison of replicated and published Table A3 estimates |
| `output/tables/standardized_effects.csv` | Unadjusted effects in control-group standard deviations (reported in the main text) |
| `output/tables/ext_assignment.csv` | Extension: video combinations in the China and USA arms, documenting the assignment rule |
| `output/tables/ext_companion.csv` | Extension A: economic vs. political companion video within the China and USA arms (with MDEs) |
| `output/tables/ext_companion_position.csv` | Extension A: robustness of the companion comparison to video position |
| `output/tables/ext_companion_mechanisms.csv` | Extension A: companion effect on each perception item (performance and democracy) |
| `output/tables/ext_authors_video_type.csv` | Reproduction of the authors' unpublished video-type models (`7. mechanism_analysis.R`), pooled and split by arm |
| `output/tables/ext_competition_independence.csv` | Extension B: test that US and Chinese videos were assigned independently in the Competition arm |
| `output/tables/ext_competition_types.csv` | Extension B: effects of economic vs. political Chinese and US videos in the Competition arm |
| `output/tables/ext_competition_videos.csv` | Extension B: effect of each video relative to its country's `pol2` video |
| `output/figures/fig1_main_effects.{pdf,png}` | Covariate-adjusted treatment effects (replicates the paper's main results figure) |
| `output/figures/fig2_companion.{pdf,png}` | Extension A: economic vs. political companion video, China and USA arms |
| `output/figures/fig3_competition_videos.{pdf,png}` | Extension B: video-level effects in the Competition arm |
| `output/figures/fig4_companion_mechanisms.{pdf,png}` | Extension A: companion effect on perceptions of the arm's country, by item |
