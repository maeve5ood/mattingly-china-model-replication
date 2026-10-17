# Which Messages Persuade? Unbundling the Effects of Chinese State Media

A replication and extension of Mattingly et al. (2025)

**Paper:** [`paper/main.pdf`](paper/main.pdf)
**Author:** Yuxuan Zhang, Harris School of Public Policy, University of Chicago

> Mattingly, Daniel C., Trevor Incerti, Changwook Ju, Colin Moreshead, Seiki Tanaka, and Hikaru Yamagishi. 2025. "Chinese State Media Persuades a Global Audience That the 'China Model' Is Superior: Evidence from a 19-Country Experiment." *American Journal of Political Science* 69(3): 1029–46. https://doi.org/10.1111/ajps.12887

The original analysis was written in R. This project reproduces the main results in Python and extends them by *unbundling the treatments*: within each treatment arm, the specific videos a respondent saw were also randomly assigned, which allows causal comparisons of individual messages that the published paper does not report.

## Main findings

- **Replication.** All 24 coefficients, standard errors, and sample sizes in the original primary results table (Mattingly et al.'s Table A3) match the published values at two decimals.
- **Individual messages differ.** Among respondents who saw a single CGTN video, the most and least persuasive of the four videos differ by 0.60 points on preference for China's political model and 0.97 points on its economic model (six-point scales), more than the overall effects of that arm (0.36 and 0.28).
- **Economic content compares well.** In two independent designs, economic CGTN videos are not detectably less persuasive than the stronger political video about China's political model, and they are more persuasive about its economic model (by 0.26 and 0.30 points).
- **The design as implemented.** Every respondent in the China arm saw the same political video, `china_pol2`, which was the least persuasive of the four CGTN videos in the Competition arm.

![Effect of each video relative to its country's pol2 video in the Competition arm](output/figures/fig2_competition_videos.png)

*Figure 2 of the paper: effect of each video relative to its country's `pol2` video, among respondents who saw one Chinese and one US video. Lines are pointwise 95% confidence intervals.*

## Data

The data are not included in this repository. Download the authors' replication package from Harvard Dataverse (https://doi.org/10.7910/DVN/CQ4FZR) and place `clean.RData` from its `Data/` folder at `data/raw/clean.RData`. This is the authors' cleaned analysis file and the only data file the code uses (see Section 3.1 of the paper for why the analysis starts from it rather than the raw survey exports).

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

The pipeline runs in about 10 seconds and has been tested on macOS (Apple silicon) and Linux. The last step, `src/05_check_paper.py`, rebuilds every reported number in `paper/main.tex` (each table row and each sentence that reports a result) from the output tables and confirms that it appears in the paper; the run fails if any number no longer matches.

To rebuild the paper after running the pipeline (requires a LaTeX distribution):

```bash
cd paper && latexmk -pdf main.tex
```

## Repository structure

```
├── run_all.py            # runs the full pipeline
├── src/
│   ├── 01_clean.py       # load the data, document the sample, check it against the paper
│   ├── 02_replicate.py   # main results (Mattingly et al.'s Table A3)
│   ├── 03_extension.py   # within-arm assignment, video-level effects, perceptions, balance
│   ├── 04_figures.py     # figures
│   └── 05_check_paper.py # checks every number in the paper against the outputs
├── paper/
│   ├── main.tex          # paper (LaTeX)
│   └── main.pdf          # compiled paper
├── data/                 # data (not tracked; see Data above)
└── output/
    ├── tables/
    └── figures/
```

## Outputs

Table, figure, and section numbers refer to [`paper/main.pdf`](paper/main.pdf).

| File | Contents | In the paper |
|---|---|---|
| `output/tables/sample_flow.csv` | Respondents screened out before randomization (6,276 randomized) | Table 1 |
| `output/tables/primary_coefficients.csv` | Effects on preference for China's political model, economic model, and world leadership, with and without covariates | Table 2 |
| `output/tables/table_a3_check.csv` | Cell-by-cell comparison of all 24 replicated estimates with Mattingly et al.'s Table A3 | Section 3.3 |
| `output/tables/standardized_effects.csv` | Unadjusted effects in control-group standard deviations | Section 3.3 |
| `output/tables/share_prefer_china.csv` | Share of respondents preferring China's model, by arm | Section 3.3 |
| `output/tables/ext_assignment.csv` | Video pairs in the China and USA arms; verifies that `pol2` was always shown | Table 4 |
| `output/tables/ext_competition_independence.csv` | Test that the US and Chinese videos were assigned independently in the Competition arm | Section 4.3 |
| `output/tables/ext_competition_videos.csv` | Effect of each video relative to its country's `pol2`, Competition arm | Figure 2, Tables 3 and A1 |
| `output/tables/ext_competition_joint_tests.csv` | Wald tests that a country's four videos are equally persuasive | Table A1 |
| `output/tables/ext_competition_range.csv` | Bias and basic bootstrap interval for the range of the four CGTN videos' effects | Section 4.5 |
| `output/tables/ext_competition_types.csv` | Economic vs. political videos, pooling each country's political videos, Competition arm | Table A2 |
| `output/tables/ext_content_type.csv` | Economic video vs. `pol1` in the single-country and Competition arms, with MDEs | Table 5, Figure 3 |
| `output/tables/ext_companion_position.csv` | Position check: the companion contrast with `pol2` in the second slot, and the effect of `pol2`'s slot | Table A4 |
| `output/tables/ext_companion_mechanisms.csv` | Companion contrast on perceptions of each arm's own country, with control-group means | Section 4.7 |
| `output/tables/ext_mechanisms_content.csv` | Economic video vs. `china_pol1` on eight perceptions of China, both designs | Figure 4 |
| `output/tables/ext_mechanisms_baseline.csv` | Control-group level of each perception of China | Section 4.7 |
| `output/tables/ext_mechanisms_videos.csv` | Each CGTN video's effect on perceptions relative to `china_pol2` | Table A3 |
| `output/tables/ext_mechanisms_index.csv` | Standardized index of the eight perceptions (equally and inverse-covariance weighted) | Table A3 |
| `output/tables/ext_mechanisms_index_loo.csv` | The index re-estimated without each of the eight items in turn | Section 4.7 |
| `output/tables/ext_balance_means.csv` | Covariate means and missing-outcome rates by video | Table A5 |
| `output/tables/ext_balance_tests.csv` | Balance, country-by-video, omnibus, and missingness tests by video | Table A5, Section 4.3 |
| `output/tables/ext_authors_video_type.csv` | The replication code's unreported video-type regression, as specified and with arms separated | Table A6 |
| `output/figures/fig1_main_effects.{pdf,png}` | Covariate-adjusted treatment effects | Figure 1 |
| `output/figures/fig2_competition_videos.{pdf,png}` | Video-level effects in the Competition arm | Figure 2 |
| `output/figures/fig3_content_type.{pdf,png}` | Economic video vs. `pol1` in two designs | Figure 3 |
| `output/figures/fig4_mechanisms.{pdf,png}` | Economic video vs. `china_pol1` on perceptions of China | Figure 4 |
