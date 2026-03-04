# Results Analysis — Output Guide

This document describes every output produced by `analyse_results.py`, what each visualisation shows, and how to interpret it.

---

## 1. Summary Tables (`tables/`)

### Files
- `{Dataset}_summary.csv` — one per dataset
- `overall_summary.csv` — all datasets combined

### Description
Each table lists every model as a row and every metric as a column. Each cell contains the **mean** and **standard deviation** of that metric across the 30 runs, formatted as `mean (std)`.

### How to interpret
- **Higher mean** = better performance for that metric.
- **Lower std** = more consistent/reproducible across runs.
- Compare rows within a table to see which model performs best on a given dataset.
- The `overall_summary.csv` file also provides raw numeric `_mean` and `_std` columns for downstream processing.

### Metrics reference
| Metric | What it measures |
|---|---|
| accuracy | Overall fraction of correct predictions |
| precision | Precision averaged across classes |
| precision_illicit | Precision for the illicit (positive) class specifically |
| recall | Recall averaged across classes |
| recall_illicit | Recall for the illicit class specifically |
| f1 | F1-score averaged across classes |
| f1_illicit | F1-score for the illicit class specifically |
| roc_auc | Area under the ROC curve |
| PRAUC | Area under the Precision-Recall curve |
| kappa | Cohen's kappa (agreement beyond chance) |

---

## 2. Precision-Recall Curves (`pr_curves/`)

### Files
- `{Dataset}_pr_curves.png` — one per dataset

### Description
Each plot overlays the Precision-Recall (PR) curves of all seven models on a single set of axes. Because there are 30 runs per model, the displayed curve is the **mean precision** at every recall level, with a **shaded band** showing plus/minus one standard deviation. The legend includes each model's mean PRAUC value.

### How to interpret
- **Higher and further right** = better. A perfect classifier hugs the top-right corner (precision = 1 at all recall levels).
- **Wider shaded band** = greater variability across runs at that recall level.
- PRAUC (shown in the legend) summarises each curve into a single number; higher is better.
- PR curves are especially informative for **imbalanced datasets** (which AML datasets are), because unlike ROC curves they are not inflated by a large number of true negatives.
- If two models have similar PRAUC but different curve shapes, one may achieve higher precision at low recall (good for flagging high-confidence cases) while the other maintains precision at higher recall (good for broad screening).

---

## 3. Confusion Matrices (`confusion_matrices/`)

### Files
- `{Dataset}_confusion_matrices.png` — one per dataset (contains a subplot per model)

### Description
Each subplot shows a 2x2 **row-normalised confusion matrix** averaged across 30 runs. Row normalisation means each row sums to 100%, so the values represent rates:

|  | Predicted Licit | Predicted Illicit |
|---|---|---|
| **True Licit** | True Negative Rate (Specificity) | False Positive Rate |
| **True Illicit** | False Negative Rate | True Positive Rate (Recall) |

Each cell is annotated with `mean% +/- std%`.

### How to interpret
- **Diagonal cells** (top-left, bottom-right) = correct predictions. Higher is better.
- **Off-diagonal cells** = errors. Lower is better.
- **Bottom-right cell** (True Positive Rate) is equivalent to recall for the illicit class. This is typically the most critical cell for AML — it tells you what proportion of illicit transactions the model catches.
- **Top-right cell** (False Positive Rate) tells you how many legitimate transactions are incorrectly flagged. In AML, a high FPR creates costly false alerts.
- The **std** values indicate how much each rate varies across the 30 runs. Lower std = more reproducible.
- Comparing models within a dataset: look for models that maximise the bottom-right cell while minimising the top-right cell.

---

## 4. Box Plots (`box_plots/`)

### Files
- `{Dataset}_box_plots.png` — one per dataset

### Description
Each figure contains one box plot per key metric (f1_illicit, PRAUC, roc_auc, kappa). Within each box plot, there is one box per model showing the distribution of that metric across 30 runs. The red diamond marker indicates the **mean**, the orange line inside the box is the **median**, the box spans the **interquartile range (IQR, Q1 to Q3)**, whiskers extend to 1.5x IQR, and dots beyond the whiskers are **outliers**.

### How to interpret
- **Box position** (higher = better performance).
- **Box width/height** (taller box = more variability between runs).
- **Outlier dots** indicate runs with unusually good or poor performance.
- **Mean vs median**: if they differ noticeably, the distribution is skewed.
- Useful for quick visual comparison of both **performance level** and **stability** across models on a given dataset.

---

## 5. Cross-Dataset Heatmaps (`heatmaps/`)

### Files
- `{metric}_heatmap.png` — one per key metric (f1_illicit, PRAUC, roc_auc, kappa)

### Description
A grid with **datasets as rows** and **models as columns**. Each cell is colour-coded and annotated with the **mean** value of that metric across 30 runs. The colour scale runs from yellow (lower) to red (higher).

### How to interpret
- Scan **columns** to see how a specific model performs across all datasets.
- Scan **rows** to see which model is best on a specific dataset.
- Look for **consistent column patterns**: a model that is dark red across all rows is a strong general performer.
- Look for **dataset-specific patterns**: some datasets may be inherently easier (entire row is dark) or harder (entire row is light).
- These heatmaps are useful for identifying whether performance differences are dataset-dependent or whether one model dominates universally.

---

## 6. Violin Plots (`violin_plots/`)

### Files
- `{Dataset}_{metric}_violin.png` — one per dataset-metric combination

### Description
Similar to box plots but show the full **kernel density estimate** of the distribution. The width of the violin at any point represents how many runs achieved that score. A small box plot is embedded inside each violin for reference.

### How to interpret
- **Shape of the violin**: a narrow violin means most runs cluster tightly around the same value (high reproducibility). A wide or bimodal violin means scores are spread out or split into two groups.
- **Bimodality** (two bumps) would suggest the model sometimes converges to different solutions across runs — a potential reproducibility concern.
- **Long tails** (violin extending far down on one side) indicate occasional poor runs.
- Compare the **density peaks** (widest points) across models to see where each model's most likely performance falls.

---

## 7. Statistical Tests (`statistical_tests/`)

### 7a. Friedman Test (`friedman_tests.csv`)

#### Description
A **non-parametric omnibus test** that checks whether at least one model performs significantly differently from the others. Two variants are included:
- **cross_dataset**: datasets are treated as blocks (n=6 blocks). Tests whether model rankings are consistent across datasets.
- **within_{Dataset}**: individual runs are treated as blocks (n=30 blocks). Tests whether models differ within a single dataset.

#### How to interpret
- **p_value < 0.05** (significant): there is evidence that at least one model differs from the rest. Proceed to post-hoc tests (Nemenyi/Wilcoxon) to find out which pairs differ.
- **p_value >= 0.05** (not significant): no strong evidence that models differ for this metric. Post-hoc tests are not warranted.
- The **chi2 statistic** indicates the strength of the effect; larger values = more evidence of differences.

### 7b. Wilcoxon Signed-Rank Tests (`wilcoxon_{Dataset}.csv`)

#### Description
**Pairwise non-parametric tests** comparing every pair of models within each dataset. Each row tests whether model_1 and model_2 have significantly different performance on a given metric, using the 30 paired run scores. **Holm-Bonferroni correction** is applied to control the family-wise error rate across all pairwise comparisons within a dataset.

#### How to interpret
- **p_value_holm < 0.05**: the two models differ significantly even after correcting for multiple comparisons.
- **effect_size_r**: the rank-biserial correlation, a measure of effect size.
  - |r| < 0.3: small effect
  - 0.3 <= |r| < 0.5: medium effect
  - |r| >= 0.5: large effect
- **mean_1 vs mean_2**: shows the direction of the difference (which model is better).
- For your reproducibility paper, focus on pairs where the corrected p-value is significant **and** the effect size is at least medium.

### 7c. Nemenyi Post-Hoc Test (`nemenyi_posthoc.csv`)

#### Description
After a significant Friedman test, the **Nemenyi test** identifies which specific pairs of models differ. It compares the average ranks of models across datasets and declares a pair significantly different if their rank difference exceeds the **critical difference (CD)**.

#### How to interpret
- **significant = True**: the two models have statistically different performance rankings across datasets.
- **rank_diff vs critical_difference**: if rank_diff > CD, the difference is significant. The larger the margin, the more confident the conclusion.
- Note: with only 6 datasets, the CD is relatively large, so this test is conservative. Within-dataset Wilcoxon tests provide complementary finer-grained evidence.

---

## 8. Coefficient of Variation (`cv_analysis/`)

### Files
- `coefficient_of_variation.csv` — full table
- `{metric}_cv_heatmap.png` — one heatmap per key metric

### Description
The **Coefficient of Variation (CV)** is defined as `(std / mean) * 100%`. It measures the relative variability of a metric across runs, normalised by the mean. This makes it directly comparable across models and datasets even when the means differ substantially.

### How to interpret
- **Lower CV = more reproducible**. A CV of 1% means the standard deviation is 1% of the mean; a CV of 20% means results vary considerably across runs.
- In the heatmap, the colour scale uses **red = high CV (poor reproducibility)** and **green = low CV (good reproducibility)**.
- Compare models within a row: which model is most stable on each dataset?
- Compare datasets within a column: on which dataset is a given model most/least reproducible?
- For a reproducibility paper, CV is one of the most direct and interpretable measures of run-to-run variability.
- Typical interpretation thresholds (context-dependent):
  - CV < 5%: excellent reproducibility
  - 5% <= CV < 15%: moderate variability
  - CV >= 15%: poor reproducibility, may warrant investigation

---

## 9. Rank Analysis (`rank_analysis/`)

### Files
- `{metric}_rank_stability.csv` — one per key metric

### Description
For each run independently, all models are ranked from 1 (best) to 7 (worst) on the given metric. The table then reports summary statistics of each model's rank distribution across the 30 runs: **mean rank, std of rank, median rank**, and **best_rank_count** (how many of the 30 runs the model achieved rank 1).

### How to interpret
- **mean_rank**: the average position of the model across runs. Closer to 1 = consistently the top performer.
- **std_rank**: how much the ranking fluctuates. Low std = the model reliably occupies the same position. High std = the model's relative performance is unstable.
- **best_rank_count**: how often the model "wins" across runs. A model with mean_rank=2 but best_rank_count=15 wins half the time but occasionally drops lower.
- This is important for reproducibility because it answers: **"If I rerun the experiment, will the same model come out on top?"** If multiple models have similar mean_rank with high std_rank, the model ranking is not robust.

---

## 10. Radar Charts (`radar_charts/`)

### Files
- `{Dataset}_radar.png` — one per dataset

### Description
Also called spider or web charts. Each model is plotted as a polygon where the six axes represent different metrics (accuracy, precision_illicit, recall_illicit, f1_illicit, roc_auc, kappa). The distance from the centre along each axis corresponds to the mean value of that metric across 30 runs. All axes range from 0 to 1.

### How to interpret
- **Larger polygon = better overall performance** across all metrics simultaneously.
- **Shape of the polygon** reveals each model's strengths and weaknesses. A model with a spike towards recall_illicit but a dip at precision_illicit catches most illicit transactions but generates many false positives.
- **Overlapping polygons** make it easy to see where one model outperforms another and where it falls behind.
- Useful for presentations and summaries, but note that the visual impression is sensitive to the order of axes. Use alongside the summary tables for precise comparisons.

---

## 11. Critical Difference Diagrams (`critical_difference/`)

### Files
- `{metric}_cd_diagram.png` — one per key metric

### Description
A visual representation of the Nemenyi post-hoc test results. Models are placed along a horizontal axis at their **average rank** (lower = better, positioned further left). A **CD bar** at the top shows the critical difference threshold. **Horizontal bars** connect groups of models whose average ranks are **not** significantly different from each other (i.e. their rank difference is less than the CD).

### How to interpret
- Models positioned **further left** have better (lower) average rank.
- If two models are **connected by a horizontal bar**, they are statistically indistinguishable based on the Nemenyi test.
- If two models are **not connected**, their performance difference is statistically significant.
- The **CD value** depends on the number of models and datasets. With only 6 datasets, the CD is relatively large, so many models may be grouped together.
- This is a standard figure in machine learning benchmarking papers and is widely understood by reviewers.

---

## 12. Convergence Analysis (`convergence/`)

### Files
- `{Dataset}_{metric}_convergence.png` — one per dataset-metric combination

### Description
Each plot shows the **cumulative mean** of a metric as runs are added one by one (run 1, then mean of runs 1-2, then mean of runs 1-3, etc.). Each model is a separate line. If the estimate has stabilised, the line should flatten out towards the right side of the plot.

### How to interpret
- **A flat line at the right end** means the mean has converged and 30 runs is sufficient for a stable estimate.
- **A line still trending up or down at run 30** suggests the estimate has not fully converged and more runs might be needed.
- **Early oscillations settling into a plateau** is the expected pattern. The point where oscillations become negligible indicates the minimum number of runs needed.
- Compare models: some may converge faster (e.g., deterministic models like SVM/RF may converge in fewer runs than stochastic models like GNNs).
- This directly supports a claim in your reproducibility paper that N=30 runs is (or is not) sufficient for reliable estimates.
- Note: this shows convergence of the mean. It does not show whether the variance estimate has also converged, which typically requires more samples.

---

## How to Regenerate Individual Sections

Edit the `SECTIONS` dictionary at the top of `analyse_results.py`. Set any section to `False` to skip it:

```python
SECTIONS = {
    "summary_tables":        True,
    "pr_curves":             False,  # skip this section
    "confusion_matrices":    True,
    "box_plots":             False,  # skip this section
    ...
}
```

Then rerun the script. Only enabled sections will be regenerated. Data loading always runs since most sections depend on it.
