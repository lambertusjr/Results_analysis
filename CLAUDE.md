# CLAUDE.md

## Project Overview

This is a **Results Analysis Pipeline** for comparing machine learning models on Anti-Money Laundering (AML) datasets. It processes multi-run experiment results and generates publication-ready tables, visualizations, and statistical tests.

## Tech Stack

- **Language**: Python 3.11
- **Core libraries**: pandas, numpy, matplotlib, seaborn, scipy.stats, sklearn.metrics
- **Data formats**: Pickle (.pkl) and NumPy (.npz) for input; CSV and PNG for output
- **Environment**: Conda (configured in `.vscode/settings.json`)

## How to Run

```bash
python analyse_results.py
```

Toggle individual analysis sections on/off via the `SECTIONS` dictionary (lines 74-87 in `analyse_results.py`). All configuration constants (datasets, models, metrics, paths) are defined at the top of the script.

## Project Structure

```
Results_analysis/
├── analyse_results.py      # Main analysis script (~1,040 lines, 18 functions)
└── output/                 # Generated outputs
    ├── tables/             # CSV summary tables
    ├── pr_curves/          # Precision-recall curve plots
    ├── confusion_matrices/ # Confusion matrix heatmaps
    ├── box_plots/          # Box plot distributions
    ├── heatmaps/           # Cross-dataset performance heatmaps
    ├── violin_plots/       # Violin distribution plots
    ├── statistical_tests/  # Friedman, Wilcoxon, Nemenyi test results
    ├── cv_analysis/        # Coefficient of variation analysis
    ├── rank_analysis/      # Rank stability analysis
    ├── radar_charts/       # Multi-metric spider plots
    ├── critical_difference/# Critical difference diagrams
    ├── convergence/        # Convergence plots over runs
    └── GUIDE.md            # Detailed output documentation
```

## Models & Datasets

- **7 models**: MLP, GCN, GAT, GIN, RF, SVM, XGB
- **6 AML datasets** (defined in `DATASETS`)
- **30 runs** per model-dataset combination (`N_RUNS = 30`)

## Key Metrics

All 10 metrics: accuracy, precision, recall, f1, precision_illicit, recall_illicit, f1_illicit, roc_auc, PRAUC, kappa

4 focus metrics for detailed analysis: **f1_illicit, PRAUC, roc_auc, kappa**

## Code Conventions

- **Modular functions**: Each of the 18 functions handles one analysis type
- **Data-flow pattern**: Load → Process → Visualize → Save (consistent across all functions)
- **Configuration at top**: Edit constants (`DATASETS`, `MODELS`, `METRIC_COLS`, `KEY_METRICS`, `RESULTS_DIR`, `OUTPUT_DIR`) to reconfigure
- **Matplotlib Agg backend**: Non-interactive plotting for batch processing
- **300 DPI output**: Publication-quality PNG figures
- **Predefined model colours**: Hex colour map in `MODEL_COLORS` for consistent legends

## Known Caveats

- **Hardcoded Windows paths**: `RESULTS_DIR` and `OUTPUT_DIR` use Windows-style paths (`C:\Users\...`). Update these for other systems.
- **No test suite**: No unit tests; the script is a standalone analysis tool.
- **No CLI arguments**: All configuration is done by editing the script directly.
- **Silent failure on missing data**: Script skips missing data files without raising exceptions.
