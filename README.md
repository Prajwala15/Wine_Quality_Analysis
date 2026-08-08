# Wine Quality Analysis

Analyzing the impact of chemical attributes on wine quality using the UCI Wine Quality dataset, combining exploratory data analysis, statistical hypothesis testing, and predictive modeling.

## Overview

This project explores what chemical properties drive wine quality ratings across red and white wines, using rigorous statistical testing to validate findings and a machine learning model to predict quality categories.

## Data

Sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/):

- **Red wine:** 1,599 samples
- **White wine:** 4,898 samples
- **Combined total:** 6,497 samples
- **After cleaning:** 5,320 samples (1,177 duplicates removed, outliers capped via IQR method)

Each sample includes 11 chemical features (fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free/total sulfur dioxide, density, pH, sulphates, alcohol) and a quality score (3–9), later bucketed into `quality_category` (Low ≤4, Medium 5–6, High ≥7).

## Methods

1. **Data Preparation** — merged datasets, deduplicated, capped outliers, engineered quality categories
2. **Exploratory Data Analysis** — descriptive statistics, distribution plots, correlation heatmaps, red vs. white comparisons
3. **Statistical Hypothesis Testing** — Pearson correlation, one-way ANOVA, independent t-tests, chi-square test (α = 0.05)
4. **Predictive Modeling** — Random Forest Classifier (300 trees) with SMOTE oversampling to address class imbalance

## Key Findings

| Finding | Evidence |
|---------|----------|
| Alcohol is the strongest predictor | r = +0.47, higher alcohol = higher quality |
| Volatile acidity hurts quality | r = -0.26, lower VA = higher quality |
| Density negatively affects quality | r = -0.34, lower density = higher quality |
| White wines slightly higher rated | Mean: 5.86 vs 5.62 for red |
| Class imbalance exists | 75% Medium, 19% High, 4% Low quality |
| Random Forest achieves 75% accuracy | Strongest on Medium quality; Low quality remains hardest to predict |

All statistical hypothesis tests returned p < 0.05, indicating every chemical feature is significantly associated with wine quality.

## Repository Structure

```
├── data/                          # Raw and cleaned datasets
├── outputs/
│   ├── descriptive_statistics.csv
│   ├── correlation_significance_table.csv
│   ├── anova_results.csv
│   ├── ttest_results.csv
│   ├── hypothesis_testing_summary.csv
│   ├── red_vs_white_comparison.csv
│   └── figures/                   # Charts and visualizations
└── RoleA_Complete_Analysis.ipynb  # Full analysis notebook
```

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install pandas numpy matplotlib seaborn scipy scikit-learn imbalanced-learn`
3. Open `RoleA_Complete_Analysis.ipynb` in Jupyter or Google Colab
4. Run all cells to reproduce the data prep, EDA, hypothesis tests, and model

## Model Performance

- **Algorithm:** Random Forest Classifier (300 estimators, SMOTE-balanced training set)
- **Test accuracy:** 75.0%
- **Top predictive features:** alcohol, free sulfur dioxide, total sulfur dioxide, volatile acidity, density

## License

Dataset courtesy of the UCI Machine Learning Repository. See [source](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/) for citation details.
