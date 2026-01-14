# Evaluation Pipeline for Small Datasets

Complete guide to the evaluation pipeline for Parkinson's Disease classification, optimized for small datasets (~70-210 samples).

---

##  Overview

The evaluation pipeline provides:

1. **Metrics with Bootstrap Confidence Intervals** - Quantify uncertainty in performance estimates
2. **Statistical Significance Testing** - Determine if model differences are real or due to chance
3. **Publication-Quality Visualizations** - 300 DPI plots for papers and presentations
4. **Comprehensive Cross-Validation** - Proper k-fold CV with OOF predictions
5. **Automated Reporting** - Markdown and JSON exports

---

##  Quick Start

### Run Full Evaluation

```bash
# Train all models and generate comprehensive evaluation
python scripts/comprehensive_evaluation.py --cv_folds 5 --generate_plots

# Skip training (use cached results), just generate reports
python scripts/comprehensive_evaluation.py --skip_baseline --generate_plots
```

### Individual Components

```python
from src.evaluation import (
    calculate_all_metrics,
    plot_multiple_roc_curves,
    compare_all_models
)

# 1. Calculate metrics with confidence intervals
from sklearn.metrics import accuracy_score
y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]

metrics = calculate_all_metrics(y_true, y_pred)
print(metrics)

# 2. Plot ROC curves
results = {
    'XGBoost': (y_true, y_proba_xgb),
    'Random Forest': (y_true, y_proba_rf)
}
plot_multiple_roc_curves(results, save_path='results/figures/roc_curves.png')

# 3. Statistical comparison
cv_results = {
    'XGBoost': {'f1': [0.82, 0.80, 0.83, 0.81, 0.82]},
    'Random Forest': {'f1': [0.79, 0.78, 0.80, 0.79, 0.80]}
}
compare_all_models(cv_results, metric='f1')
```

---

##  Metrics

### Basic Metrics

```python
from src.evaluation import calculate_all_metrics

metrics = calculate_all_metrics(
    y_true=y_true,
    y_pred=y_pred,
    y_proba=y_proba  # Optional, for AUC
)

# Returns:
# {
#     'accuracy': 0.85,
#     'precision': 0.83,
#     'recall': 0.87,
#     'f1': 0.85,
#     'balanced_accuracy': 0.84,
#     'mcc': 0.70,
#     'auc_roc': 0.92,
#     'true_positive': 13,
#     'true_negative': 10,
#     'false_positive': 2,
#     'false_negative': 3,
#     'true_positive_rate': 0.81,
#     'true_negative_rate': 0.83,
#     'false_positive_rate': 0.17
# }
```

### Bootstrap Confidence Intervals

Critical for small datasets to understand metric uncertainty.

```python
from src.evaluation import calculate_metrics_with_confidence_intervals

metrics_with_ci = calculate_metrics_with_confidence_intervals(
    y_true=y_true,
    y_pred=y_pred,
    y_proba=y_proba,
    n_bootstrap=1000,  # Number of bootstrap iterations
    confidence_level=0.95
)

# Returns:
# {
#     'accuracy': {
#         'value': 0.85,
#         'ci_lower': 0.78,
#         'ci_upper': 0.91,
#         'std': 0.04
#     },
#     'f1': {...},
#     'auc_roc': {...}
# }

# Print formatted report
from src.evaluation import print_metrics_report
print_metrics_report(metrics_with_ci, title="Model Performance")
```

**Output:**
```
============================================================
                  Classification Report
============================================================

Metric                      Value               95% CI
------------------------------------------------------------
accuracy                   0.8500    [0.7800, 0.9100]
f1                         0.8512    [0.7802, 0.9101]
auc_roc                    0.9200    [0.8700, 0.9600]
------------------------------------------------------------
```

---

##  Visualizations

### 1. Performance Comparison Bar Chart

Compare multiple models across multiple metrics with error bars.

```python
from src.evaluation import plot_performance_comparison

results = {
    'XGBoost': {
        'accuracy': {'value': 0.85, 'std': 0.03},
        'f1': {'value': 0.83, 'std': 0.04},
        'auc_roc': {'value': 0.91, 'std': 0.02}
    },
    'Random Forest': {...},
    'SVM': {...}
}

plot_performance_comparison(
    results,
    metrics=['accuracy', 'f1', 'auc_roc'],
    save_path='results/figures/performance_comparison.png'
)
```

**Features:**
- Side-by-side metrics
- Error bars (std across CV folds)
- Value labels on bars
- Professional styling (300 DPI)

### 2. ROC Curves (Multi-Model)

All models in one plot with AUC in legend.

```python
from src.evaluation import plot_multiple_roc_curves

predictions = {
    'XGBoost': (y_true, y_proba_xgb),
    'Random Forest': (y_true, y_proba_rf),
    'SVM': (y_true, y_proba_svm)
}

plot_multiple_roc_curves(
    predictions,
    save_path='results/figures/roc_curves.png',
    title='ROC Curves - Model Comparison'
)
```

**Features:**
- All models in one figure
- AUC values in legend
- Shaded area under curves
- Diagonal reference line

### 3. Confusion Matrices (2x2 Grid)

Four confusion matrices with counts, percentages, sensitivity, specificity.

```python
from src.evaluation import plot_confusion_matrix_grid

predictions = {
    'Tabular XGBoost': (y_true, y_pred_tab),
    'Image XGBoost': (y_true, y_pred_img),
    'Late Fusion': (y_true, y_pred_fusion),
    'Stacking': (y_true, y_pred_stacking)
}

plot_confusion_matrix_grid(
    predictions,
    model_names=list(predictions.keys())[:4],  # First 4
    save_path='results/figures/confusion_matrices.png'
)
```

**Features:**
- 2x2 grid layout
- Counts (heatmap colors)
- Percentages (red text)
- Sensitivity/Specificity in titles
- Professional color scheme

### 4. Feature Importance Comparison

Side-by-side top 15 features from tabular and image models.

```python
from src.evaluation import plot_feature_importance_comparison

plot_feature_importance_comparison(
    tabular_importance=tab_model.feature_importances_,
    tabular_feature_names=tab_feature_names,
    image_importance=img_model.feature_importances_,
    image_feature_names=img_feature_names,
    tabular_model_name='Tabular XGBoost',
    image_model_name='Image XGBoost',
    top_n=15,
    save_path='results/figures/feature_importance.png'
)
```

### 5. t-SNE Visualization

Visualize learned representations, color by class, mark misclassifications.

```python
from src.evaluation import plot_tsne_visualization

plot_tsne_visualization(
    X=X_tabular,
    y=y_true,
    y_pred=y_pred,  # Optional, marks misclassifications with X
    feature_name='Tabular Features',
    save_path='results/figures/tsne_tabular.png'
)
```

**Features:**
- 2D t-SNE embedding
- Colored by true class
- Misclassifications marked with red X
- Two plots: (1) by class, (2) with errors marked

### 6. SHAP Summary Plot

Global feature importance with impact direction (positive/negative).

```python
from src.evaluation import plot_shap_summary

plot_shap_summary(
    model=xgboost_model,
    X=X_tabular,
    feature_names=tab_feature_names,
    max_features=20,
    save_path='results/figures/shap_summary.png',
    plot_type='bar'  # or 'dot' for beeswarm
)
```

**Features:**
- TreeExplainer for tree models
- KernelExplainer fallback for others
- Top N features
- Impact direction (dot plot)
- Bar chart for ranking

**Note:** Requires `shap` package:
```bash
pip install shap
```

### 7. Generate All Plots

Convenience function to generate all requested plots.

```python
from src.evaluation import generate_all_plots

results = {
    'cv_results': cv_results_dict,
    'predictions': predictions_dict,
    'feature_importance': {
        'tabular': {'importance': tab_importance},
        'image': {'importance': img_importance}
    }
}

generate_all_plots(
    results=results,
    X_tabular=X_tab,
    X_image=X_img,
    y=y,
    tabular_feature_names=tab_feature_names,
    image_feature_names=img_feature_names,
    output_dir='results/figures/'
)
```

**Generates:**
1. `performance_comparison.png` - Bar chart
2. `roc_curves.png` - All ROC curves
3. `confusion_matrices.png` - 2x2 grid
4. `feature_importance.png` - Side-by-side
5. `tsne_tabular.png` - Tabular t-SNE
6. `tsne_image.png` - Image t-SNE
7. `shap_tabular.png` - Tabular SHAP (if model available)
8. `shap_image.png` - Image SHAP (if model available)

---

##  Statistical Tests

### McNemar's Test (Model Comparison)

Test if two models have significantly different error rates.

```python
from src.evaluation import mcnemars_test

result = mcnemars_test(
    y_true=y_true,
    y_pred1=y_pred_xgb,
    y_pred2=y_pred_rf,
    model1_name="XGBoost",
    model2_name="Random Forest"
)

# Returns:
# {
#     'statistic': 2.5,
#     'p_value': 0.025,
#     'significant': True,  # p < 0.05
#     'contingency_table': [[10, 2], [3, 12]],
#     'model1_accuracy': 0.85,
#     'model2_accuracy': 0.79
# }
```

**Output:**
```
======================================================================
McNemar's Test: XGBoost vs Random Forest
======================================================================
Contingency Table:
                Random Forest
                Correct   Wrong
XGBoost     Correct      10         2
XGBoost     Wrong         3         12

XGBoost Accuracy: 0.8500
Random Forest Accuracy: 0.7900

Chi-squared: 2.5000
P-value: 0.025000

 Significant difference (p < 0.05)
======================================================================
```

### Wilcoxon Signed-Rank Test (CV Scores)

Non-parametric test for paired CV scores.

```python
from src.evaluation import wilcoxon_signed_rank_test

result = wilcoxon_signed_rank_test(
    scores1=[0.82, 0.80, 0.83, 0.81, 0.82],  # XGBoost CV scores
    scores2=[0.79, 0.78, 0.80, 0.79, 0.80],   # RF CV scores
    model1_name="XGBoost",
    model2_name="Random Forest"
)
```

### Comprehensive Pairwise Comparison

Test all model pairs with multiple comparison correction.

```python
from src.evaluation import compare_all_models

cv_results = {
    'XGBoost': {'f1': [0.82, 0.80, 0.83, 0.81, 0.82]},
    'Random Forest': {'f1': [0.79, 0.78, 0.80, 0.79, 0.80]},
    'SVM': {'f1': [0.76, 0.75, 0.77, 0.76, 0.76]},
    'Logistic Regression': {'f1': [0.74, 0.73, 0.75, 0.74, 0.74]}
}

results = compare_all_models(
    cv_results,
    metric='f1',
    alpha=0.05,
    correction_method='holm'  # or 'bonferroni'
)

# Generates summary table and prints results
# Also includes adjusted p-values
```

**Output:**
```
======================================================================
PAIRWISE COMPARISON SUMMARY
======================================================================
Model 1                     Model 2                     P-value  Significant
----------------------------------------------------------------------
XGBoost                     Random Forest               0.012345   Yes
XGBoost                     SVM                         0.000123   Yes
XGBoost                     Logistic Regression         0.000012   Yes
Random Forest               SVM                         0.045678   No
Random Forest               Logistic Regression         0.002345   Yes
SVM                         Logistic Regression         0.123456   No
======================================================================
```

### Bootstrap Performance Difference

Non-parametric test for performance difference.

```python
from src.evaluation import bootstrap_performance_difference
from sklearn.metrics import accuracy_score

result = bootstrap_performance_difference(
    y_true=y_true,
    y_pred1=y_pred_xgb,
    y_pred2=y_pred_rf,
    metric_func=accuracy_score,
    n_bootstrap=10000,
    alpha=0.05
)

# Returns:
# {
#     'point_difference': 0.06,
#     'mean_difference': 0.058,
#     'ci_lower': 0.02,
#     'ci_upper': 0.10,
#     'significant': True  # CI doesn't contain 0
# }
```

---

##  Cross-Validation

### Single Model CV

```python
from src.evaluation import cross_validate_model
from xgboost import XGBClassifier

model = XGBClassifier(max_depth=3, n_estimators=100)

cv_scores = cross_validate_model(
    model=model,
    X=X_tabular,
    y=y,
    n_splits=5,
    verbose=True
)

# Returns:
# {
#     'accuracy': [0.85, 0.83, 0.84, 0.82, 0.85],
#     'precision': [0.83, 0.81, 0.82, 0.80, 0.83],
#     'recall': [...],
#     'f1': [...],
#     'auc_roc': [...]
# }
```

### Multiple Models CV

```python
from src.evaluation import cross_validate_multiple_models

models = {
    'XGBoost': XGBClassifier(max_depth=3, n_estimators=100),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=8)
}

results = cross_validate_multiple_models(
    models=models,
    X=X_tabular,
    y=y,
    n_splits=5
)

# Prints comparison table
```

**Output:**
```
================================================================================
MODEL COMPARISON
================================================================================
Model                               Accuracy          F1              AUC
--------------------------------------------------------------------------------
XGBoost                           0.8380±0.0234    0.8312±0.0245    0.8912±0.0198
Random Forest                     0.8023±0.0312    0.7956±0.0323    0.8623±0.0256
================================================================================
```

---

##  Reporting

### Generate Markdown Table

```python
from src.evaluation import generate_markdown_table

results = {
    'XGBoost': {
        'accuracy': {'value': 0.85, 'ci_lower': 0.80, 'ci_upper': 0.90},
        'f1': {'value': 0.83, 'ci_lower': 0.78, 'ci_upper': 0.88}
    },
    'Random Forest': {...}
}

markdown_table = generate_markdown_table(
    results_dict=results,
    metrics=['accuracy', 'f1'],
    title="Model Performance Comparison"
)

print(markdown_table)
```

**Output:**
```markdown
## Model Performance Comparison

| Model | Accuracy | F1 |
|-------|----------|-----|
| Random Forest | 0.8000 (0.7500-0.8500) | 0.7900 (0.7400-0.8400) |
| XGBoost | 0.8500 (0.8000-0.9000) | 0.8300 (0.7800-0.8800) |
```

### Export to JSON

```python
from src.evaluation import generate_results_json

results = {
    'cv_results': cv_results_dict,
    'statistical_tests': stat_results_dict,
    'metadata': {
        'dataset': 'PD Classification',
        'cv_folds': 5
    }
}

generate_results_json(
    results=results,
    output_path='results/metrics/all_results.json'
)
```

### Generate Summary Report

```python
from src.evaluation import generate_summary_report

generate_summary_report(
    results=results,
    output_path='results/metrics/summary_report.md'
)
```

**Generates:**
- Executive summary
- Key findings
- Best performing model
- Performance comparison table
- 95% confidence intervals

### Generate Statistical Report

```python
from src.evaluation import generate_statistical_report

generate_statistical_report(
    statistical_results=stat_results,
    output_path='results/metrics/statistical_report.md'
)
```

**Generates:**
- Summary of tests performed
- Significance level and correction method
- Detailed pairwise comparison table
- Conclusions about significant differences

---

##  Output Structure

Running the comprehensive evaluation generates:

```
results/evaluation/
 all_results.json              # All results in JSON
 summary_report.md             # Markdown summary
 statistical_report.md         # Statistical testing results
 figures/                      # All plots (300 DPI PNG)
    performance_comparison.png
    roc_curves.png
    confusion_matrices.png
    feature_importance.png
    tsne_tabular.png
    tsne_image.png
    shap_tabular.png
    shap_image.png
 baselines/                    # Baseline model results
    baseline_results.csv
    cv_scores_detailed.csv
 fusion/                       # Fusion model results
     fusion_comparison.csv
     cv_scores_detailed.csv
```

---

##  Best Practices for Small Datasets

### 1. Always Use Cross-Validation

 **BAD:** Single train/test split
```python
X_train, X_test, y_train, y_test = train_test_split(X, y)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)  # Unreliable!
```

 **GOOD:** 5-fold stratified CV
```python
cv_scores = cross_validate_model(model, X, y, n_splits=5)
mean_score = np.mean(cv_scores['accuracy'])
std_score = np.std(cv_scores['accuracy'])
print(f"Accuracy: {mean_score:.4f} (±{std_score:.4f})")
```

### 2. Report Confidence Intervals

 **BAD:** Point estimate only
```python
print(f"Accuracy: {0.85}")  # No uncertainty!
```

 **GOOD:** With 95% CI
```python
metrics = calculate_metrics_with_confidence_intervals(y_true, y_pred)
acc = metrics['accuracy']
print(f"Accuracy: {acc['value']:.4f} (95% CI: {acc['ci_lower']:.4f}-{acc['ci_upper']:.4f})")
# Accuracy: 0.8500 (95% CI: 0.7800-0.9100)
```

### 3. Statistical Significance Testing

 **BAD:** Claim improvement without testing
```python
if acc_xgb > acc_rf:
    print("XGBoost is better!")  # Maybe due to chance!
```

 **GOOD:** Statistical test
```python
result = mcnemars_test(y_true, y_pred_xgb, y_pred_rf)
if result['significant']:
    print(f"XGBoost significantly better (p={result['p_value']:.4f})")
else:
    print("No significant difference")
```

### 4. Multiple Comparison Correction

When testing multiple models, correct for multiple comparisons.

```python
# Without correction: inflated Type I error
compare_all_models(cv_results, correction_method=None)

# With correction: control family-wise error rate
compare_all_models(cv_results, correction_method='holm')
```

### 5. Visualization for Interpretability

- Use SHAP for feature importance
- t-SNE for understanding representations
- Confusion matrices for error analysis
- ROC curves for threshold selection

---

##  Troubleshooting

### "SHAP not installed"

**Solution:**
```bash
pip install shap
```

### "t-SNE fails with small dataset"

**Cause:** Perplexity must be < n_samples

**Solution:** Automatically handled in our code:
```python
perplexity = min(30, X.shape[0] - 1)
tsne = TSNE(perplexity=perplexity)
```

### "Confidence intervals are very wide"

**Meaning:** High uncertainty due to small sample size

**Solutions:**
1. Increase n_bootstrap (e.g., 2000-5000)
2. Use stratified CV (more robust)
3. Accept uncertainty (report honestly)
4. Collect more data

### "McNemar's test p-value is exactly 0.05"

**Edge case:** p = 0.05 is borderline

**Approach:**
- Report exact p-value
- Be cautious in claims
- Collect more data if possible
- Use bootstrap test as alternative

---

##  API Reference

See individual module docstrings:

- **[metrics.py](../src/evaluation/metrics.py)** - All metric calculations
- **[visualizations.py](../src/evaluation/visualizations.py)** - Plotting functions
- **[cross_validation.py](../src/evaluation/cross_validation.py)** - CV utilities
- **[statistical_tests.py](../src/evaluation/statistical_tests.py)** - Statistical tests

---

##  Related Documentation

- **[Fusion Guide](FUSION_GUIDE.md)** - Multi-modal fusion methods
- **[Model Config](../configs/model_config.yaml)** - Model hyperparameters
- **[Fusion Config](../configs/fusion_config.yaml)** - Fusion settings

---

**For help:**
- Olivia Liau: oliau@usc.edu
- Project: MoE-MultiSDAR-PD-Classification
