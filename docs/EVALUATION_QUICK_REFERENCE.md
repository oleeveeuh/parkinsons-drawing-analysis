# Evaluation Pipeline - Quick Reference

Quick reference for the evaluation pipeline. See [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) for comprehensive documentation.

---

##  Quick Start

### Run Full Evaluation
```bash
python scripts/comprehensive_evaluation.py --cv_folds 5 --generate_plots
```

### Python API
```python
from src.evaluation import *

# Metrics
metrics = calculate_all_metrics(y_true, y_pred, y_proba)

# Bootstrap CI
metrics_ci = calculate_metrics_with_confidence_intervals(y_true, y_pred, y_proba)

# Plots
plot_multiple_roc_curves(predictions_dict)
plot_confusion_matrix_grid(predictions_dict, model_names)
plot_performance_comparison(results_dict)

# Statistical tests
result = mcnemars_test(y_true, y_pred1, y_pred2)
results = compare_all_models(cv_results, metric='f1')

# Cross-validation
cv_scores = cross_validate_model(model, X, y, n_splits=5)

# Generate all
generate_all_plots(results, X_tab, X_img, y, tab_features, img_features)
```

---

##  Common Tasks

### Calculate Metrics with 95% CI
```python
from src.evaluation import calculate_metrics_with_confidence_intervals

metrics = calculate_metrics_with_confidence_intervals(
    y_true, y_pred, y_proba,
    n_bootstrap=1000,
    confidence_level=0.95
)

# Access results
acc = metrics['accuracy']
print(f"Accuracy: {acc['value']:.4f} (95% CI: {acc['ci_lower']:.4f}-{acc['ci_upper']:.4f})")
```

### Plot ROC Curves (All Models)
```python
from src.evaluation import plot_multiple_roc_curves

predictions = {
    'XGBoost': (y_true, y_proba_xgb),
    'Random Forest': (y_true, y_proba_rf),
    'SVM': (y_true, y_proba_svm)
}

plot_multiple_roc_curves(predictions, save_path='results/figures/roc_curves.png')
```

### Statistical Model Comparison
```python
from src.evaluation import compare_all_models

cv_results = {
    'XGBoost': {'f1': [0.82, 0.80, 0.83, 0.81, 0.82]},
    'Random Forest': {'f1': [0.79, 0.78, 0.80, 0.79, 0.80]},
    'SVM': {'f1': [0.76, 0.75, 0.77, 0.76, 0.76]}
}

results = compare_all_models(cv_results, metric='f1', correction_method='holm')
```

### Export Results
```python
from src.evaluation import generate_results_json, generate_summary_report

generate_results_json(results, 'results/metrics/all_results.json')
generate_summary_report(results, 'results/metrics/summary_report.md')
```

---

##  Plotting Functions

| Function | Purpose | Output |
|----------|---------|--------|
| `plot_performance_comparison()` | Bar chart with error bars | `performance_comparison.png` |
| `plot_multiple_roc_curves()` | All ROC curves in one plot | `roc_curves.png` |
| `plot_confusion_matrix_grid()` | 2x2 confusion matrices | `confusion_matrices.png` |
| `plot_feature_importance_comparison()` | Top 15 features side-by-side | `feature_importance.png` |
| `plot_tsne_visualization()` | t-SNE with misclassifications | `tsne_*.png` |
| `plot_shap_summary()` | SHAP feature importance | `shap_*.png` |
| `generate_all_plots()` | Generate all above | All plots |

**All plots saved at 300 DPI PNG.**

---

##  Statistical Tests

| Test | Use Case | Function |
|------|----------|----------|
| **McNemar's** | Compare 2 classifiers on same test | `mcnemars_test()` |
| **Paired t-test** | Compare CV scores (parametric) | `paired_t_test()` |
| **Wilcoxon** | Compare CV scores (non-parametric) | `wilcoxon_signed_rank_test()` |
| **ANOVA** | Compare 3+ models (parametric) | `anova_test()` |
| **Friedman** | Compare 3+ models (non-parametric) | `friedman_test()` |
| **Bootstrap** | Performance difference (non-parametric) | `bootstrap_performance_difference()` |

**Multiple comparison correction:**
- `bonferroni_correction()` - Conservative
- `holm_bonferroni_correction()` - Less conservative (recommended)
- `compare_all_models()` - Automatic pairwise with correction

---

##  Output Format

### JSON Export
```json
{
  "baseline_results": {...},
  "fusion_results": {...},
  "statistical_tests": {...},
  "metadata": {
    "cv_folds": 5,
    "n_tabular_samples": 77,
    "dataset": "PD Classification"
  }
}
```

### Metrics with CI
```python
{
  'accuracy': {
    'value': 0.85,
    'ci_lower': 0.78,
    'ci_upper': 0.91,
    'std': 0.04
  },
  'f1': {...},
  'auc_roc': {...}
}
```

---

##  Best Practices

### For Small Datasets (~70 samples)

 **DO:**
- Use 5-fold stratified CV (minimum)
- Report bootstrap confidence intervals (1000 iterations)
- Apply multiple comparison correction
- Use non-parametric tests (Wilcoxon, Friedman)
- Report effect sizes (Cohen's d, r)
- Visualize with SHAP, t-SNE

 **DON'T:**
- Use single train/test split
- Report point estimates only
- Claim improvement without statistical test
- Compare multiple models without correction
- Ignore uncertainty in metrics

---

##  File Locations

```
src/evaluation/
 __init__.py                 # Exports all functions
 metrics.py                  # Metrics with bootstrap CI
 visualizations.py           # All plotting functions
 cross_validation.py         # CV utilities
 statistical_tests.py        # Statistical tests

scripts/
 comprehensive_evaluation.py # Full evaluation pipeline

docs/
 EVALUATION_GUIDE.md                 # Comprehensive guide
 EVALUATION_IMPLEMENTATION_SUMMARY.md # Implementation summary
 EVALUATION_QUICK_REFERENCE.md       # This file
```

---

##  Full Documentation

- **[EVALUATION_GUIDE.md](EVALUATION_GUIDE.md)** - Comprehensive documentation (500+ lines)
- **[EVALUATION_IMPLEMENTATION_SUMMARY.md](EVALUATION_IMPLEMENTATION_SUMMARY.md)** - Implementation details
- **[FUSION_GUIDE.md](FUSION_GUIDE.md)** - Fusion methods guide

---

##  Common Issues

**Q: SHAP not installed?**
```bash
pip install shap
```

**Q: Wide confidence intervals?**
- Increase `n_bootstrap` (2000-5000)
- Expected with small datasets
- Report honestly

**Q: McNemar's test p = 0.05?**
- Borderline case
- Report exact p-value
- Use bootstrap test as alternative

**Q: t-SNE fails?**
- Automatic perplexity adjustment included
- `perplexity = min(30, n_samples - 1)`

---

##  Pro Tips

1. **Use `generate_all_plots()`** - Creates all requested plots at once
2. **Always use CV** - Single split unreliable for small datasets
3. **Report CI** - `calculate_metrics_with_confidence_intervals()`
4. **Statistical tests** - `compare_all_models()` with correction
5. **Export results** - `generate_results_json()` + `generate_summary_report()`

---

**For help:** oliau@usc.edu
