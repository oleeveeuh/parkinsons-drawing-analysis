"""
Evaluation module for PD classification.

Provides comprehensive evaluation tools including:
- Metrics with bootstrap confidence intervals
- Statistical significance testing (McNemar's, Wilcoxon, etc.)
- Publication-quality visualizations (300 DPI)
- Cross-validation utilities
"""

from .metrics import (
    calculate_all_metrics,
    calculate_metrics_with_confidence_intervals,
    bootstrap_confidence_interval,
    print_metrics_report,
    mcnemars_test,
    generate_markdown_table,
    generate_results_json,
    generate_summary_report
)
from .visualizations import (
    plot_performance_comparison,
    plot_multiple_roc_curves,
    plot_confusion_matrix_grid,
    plot_confusion_matrix,
    plot_feature_importance_comparison,
    plot_tsne_visualization,
    plot_shap_summary,
    plot_shap_dependence,
    plot_cross_validation_scores,
    plot_learning_curve,
    plot_calibration_comparison,
    plot_roc_curve,
    generate_all_plots
)
from .cross_validation import (
    cross_validate_model,
    cross_validate_multiple_models,
    nested_cross_validation,
    cross_validate_with_stratification
)
from .statistical_tests import (
    paired_t_test,
    wilcoxon_signed_rank_test,
    mcnemars_test as mcnemars_test_stats,
    anova_test,
    friedman_test,
    bonferroni_correction,
    holm_bonferroni_correction,
    compare_all_models,
    bootstrap_performance_difference,
    generate_statistical_report
)

__all__ = [
    # Metrics
    'calculate_all_metrics',
    'calculate_metrics_with_confidence_intervals',
    'bootstrap_confidence_interval',
    'print_metrics_report',
    'mcnemars_test',
    'generate_markdown_table',
    'generate_results_json',
    'generate_summary_report',
    # Visualizations
    'plot_performance_comparison',
    'plot_multiple_roc_curves',
    'plot_confusion_matrix_grid',
    'plot_confusion_matrix',
    'plot_feature_importance_comparison',
    'plot_tsne_visualization',
    'plot_shap_summary',
    'plot_shap_dependence',
    'plot_cross_validation_scores',
    'plot_learning_curve',
    'plot_calibration_comparison',
    'plot_roc_curve',
    'generate_all_plots',
    # Cross-validation
    'cross_validate_model',
    'cross_validate_multiple_models',
    'nested_cross_validation',
    'cross_validate_with_stratification',
    # Statistical tests
    'paired_t_test',
    'wilcoxon_signed_rank_test',
    'mcnemars_test_stats',
    'anova_test',
    'friedman_test',
    'bonferroni_correction',
    'holm_bonferroni_correction',
    'compare_all_models',
    'bootstrap_performance_difference',
    'generate_statistical_report'
]
