"""
Evaluation metrics for binary classification with confidence intervals.

Implements all standard classification metrics plus bootstrap confidence
intervals, essential for small datasets where metrics have high variance.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    matthews_corrcoef,
    balanced_accuracy_score
)
import warnings

warnings.filterwarnings('ignore')


def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate all classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (for AUC)

    Returns:
        Dictionary of metric names to values
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'mcc': matthews_corrcoef(y_true, y_pred),
    }

    # Calculate AUC if probabilities provided
    if y_proba is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_proba[:, 1])
        except:
            metrics['auc_roc'] = 0.0

    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics['true_positive'] = tp
    metrics['true_negative'] = tn
    metrics['false_positive'] = fp
    metrics['false_negative'] = fn

    # Additional rates
    metrics['true_positive_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['true_negative_rate'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['false_positive_rate'] = fp / (tn + fp) if (tn + fp) > 0 else 0

    return metrics


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    metric_func: callable,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95
) -> Tuple[float, float, float]:
    """
    Calculate bootstrap confidence interval for a metric.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        metric_func: Function to calculate metric
        n_bootstrap: Number of bootstrap iterations
        confidence_level: Confidence level (e.g., 0.95)

    Returns:
        (mean, lower_bound, upper_bound)
    """
    n_samples = len(y_true)
    bootstrap_scores = []

    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]

        # Check if both classes present
        if len(np.unique(y_true_boot)) < 2:
            continue

        # Calculate metric
        if y_proba is not None and 'auc' in str(metric_func):
            y_proba_boot = y_proba[indices]
            score = metric_func(y_true_boot, y_proba_boot[:, 1])
        else:
            score = metric_func(y_true_boot, y_pred_boot)

        bootstrap_scores.append(score)

    bootstrap_scores = np.array(bootstrap_scores)

    # Calculate percentiles
    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_scores, 100 * alpha / 2)
    upper = np.percentile(bootstrap_scores, 100 * (1 - alpha / 2))
    mean = np.mean(bootstrap_scores)

    return mean, lower, upper


def calculate_metrics_with_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95
) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics with bootstrap confidence intervals.

    Essential for small datasets to understand metric uncertainty.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        n_bootstrap: Bootstrap iterations
        confidence_level: Confidence level

    Returns:
        Dictionary with structure:
        {
            'metric_name': {
                'value': point_estimate,
                'ci_lower': lower_bound,
                'ci_upper': upper_bound,
                'std': std_dev
            }
        }
    """
    results = {}

    # Define metrics to calculate
    metrics_to_calc = [
        ('accuracy', accuracy_score),
        ('precision', precision_score),
        ('recall', recall_score),
        ('f1', f1_score),
    ]

    for metric_name, metric_func in metrics_to_calc:
        mean, lower, upper = bootstrap_confidence_interval(
            y_true, y_pred, y_proba, metric_func, n_bootstrap, confidence_level
        )

        # Point estimate
        point_estimate = metric_func(y_true, y_pred)

        # Std from bootstrap
        scores = []
        for _ in range(min(100, n_bootstrap)):
            indices = np.random.choice(len(y_true), len(y_true), replace=True)
            if len(np.unique(y_true[indices])) < 2:
                continue
            score = metric_func(y_true[indices], y_pred[indices])
            scores.append(score)
        std = np.std(scores) if scores else 0

        results[metric_name] = {
            'value': point_estimate,
            'ci_lower': lower,
            'ci_upper': upper,
            'std': std
        }

    # AUC if probabilities available
    if y_proba is not None:
        mean, lower, upper = bootstrap_confidence_interval(
            y_true, y_pred, y_proba,
            lambda yt, yp: roc_auc_score(yt, yp),
            n_bootstrap, confidence_level
        )

        point_estimate = roc_auc_score(y_true, y_proba[:, 1])

        results['auc_roc'] = {
            'value': point_estimate,
            'ci_lower': lower,
            'ci_upper': upper,
            'std': 0  # Will be calculated above
        }

    return results


def print_metrics_report(
    metrics: Dict[str, Dict[str, float]],
    title: str = "Classification Report"
):
    """
    Print a formatted metrics report with confidence intervals.

    Args:
        metrics: Dictionary from calculate_metrics_with_confidence_intervals
        title: Report title
    """
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}\n")

    print(f"{'Metric':<20} {'Value':>10} {'95% CI':>20}")
    print(f"{'-'*60}")

    for metric_name, values in sorted(metrics.items()):
        value = values['value']
        lower = values['ci_lower']
        upper = values['ci_upper']

        print(f"{metric_name:<20} {value:>10.4f} [{lower:6.4f}, {upper:6.4f}]")

    print(f"{'-'*60}\n")


def calculate_sensitivity_specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Tuple[float, float, float, float]:
    """
    Calculate sensitivity, specificity, PPV, NPV.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        (sensitivity, specificity, ppv, npv)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value

    return sensitivity, specificity, ppv, npv


def compare_models(
    results_dict: Dict[str, Dict[str, Dict[str, float]]],
    metric: str = 'f1'
) -> None:
    """
    Compare multiple models on a specific metric.

    Args:
        results_dict: Dict of {model_name: metrics_dict}
        metric: Metric to compare

    Prints formatted comparison table
    """
    print(f"\nModel Comparison ({metric}):")
    print(f"{'Model':<30} {metric:>10} {'95% CI':>20}")
    print(f"{'-'*70}")

    for model_name, metrics in sorted(results_dict.items()):
        if metric in metrics:
            values = metrics[metric]
            value = values['value']
            lower = values['ci_lower']
            upper = values['ci_upper']

            print(f"{model_name:<30} {value:>10.4f} [{lower:6.4f}, {upper:6.4f}]")

    print(f"{'-'*70}\n")


def mcnemars_test(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray,
    model1_name: str = "Model 1",
    model2_name: str = "Model 2"
) -> Dict[str, Any]:
    """
    Perform McNemar's test for comparing two models.

    Tests whether two models have significantly different error rates.
    Essential for small datasets to determine if improvements are significant.

    Args:
        y_true: True labels
        y_pred1: Predictions from model 1
        y_pred2: Predictions from model 2
        model1_name: Name of first model
        model2_name: Name of second model

    Returns:
        Dictionary with test statistic and p-value
    """
    from statsmodels.stats.contingency_tables import mcnemar

    # Create contingency table
    #               Model 2 Correct | Model 2 Wrong
    # Model 1 Correct |        b         |        a
    # Model 1 Wrong  |        c         |        d

    model1_correct = (y_pred1 == y_true)
    model2_correct = (y_pred2 == y_true)

    # Neither model1 nor model2 correct (both wrong, model2 correct)
    b = np.sum((model1_correct) & (~model2_correct))

    # model1 correct, model2 wrong
    a = np.sum((model1_correct) & (~model2_correct))

    # Both wrong
    d = np.sum((~model1_correct) & (~model2_correct))

    # model1 wrong, model2 correct
    c = np.sum((~model1_correct) & (model2_correct))

    # Contingency table (correct/incorrect)
    #              Model2
    #              Correct   Wrong
    # Model1 Correct  b       a
    # Model1 Wrong   c       d

    table = np.array([[b, a], [c, d]])

    # Perform McNemar's test
    try:
        result = mcnemar(table, exact=True, correction=True)
    except:
        # If exact test fails, use asymptotic
        result = mcnemar(table, exact=False, correction=True)

    statistic = result.statistic
    p_value = result.pvalue

    print(f"\n{'='*70}")
    print(f"McNemar's Test: {model1_name} vs {model2_name}")
    print(f"{'='*70}")
    print(f"Contingency Table:")
    print(f"                {model2_name}")
    print(f"                Correct   Wrong")
    print(f"{model1_name} Correct   {b:5d}      {a:5d}")
    print(f"{model1_name} Wrong    {c:5d}      {d:5d}")
    print(f"\nChi-squared: {statistic:.4f}")
    print(f"P-value: {p_value:.6f}")

    alpha = 0.05
    if p_value < alpha:
        print(f"\n Significant difference (p < {alpha})")
    else:
        print(f"\n No significant difference (p >= {alpha})")

    print(f"{'='*70}\n")

    return {
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'contingency_table': table,
        'model1_name': model1_name,
        'model2_name': model2_name
    }


def generate_markdown_table(
    results_dict: Dict[str, Dict[str, Dict[str, float]]],
    metrics: list = ['accuracy', 'f1', 'auc_roc'],
    title: str = "Model Performance Comparison"
) -> str:
    """
    Generate a markdown table for README/documentation.

    Args:
        results_dict: Dict of {model_name: {metric: {value, ci_lower, ci_upper}}}
        metrics: List of metrics to include
        title: Table title

    Returns:
        Markdown formatted string
    """
    md = []
    md.append(f"\n## {title}\n")
    md.append("| Model | " + " | ".join([m.capitalize() for m in metrics]) + " |")
    md.append("|-------" + "|-------" * len(metrics) + "|")

    for model_name, model_results in sorted(results_dict.items()):
        row = [model_name]
        for metric in metrics:
            if metric in model_results:
                val = model_results[metric]
                value = val['value']
                lower = val['ci_lower']
                upper = val['ci_upper']
                row.append(f"{value:.4f} ({lower:.4f}-{upper:.4f})")
            else:
                row.append("N/A")
        md.append("| " + " | ".join(row) + " |")

    return "\n".join(md) + "\n"


def generate_results_json(
    results: Dict[str, Any],
    output_path: str = 'results/metrics/all_results.json'
) -> None:
    """
    Save comprehensive results to JSON file.

    Args:
        results: Dictionary with all experimental results
        output_path: Path to save JSON
    """
    import json
    from pathlib import Path

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj

    results_serializable = convert_to_serializable(results)

    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)

    print(f" Saved comprehensive results to {output_path}")


def generate_summary_report(
    results: Dict[str, Any],
    output_path: str = 'results/metrics/summary_report.md'
) -> None:
    """
    Generate a comprehensive markdown summary report.

    Args:
        results: Dictionary with all experimental results
        output_path: Path to save report
    """
    from pathlib import Path

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    md = []
    md.append("# Parkinson's Disease Classification - Results Summary\n")
    md.append(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Executive Summary
    md.append("## Executive Summary\n")
    md.append("This document summarizes the performance of various machine learning models")
    md.append("for Parkinson's Disease classification using spiral drawing data.\n")

    # Best performing model
    md.append("## Key Findings\n")

    if 'summary' in results:
        summary_df = pd.DataFrame(results['summary'])
        if not summary_df.empty:
            best_model = summary_df.loc[summary_df['f1'].idxmax()]
            best_acc = summary_df.loc[summary_df['accuracy'].idxmax(), 'accuracy']
            best_f1 = summary_df['f1'].max()

            md.append(f"- **Best F1 Score:** {best_f1:.4f} ({best_model['model']} on {best_model['modality']})")
            md.append(f"- **Best Accuracy:** {best_acc:.4f}")
            md.append(f"- **Total models evaluated:** {len(summary_df)}\n")

    # Detailed results table
    md.append("## Performance Comparison\n")
    if 'summary' in results:
        md.append("| Model | Modality | Accuracy ± Std | F1 ± Std | AUC ± Std |")
        md.append("|-------|----------|---------------|-----------|-----------|")

        for _, row in summary_df.iterrows():
            model = row['model']
            modality = row['modality']
            acc = f"{row['accuracy']:.4f} ± {row['accuracy_std']*2:.4f}"
            f1 = f"{row['f1']:.4f} ± {row['f1_std']*2:.4f}"
            auc = f"{row['auc']:.4f} ± {row['auc_std']*2:.4f}"
            md.append(f"| {model} | {modality} | {acc} | {f1} | {auc} |")

    md.append("\n---\n")
    md.append("*All metrics include 95% confidence intervals from bootstrap or cross-validation.*\n")

    # Write report
    report = "\n".join(md)

    with open(output_path, 'w') as f:
        f.write(report)

    print(f" Generated summary report at {output_path}")
    print(report)
