"""
Publication-quality visualizations for model evaluation.

Creates professional plots suitable for README, presentations, and papers.
All plots saved at 300 DPI PNG for high-resolution output.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.manifold import TSNE
import pandas as pd

# Set publication-quality style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10


def plot_performance_comparison(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = ['accuracy', 'f1', 'auc_roc'],
    save_path: str = 'results/figures/performance_comparison.png',
    figsize: Tuple[int, int] = (14, 5)
):
    """
    Create performance comparison bar chart for all models.

    Shows multiple metrics side-by-side with error bars (std across folds).

    Args:
        results: Dict of {model_name: {metric: value, metric_std: std}}
        metrics: List of metrics to plot
        save_path: Path to save figure
        figsize: Figure size (width, height)
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)

    # Ensure axes is always iterable
    if len(metrics) == 1:
        axes = [axes]

    model_names = list(results.keys())

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        # Extract values and stds
        values = []
        stds = []
        for model_name in model_names:
            if metric in results[model_name]:
                values.append(results[model_name][metric].get('value', results[model_name][metric]))
                stds.append(results[model_name][metric].get('std', 0))
            else:
                values.append(0)
                stds.append(0)

        # Create bar chart
        x_pos = np.arange(len(model_names))
        bars = ax.bar(x_pos, values, yerr=stds, capsize=5, alpha=0.8,
                      color=sns.color_palette("husl", len(model_names)))

        # Customize
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.0])

        # Add value labels on bars
        for i, (bar, val, std) in enumerate(zip(bars, values, stds)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Performance comparison saved to {save_path}")


def plot_multiple_roc_curves(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: str = 'results/figures/roc_curves.png',
    figsize: Tuple[int, int] = (10, 8),
    title: str = 'ROC Curves - Model Comparison'
):
    """
    Plot ROC curves for all models in one figure.

    Includes AUC values in legend and shaded area under curves.

    Args:
        results: Dict of {model_name: (y_true, y_proba)}
        save_path: Path to save figure
        figsize: Figure size
        title: Plot title
    """
    plt.figure(figsize=figsize)

    colors = sns.color_palette("husl", len(results))

    for (model_name, (y_true, y_proba)), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve with shaded area
        plt.plot(fpr, tpr, color=color, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
        plt.fill_between(fpr, tpr, alpha=0.1, color=color)

    # Diagonal reference line
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" ROC curves saved to {save_path}")


def plot_confusion_matrix_grid(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    model_names: List[str],
    class_names: List[str] = ['Healthy', 'Parkinson\'s'],
    save_path: str = 'results/figures/confusion_matrices.png',
    figsize: Tuple[int, int] = (14, 12)
):
    """
    Plot confusion matrices in a 2x2 grid.

    Shows both counts and percentages, annotated with sensitivity/specificity.

    Args:
        results: Dict of {model_name: (y_true, y_pred)}
        model_names: List of 4 model names to plot (for 2x2 grid)
        class_names: Names for classes
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.ravel()

    for idx, model_name in enumerate(model_names[:4]):  # Only first 4 models
        if model_name not in results:
            continue

        y_true, y_pred = results[model_name]
        cm = confusion_matrix(y_true, y_pred)

        # Calculate percentages
        cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        # Calculate sensitivity and specificity
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   cbar=True, square=True, linewidths=1)

        # Add percentage annotations
        for i in range(2):
            for j in range(2):
                text = axes[idx].text(j + 0.5, i + 0.7, f'({cm_pct[i, j]:.1f}%)',
                                     ha='center', va='center', color='red', fontsize=9)

        axes[idx].set_title(f'{model_name}\nSensitivity: {sensitivity:.3f}, Specificity: {specificity:.3f}',
                          fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Predicted Label', fontsize=10)
        axes[idx].set_ylabel('True Label', fontsize=10)
        axes[idx].set_xticklabels(class_names, rotation=0)
        axes[idx].set_yticklabels(class_names, rotation=0)

    plt.suptitle('Confusion Matrices Comparison', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Confusion matrices saved to {save_path}")


def plot_feature_importance_comparison(
    tabular_importance: np.ndarray,
    tabular_feature_names: List[str],
    image_importance: np.ndarray,
    image_feature_names: List[str],
    tabular_model_name: str = 'Tabular Model',
    image_model_name: str = 'Image Model',
    top_n: int = 15,
    save_path: str = 'results/figures/feature_importance.png',
    figsize: Tuple[int, int] = (16, 7)
):
    """
    Plot top features from tabular and image models side-by-side.

    Args:
        tabular_importance: Feature importance for tabular model
        tabular_feature_names: Feature names for tabular
        image_importance: Feature importance for image model
        image_feature_names: Feature names for image
        tabular_model_name: Name for tabular model title
        image_model_name: Name for image model title
        top_n: Number of top features to show
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Tabular features
    tab_indices = np.argsort(tabular_importance)[::-1][:top_n]
    tab_top_importance = tabular_importance[tab_indices][::-1]
    tab_top_names = [tabular_feature_names[i] for i in tab_indices[::-1]]

    axes[0].barh(range(top_n), tab_top_importance, color='steelblue', alpha=0.8)
    axes[0].set_yticks(range(top_n))
    axes[0].set_yticklabels(tab_top_names, fontsize=9)
    axes[0].set_xlabel('Importance Score', fontsize=11)
    axes[0].set_title(f'Top {top_n} Features - {tabular_model_name}', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')
    axes[0].invert_yaxis()

    # Image features
    img_indices = np.argsort(image_importance)[::-1][:top_n]
    img_top_importance = image_importance[img_indices][::-1]
    img_top_names = [image_feature_names[i] for i in img_indices[::-1]]

    axes[1].barh(range(top_n), img_top_importance, color='coral', alpha=0.8)
    axes[1].set_yticks(range(top_n))
    axes[1].set_yticklabels(img_top_names, fontsize=9)
    axes[1].set_xlabel('Importance Score', fontsize=11)
    axes[1].set_title(f'Top {top_n} Features - {image_model_name}', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')
    axes[1].invert_yaxis()

    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Feature importance comparison saved to {save_path}")


def plot_tsne_visualization(
    X: np.ndarray,
    y: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
    feature_name: str = 'Features',
    save_path: str = 'results/figures/tsne_visualization.png',
    figsize: Tuple[int, int] = (12, 5),
    random_state: int = 42
):
    """
    Create t-SNE visualization of learned representations.

    Colors points by true class, marks misclassifications with X.

    Args:
        X: Feature matrix (N, D)
        y: True labels
        y_pred: Predicted labels (optional, for marking misclassifications)
        feature_name: Name for title
        save_path: Path to save figure
        figsize: Figure size
        random_state: Random seed for t-SNE
    """
    # Apply t-SNE
    print(f"Applying t-SNE to {X.shape} features...")
    tsne = TSNE(
        n_components=2,
        random_state=random_state,
        perplexity=min(30, max(5, X.shape[0]-1))  # Ensure perplexity < n_samples
    )
    X_tsne = tsne.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Colored by true class
    scatter = axes[0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis',
                              alpha=0.7, s=100, edgecolors='black', linewidth=0.5)
    axes[0].set_title(f't-SNE: {feature_name}\nColored by True Class', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('t-SNE Component 1', fontsize=10)
    axes[0].set_ylabel('t-SNE Component 2', fontsize=10)
    plt.colorbar(scatter, ax=axes[0], label='Class (0=Healthy, 1=Parkinson)')

    # Plot 2: Mark misclassifications if predictions provided
    if y_pred is not None:
        correct = y == y_pred
        misclassified = y != y_pred

        # Plot correct classifications
        axes[1].scatter(X_tsne[correct, 0], X_tsne[correct, 1],
                       c=y[correct], cmap='viridis', alpha=0.7,
                       s=100, edgecolors='black', linewidth=0.5, label='Correct')

        # Plot misclassifications with red X
        axes[1].scatter(X_tsne[misclassified, 0], X_tsne[misclassified, 1],
                       c=y[misclassified], cmap='viridis', alpha=0.9,
                       s=150, marker='X', edgecolors='red', linewidth=2,
                       label=f'Misclassified ({misclassified.sum()})')

        axes[1].set_title(f't-SNE: {feature_name}\nMark Misclassifications',
                         fontsize=12, fontweight='bold')
        axes[1].set_xlabel('t-SNE Component 1', fontsize=10)
        axes[1].set_ylabel('t-SNE Component 2', fontsize=10)
        axes[1].legend(loc='best', fontsize=9)
    else:
        # Duplicate first plot if no predictions
        scatter = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis',
                                 alpha=0.7, s=100, edgecolors='black', linewidth=0.5)
        axes[1].set_title(f't-SNE: {feature_name}\nColored by True Class', fontsize=12)
        plt.colorbar(scatter, ax=axes[1], label='Class')

    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" t-SNE visualization saved to {save_path}")


def plot_shap_summary(
    model: Any,
    X: np.ndarray,
    feature_names: List[str],
    max_features: int = 20,
    save_path: str = 'results/figures/shap_summary.png',
    figsize: Tuple[int, int] = (10, 8),
    plot_type: str = 'bar'
):
    """
    Create SHAP summary plot for model interpretability.

    Shows global feature importance and impact direction (positive/negative).

    Args:
        model: Trained model (must have predict_proba method)
        X: Feature matrix (use subset for speed)
        feature_names: List of feature names
        max_features: Maximum features to show
        save_path: Path to save figure
        figsize: Figure size
        plot_type: 'bar' for bar chart, 'dot' for beeswarm plot
    """
    try:
        import shap
    except ImportError:
        print("Warning: SHAP not installed. Install with: pip install shap")
        return

    print(f"Computing SHAP values for {X.shape} samples...")

    # Use subset for speed
    if X.shape[0] > 500:
        sample_idx = np.random.choice(X.shape[0], 500, replace=False)
        X_sample = X[sample_idx]
    else:
        X_sample = X

    # Create SHAP explainer
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # For binary classification, use positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    except:
        # Fallback to KernelExplainer for non-tree models
        explainer = shap.KernelExplainer(model.predict_proba, shap.kmeans(X_sample, 20))
        shap_values = explainer.shap_values(X_sample, nsamples=100)

        # For binary classification, use positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

    # Create summary plot
    plt.figure(figsize=figsize)

    if plot_type == 'bar':
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                         max_features=max_features, plot_type='bar', show=False)
    else:  # beeswarm
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                         max_features=max_features, show=False)

    plt.title(f'SHAP Feature Importance - Global Interpretability',
             fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" SHAP summary plot saved to {save_path}")


def plot_shap_dependence(
    model: Any,
    X: np.ndarray,
    feature_names: List[str],
    top_features: List[str],
    save_path: str = 'results/figures/shap_dependence.png',
    figsize: Tuple[int, int] = (14, 10)
):
    """
    Create SHAP dependence plots for top features.

    Shows how feature values affect predictions.

    Args:
        model: Trained model
        X: Feature matrix
        feature_names: List of feature names
        top_features: List of feature names to plot
        save_path: Path to save figure
        figsize: Figure size
    """
    try:
        import shap
    except ImportError:
        print("Warning: SHAP not installed. Install with: pip install shap")
        return

    # Use subset for speed
    if X.shape[0] > 500:
        sample_idx = np.random.choice(X.shape[0], 500, replace=False)
        X_sample = X[sample_idx]
    else:
        X_sample = X

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # For binary classification, use positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Get feature indices
    feature_indices = [feature_names.index(f) for f in top_features if f in feature_names]

    # Create dependence plots
    n_features = len(feature_indices)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.ravel() if n_features > 1 else [axes]

    for idx, feat_idx in enumerate(feature_indices):
        if idx >= len(axes):
            break

        # Scatter plot: feature value vs SHAP value
        axes[idx].scatter(X_sample[:, feat_idx], shap_values[:, feat_idx],
                         alpha=0.6, s=30, c=shap_values[:, feat_idx],
                         cmap='coolwarm', edgecolors='black', linewidth=0.3)

        axes[idx].set_xlabel(feature_names[feat_idx], fontsize=10)
        axes[idx].set_ylabel('SHAP Value', fontsize=10)
        axes[idx].set_title(f'{feature_names[feat_idx]}', fontsize=11, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)

        # Add trend line
        z = np.polyfit(X_sample[:, feat_idx], shap_values[:, feat_idx], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(X_sample[:, feat_idx].min(), X_sample[:, feat_idx].max(), 100)
        axes[idx].plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)

    # Hide unused subplots
    for idx in range(len(feature_indices), len(axes)):
        axes[idx].axis('off')

    plt.suptitle('SHAP Dependence Plots - Feature Impact Direction',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" SHAP dependence plots saved to {save_path}")


def plot_cross_validation_scores(
    scores: Dict[str, List[float]],
    metric: str = 'accuracy',
    save_path: str = 'results/figures/cv_scores.png',
    figsize: Tuple[int, int] = (12, 6)
):
    """
    Plot cross-validation scores across folds as box plots.

    Args:
        scores: Dict of {model_name: [score_fold1, score_fold2, ...]}
        metric: Metric name for labels
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Prepare data
    data = [scores[model] for model in scores.keys()]
    labels = list(scores.keys())

    # Create box plot
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True,
                    medianprops=dict(linewidth=2, color='black'),
                    boxprops=dict(linewidth=1.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5))

    # Color the boxes
    colors = sns.color_palette("husl", len(labels))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel(f'{metric.replace("_", " ").title()}', fontsize=12)
    ax.set_title(f'Cross-Validation {metric.replace("_", " ").title()} Scores',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" CV scores plot saved to {save_path}")


def plot_learning_curve(
    train_scores: List[float],
    val_scores: List[float],
    train_sizes: List[int],
    model_name: str = 'Model',
    save_path: str = 'results/figures/learning_curve.png',
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot learning curve showing performance vs training size.

    Useful for diagnosing bias/variance problems.

    Args:
        train_scores: Training scores across increasing training sizes
        val_scores: Validation scores across increasing training sizes
        train_sizes: Number of training samples used
        model_name: Name for title
        save_path: Path to save figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    plt.plot(train_sizes, train_scores, 'o-', color='blue', label='Training Score', linewidth=2)
    plt.plot(train_sizes, val_scores, 'o-', color='red', label='Validation Score', linewidth=2)

    plt.xlabel('Training Set Size', fontsize=12)
    plt.ylabel('Score (F1)', fontsize=12)
    plt.title(f'Learning Curve - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])

    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Learning curve saved to {save_path}")


def plot_calibration_comparison(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: str = 'results/figures/calibration_curves.png',
    figsize: Tuple[int, int] = (10, 8),
    n_bins: int = 10
):
    """
    Plot calibration curves for multiple models.

    Shows how well-calibrated predicted probabilities are.

    Args:
        results: Dict of {model_name: (y_true, y_proba)}
        save_path: Path to save figure
        figsize: Figure size
        n_bins: Number of bins for calibration
    """
    from sklearn.calibration import calibration_curve

    plt.figure(figsize=figsize)

    colors = sns.color_palette("husl", len(results))

    for (model_name, (y_true, y_proba)), color in zip(results.items(), colors):
        prob_true, prob_pred = calibration_curve(y_true, y_proba[:, 1], n_bins=n_bins)

        plt.plot(prob_pred, prob_true, marker='o', linewidth=2, color=color,
                label=model_name, alpha=0.8)

    # Perfectly calibrated reference line
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2,
            label='Perfectly Calibrated', alpha=0.8)

    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('True Probability', fontsize=12)
    plt.title('Calibration Curves - Reliability Diagram', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Calibration curves saved to {save_path}")


def generate_all_plots(
    results: Dict[str, Any],
    X_tabular: np.ndarray,
    X_image: np.ndarray,
    y: np.ndarray,
    tabular_feature_names: List[str],
    image_feature_names: List[str],
    output_dir: str = 'results/figures/'
):
    """
    Generate all visualization plots for the project.

    This is a convenience function that creates all requested plots.

    Args:
        results: Dictionary containing all experimental results
        X_tabular: Tabular feature matrix
        X_image: Image feature matrix
        y: True labels
        tabular_feature_names: Names of tabular features
        image_feature_names: Names of image features
        output_dir: Directory to save plots
    """
    print("\n" + "="*70)
    print("GENERATING ALL VISUALIZATIONS")
    print("="*70)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 1. Performance comparison bar chart
    if 'cv_results' in results:
        print("\n[1/6] Performance comparison bar chart...")
        plot_performance_comparison(
            results['cv_results'],
            save_path=f'{output_dir}/performance_comparison.png'
        )

    # 2. ROC curves (all models)
    if 'predictions' in results:
        print("\n[2/6] ROC curves...")
        plot_multiple_roc_curves(
            results['predictions'],
            save_path=f'{output_dir}/roc_curves.png'
        )

    # 3. Confusion matrices (2x2 grid)
    if 'predictions' in results:
        print("\n[3/6] Confusion matrices...")
        model_names = list(results['predictions'].keys())[:4]
        plot_confusion_matrix_grid(
            results['predictions'],
            model_names,
            save_path=f'{output_dir}/confusion_matrices.png'
        )

    # 4. Feature importance comparison
    if 'feature_importance' in results:
        print("\n[4/6] Feature importance plots...")
        plot_feature_importance_comparison(
            results['feature_importance']['tabular']['importance'],
            tabular_feature_names,
            results['feature_importance']['image']['importance'],
            image_feature_names,
            save_path=f'{output_dir}/feature_importance.png'
        )

    # 5. t-SNE visualization
    print("\n[5/6] t-SNE visualization...")
    plot_tsne_visualization(
        X_tabular, y,
        feature_name='Tabular Features',
        save_path=f'{output_dir}/tsne_tabular.png'
    )

    plot_tsne_visualization(
        X_image, y,
        feature_name='Image Features',
        save_path=f'{output_dir}/tsne_image.png'
    )

    # 6. SHAP plots (if model available)
    if 'best_model' in results:
        print("\n[6/6] SHAP summary plots...")
        try:
            # Tabular model SHAP
            if 'best_tabular_model' in results:
                plot_shap_summary(
                    results['best_tabular_model'],
                    X_tabular,
                    tabular_feature_names,
                    save_path=f'{output_dir}/shap_tabular.png'
                )

            # Image model SHAP
            if 'best_image_model' in results:
                plot_shap_summary(
                    results['best_image_model'],
                    X_image,
                    image_feature_names,
                    save_path=f'{output_dir}/shap_image.png'
                )
        except Exception as e:
            print(f"Warning: Could not generate SHAP plots: {e}")

    print(f"\n{'='*70}")
    print(f" All visualizations saved to {output_dir}")
    print(f"{'='*70}\n")


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = 'Model',
    save_path: str = 'results/figures/roc_curve.png',
    figsize: Tuple[int, int] = (8, 6)
):
    """
    Plot single ROC curve with AUC score.

    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities (can be 1D or 2D with positive class in column 1)
        model_name: Name for title and legend
        save_path: Path to save figure
        figsize: Figure size
    """
    from sklearn.metrics import roc_curve, auc

    # Handle both 1D and 2D probability arrays
    if y_proba.ndim == 2:
        if y_proba.shape[1] == 2:
            # Take probability of positive class (column 1)
            y_proba = y_proba[:, 1]
        else:
            raise ValueError(f"Expected 2 columns for probabilities, got {y_proba.shape[1]}")

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" ROC curve saved to {save_path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = 'Model',
    save_path: str = 'results/figures/confusion_matrix.png',
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = 'Blues'
):
    """
    Plot confusion matrix with annotations.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name for title
        save_path: Path to save figure
        figsize: Figure size
        cmap: Colormap name
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar_kws={"label": "Count"},
                square=True, linewidths=1, linecolor='white')

    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')

    # Class labels
    tick_labels = ['Healthy', 'PD']
    plt.xticks(np.arange(2) + 0.5, tick_labels, rotation=0)
    plt.yticks(np.arange(2) + 0.5, tick_labels, rotation=0)

    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Confusion matrix saved to {save_path}")
