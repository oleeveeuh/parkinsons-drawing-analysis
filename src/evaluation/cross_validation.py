"""
Stratified k-fold cross-validation for small datasets.

Proper cross-validation is critical for small datasets to get reliable
performance estimates and detect overfitting.
"""

import numpy as np
from typing import Dict, List, Any, Callable, Optional, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
import warnings

warnings.filterwarnings('ignore')

from .metrics import calculate_all_metrics, calculate_metrics_with_confidence_intervals


def cross_validate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    X_image: Optional[np.ndarray] = None,
    n_splits: int = 5,
    random_state: int = 42,
    verbose: bool = True
) -> Dict[str, List[float]]:
    """
    Perform stratified k-fold cross-validation.

    Args:
        model: Model instance with fit() and predict() methods
        X: Tabular features
        y: Labels
        X_image: Optional image features (for fusion models)
        n_splits: Number of folds
        random_state: Random seed
        verbose: Whether to print progress

    Returns:
        Dictionary of metric lists across folds
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    metrics_history = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'auc_roc': []
    }

    if verbose:
        print(f"\nPerforming {n_splits}-fold cross-validation...")
        print(f"Total samples: {len(X)}")
        print(f"Samples per fold: ~{len(X) // n_splits}")

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        if verbose:
            print(f"\nFold {fold}/{n_splits}")

        # Split data
        if X_image is not None:
            # Fusion model
            X_train_tab, X_test_tab = X[train_idx], X[test_idx]
            X_train_img, X_test_img = X_image[train_idx], X_image[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Clone model to get fresh instance
            model_fold = clone(model.model if hasattr(model, 'model') else model)

            # Fit
            if hasattr(model, 'fit'):  # Check if it's a fusion model
                model_fold.fit(X_train_tab, X_train_img, y_train)
                y_pred = model_fold.predict(X_test_tab, X_test_img)
                y_proba = model_fold.predict_proba(X_test_tab, X_test_img)
            else:
                raise ValueError("Invalid model type")
        else:
            # Single modality
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Clone model
            model_fold = clone(model.model if hasattr(model, 'model') else model)

            # Fit
            model_fold.fit(X_train, y_train)
            y_pred = model_fold.predict(X_test)
            y_proba = model_fold.predict_proba(X_test)

        # Calculate metrics
        fold_metrics = calculate_all_metrics(y_test, y_pred, y_proba)

        for metric_name in metrics_history.keys():
            if metric_name in fold_metrics:
                metrics_history[metric_name].append(fold_metrics[metric_name])

        if verbose:
            print(f"  Accuracy: {fold_metrics['accuracy']:.4f}")
            print(f"  F1: {fold_metrics['f1']:.4f}")
            if 'auc_roc' in fold_metrics:
                print(f"  AUC: {fold_metrics['auc_roc']:.4f}")

    # Print summary statistics
    if verbose:
        print(f"\n{'-'*60}")
        print("Cross-Validation Summary:")
        print(f"{'-'*60}")
        for metric_name, values in metrics_history.items():
            if values:
                mean = np.mean(values)
                std = np.std(values)
                print(f"{metric_name:<15}: {mean:.4f} (+/- {std:.4f})")

    return metrics_history


def cross_validate_multiple_models(
    models: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    X_image: Optional[np.ndarray] = None,
    n_splits: int = 5,
    random_state: int = 42
) -> Dict[str, Dict[str, List[float]]]:
    """
    Cross-validate multiple models for comparison.

    Args:
        models: Dictionary of {model_name: model_instance}
        X: Tabular features
        y: Labels
        X_image: Optional image features
        n_splits: Number of folds
        random_state: Random seed

    Returns:
        Nested dict of {model_name: {metric_name: [values]}}
    """
    results = {}

    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Cross-validating: {model_name}")
        print(f"{'='*60}")

        try:
            cv_results = cross_validate_model(
                model, X, y, X_image,
                n_splits=n_splits,
                random_state=random_state,
                verbose=True
            )
            results[model_name] = cv_results

        except Exception as e:
            print(f"Error with {model_name}: {e}")
            continue

    # Print comparison table
    print(f"\n\n{'='*80}")
    print("MODEL COMPARISON")
    print(f"{'='*80}")
    print(f"{'Model':<30} {'Accuracy':>12} {'F1':>12} {'AUC':>12}")
    print(f"{'-'*80}")

    for model_name, metrics in results.items():
        acc_mean = np.mean(metrics['accuracy']) if metrics['accuracy'] else 0
        acc_std = np.std(metrics['accuracy']) if metrics['accuracy'] else 0

        f1_mean = np.mean(metrics['f1']) if metrics['f1'] else 0
        f1_std = np.std(metrics['f1']) if metrics['f1'] else 0

        auc_mean = np.mean(metrics['auc_roc']) if metrics['auc_roc'] else 0
        auc_std = np.std(metrics['auc_roc']) if metrics['auc_roc'] else 0

        print(f"{model_name:<30} {acc_mean:>6.4f}±{acc_std:<4.4f} "
              f"{f1_mean:>6.4f}±{f1_std:<4.4f} "
              f"{auc_mean:>6.4f}±{auc_std:<4.4f}")

    print(f"{'='*80}\n")

    return results


def nested_cross_validation(
    model_class: type,
    param_grid: Dict[str, List[Any]],
    X: np.ndarray,
    y: np.ndarray,
    outer_splits: int = 5,
    inner_splits: int = 3,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Perform nested cross-validation for unbiased model evaluation.

    Outer loop: Evaluate model performance
    Inner loop: Hyperparameter tuning

    Args:
        model_class: Model class to instantiate
        param_grid: Hyperparameter grid
        X: Features
        y: Labels
        outer_splits: Number of outer folds
        inner_splits: Number of inner folds
        random_state: Random seed

    Returns:
        Dictionary with outer fold scores and best params
    """
    from sklearn.model_selection import GridSearchCV

    outer_skf = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    inner_skf = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)

    outer_scores = {
        'test_score': [],
        'best_params': []
    }

    print(f"\nNested Cross-Validation:")
    print(f"Outer folds: {outer_splits}, Inner folds: {inner_splits}")

    for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(outer_skf.split(X, y), 1):
        print(f"\nOuter Fold {outer_fold}/{outer_splits}")

        X_outer_train, X_outer_test = X[outer_train_idx], X[outer_test_idx]
        y_outer_train, y_outer_test = y[outer_train_idx], y[outer_test_idx]

        # Inner loop: hyperparameter tuning
        model = model_class()
        clf = GridSearchCV(
            model,
            param_grid,
            cv=inner_skf,
            scoring='f1',
            n_jobs=-1
        )

        clf.fit(X_outer_train, y_outer_train)

        # Evaluate on outer test set
        best_model = clf.best_estimator_
        score = best_model.score(X_outer_test, y_outer_test)

        outer_scores['test_score'].append(score)
        outer_scores['best_params'].append(clf.best_params_)

        print(f"  Best F1: {clf.best_score_:.4f}")
        print(f"  Test score: {score:.4f}")
        print(f"  Best params: {clf.best_params_}")

    # Summary
    print(f"\n{'-'*60}")
    print("Nested CV Results:")
    print(f"{'-'*60}")
    print(f"Mean test score: {np.mean(outer_scores['test_score']):.4f} "
          f"(±{np.std(outer_scores['test_score']):.4f})")

    return outer_scores


def cross_validate_with_stratification(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int = 42
) -> Dict[str, List[float]]:
    """
    Perform repeated stratified k-fold cross-validation.

    More robust estimates by repeating CV with different random seeds.

    Args:
        model: Model instance
        X: Features
        y: Labels
        n_splits: Number of folds per repeat
        n_repeats: Number of repetitions
        random_state: Random seed

    Returns:
        Dictionary of metric lists across all folds and repeats
    """
    try:
        from sklearn.model_selection import RepeatedStratifiedKFold
        rskf = RepeatedStratifiedKFold(
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=random_state
        )
    except ImportError:
        # Fallback to regular stratified k-fold
        print("Warning: RepeatedStratifiedKFold not available, using regular StratifiedKFold")
        from sklearn.model_selection import StratifiedKFold
        rskf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    metrics_history = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    total_folds = n_splits * n_repeats

    print(f"\nRepeated Stratified K-Fold CV:")
    print(f"Folds per repeat: {n_splits}, Repeats: {n_repeats}")
    print(f"Total folds: {total_folds}")

    for fold, (train_idx, test_idx) in enumerate(rskf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Clone and fit model
        model_fold = clone(model.model if hasattr(model, 'model') else model)
        model_fold.fit(X_train, y_train)

        # Predict
        y_pred = model_fold.predict(X_test)
        y_proba = model_fold.predict_proba(X_test)

        # Metrics
        fold_metrics = calculate_all_metrics(y_test, y_pred, y_proba)

        for metric_name in metrics_history.keys():
            if metric_name in fold_metrics:
                metrics_history[metric_name].append(fold_metrics[metric_name])

        if fold % 5 == 0:
            print(f"Completed {fold}/{total_folds} folds")

    # Summary
    print(f"\n{'-'*60}")
    print("Summary Statistics:")
    for metric_name, values in metrics_history.items():
        if values:
            print(f"{metric_name:<15}: {np.mean(values):.4f} (±{np.std(values):.4f})")

    return metrics_history
