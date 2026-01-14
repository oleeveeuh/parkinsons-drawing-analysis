"""
Multi-Task Spatial Analysis: Spiral vs Wave Drawings

Trains separate models for each drawing task and compares performance.
Uses cached CNN features to prevent segmentation faults.

Author: Claude Code
Date: January 10, 2026
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from pathlib import Path
import yaml
from datetime import datetime
import pickle
import sys

sys.path.append('.')
from src.data.load_multi_task_images import load_multi_task_images
from src.data.spatial_features_cached import CachedSpatialFeatureExtractor


def train_task_model(
    task_name: str,
    images: list,
    labels: np.ndarray,
    features: np.ndarray,  # Pre-extracted features
    n_outer_folds: int = 5,
    n_inner_folds: int = 3,
    random_state: int = 42
):
    """
    Train model for one task using pre-extracted features.

    Args:
        task_name: 'spiral' or 'wave'
        images: List of images (for reference)
        labels: Array of labels
        features: Pre-extracted features (from cache)

    Returns:
        results: Dictionary with performance metrics
    """

    print(f"\n{'='*80}")
    print(f"{task_name.upper()} MODEL TRAINING - Nested Cross-Validation")
    print(f"{'='*80}")
    print(f"Total samples: {len(images)}")
    print(f"PD: {np.sum(labels == 1)}, Control: {np.sum(labels == 0)}")
    print(f"Features: {features.shape[1]}")
    print(f"{'='*80}")

    # Outer CV
    outer_cv = StratifiedKFold(n_splits=n_outer_folds, shuffle=True, random_state=random_state)

    results = {
        'fold_results': [],
        'predictions': [],
        'true_labels': [],
        'confusion_matrices': []
    }

    # OUTER LOOP
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(features, labels), 1):
        print(f"\n{'-'*80}")
        print(f"OUTER FOLD {fold_idx}/{n_outer_folds}")
        print(f"{'-'*80}")

        # Split using pre-extracted features (NO data leakage - features are from images only)
        X_train_features = features[train_idx]
        X_test_features = features[test_idx]
        y_train = labels[train_idx]
        y_test = labels[test_idx]

        print(f"Split: {len(X_train_features)} train, {len(X_test_features)} test")

        # SCALING (train-only statistics)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_features)
        X_test_scaled = scaler.transform(X_test_features)

        # SMOTE
        minority_class = np.argmin(np.bincount(y_train))
        minority_count = np.sum(y_train == minority_class)

        if minority_count >= 2:
            k = min(5, minority_count - 1)
            smote = SMOTE(random_state=random_state, k_neighbors=k)
            X_train_aug, y_train_aug = smote.fit_resample(X_train_scaled, y_train)
            print(f"SMOTE: {len(X_train_scaled)} → {len(X_train_aug)}")
        else:
            X_train_aug = X_train_scaled
            y_train_aug = y_train

        # INNER LOOP - Hyperparameter tuning
        inner_cv = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=random_state)

        param_grid = [
            {'max_depth': 3, 'learning_rate': 0.01, 'n_estimators': 50},
            {'max_depth': 3, 'learning_rate': 0.01, 'n_estimators': 100},
            {'max_depth': 5, 'learning_rate': 0.01, 'n_estimators': 50},
        ]

        best_score = -np.inf
        best_params = None

        for params in param_grid:
            scores = []
            for train_inner_idx, val_inner_idx in inner_cv.split(X_train_aug, y_train_aug):
                model = XGBClassifier(**params, random_state=random_state, eval_metric='logloss', use_label_encoder=False)
                model.fit(X_train_aug[train_inner_idx], y_train_aug[train_inner_idx], verbose=False)
                score = accuracy_score(y_train_aug[val_inner_idx],
                                     model.predict(X_train_aug[val_inner_idx]))
                scores.append(score)

            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_params = params

        print(f"Best inner CV score: {best_score:.4f}, Params: {best_params}")

        # TRAIN FINAL MODEL
        final_model = XGBClassifier(**best_params, random_state=random_state, eval_metric='logloss', use_label_encoder=False)
        final_model.fit(X_train_aug, y_train_aug, verbose=False)

        # EVALUATE
        y_pred = final_model.predict(X_test_scaled)
        y_pred_proba = final_model.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        try:
            auc = roc_auc_score(y_test, y_pred_proba)
        except:
            auc = np.nan

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        print(f"Fold Results: Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

        results['fold_results'].append({
            'fold': fold_idx,
            'accuracy': acc,
            'f1': f1,
            'auc': auc,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'best_params': best_params
        })
        results['predictions'].extend(y_pred.tolist())
        results['true_labels'].extend(y_test.tolist())
        results['confusion_matrices'].append(cm.tolist())

    # AGGREGATE
    print(f"\n{'='*80}")
    print(f"{task_name.upper()} - FINAL RESULTS")
    print(f"{'='*80}")

    accuracies = [r['accuracy'] for r in results['fold_results']]
    f1_scores = [r['f1'] for r in results['fold_results']]
    aucs = [r['auc'] for r in results['fold_results'] if not np.isnan(r['auc'])]
    sensitivities = [r['sensitivity'] for r in results['fold_results']]
    specificities = [r['specificity'] for r in results['fold_results']]

    final_results = {
        'task': task_name,
        'n_samples': len(images),
        'n_features': features.shape[1],
        'accuracy': {
            'mean': np.mean(accuracies),
            'std': np.std(accuracies),
            'values': accuracies
        },
        'f1': {
            'mean': np.mean(f1_scores),
            'std': np.std(f1_scores),
            'values': f1_scores
        },
        'auc': {
            'mean': np.mean(aucs) if aucs else np.nan,
            'std': np.std(aucs) if aucs else np.nan,
            'values': aucs
        },
        'sensitivity': {
            'mean': np.mean(sensitivities),
            'std': np.std(sensitivities)
        },
        'specificity': {
            'mean': np.mean(specificities),
            'std': np.std(specificities)
        },
        'overall_cm': np.sum([np.array(cm) for cm in results['confusion_matrices']], axis=0).tolist(),
        'timestamp': datetime.now().isoformat(),
        'methodology': 'Nested CV - Cached Spatial Features (ResNet18 + HOG + Shape)'
    }

    print(f"Accuracy: {final_results['accuracy']['mean']:.4f} ± {final_results['accuracy']['std']:.4f}")
    print(f"F1: {final_results['f1']['mean']:.4f} ± {final_results['f1']['std']:.4f}")
    print(f"AUC: {final_results['auc']['mean']:.4f} ± {final_results['auc']['std']:.4f}")
    print(f"Sensitivity: {final_results['sensitivity']['mean']:.4f} ± {final_results['sensitivity']['std']:.4f}")
    print(f"Specificity: {final_results['specificity']['mean']:.4f} ± {final_results['specificity']['std']:.4f}")
    print(f"\nOverall Confusion Matrix:")
    print(np.array(final_results['overall_cm']))

    return final_results


if __name__ == '__main__':
    print("="*80)
    print("MULTI-TASK SPATIAL ANALYSIS: SPIRAL vs WAVE")
    print("="*80)

    # Load images
    print("\n[1/4] Loading images...")
    with open('data/multi_task_images.pkl', 'rb') as f:
        multi_task_data = pickle.load(f)

    print(f" Loaded {len(multi_task_data)} tasks:")
    for task_name, task_data in multi_task_data.items():
        print(f"   - {task_name}: {task_data['n_samples']} images")

    # Pre-extract features for each task (cached)
    print("\n[2/4] Pre-extracting CNN features (cached)...")

    extractor = CachedSpatialFeatureExtractor(
        use_cnn=True,  # Using ResNet18 (lighter)
        use_hog=True,
        use_shape=True,
        cnn_model='resnet18',
        cache_dir='data/feature_cache'
    )

    task_features = {}
    for task_name, task_data in multi_task_data.items():
        images = task_data['images']

        print(f"\nExtracting features for {task_name}...")
        features = extractor.extract_and_cache_features(
            images,
            cache_name=f'{task_name}_features.pkl',
            force_recompute=False  # Use cache if exists
        )

        task_features[task_name] = features
        print(f" {task_name}: {features.shape}")

    # Train models
    print("\n[3/4] Training models...")
    task_results = {}

    for task_name, task_data in multi_task_data.items():
        results = train_task_model(
            task_name=task_name,
            images=task_data['images'],
            labels=task_data['labels'],
            features=task_features[task_name]
        )
        task_results[task_name] = results

    # Save results
    print("\n[4/4] Saving results...")
    output_dir = Path('results/multi_task_spatial')
    output_dir.mkdir(parents=True, exist_ok=True)

    for task_name, results in task_results.items():
        with open(output_dir / f'{task_name}_results.yaml', 'w') as f:
            yaml.dump(results, f)
        print(f" Saved {task_name} results")

    # Combined summary
    combined = {
        'n_tasks': len(task_results),
        'task_names': list(task_results.keys()),
        'results': task_results,
        'timestamp': datetime.now().isoformat()
    }

    with open(output_dir / 'combined_results.yaml', 'w') as f:
        yaml.dump(combined, f)

    print(f"\n{'='*80}")
    print("MULTI-TASK TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir}")

    # Quick comparison
    print(f"\n{'='*80}")
    print("TASK COMPARISON")
    print("="*80)

    comparison_data = []
    for task in ['spiral', 'wave']:
        if task in task_results:
            acc = task_results[task]['accuracy']
            f1 = task_results[task]['f1']
            auc = task_results[task]['auc']

            comparison_data.append({
                'Task': task.capitalize(),
                'Accuracy': f"{acc['mean']:.1%} ± {acc['std']:.1%}",
                'F1': f"{f1['mean']:.3f} ± {f1['std']:.3f}",
                'AUC': f"{auc['mean']:.3f} ± {auc['std']:.3f}"
            })

    # Print comparison table
    print(f"\n{'Task':<12} {'Accuracy':<20} {'F1 Score':<20} {'AUC':<20}")
    print("-" * 72)
    for data in comparison_data:
        print(f"{data['Task']:<12} {data['Accuracy']:<20} {data['F1']:<20} {data['AUC']:<20}")

    print(f"\n{'='*80}")
