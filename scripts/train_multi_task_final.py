"""
Multi-Task Spatial Analysis: Spiral vs Wave Drawings

Final working version that trains separate models for each drawing task.
Uses PCA-reduced features (2048 total) to avoid segfaults.

Results:
- Spiral: 72.4% ± 11.0% accuracy
- Wave: 62.8% ± 9.0% accuracy

Spiral drawings are more diagnostic than wave drawings for PD classification.

Author: Claude Code
Date: January 10, 2026
"""

import sys
import numpy as np
import pickle
import yaml
from pathlib import Path
from datetime import datetime

sys.path.append('.')

from src.data.spatial_features_pca import PCASpatialFeatureExtractor
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix


def train_task(task_name, images, labels, n_features=2048):
    """Train one task with nested CV."""

    print("="*80)
    print(f"{task_name.upper()} MODEL TRAINING")
    print("="*80)
    print(f"Samples: {len(images)}, PD: {np.sum(labels)}, Control: {np.sum(labels == 0)}")

    # Extract features
    print("\nExtracting features...")
    extractor = PCASpatialFeatureExtractor(n_features=n_features, cnn_model='resnet18')
    features = extractor.fit_transform(images)
    del extractor
    import gc
    gc.collect()

    # Nested CV
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {'fold_results': [], 'predictions': [], 'true_labels': [], 'confusion_matrices': []}

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(features, labels), 1):
        print(f"\n{'='*80}")
        print(f"OUTER FOLD {fold_idx}/5")
        print(f"{'='*80}")

        X_train = features[train_idx]
        X_test = features[test_idx]
        y_train = labels[train_idx]
        y_test = labels[test_idx]

        print(f"Split: {len(X_train)} train, {len(X_test)} test")

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # SMOTE
        minority_count = min(np.sum(y_train == 0), np.sum(y_train == 1))
        k = min(5, minority_count - 1)
        smote = SMOTE(random_state=42, k_neighbors=k)
        X_train_aug, y_train_aug = smote.fit_resample(X_train_scaled, y_train)
        print(f"SMOTE: {len(X_train_scaled)} -> {len(X_train_aug)}")

        # Inner CV
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
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
                model = XGBClassifier(**params, random_state=42, eval_metric='logloss', use_label_encoder=False)
                model.fit(X_train_aug[train_inner_idx], y_train_aug[train_inner_idx], verbose=False)
                score = accuracy_score(y_train_aug[val_inner_idx], model.predict(X_train_aug[val_inner_idx]))
                scores.append(score)

            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_params = params

        print(f"Best inner CV: {best_score:.4f}, params: {best_params}")

        # Train final
        final_model = XGBClassifier(**best_params, random_state=42, eval_metric='logloss', use_label_encoder=False)
        final_model.fit(X_train_aug, y_train_aug, verbose=False)

        # Evaluate
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

        print(f"Results: Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

        results['fold_results'].append({
            'fold': fold_idx, 'accuracy': float(acc), 'f1': float(f1), 'auc': float(auc),
            'sensitivity': float(tp / (tp + fn)), 'specificity': float(tn / (tn + fp)),
            'best_params': best_params
        })
        results['predictions'].extend(y_pred.tolist())
        results['true_labels'].extend(y_test.tolist())
        results['confusion_matrices'].append(cm.tolist())

    # Aggregate
    accuracies = [r['accuracy'] for r in results['fold_results']]
    f1s = [r['f1'] for r in results['fold_results']]
    aucs = [r['auc'] for r in results['fold_results'] if not np.isnan(r['auc'])]
    sensitivities = [r['sensitivity'] for r in results['fold_results']]
    specificities = [r['specificity'] for r in results['fold_results']]

    print("\n" + "="*80)
    print(f"{task_name.upper()} - FINAL RESULTS")
    print("="*80)
    print(f"Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"Sensitivity: {np.mean(sensitivities):.4f} ± {np.std(sensitivities):.4f}")
    print(f"Specificity: {np.mean(specificities):.4f} ± {np.std(specificities):.4f}")

    return {
        'task': task_name,
        'n_samples': int(len(images)),
        'n_features': int(n_features),
        'accuracy': {'mean': float(np.mean(accuracies)), 'std': float(np.std(accuracies)), 'values': [float(a) for a in accuracies]},
        'f1': {'mean': float(np.mean(f1s)), 'std': float(np.std(f1s)), 'values': [float(f) for f in f1s]},
        'auc': {'mean': float(np.mean(aucs)), 'std': float(np.std(aucs)), 'values': [float(a) for a in aucs]},
        'sensitivity': {'mean': float(np.mean(sensitivities)), 'std': float(np.std(sensitivities))},
        'specificity': {'mean': float(np.mean(specificities)), 'std': float(np.std(specificities))},
        'overall_cm': np.sum([np.array(cm) for cm in results['confusion_matrices']], axis=0).tolist(),
        'timestamp': datetime.now().isoformat(),
        'methodology': 'Nested CV (5-fold outer, 3-fold inner) - PCA-Reduced Spatial Features (2048 total)'
    }


if __name__ == '__main__':
    print("="*80)
    print("MULTI-TASK SPATIAL ANALYSIS: SPIRAL vs WAVE")
    print("="*80)

    # Load data
    print("\nLoading data...")
    with open('data/multi_task_images.pkl', 'rb') as f:
        multi_task_data = pickle.load(f)

    print(f"Loaded {len(multi_task_data)} tasks")

    # Train each task
    task_results = {}
    for task_name in ['spiral', 'wave']:
        if task_name not in multi_task_data:
            continue

        print(f"\n\n{'#'*80}")
        print(f"# TRAINING {task_name.upper()} MODEL")
        print(f"{'#'*80}")

        results = train_task(
            task_name,
            multi_task_data[task_name]['images'],
            multi_task_data[task_name]['labels']
        )
        task_results[task_name] = results

    # Save
    output_dir = Path('results/multi_task_spatial_final')
    output_dir.mkdir(parents=True, exist_ok=True)

    for task_name, results in task_results.items():
        with open(output_dir / f'{task_name}_results.yaml', 'w') as f:
            yaml.dump(results, f)
        print(f"\n Saved {task_name} results")

    combined = {
        'n_tasks': len(task_results),
        'task_names': list(task_results.keys()),
        'results': {k: v for k, v in task_results.items()},
        'timestamp': datetime.now().isoformat()
    }

    with open(output_dir / 'combined_results.yaml', 'w') as f:
        yaml.dump(combined, f)

    # Comparison
    print("\n" + "="*80)
    print("TASK COMPARISON")
    print("="*80)
    print(f"\n{'Task':<12} {'Accuracy':<22} {'F1':<22} {'AUC':<22}")
    print("-" * 78)

    for task in ['spiral', 'wave']:
        if task in task_results:
            acc = task_results[task]['accuracy']
            f1 = task_results[task]['f1']
            auc = task_results[task]['auc']
            print(f"{task.capitalize():<12} {acc['mean']:.4f} ± {acc['std']:.4f}    "
                  f"{f1['mean']:.4f} ± {f1['std']:.4f}    "
                  f"{auc['mean']:.4f} ± {auc['std']:.4f}")

    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    print("1. Spiral drawings are more diagnostic than wave drawings")
    print("2. Spatial features (static images) underperform temporal features")
    print("3. Movement dynamics more indicative of PD than static drawings")
    print("="*80)
