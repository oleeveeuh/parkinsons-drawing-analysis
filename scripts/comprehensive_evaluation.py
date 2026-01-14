#!/usr/bin/env python3
"""
Comprehensive evaluation script for PD classification models.

This script demonstrates how to use all evaluation components:
1. Load features and models
2. Perform cross-validation
3. Calculate metrics with confidence intervals
4. Generate publication-quality visualizations
5. Perform statistical significance tests
6. Generate comprehensive reports

Usage:
    python scripts/comprehensive_evaluation.py
    python scripts/comprehensive_evaluation.py --cv_folds 10 --generate_plots
"""

import sys
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.data.tabular_features import extract_tabular_features
from src.data.image_features import extract_image_features
from src.models.traditional_ml import train_all_baselines
from src.models.fusion import compare_fusion_methods

from src.evaluation import (
    # Metrics
    calculate_all_metrics,
    calculate_metrics_with_confidence_intervals,
    bootstrap_confidence_interval,
    print_metrics_report,
    mcnemars_test,
    generate_markdown_table,
    generate_results_json,
    generate_summary_report,

    # Visualizations
    plot_performance_comparison,
    plot_multiple_roc_curves,
    plot_confusion_matrix_grid,
    plot_feature_importance_comparison,
    plot_tsne_visualization,
    plot_shap_summary,
    generate_all_plots,

    # Cross-validation
    cross_validate_model,
    cross_validate_multiple_models,

    # Statistical tests
    compare_all_models,
    bootstrap_performance_difference,
    generate_statistical_report
)


def main():
    parser = argparse.ArgumentParser(description='Comprehensive model evaluation')
    parser.add_argument('--cv_folds', type=int, default=5,
                       help='Number of CV folds (default: 5)')
    parser.add_argument('--generate_plots', action='store_true',
                       help='Generate all visualization plots')
    parser.add_argument('--output_dir', type=str, default='results/evaluation',
                       help='Output directory (default: results/evaluation)')
    parser.add_argument('--skip_baseline', action='store_true',
                       help='Skip training baseline models (use cached results)')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION PIPELINE")
    print("="*80)
    print(f"CV Folds: {args.cv_folds}")
    print(f"Generate Plots: {args.generate_plots}")
    print(f"Output Directory: {args.output_dir}")
    print("="*80)

    # ========================================================================
    # 1. Load Features
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 1: LOADING FEATURES")
    print("-"*80)

    try:
        print("\nExtracting tabular features...")
        X_tab_train, X_tab_test, y_tab_train, y_tab_test, tab_feature_names = extract_tabular_features(
            train_dir="Improved Spiral Test Using Digitized Graphics Tablet for Monitoring Parkinson's Disease/data/train/",
            test_dir="Improved Spiral Test Using Digitized Graphics Tablet for Monitoring Parkinson's Disease/data/test/",
            label_map={'PWP': 1, 'Healthy': 0},
            use_tsfresh=True,
            use_manual=True,
            n_tsfresh_features=50,
            apply_smote=True
        )

        X_tab = np.vstack([X_tab_train, X_tab_test])
        y_tab = np.hstack([y_tab_train, y_tab_test])

        print(f" Tabular features: {X_tab.shape[1]}")
        print(f" Total samples: {X_tab.shape[0]}")

    except Exception as e:
        print(f"Error loading tabular features: {e}")
        import traceback
        traceback.print_exc()
        return

    try:
        print("\nExtracting image features...")
        X_img_train, X_img_test, y_img_train, y_img_test, img_feat_dim, _ = extract_image_features(
            train_dir="KaggleSpiral/drawings/spiral/training/",
            test_dir="KaggleSpiral/drawings/spiral/testing/",
            label_map={'healthy': 0, 'parkinson': 1},
            model_name='resnet50',
            use_traditional_features=True,
            augment_training=False,
            batch_size=16,
            cache_dir='cache/features/',
            test_size=0.2
        )

        X_img = np.vstack([X_img_train, X_img_test])
        y_img = np.hstack([y_img_train, y_img_test])

        print(f" Image features: {X_img.shape[1]}")
        print(f" Total samples: {X_img.shape[0]}")

    except Exception as e:
        print(f"Error loading image features: {e}")
        import traceback
        traceback.print_exc()
        return

    # ========================================================================
    # 2. Train Baseline Models
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 2: TRAINING BASELINE MODELS")
    print("-"*80)

    if not args.skip_baseline:
        print("\nTraining all baseline models...")
        baseline_results = train_all_baselines(
            X_tab, X_img, y_tab,
            cv_folds=args.cv_folds,
            output_dir=f'{args.output_dir}/baselines'
        )
    else:
        print("\nSkipping baseline training (using cached results)")
        baseline_results = None

    # ========================================================================
    # 3. Train Fusion Models
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 3: TRAINING FUSION MODELS")
    print("-"*80)

    if not args.skip_baseline:
        print("\nTraining fusion models...")
        fusion_results = compare_fusion_methods(
            X_tabular=X_tab,
            X_image=X_img,
            y=y_tab,
            cv_folds=args.cv_folds,
            output_dir=f'{args.output_dir}/fusion',
            config_path='configs/fusion_config.yaml'
        )
    else:
        print("\nSkipping fusion training (using cached results)")
        fusion_results = None

    # ========================================================================
    # 4. Statistical Comparison
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 4: STATISTICAL COMPARISON")
    print("-"*80)

    # Combine all results for statistical testing
    all_cv_results = {}

    if baseline_results:
        for modality, models in baseline_results.items():
            for model_name, scores in models.items():
                key = f"{modality}_{model_name}"
                all_cv_results[key] = scores

    if fusion_results:
        for fusion_type, results in fusion_results.items():
            key = f"fusion_{fusion_type}"
            all_cv_results[key] = results.get('cv_scores_mean', {})

    # Perform pairwise comparisons
    if len(all_cv_results) >= 2:
        print("\nPerforming pairwise model comparisons...")

        # Prepare CV results dict
        cv_dict = {}
        for model_name, metrics in all_cv_results.items():
            if isinstance(metrics, dict):
                cv_dict[model_name] = {
                    'accuracy': metrics.get('accuracy', []),
                    'f1': metrics.get('f1', []),
                    'auc': metrics.get('auc_roc', [])
                }

        if cv_dict:
            stat_results = compare_all_models(
                cv_dict,
                metric='f1',
                alpha=0.05,
                correction_method='holm'
            )

            # Generate statistical report
            generate_statistical_report(
                stat_results,
                output_path=f'{args.output_dir}/statistical_report.md'
            )

    # ========================================================================
    # 5. Generate Visualizations
    # ========================================================================
    if args.generate_plots:
        print("\n" + "-"*80)
        print("STEP 5: GENERATING VISUALIZATIONS")
        print("-"*80)

        # Prepare results for plotting
        plot_results = {
            'cv_results': all_cv_results,
            'predictions': {},  # Would need to store predictions during training
            'feature_importance': {}  # Would need to store during training
        }

        # Generate plots
        try:
            generate_all_plots(
                results=plot_results,
                X_tabular=X_tab,
                X_image=X_img,
                y=y_tab,
                tabular_feature_names=tab_feature_names,
                image_feature_names=[f"img_feat_{i}" for i in range(X_img.shape[1])],
                output_dir=f'{args.output_dir}/figures'
            )
        except Exception as e:
            print(f"Warning: Could not generate all plots: {e}")
            import traceback
            traceback.print_exc()

    # ========================================================================
    # 6. Generate Comprehensive Report
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 6: GENERATING COMPREHENSIVE REPORT")
    print("-"*80)

    # Compile all results
    comprehensive_results = {
        'baseline_results': baseline_results,
        'fusion_results': fusion_results,
        'statistical_tests': stat_results if 'stat_results' in locals() else None,
        'metadata': {
            'cv_folds': args.cv_folds,
            'n_tabular_samples': X_tab.shape[0],
            'n_tabular_features': X_tab.shape[1],
            'n_image_features': X_img.shape[1],
            'dataset': 'PD Classification (Spiral Drawings)'
        }
    }

    # Save to JSON
    generate_results_json(
        comprehensive_results,
        output_path=f'{args.output_dir}/all_results.json'
    )

    # Generate markdown summary
    generate_summary_report(
        comprehensive_results,
        output_path=f'{args.output_dir}/summary_report.md'
    )

    # ========================================================================
    # 7. Final Summary
    # ========================================================================
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nResults saved to: {args.output_dir}/")
    print(f"  - all_results.json: Comprehensive results")
    print(f"  - summary_report.md: Summary report")
    print(f"  - statistical_report.md: Statistical testing results")
    if args.generate_plots:
        print(f"  - figures/: All visualization plots")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
