"""
Example: Using the tabular feature extractor for PD spiral data.

This demonstrates how to extract features from your semicolon-separated .txt files
and train a classifier.
"""

import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from src.data.tabular_features import (
    extract_tabular_features,
    visualize_feature_distributions,
    TabularFeatureExtractor
)
from src.models import XGBoostClassifier
from src.evaluation import (
    calculate_metrics_with_confidence_intervals,
    print_metrics_report,
    plot_feature_importance
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Update these paths to match your data structure
IMPROVED_SPIRAL_TRAIN = "/Users/olivialiau/MoE-MultiSDAR-PD-Classification/Improved Spiral Test Using Digitized Graphics Tablet for Monitoring Parkinson's Disease/data/train/"
IMPROVED_SPIRAL_TEST = "/Users/olivialiau/MoE-MultiSDAR-PD-Classification/Improved Spiral Test Using Digitized Graphics Tablet for Monitoring Parkinson's Disease/data/test/"

# Label mapping: filename pattern -> label
# PWP = Parkinson's Disease Patients (label 1)
# Healthy = Control subjects (label 0)
LABEL_MAP = {'PWP': 1, 'Healthy': 0}

# ============================================================================
# METHOD 1: Quick extraction from directories
# ============================================================================

def quick_extract_and_train():
    """One-line feature extraction and training."""

    print("\n" + "="*70)
    print("METHOD 1: Quick Extraction from Directories")
    print("="*70)

    # Extract features (handles everything automatically)
    X_train, X_test, y_train, y_test, feature_names = extract_tabular_features(
        train_dir=IMPROVED_SPIRAL_TRAIN,
        test_dir=IMPROVED_SPIRAL_TEST,
        label_map=LABEL_MAP,
        use_tsfresh=True,        # Use TSFresh automated features
        use_manual=True,         # Use PD-specific manual features
        n_tsfresh_features=50,   # Limit TSFresh to 50 features
        apply_smote=True,        # Apply SMOTE augmentation
        test_size=0.2
    )

    # Train model
    print("\nTraining XGBoost classifier...")
    model = XGBoostClassifier(
        n_estimators=50,
        max_depth=3,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    metrics = calculate_metrics_with_confidence_intervals(
        y_test, y_pred, y_proba,
        n_bootstrap=1000,
        confidence_level=0.95
    )

    print_metrics_report(metrics, title="XGBoost Performance")

    # Feature importance
    feature_importance = model.get_feature_importance()
    plot_feature_importance(
        feature_importance,
        feature_names,
        model_name="XGBoost",
        top_n=20,
        save_path="../results/figures/tabular_feature_importance.png"
    )

    return X_train, X_test, y_train, y_test, feature_names, model


# ============================================================================
# METHOD 2: Step-by-step (more control)
# ============================================================================

def step_by_step_extraction():
    """Detailed step-by-step extraction with more control."""

    print("\n" + "="*70)
    print("METHOD 2: Step-by-Step Extraction")
    print("="*70)

    # Initialize extractor
    extractor = TabularFeatureExtractor(
        use_tsfresh=True,
        use_manual_features=True,
        n_tsfresh_features=50,
        normalize=True
    )

    # Load and extract training features
    print("\nLoading training data...")
    from src.data.tabular_features import load_tabular_data_directory

    train_files, y_train = load_tabular_data_directory(
        IMPROVED_SPIRAL_TRAIN,
        label_map=LABEL_MAP
    )

    print(f"Found {len(train_files)} training files")
    print(f"Class distribution: {np.bincount(y_train)}")

    # Extract features with SMOTE
    X_train, y_train_augmented = extractor.fit_transform(
        train_files,
        y_train,
        apply_smote=True
    )

    # Load and extract test features
    print("\nLoading test data...")
    test_files, y_test = load_tabular_data_directory(
        IMPROVED_SPIRAL_TEST,
        label_map=LABEL_MAP
    )

    X_test = extractor.transform(test_files)

    feature_names = extractor.get_feature_names()

    print(f"\nFeature extraction complete!")
    print(f"  Training shape: {X_train.shape}")
    print(f"  Test shape: {X_test.shape}")
    print(f"  Features: {len(feature_names)}")

    return X_train, X_test, y_train_augmented, y_test, feature_names, extractor


# ============================================================================
# METHOD 3: Using alldata (unsplit) directory
# ============================================================================

def use_alldirectory_with_split():
    """Use the alldata directory and split manually."""

    print("\n" + "="*70)
    print("METHOD 3: Using alldata Directory with Manual Split")
    print("="*70)

    ALLDATA_DIR = "/Users/olivialiau/MoE-MultiSDAR-PD-Classification/Improved Spiral Test Using Digitized Graphics Tablet for Monitoring Parkinson's Disease/data/alldata/"

    # Extract features (will automatically split if no test_dir provided)
    X_train, X_test, y_train, y_test, feature_names = extract_tabular_features(
        train_dir=ALLDATA_DIR,
        test_dir=None,  # None = auto-split from alldata
        label_map=LABEL_MAP,
        use_tsfresh=True,
        use_manual=True,
        n_tsfresh_features=50,
        apply_smote=False,  # Don't apply SMOTE yet (small dataset)
        test_size=0.2  # 20% for testing
    )

    print(f"\nData split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Test: {len(X_test)} samples")

    return X_train, X_test, y_train, y_test, feature_names


# ============================================================================
# Visualize feature distributions
# ============================================================================

def visualize_features(X_train, y_train, feature_names):
    """Visualize most discriminative features."""

    print("\nGenerating feature distribution plots...")

    visualize_feature_distributions(
        X_train,
        y_train,
        feature_names,
        top_n=20,
        save_path="../results/figures/tabular_feature_distributions.png"
    )


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    # Choose your method

    # METHOD 1: Quick (recommended)
    X_train, X_test, y_train, y_test, feature_names, model = quick_extract_and_train()

    # Visualize features
    visualize_features(X_train, y_train, feature_names)

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total features extracted: {len(feature_names)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Feature names: {feature_names[:10]}...")  # Show first 10

    # Most important features (PD-specific)
    feature_importance = model.get_feature_importance()
    top_indices = np.argsort(feature_importance)[-10:][::-1]

    print("\nTop 10 Most Important Features:")
    for idx, feat_idx in enumerate(top_indices, 1):
        print(f"  {idx}. {feature_names[feat_idx]}: {feature_importance[feat_idx]:.4f}")

    print("\n Analysis complete!")
