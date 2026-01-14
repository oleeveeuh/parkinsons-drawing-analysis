"""
Example: Using the image feature extractor for PD spiral images.

Demonstrates how to extract features from grayscale spiral images
using pretrained ResNet50 (frozen) - no fine-tuning needed.
"""

import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from src.data.image_features import (
    extract_image_features,
    ImageFeatureExtractor,
    load_image_directory
)
from src.models import XGBoostClassifier
from src.evaluation import (
    calculate_metrics_with_confidence_intervals,
    print_metrics_report
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Update these paths to match your data structure
# Expected structure:
# data_dir/
#   healthy/          # or control/
#     img1.png
#     img2.png
#   parkinsons/       # or pd/, pwp/
#     img1.png
#     img2.png

IMAGE_DATA_DIR = "/path/to/your/spiral/images/"

# ============================================================================
# METHOD 1: Quick extraction from directories
# ============================================================================

def quick_extract_and_train():
    """One-line feature extraction and training."""

    print("\n" + "="*70)
    print("METHOD 1: Quick Extraction from Directories")
    print("="*70)

    # Extract features (handles everything automatically)
    X_train, X_test, y_train, y_test, feature_dim, extractor = extract_image_features(
        train_dir=IMAGE_DATA_DIR,
        test_dir=None,  # Auto-split if None
        label_map={'healthy': 0, 'parkinsons': 1},  # Customize as needed
        model_name='resnet50',
        use_traditional_features=True,
        augment_training=True,
        augment_factor=2,
        batch_size=16,
        cache_dir='cache/features/',
        test_size=0.2
    )

    # Visualize features with t-SNE
    print("\nGenerating t-SNE visualization...")
    extractor.visualize_features(
        X_train,
        y_train,
        method='tsne',
        save_path='../results/figures/image_features_tsne.png'
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

    print_metrics_report(metrics, title="XGBoost Performance on Image Features")

    return X_train, X_test, y_train, y_test, extractor, model


# ============================================================================
# METHOD 2: Step-by-step (more control)
# ============================================================================

def step_by_step_extraction():
    """Detailed step-by-step extraction with more control."""

    print("\n" + "="*70)
    print("METHOD 2: Step-by-Step Extraction")
    print("="*70)

    # Initialize extractor
    extractor = ImageFeatureExtractor(
        model_name='resnet50',
        use_traditional_features=True,
        cache_dir='cache/features/'
    )

    # Load image paths
    train_paths, y_train = load_image_directory(
        IMAGE_DATA_DIR,
        label_map={'healthy': 0, 'parkinsons': 1}
    )

    print(f"Found {len(train_paths)} images")
    print(f"Class distribution: {np.bincount(y_train)}")

    # Show some sample paths
    print(f"\nSample paths:")
    for path in train_paths[:5]:
        print(f"  {path}")

    # Extract features with augmentation
    X_train, y_train_augmented = extractor.extract_features(
        train_paths,
        y_train,
        augment=True,
        augment_factor=2,
        batch_size=16,
        use_cache=True
    )

    # For demonstration, split train/test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train_augmented,
        test_size=0.2,
        stratify=y_train_augmented,
        random_state=42
    )

    print(f"\nFeature extraction complete!")
    print(f"  Original images: {len(train_paths)}")
    print(f"  After augmentation: {X_train.shape[0]}")
    print(f"  Feature dimension: {X_train.shape[1]}")

    return X_train, X_test, y_train, y_test, extractor


# ============================================================================
# Visualize Augmented Images
# ============================================================================

def visualize_augmentation():
    """Show example augmented images."""

    print("\n" + "="*70)
    print("VISUALIZING AUGMENTATION")
    print("="*70)

    from torch.utils.data import DataLoader
    from torchvision import transforms

    # Load some images
    paths, labels = load_image_directory(IMAGE_DATA_DIR)

    # Create dataset with augmentation
    from src.data.image_features import SpiralImageDataset

    dataset = SpiralImageDataset(
        image_paths=paths[:5],  # Just 5 samples
        labels=labels[:5],
        augment=True,
        augment_factor=2
    )

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Plot
    fig, axes = plt.subplots(5, 3, figsize=(12, 20))

    for idx, (img, label) in enumerate(dataloader):
        # Convert to numpy
        img_np = img.squeeze().permute(1, 2, 0).numpy()

        # Plot
        axes[idx, 0].imshow(img_np, cmap='gray')
        axes[idx, 0].set_title(f'Original\n{"PD" if label.item() == 1 else "Control"}')
        axes[idx, 0].axis('off')

        # Augmented versions would be in subsequent iterations
        # (omitted for brevity)

    plt.tight_layout()
    plt.savefig('../results/figures/augmented_samples.png', dpi=300)
    print(" Augmentation samples saved")
    plt.show()


# ============================================================================
# Compare Different Configurations
# ============================================================================

def compare_configurations():
    """Compare ResNet18 vs ResNet50, with/without traditional features."""

    print("\n" + "="*70)
    print("COMPARING DIFFERENT CONFIGURATIONS")
    print("="*70)

    configs = [
        ('resnet18', False, 'ResNet18 only'),
        ('resnet50', False, 'ResNet50 only'),
        ('resnet50', True, 'ResNet50 + Traditional'),
    ]

    results = {}

    for model_name, use_traditional, config_name in configs:
        print(f"\n{'='*70}")
        print(f"Configuration: {config_name}")
        print(f"{'='*70}")

        try:
            X_train, X_test, y_train, y_test, feature_dim, extractor = extract_image_features(
                train_dir=IMAGE_DATA_DIR,
                model_name=model_name,
                use_traditional_features=use_traditional,
                augment_training=True,
                augment_factor=2,
                cache_dir='cache/features/'
            )

            # Train simple model
            model = XGBoostClassifier(max_depth=3, n_estimators=50)
            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_test)
            acc = np.mean(y_pred == y_test)

            results[config_name] = {
                'feature_dim': feature_dim,
                'accuracy': acc,
                'n_samples': len(X_train)
            }

            print(f"\nResults for {config_name}:")
            print(f"  Feature dimension: {feature_dim}")
            print(f"  Training samples: {len(X_train)}")
            print(f"  Test accuracy: {acc:.4f}")

        except Exception as e:
            print(f"Error with {config_name}: {e}")
            continue

    # Summary comparison
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    for config_name, metrics in results.items():
        print(f"\n{config_name}:")
        print(f"  Features: {metrics['feature_dim']}")
        print(f"  Samples: {metrics['n_samples']}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")

    return results


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    # Choose your method

    # METHOD 1: Quick (recommended)
    X_train, X_test, y_train, y_test, extractor, model = quick_extract_and_train()

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total features extracted: {X_train.shape[1]}")
    print(f"  - CNN features: {extractor.cnn_feature_dim}")
    if extractor.use_traditional_features:
        print(f"  - Traditional features: {X_train.shape[1] - extractor.cnn_feature_dim}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Feature breakdown
    if extractor.use_traditional_features:
        print("\nTraditional feature breakdown:")
        print("  - HOG: ~1764 features")
        print("  - Spiral-specific: 9 features")
        print("    * Circularity, roundness, compactness")
        print("    * Solidity, eccentricity")
        print("    * Line thickness variance (PD-specific)")
        print("    * Radius variance (micrographia indicator)")

    print("\n Analysis complete!")
    print("\nNext steps:")
    print("  1. Try different augmentations")
    print("  2. Feature selection to reduce dimensionality")
    print("  3. Combine with tabular features for fusion")
