# Fusion Methods for Small Datasets

Complete guide to multi-modal fusion methods for combining tabular and image features, optimized for small datasets (~70-210 samples).

---

##  Overview

Fusion methods combine information from multiple modalities (tabular time-series + images) to improve classification performance.

**Key Challenge**: Small datasets (~70 samples) require simple, robust fusion methods that won't overfit.

---

##  Fusion Methods

### 1. Late Fusion (Ensemble)  **RECOMMENDED**

**Approach:** Train separate models on each modality, combine predictions.

```
Tabular Features → Tabular Model → Prediction P₁
                                         ↘
                                        Combine → Final Prediction
                                         ↗
Image Features → Image Model → Prediction P₂
```

**Combination Methods:**
- **Simple Average**: `P_final = (P₁ + P₂) / 2`
- **Weighted Average**: `P_final = w·P₁ + (1-w)·P₂` (optimize w with CV)
- **Majority Voting**: Final class = mode(class₁, class₂)

**Pros:**
-  Most robust for small datasets
-  Handles missing modalities naturally
-  Easy to interpret
-  Can weight modalities differently

**Cons:**
-  Doesn't capture cross-modal relationships
-  Requires training multiple models

**When to Use:**
- Small datasets (<200 samples)
- When modalities have different reliability
- When you might have missing data

**Code Example:**
```python
from src.models.fusion import LateFusion

fusion = LateFusion(
    fusion_method='weighted_average',
    optimize_weights=True,
    cv_folds=5
)

fusion.fit(X_tabular, X_image, y)
predictions = fusion.predict(X_tabular, X_image)
```

---

### 2. Early Fusion (Feature Concatenation)

**Approach:** Concatenate features, apply feature selection, train single model.

```
Tabular Features (100) 
                       → Concatenate → Select Top 150 → Single Model → Prediction
Image Features (3800) 
```

**Critical Components:**
- **Feature Selection**: Must reduce dimensionality!
  - SelectKBest with ANOVA F-test
  - Mutual information
  - Keep top 100-150 features (down from ~3900)
- **Normalization**: StandardScaler before selection
- **Model**: Conservative XGBoost (max_depth=4)

**Feature Selection Details:**
```python
# Original: 100 + 3800 = 3900 features
# After selection: 150 features
#   - Tabular: ~40-60 features selected
#   - Image: ~90-110 features selected
```

**Pros:**
-  Captures cross-modal relationships
-  Single model to train
-  Feature selection provides interpretability

**Cons:**
-  Requires careful feature selection
-  Can overfit with too many features
-  Sensitive to feature selection method

**When to Use:**
- Moderate datasets (100-500 samples)
- When features from both modalities are complementary
- When you want to understand which features matter most

**Code Example:**
```python
from src.models.fusion import EarlyFusion

fusion = EarlyFusion(
    n_features=150,  # Critical!
    feature_selection_method='selectkbest'
)

fusion.fit(X_tabular, X_image, y)
predictions = fusion.predict(X_tabular, X_image)

# See which features were selected
indices, names = fusion.get_selected_features()
print(f"Selected {len(indices)} features")
print(f"Tabular: {(indices < 100).sum()}, Image: {(indices >= 100).sum()}")
```

---

### 3. Stacking (Meta-Learning)

**Approach:** Use predictions as meta-features, train meta-learner.

```
Tabular → Tabular Model 
                              → Meta-Features → Meta-Learner → Final Prediction
Image → Image Model 
```

**Critical: Out-of-Fold (OOF) Predictions**
- Uses k-fold CV to generate predictions
- Prevents data leakage
- Meta-learner trained on OOF predictions only

**Meta-Learner:**
- Simple Logistic Regression (L2 regularization)
- Linear to prevent overfitting
- Only 4 input features: P₁(class 0), P₁(class 1), P₂(class 0), P₂(class 1)

**Pros:**
-  Can learn complex combinations
-  Often best performance
-  OOF prevents overfitting

**Cons:**
-  Most complex
-  Requires careful CV implementation
-  Less interpretable

**When to Use:**
- When simple fusion isn't enough
- When you have enough data for meta-learning (>100 samples)
- When you want maximum performance

**Code Example:**
```python
from src.models.fusion import StackingFusion

fusion = StackingFusion(
    cv_folds=5,  # For OOF predictions
    meta_learner_type='logistic_regression'
)

fusion.fit(X_tabular, X_image, y)
predictions = fusion.predict(X_tabular, X_image)
```

---

##  Comparison of Methods

| Method | Complexity | Overfitting Risk | Data Needed | Performance | Interpretability |
|--------|-----------|------------------|-------------|-------------|------------------|
| Late Fusion |  Low |  Low | ~70+ | Good |  High |
| Early Fusion |  Medium |  Medium | ~100+ | Better |  Medium |
| Stacking |  High |  Medium | ~150+ | Best |  Low |

---

##  Performance Expectations

**Expected improvements over single modality:**

| Method | vs Tabular Only | vs Image Only |
|--------|----------------|---------------|
| Late Fusion | +3-5% | +5-8% |
| Early Fusion | +5-8% | +3-5% |
| Stacking | +5-10% | +5-8% |

**Actual performance depends on:**
- Quality of individual modalities
- Complementarity of information
- Dataset size
- Feature selection quality

**Good results:**
- Accuracy >80% with narrow std (<0.05)
- F1 >0.80 with narrow std
- Clear improvement over single modality

**Warning signs:**
- No improvement over best single modality → Modalities not complementary
- Very wide std → Need more data or simpler method
- Early fusion much worse → Feature selection failed

---

##  Usage

### Train All Fusion Methods

```bash
python scripts/train_fusion.py
```

This will:
1. Load tabular and image features
2. Train all 3 fusion methods with 5-fold CV
3. Compare performance
4. Save results to `results/fusion/fusion_comparison.csv`

### Train Specific Fusion Method

```bash
# Late fusion (ensemble)
python scripts/train_fusion.py --fusion_type late_fusion

# Early fusion (concatenation)
python scripts/train_fusion.py --fusion_type early_fusion --n_features 150

# Stacking (meta-learning)
python scripts/train_fusion.py --fusion_type stacking
```

### With Custom Parameters

```bash
# 10-fold CV for robust estimates
python scripts/train_fusion.py --cv_folds 10

# Early fusion with 200 features
python scripts/train_fusion.py --fusion_type early_fusion --n_features 200

# Late fusion with weight optimization
python scripts/train_fusion.py --fusion_type late_fusion --optimize_weights
```

---

##  Output

Results saved to `results/fusion/`:

```
results/fusion/
 fusion_comparison.csv        # Comparison of all 3 methods
 late_fusion_cv_results.csv   # Per-fold results
 early_fusion_cv_results.csv  # Per-fold results
 stacking_cv_results.csv      # Per-fold results
```

**fusion_comparison.csv Format:**
```csv
method,accuracy,accuracy_std,f1,f1_std,auc,auc_std
late_fusion,0.8234,0.0345,0.8198,0.0356,0.8502,0.0312
early_fusion,0.8102,0.0423,0.8056,0.0434,0.8321,0.0398
stacking,0.8356,0.0298,0.8321,0.0301,0.8612,0.0276
```

---

##  Best Practices

### 1. Start Simple

```python
# Step 1: Try late fusion first (most robust)
fusion = LateFusion(fusion_method='simple_average')

# Step 2: If that works, try weighted average
fusion = LateFusion(fusion_method='weighted_average', optimize_weights=True)

# Step 3: If still not good enough, try stacking
fusion = StackingFusion()
```

### 2. Always Use Cross-Validation

- Never evaluate on single train/test split
- Use stratified 5-fold CV minimum
- Report mean ± std across folds

### 3. For Early Fusion: Feature Selection is Critical

```python
#  BAD: No feature selection (3900 features, will overfit)
fusion = EarlyFusion(n_features=X.shape[0])  # Uses all features

#  GOOD: Select top 150 features
fusion = EarlyFusion(n_features=150)

#  BETTER: Tune number of features
for n_features in [50, 100, 150, 200]:
    fusion = EarlyFusion(n_features=n_features)
    # Evaluate with CV...
```

### 4. Monitor Overfitting

```python
# Check if train >> test accuracy
train_pred = fusion.predict(X_tab_train, X_img_train)
test_pred = fusion.predict(X_tab_test, X_img_test)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)

if train_acc - test_acc > 0.10:
    print("Warning: Overfitting detected!")
```

### 5. Compare Against Single Modality

```python
# Train tabular-only model
tab_model = XGBoostClassifier()
tab_model.fit(X_tab, y)
tab_acc = tab_model.score(X_tab, y)

# Train fusion model
fusion = LateFusion()
fusion.fit(X_tab, X_img, y)
fusion_acc = fusion.score(X_tab, X_img, y)

if fusion_acc <= tab_acc:
    print("Fusion doesn't help. Modalities not complementary.")
```

---

##  Troubleshooting

### "Early fusion performs worse than single modality"

**Causes:**
- Too many features (overfitting)
- Wrong feature selection method
- Features not scaled properly

**Solutions:**
```python
# Reduce number of features
fusion = EarlyFusion(n_features=50)  # Instead of 150

# Try different feature selection
fusion = EarlyFusion(
    feature_selection_method='mutual_info'  # Instead of 'selectkbest'
)

# Ensure normalization
fusion = EarlyFusion(normalize_features=True)
```

### "Stacking overfits badly"

**Causes:**
- Meta-learner too complex
- Not using OOF predictions
- CV folds too few

**Solutions:**
```python
# Use simpler meta-learner
fusion = StackingFusion(meta_learner_type='logistic_regression')

# Increase CV folds
fusion = StackingFusion(cv_folds=10)

# Add regularization
# (Edit configs/fusion_config.yaml)
```

### "Late fusion weights are extreme (0 or 1)"

**Meaning:** One modality is much better than the other.

**Solutions:**
```python
# Check if one modality is useless
tab_acc = accuracy_score(y, tab_model.predict(X_tab))
img_acc = accuracy_score(y, img_model.predict(X_img))

if abs(tab_acc - img_acc) > 0.20:
    print("Modalities have very different performance.")
    print("Consider using only the better modality.")
```

### "All fusion methods perform similarly"

**Meaning:** Modalities are redundant or not complementary.

**Solutions:**
- Check correlation between modality predictions
- Try feature selection to find complementary features
- Accept that single modality might be best

---

##  Configuration

All fusion methods use `configs/fusion_config.yaml`:

```yaml
late_fusion:
  tabular_model:
    type: "xgboost"
    params:
      max_depth: 3
      n_estimators: 100
  fusion_method: "weighted_average"

early_fusion:
  feature_selection:
    enabled: true
    method: "selectkbest"
    k_features: 150

stacking:
  base_models:
    - type: "xgboost"  # Tabular
    - type: "xgboost"  # Image
  meta_learner:
    type: "logistic_regression"
```

Edit this file to customize fusion behavior.

---

##  Quick Reference

**Late Fusion Examples:**
```python
# Simple average
fusion = LateFusion(fusion_method='simple_average')

# Weighted average with optimization
fusion = LateFusion(
    fusion_method='weighted_average',
    optimize_weights=True,
    cv_folds=5
)

# Majority voting
fusion = LateFusion(fusion_method='majority_voting')
```

**Early Fusion Examples:**
```python
# Select top 100 features
fusion = EarlyFusion(n_features=100)

# Use mutual information
fusion = EarlyFusion(
    n_features=100,
    feature_selection_method='mutual_info'
)
```

**Stacking Examples:**
```python
# Standard stacking
fusion = StackingFusion(cv_folds=5)

# With more conservative base models
fusion = StackingFusion(
    tabular_model_type='xgboost',
    image_model_type='xgboost',
    meta_learner_type='logistic_regression'
)
```

---

##  Related Documentation

- **[Traditional ML](src/models/traditional_ml.py)** - Single-modality models
- **[Model Config](configs/model_config.yaml)** - Model hyperparameters
- **[Fusion Config](configs/fusion_config.yaml)** - Fusion settings
- **[Train Baselines](scripts/train_baselines.py)** - Train single-modality models

---

**For help:**
- Olivia Liau: oliau@usc.edu
- Project: MoE-MultiSDAR-PD-Classification
