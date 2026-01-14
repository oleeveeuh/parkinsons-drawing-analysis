# Scripts - Training and Evaluation Scripts

This directory contains executable scripts for training and evaluating models.

## Available Scripts

### train_baselines.py

Train all baseline models (Logistic Regression, Random Forest, XGBoost, SVM, MLP) using stratified k-fold cross-validation on both tabular and image features.

**Usage:**

```bash
# Basic usage (5-fold CV)
python scripts/train_baselines.py

# With hyperparameter tuning (GridSearchCV)
python scripts/train_baselines.py --hyperparameter_tuning

# Use 10-fold CV instead of 5
python scripts/train_baselines.py --cv_folds 10

# Train on tabular features only
python scripts/train_baselines.py --tabular_only

# Train on image features only
python scripts/train_baselines.py --image_only

# Custom output directory
python scripts/train_baselines.py --output_dir results/my_experiment
```

**What it does:**

1. **Loads Features** - Extracts tabular and image features (or uses cached features)
2. **Trains Models** - Trains 5 different models with 5-fold stratified CV:
   - Logistic Regression (linear baseline)
   - Random Forest (non-linear, feature importance)
   - XGBoost (gradient boosting, strong baseline)
   - SVM with RBF kernel (non-linear, good for small data)
   - MLP (2 hidden layers with heavy dropout)
3. **Hyperparameter Tuning** (optional) - Uses GridSearchCV to find best hyperparameters
4. **Saves Results** - Outputs to `results/baselines/`:
   - `baseline_results.csv` - Summary of all models with mean ± std metrics
   - `cv_scores_detailed.csv` - Detailed scores for each CV fold
   - `tabular/{model}_feature_importance.csv` - Top features for each model
   - `image/{model}_feature_importance.csv` - Top features for each model

**Output:**

```
results/baselines/
 baseline_results.csv          # Summary: modality, model, accuracy±std, f1±std, auc±std
 cv_scores_detailed.csv        # Per-fold scores for all metrics
 tabular/
    lr_feature_importance.csv
    rf_feature_importance.csv
    xgboost_feature_importance.csv
    ...
 image/
     lr_feature_importance.csv
     rf_feature_importance.csv
     ...
```

**Models and Hyperparameters:**

All models use conservative hyperparameters to prevent overfitting on small datasets (~70-210 samples):

- **Logistic Regression**: L2 regularization, C=1.0
- **Random Forest**: max_depth=8, min_samples_leaf=3
- **XGBoost**: max_depth=3, min_child_weight=3, reg_alpha=0.1
- **SVM**: RBF kernel, C=1.0, probability=True
- **MLP**: 2 hidden layers (128, 64), dropout=0.4, early stopping

See `configs/model_config.yaml` for all hyperparameters and tuning grids.

**Expected Runtime:**

- Without hyperparameter tuning: 10-20 minutes
- With hyperparameter tuning: 30-60 minutes

**Cross-Validation:**

Uses stratified 5-fold CV by default:
- Maintains class distribution in each fold
- Reports mean ± standard deviation across folds
- Small datasets benefit from robust estimates

**Interpreting Results:**

Good results for small datasets:
- **Accuracy**: 75-85% with narrow std (<0.05)
- **F1 Score**: 0.75-0.85 with narrow std
- **AUC**: >0.80 with narrow std

Warning signs:
- **Very wide std** (>0.10): High variance, need more data or simpler model
- **Train >> Test**: Overfitting, reduce model complexity
- **All models similar**: Features may not be discriminative

---

## Configuration

All scripts use `configs/model_config.yaml` for hyperparameters:

```yaml
random_forest:
  n_estimators: 100
  max_depth: 8
  min_samples_split: 5
  min_samples_leaf: 3

xgboost:
  n_estimators: 100
  max_depth: 3
  learning_rate: 0.1
  min_child_weight: 3
```

Edit this file to customize model behavior.

---

## Tips for Small Datasets

1. **Use Conservative Hyperparameters**
   - Smaller `max_depth` (3-8 for trees, 2-3 for XGBoost)
   - Higher `min_samples_leaf` (3-5)
   - Regularization (L1/L2 penalties)

2. **Always Use Cross-Validation**
   - 5-fold stratified CV is minimum
   - 10-fold for very small datasets (<50 samples)
   - Never rely on single train/test split

3. **Report Confidence Intervals**
   - Mean ± std across CV folds
   - Shows metric uncertainty
   - Critical for small datasets

4. **Feature Selection**
   - Too many features → overfitting
   - Use feature importance to select top 20-50
   - Domain knowledge (PD-specific features)

5. **Data Augmentation** (use carefully)
   - SMOTE for tabular (synthetic samples)
   - Geometric for images (rotation, flip)
   - Don't over-augment (2-3x max)

---

## Troubleshooting

**"Out of memory"**
- Reduce `batch_size` in image feature extraction
- Use smaller models (reduce `max_depth`)
- Disable image features and use tabular only

**"Very low accuracy" (<60%)
- Check data quality and labels
- Verify feature extraction worked correctly
- Try simpler models (Logistic Regression first)
- Check for data leakage

**"Very high std" (>0.10)
- Increase CV folds (10 instead of 5)
- Use simpler models
- Reduce feature count
- Get more data

**"XGBoost not installed"**
```bash
pip install xgboost
```

---

## Next Steps

After training baselines:

1. **Compare Models** - Check `baseline_results.csv`
2. **Analyze Features** - Look at `feature_importance.csv` files
3. **Select Best Model** - Choose based on F1 and AUC (not just accuracy)
4. **Try Fusion** - Combine tabular + image features
5. **Hyperparameter Tuning** - Run with `--hyperparameter_tuning`

For fusion models, see:
```python
from src.models.fusion import EarlyFusion, LateFusion
```

---

**For help or questions:**
- Olivia Liau: oliau@usc.edu
- Project: MoE-MultiSDAR-PD-Classification
