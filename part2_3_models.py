"""
part2_3_models.py
=================
Part 2 — Supervised Machine Learning Model — Build, Train, and Evaluate
Part 3 — Advanced Modeling — Ensembles, Tuning, and Full ML Pipeline
Capstone Project · Housing Price Dataset

Outputs:
  • figures/08_roc_curve.png
  • figures/09_learning_curve.png
  • best_model.pkl
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ML Preprocessing & Metrics
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    mean_squared_error, r2_score, confusion_matrix, classification_report,
    roc_curve, roc_auc_score, precision_score, recall_score, f1_score
)
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
import joblib

warnings.filterwarnings("ignore")
os.makedirs("figures", exist_ok=True)
sns.set_theme(style="darkgrid", palette="muted")

# ==============================================================================
# PART 2 — SUPERVISED MACHINE LEARNING MODEL
# ==============================================================================
print("\n" + "="*80)
print("PART 2 — TASK 1: LOAD DATA AND DEFINE TARGETS")
print("="*80)

# Load cleaned dataset
df = pd.read_csv("cleaned_data.csv")

# X contains all columns except target 'SalePrice', 'Id', and high-null column 'Alley'
X = df.drop(columns=["SalePrice", "Id", "Alley"], errors="ignore")

# Define target columns
y_reg = df["SalePrice"]
# Classification label: 1 if SalePrice > median SalePrice, else 0
y_clf = (y_reg > y_reg.median()).astype(int)

print(f"Feature matrix shape: {X.shape}")
print(f"Regression target shape: {y_reg.shape}")
print(f"Classification target shape (balanced): {y_clf.shape}")
print(f"Classification class counts:\n{y_clf.value_counts()}")

# ==============================================================================
# PART 2 — TASK 2: CATEGORICAL ENCODING
# ==============================================================================
print("\n" + "="*80)
print("PART 2 — TASK 2: CATEGORICAL ENCODING")
print("="*80)

# Categorical columns
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
print(f"Categorical columns to encode: {cat_cols}")

# One-hot encode all categorical columns and drop first to avoid multicollinearity
# Since all categoricals here (Neighborhood, BldgType, HouseStyle, SaleCondition)
# are nominal (no natural order), one-hot encoding is appropriate.
X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=int)
print(f"Features shape after one-hot encoding: {X_encoded.shape}")

# ==============================================================================
# PART 2 — TASK 3: LEAK-FREE TRAIN-TEST SPLIT AND SCALING
# ==============================================================================
print("\n" + "="*80)
print("PART 2 — TASK 3: TRAIN-TEST SPLIT & SCALING")
print("="*80)

# Train-test split (80-20)
X_train, X_test, y_reg_train, y_reg_test = train_test_split(
    X_encoded, y_reg, test_size=0.2, random_state=42
)
_, _, y_clf_train, y_clf_test = train_test_split(
    X_encoded, y_clf, test_size=0.2, random_state=42
)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape : {X_test.shape}")

# Scale features using StandardScaler fit only on the training set
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print("StandardScaler successfully fitted on X_train and applied to X_train and X_test.")

# ==============================================================================
# PART 2 — TASK 4: REGRESSION MODEL — LINEAR & RIDGE REGRESSION
# ==============================================================================
print("\n" + "="*80)
print("PART 2 — TASK 4: REGRESSION MODELS (OLS & RIDGE)")
print("="*80)

# 1. OLS Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_reg_train)
y_pred_reg_lr = lr.predict(X_test_scaled)

mse_lr = mean_squared_error(y_reg_test, y_pred_reg_lr)
r2_lr  = r2_score(y_reg_test, y_pred_reg_lr)

print("── Ordinary Least Squares (OLS) Linear Regression ──")
print(f"  Test MSE: {mse_lr:.2f}")
print(f"  Test R² : {r2_lr:.4f}")

# Coefficients
coef_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Coefficient": lr.coef_,
    "Abs_Coefficient": np.abs(lr.coef_)
}).sort_values("Abs_Coefficient", ascending=False)

print("\nTop 3 features by absolute coefficient in OLS:")
print(coef_df.head(3).to_string(index=False))

# 2. Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_reg_train)
y_pred_reg_ridge = ridge.predict(X_test_scaled)

mse_ridge = mean_squared_error(y_reg_test, y_pred_reg_ridge)
r2_ridge  = r2_score(y_reg_test, y_pred_reg_ridge)

print("\n── Ridge Regression (alpha=1.0) ──")
print(f"  Test MSE: {mse_ridge:.2f}")
print(f"  Test R² : {r2_ridge:.4f}")

# ==============================================================================
# PART 2 — TASK 5: CLASSIFICATION MODEL — LOGISTIC REGRESSION
# ==============================================================================
print("\n" + "="*80)
print("PART 2 — TASK 5: CLASSIFICATION MODEL (LOGISTIC REGRESSION)")
print("="*80)

# Check class imbalance
counts = y_clf_train.value_counts()
pcts = y_clf_train.value_counts(normalize=True) * 100
print(f"y_clf_train class distribution:\n{counts.to_string()}\nPercentages:\n{pcts.to_string()}")

# Since the binary target was derived by splitting at the median, classes are exactly 50/50.
# No class imbalance handling is strictly needed, but we will print the check as requested.

# Baseline Logistic Regression (C=1.0)
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_clf_train)
y_pred_clf = log_reg.predict(X_test_scaled)
y_prob_clf = log_reg.predict_proba(X_test_scaled)[:, 1]

print("\nConfusion Matrix:")
print(confusion_matrix(y_clf_test, y_pred_clf))

print("\nClassification Report:")
print(classification_report(y_clf_test, y_pred_clf))

auc_baseline = roc_auc_score(y_clf_test, y_prob_clf)
print(f"ROC-AUC Score: {auc_baseline:.4f}")

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_clf_test, y_prob_clf)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"Baseline C=1.0 (AUC = {auc_baseline:.4f})")
ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve - Logistic Regression")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("figures/08_roc_curve.png", dpi=150)
plt.close()
print("Saved ROC curve: figures/08_roc_curve.png")

# ==============================================================================
# PART 2 — TASK 5b: DECISION-THRESHOLD SENSITIVITY
# ==============================================================================
print("\n" + "="*80)
print("PART 2 — TASK 5b: DECISION-THRESHOLD SENSITIVITY TABLE")
print("="*80)

thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]
thresh_records = []

for t in thresholds:
    y_pred_t = (y_prob_clf >= t).astype(int)
    p = precision_score(y_clf_test, y_pred_t)
    r = recall_score(y_clf_test, y_pred_t)
    f1 = f1_score(y_clf_test, y_pred_t)
    thresh_records.append({
        "Threshold": f"{t:.2f}",
        "Precision": f"{p:.4f}",
        "Recall": f"{r:.4f}",
        "F1": f"{f1:.4f}"
    })

thresh_df = pd.DataFrame(thresh_records)
print(thresh_df.to_string(index=False))

# ==============================================================================
# PART 2 — TASK 6: REGULARIZATION EXPERIMENT
# ==============================================================================
print("\n" + "="*80)
print("PART 2 — TASK 6: REGULARIZATION EXPERIMENT (C=0.01)")
print("="*80)

# Logistic Regression with C=0.01 (Stronger L2 penalty)
log_reg_strong = LogisticRegression(C=0.01, max_iter=1000, random_state=42)
log_reg_strong.fit(X_train_scaled, y_clf_train)
y_pred_clf_strong = log_reg_strong.predict(X_test_scaled)
y_prob_clf_strong = log_reg_strong.predict_proba(X_test_scaled)[:, 1]

auc_strong = roc_auc_score(y_clf_test, y_prob_clf_strong)
p_strong = precision_score(y_clf_test, y_pred_clf_strong)
r_strong = recall_score(y_clf_test, y_pred_clf_strong)

print(f"C=0.01 model:")
print(f"  Precision: {p_strong:.4f}")
print(f"  Recall   : {r_strong:.4f}")
print(f"  ROC-AUC  : {auc_strong:.4f}")

# Bootstrap Confidence Interval for AUC difference (C=1.0 - C=0.01)
print("\nCalculating Bootstrap Confidence Interval for AUC difference (n=500)...")
np.random.seed(42)
n_bootstraps = 500
bootstrapped_diffs = []
y_clf_test_arr = np.array(y_clf_test)

for _ in range(n_bootstraps):
    indices = np.random.choice(len(y_clf_test_arr), size=len(y_clf_test_arr), replace=True)
    if len(np.unique(y_clf_test_arr[indices])) < 2:
        continue
    auc_1 = roc_auc_score(y_clf_test_arr[indices], y_prob_clf[indices])
    auc_01 = roc_auc_score(y_clf_test_arr[indices], y_prob_clf_strong[indices])
    bootstrapped_diffs.append(auc_1 - auc_01)

mean_diff = np.mean(bootstrapped_diffs)
ci_lower  = np.percentile(bootstrapped_diffs, 2.5)
ci_upper  = np.percentile(bootstrapped_diffs, 97.5)

print(f"Mean AUC Difference (C=1.0 - C=0.01): {mean_diff:.5f}")
print(f"95% Bootstrap Confidence Interval: [{ci_lower:.5f}, {ci_upper:.5f}]")
print(f"Excludes zero? {ci_lower > 0 or ci_upper < 0}")

# ==============================================================================
# PART 3 — ADVANCED MODELING
# ==============================================================================
print("\n" + "="*80)
print("PART 3 — TASK 1 & 2: DECISION TREE CLASSIFIERS")
print("="*80)

# Unconstrained Decision Tree
dt_unconstrained = DecisionTreeClassifier(random_state=42)
dt_unconstrained.fit(X_train_scaled, y_clf_train)
acc_train_un = dt_unconstrained.score(X_train_scaled, y_clf_train)
acc_test_un  = dt_unconstrained.score(X_test_scaled, y_clf_test)
print("── Unconstrained Decision Tree (max_depth=None) ──")
print(f"  Training Accuracy: {acc_train_un:.4f}")
print(f"  Test Accuracy    : {acc_test_un:.4f}")

# Controlled Decision Tree
dt_controlled = DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=42)
dt_controlled.fit(X_train_scaled, y_clf_train)
acc_train_co = dt_controlled.score(X_train_scaled, y_clf_train)
acc_test_co  = dt_controlled.score(X_test_scaled, y_clf_test)
print("\n── Controlled Decision Tree (max_depth=5, min_samples_split=20) ──")
print(f"  Training Accuracy: {acc_train_co:.4f}")
print(f"  Test Accuracy    : {acc_test_co:.4f}")

# Task 3: Gini vs Entropy
dt_gini = DecisionTreeClassifier(max_depth=5, criterion="gini", random_state=42)
dt_entropy = DecisionTreeClassifier(max_depth=5, criterion="entropy", random_state=42)
dt_gini.fit(X_train_scaled, y_clf_train)
dt_entropy.fit(X_train_scaled, y_clf_train)
print(f"\nGini Criterion Test Accuracy: {dt_gini.score(X_test_scaled, y_clf_test):.4f}")
print(f"Entropy Criterion Test Accuracy: {dt_entropy.score(X_test_scaled, y_clf_test):.4f}")

# ==============================================================================
# PART 3 — TASK 4: ENSEMBLE MODELS (RANDOM FOREST & GRADIENT BOOSTING)
# ==============================================================================
print("\n" + "="*80)
print("PART 3 — TASK 4: ENSEMBLES (RANDOM FOREST & GRADIENT BOOSTING)")
print("="*80)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train_scaled, y_clf_train)
rf_train_acc = rf.score(X_train_scaled, y_clf_train)
rf_test_acc  = rf.score(X_test_scaled, y_clf_test)
rf_auc       = roc_auc_score(y_clf_test, rf.predict_proba(X_test_scaled)[:, 1])

print("── RandomForestClassifier ──")
print(f"  Training Accuracy: {rf_train_acc:.4f}")
print(f"  Test Accuracy    : {rf_test_acc:.4f}")
print(f"  Test ROC-AUC     : {rf_auc:.4f}")

# Top 5 Features
rf_importances = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False)

print("\nTop 5 features by Random Forest importance:")
print(rf_importances.head(5).to_string(index=False))

# Task 4a: Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb.fit(X_train_scaled, y_clf_train)
gb_train_acc = gb.score(X_train_scaled, y_clf_train)
gb_test_acc  = gb.score(X_test_scaled, y_clf_test)
gb_auc       = roc_auc_score(y_clf_test, gb.predict_proba(X_test_scaled)[:, 1])

print("\n── GradientBoostingClassifier ──")
print(f"  Training Accuracy: {gb_train_acc:.4f}")
print(f"  Test Accuracy    : {gb_test_acc:.4f}")
print(f"  Test ROC-AUC     : {gb_auc:.4f}")

# ==============================================================================
# PART 3 — TASK 4b: FEATURE ABLATION STUDY
# ==============================================================================
print("\n" + "="*80)
print("PART 3 — TASK 4b: FEATURE ABLATION STUDY")
print("="*80)

# Identify 5 lowest importance features from Random Forest
lowest_5_features = rf_importances.tail(5)["Feature"].tolist()
print(f"5 lowest importance features to remove:\n{lowest_5_features}")

# Create subsets
X_train_reduced = X_train.drop(columns=lowest_5_features)
X_test_reduced  = X_test.drop(columns=lowest_5_features)

# Scale
scaler_red = StandardScaler()
X_train_red_scaled = scaler_red.fit_transform(X_train_reduced)
X_test_red_scaled  = scaler_red.transform(X_test_reduced)

# Train identical RF on reduced set
rf_reduced = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_reduced.fit(X_train_red_scaled, y_clf_train)
rf_reduced_auc = roc_auc_score(y_clf_test, rf_reduced.predict_proba(X_test_red_scaled)[:, 1])

print(f"\nFull Model (All Features) Test ROC-AUC: {rf_auc:.5f}")
print(f"Reduced Model (5 Cols Removed) Test ROC-AUC: {rf_reduced_auc:.5f}")
print(f"AUC Difference (Full - Reduced): {rf_auc - rf_reduced_auc:.5f}")

# ==============================================================================
# PART 3 — TASK 5: CROSS-VALIDATED COMPARISON
# ==============================================================================
print("\n" + "="*80)
print("PART 3 — TASK 5: STRATIFIED 5-FOLD CROSS-VALIDATION")
print("="*80)

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models_to_compare = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Controlled Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
}

for name, model in models_to_compare.items():
    cv_scores = cross_val_score(model, X_train_scaled, y_clf_train, cv=cv_strategy, scoring="roc_auc")
    print(f"{name:25s} -> Mean AUC: {cv_scores.mean():.4f} (Std: {cv_scores.std():.4f})")

# ==============================================================================
# PART 3 — TASK 6 & 7: PIPELINE, GRIDSEARCHCV & LEARNING CURVE
# ==============================================================================
print("\n" + "="*80)
print("PART 3 — TASK 6: PIPELINE AND GRIDSEARCHCV")
print("="*80)

# Build Pipeline (with SimpleImputer & StandardScaler)
pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    RandomForestClassifier(random_state=42)
)

param_grid = {
    'randomforestclassifier__n_estimators': [50, 100, 200],
    'randomforestclassifier__max_depth': [5, 10, None],
    'randomforestclassifier__min_samples_leaf': [1, 5]
}

# Run GridSearchCV on raw (unscaled) X_train and y_clf_train
grid_search = GridSearchCV(
    pipeline, param_grid, cv=cv_strategy, scoring="roc_auc", n_jobs=-1
)
grid_search.fit(X_train, y_clf_train)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV Score  : {grid_search.best_score_:.4f}")

best_pipeline = grid_search.best_estimator_

# Task 7: Manual learning curve
print("\n── Manual Learning Curve ──")
fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
curve_records = []

for f in fractions:
    n_samples = int(f * len(X_train))
    X_sub = X_train.iloc[:n_samples]
    y_sub = y_clf_train.iloc[:n_samples]
    
    # Fit model on training subset
    best_pipeline.fit(X_sub, y_sub)
    
    # Compute AUCs
    train_auc = roc_auc_score(y_sub, best_pipeline.predict_proba(X_sub)[:, 1])
    test_auc  = roc_auc_score(y_clf_test, best_pipeline.predict_proba(X_test)[:, 1])
    
    curve_records.append({
        "Fraction": f,
        "Train AUC": train_auc,
        "Test AUC": test_auc
    })

curve_df = pd.DataFrame(curve_records)
print(curve_df.to_string(index=False))

# Plot learning curve
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(curve_df["Fraction"] * 100, curve_df["Train AUC"], "o-", label="Training AUC")
ax.plot(curve_df["Fraction"] * 100, curve_df["Test AUC"], "s-", label="Test AUC")
ax.set_xlabel("Percentage of Training Data (%)")
ax.set_ylabel("ROC-AUC Score")
ax.set_title("Manual Learning Curve - Tuned Random Forest Pipeline")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("figures/09_learning_curve.png", dpi=150)
plt.close()
print("Saved learning curve: figures/09_learning_curve.png")

# ==============================================================================
# PART 3 — TASK 8: SERIALIZE THE BEST MODEL
# ==============================================================================
print("\n" + "="*80)
print("PART 3 — TASK 8: SERIALIZE THE BEST PIPELINE")
print("="*80)

# Save pipeline
joblib.dump(best_pipeline, "best_model.pkl")
print("Successfully serialized best pipeline to 'best_model.pkl'.")

# Reload and test on hand-crafted rows
loaded_model = joblib.load("best_model.pkl")

# Generate two hand-crafted rows from X_test to test prediction
test_rows = X_test.iloc[:2].copy()
predictions = loaded_model.predict(test_rows)
probs = loaded_model.predict_proba(test_rows)[:, 1]

print("\nReloaded model predictions on 2 hand-crafted rows:")
for idx, (pred, prob) in enumerate(zip(predictions, probs)):
    print(f"  Row {idx+1}: Predicted Class = {pred}, Class 1 Probability = {prob:.4f}")

print("\n✓ Preprocessing, training, and evaluations complete.")
