#!/usr/bin/env python3
"""
Stage 2 XGBoost with Hyperparameter Tuning
Target: 99% Accuracy on Multi-class
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import glob
import os
from scipy.stats import uniform, randint

from src.data.advanced_preprocessor import AdvancedPreprocessor

print("=" * 100)
print("🎯 STAGE 2 XGBOOST - HYPERPARAMETER TUNED")
print("=" * 100)

# Initialize preprocessor
preprocessor = AdvancedPreprocessor()

# Load and preprocess data
print("\n📂 Loading data...")
files = glob.glob("data/enhanced/file_*.csv")
dfs = [preprocessor.load_and_clean(f) for f in files]
df = pd.concat(dfs, ignore_index=True)

# Create advanced features
df = preprocessor.create_advanced_features(df)

# Filter only attack samples
attack_df = df[df['Attack_Type'] != 'Benign'].copy()
print(f"\n🎯 Attack samples: {len(attack_df):,}")

# Prepare features (use all for Stage 2)
feature_cols = [c for c in attack_df.columns if c not in ['Attack_Type']]
X = attack_df[feature_cols].values
y = attack_df['Attack_Type'].values

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"\n📊 Attack classes: {len(le.classes_)}")
for cls in le.classes_:
    count = sum(y == cls)
    print(f"   {cls}: {count}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n📊 Training set: {len(X_train):,}")
print(f"📊 Test set: {len(X_test):,}")

# ============================================
# HYPERPARAMETER TUNING FOR XGBOOST
# ============================================
print("\n🔍 Hyperparameter Tuning for XGBoost...")

# Define parameter distribution
param_dist = {
    'n_estimators': randint(500, 2000),
    'max_depth': randint(6, 20),
    'learning_rate': uniform(0.005, 0.1),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'min_child_weight': randint(1, 10),
    'gamma': uniform(0, 0.5),
    'reg_alpha': uniform(0, 1),
    'reg_lambda': uniform(0.5, 2)
}

# Base model
xgb_model = xgb.XGBClassifier(
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss',
    tree_method='hist'
)

# Randomized search
random_search = RandomizedSearchCV(
    xgb_model,
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2,
    random_state=42
)

print("\n🚀 Training XGBoost with hyperparameter tuning...")
random_search.fit(X_train_scaled, y_train)

print(f"\n✅ Best parameters found:")
for param, value in random_search.best_params_.items():
    print(f"   {param}: {value}")

print(f"✅ Best CV accuracy: {random_search.best_score_:.6f} ({random_search.best_score_*100:.4f}%)")

# Best model
best_xgb = random_search.best_estimator_

# Evaluate
y_pred = best_xgb.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Test Accuracy: {accuracy:.6f} ({accuracy*100:.4f}%)")

# Detailed classification report
print("\n📊 Classification Report:")
y_test_labels = le.inverse_transform(y_test)
y_pred_labels = le.inverse_transform(y_pred)
print(classification_report(y_test_labels, y_pred_labels))

# Per-class accuracy
print("\n📊 Per-Class Accuracy:")
cm = confusion_matrix(y_test_labels, y_pred_labels, labels=le.classes_)
for i, cls in enumerate(le.classes_):
    tp = cm[i, i]
    total = cm[i, :].sum()
    class_acc = tp / total if total > 0 else 0
    print(f"   {cls:30} : {class_acc*100:6.2f}% ({tp}/{total})")

# ============================================
# SAVE MODEL
# ============================================
print("\n💾 Saving tuned XGBoost model...")
os.makedirs("models", exist_ok=True)

model_data = {
    'model': best_xgb,
    'scaler': scaler,
    'encoder': le,
    'feature_cols': feature_cols,
    'best_params': random_search.best_params_,
    'accuracy': accuracy,
    'cv_score': random_search.best_score_
}

joblib.dump(model_data, "models/stage2_tuned_99.pkl")
joblib.dump(best_xgb, "models/stage2_xgboost.pkl")  # For dashboard
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(le, "models/stage2_encoder.pkl")

print(f"✅ Model saved to: models/stage2_tuned_99.pkl")

if accuracy >= 0.99:
    print("\n🎉🎉🎉 TARGET ACHIEVED: 99%+ ACCURACY! 🎉🎉🎉")
else:
    gap = (0.99 - accuracy) * 100
    print(f"\n📈 Current: {accuracy*100:.2f}% | Target: 99% | Gap: {gap:.2f}%")
