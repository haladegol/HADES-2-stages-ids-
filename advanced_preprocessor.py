#!/usr/bin/env python3
"""
Advanced Preprocessing with Feature Selection Techniques
For achieving 99% accuracy in both stages
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score
import joblib
import logging
from scipy import stats

class AdvancedPreprocessor:
    """
    Advanced preprocessing with multiple feature selection techniques
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scaler = RobustScaler()  # Robust to outliers
        self.selected_features = None
        self.feature_scores = {}
        
    def load_and_clean(self, file_path, nrows=None):
        """Load data with advanced cleaning"""
        self.logger.info(f"Loading data from {file_path}")
        
        df = pd.read_csv(file_path, nrows=nrows, low_memory=False)
        
        # Convert to numeric
        for col in df.columns:
            if col != 'Attack_Type':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle outliers using IQR method
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            df[col] = df[col].clip(lower_bound, upper_bound)
        
        # Remove duplicates and NaN
        df = df.drop_duplicates()
        df = df.dropna()
        
        self.logger.info(f"Loaded {len(df)} clean samples")
        return df
    
    def create_advanced_features(self, df):
        """
        Create advanced engineered features
        """
        self.logger.info("Creating advanced features...")
        
        # Statistical features
        iat_cols = [c for c in df.columns if 'IAT' in c]
        if len(iat_cols) > 5:
            df['IAT_kurtosis'] = df[iat_cols].kurtosis(axis=1)
            df['IAT_skew'] = df[iat_cols].skew(axis=1)
        
        # Ratio features with safety
        ratio_pairs = [
            ('Fwd IAT Mean', 'Bwd IAT Mean', 'IAT_Ratio_1'),
            ('Active Mean', 'Idle Mean', 'Active_Idle_Ratio'),
            ('Flow IAT Max', 'Flow IAT Min', 'IAT_Range_Ratio'),
            ('Fwd IAT Max', 'Fwd IAT Min', 'Fwd_IAT_Spread')
        ]
        
        for col1, col2, name in ratio_pairs:
            if col1 in df.columns and col2 in df.columns:
                df[name] = (df[col1] + 1) / (df[col2] + 1)
        
        # Polynomial features of top indicators
        if 'Fwd Seg Size Min' in df.columns:
            df['FwdSeg_poly2'] = df['Fwd Seg Size Min'] ** 2
            df['FwdSeg_poly3'] = df['Fwd Seg Size Min'] ** 3
            df['FwdSeg_log'] = np.log1p(df['Fwd Seg Size Min'])
        
        return df
    
    def select_features_filter(self, X, y, k=30):
        """
        Method 1: Filter-based selection (Mutual Information)
        """
        self.logger.info("Applying filter-based selection (Mutual Information)...")
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        
        # Get scores
        scores = selector.scores_
        self.feature_scores['mutual_info'] = dict(zip(range(len(scores)), scores))
        
        return X_selected, selector.get_support(indices=True)
    
    def select_features_wrapper(self, X, y, n_features=25):
        """
        Method 2: Wrapper-based selection (RFE)
        """
        self.logger.info("Applying wrapper-based selection (RFE)...")
        estimator = RandomForestClassifier(n_estimators=100, n_jobs=-1)
        selector = RFE(estimator, n_features_to_select=n_features, step=5)
        X_selected = selector.fit_transform(X, y)
        
        return X_selected, selector.get_support(indices=True)
    
    def select_features_embedded(self, X, y, threshold=0.01):
        """
        Method 3: Embedded selection (Random Forest importance)
        """
        self.logger.info("Applying embedded selection (RF Importance)...")
        rf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
        rf.fit(X, y)
        
        # Get importance scores
        importances = rf.feature_importances_
        self.feature_scores['rf_importance'] = dict(zip(range(len(importances)), importances))
        
        # Select features above threshold
        selector = SelectFromModel(rf, threshold=threshold, prefit=True)
        X_selected = selector.transform(X)
        
        return X_selected, selector.get_support(indices=True)
    
    def ensemble_feature_selection(self, X, y, feature_names, top_k=25):
        """
        Ensemble of all selection methods
        """
        self.logger.info("Applying ensemble feature selection...")
        
        # Get selections from all methods
        _, idx1 = self.select_features_filter(X, y, k=top_k+5)
        _, idx2 = self.select_features_wrapper(X, y, n_features=top_k+5)
        _, idx3 = self.select_features_embedded(X, y, threshold=0.005)
        
        # Combine selections (features chosen by at least 2 methods)
        from collections import Counter
        all_indices = list(idx1) + list(idx2) + list(idx3)
        vote_counts = Counter(all_indices)
        
        # Select features with at least 2 votes
        ensemble_indices = [i for i, count in vote_counts.items() if count >= 2]
        ensemble_indices = sorted(ensemble_indices)[:top_k]
        
        selected_features = [feature_names[i] for i in ensemble_indices]
        X_selected = X[:, ensemble_indices]
        
        self.logger.info(f"Selected {len(selected_features)} features via ensemble")
        return X_selected, ensemble_indices, selected_features
    
    def prepare_for_training(self, df, target_col='is_attack', feature_cols=None):
        """
        Complete preparation pipeline
        """
        # Create binary labels
        if 'Attack_Type' in df.columns:
            df[target_col] = (df['Attack_Type'] != 'Benign').astype(int)
        
        # Get feature columns
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c not in ['Attack_Type', target_col]]
        
        X = df[feature_cols].values
        y = df[target_col].values
        
        # Apply ensemble feature selection
        X_selected, indices, selected_features = self.ensemble_feature_selection(
            X, y, feature_cols, top_k=25
        )
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_selected)
        
        return X_scaled, y, selected_features, indices
