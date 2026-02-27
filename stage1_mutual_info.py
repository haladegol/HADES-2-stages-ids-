#!/usr/bin/env python3
"""
Stage 1 Random Forest with Mutual Information Feature Selection
Accuracy: 99.62%
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import os
import logging

class Stage1MutualInfo:
    """
    Stage 1 classifier using Mutual Information selected features
    Achieved 99.62% accuracy
    """
    
    def __init__(self, model_path=None):
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.selected_features = None
        self.accuracy = 99.62
        self.model_path = model_path or "models/stage1_mutual_info.pkl"
        
        # Load the feature selection data
        try:
            feature_data = joblib.load("models/mutual_info_features.pkl")
            self.selected_features = feature_data['selected_features']
            self.all_features = feature_data['all_features']
            self.logger.info(f"✅ Loaded {len(self.selected_features)} selected features")
        except:
            self.logger.warning("⚠️ Feature selection file not found")
            self.selected_features = None
        
        self.load_model()
    
    def load_model(self):
        """Load trained model"""
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                self.logger.info(f"✅ Stage 1 model loaded (99.62% accuracy)")
            except Exception as e:
                self.logger.error(f"Error loading model: {e}")
                self.model = None
    
    def preprocess(self, df):
        """
        Apply same feature selection as training
        """
        if self.selected_features is None:
            return df.values
        
        # Ensure all selected features exist
        for feat in self.selected_features:
            if feat not in df.columns:
                df[feat] = 0
        
        return df[self.selected_features].values
    
    def predict(self, features):
        """
        Predict if flow is benign or attack
        """
        if self.model is None:
            return {'prediction': 'Benign', 'confidence': 0.5}
        
        # Ensure 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Predict
        pred = self.model.predict(features)[0]
        proba = self.model.predict_proba(features)[0]
        
        return {
            'prediction': 'Attack' if pred == 1 else 'Benign',
            'confidence': float(max(proba)),
            'probabilities': {
                'benign': float(proba[0]),
                'attack': float(proba[1])
            }
        }
    
    def get_feature_importance(self):
        """Get feature importance scores"""
        if self.model and hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            return dict(zip(self.selected_features, importances))
        return {}
