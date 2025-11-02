"""
Transformadores personalizados para preprocesamiento de datos.
Compatible con pipelines de scikit-learn mediante BaseEstimator y TransformerMixin.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Optional


class DataCleanerTransformer(BaseEstimator, TransformerMixin):
    """Reemplaza valores inválidos por NaN en columnas de texto."""
    
    INVALID_VALUES = ['NULL', 'NaN', 'nan', 'error', 'invalid', 'bad', 
                      'unknown', '?', 'ERROR', 'N/A', '', ' ']
    
    def __init__(self, exclude_columns: Optional[List[str]] = None):
        self.exclude_columns = exclude_columns or []
    
    def fit(self, X, y=None):
        """No requiere ajuste, retorna el transformador sin cambios."""
        return self
    
    def transform(self, X):
        """Detecta y reemplaza valores inválidos en columnas de texto."""
        X_clean = X.copy()
        for col in X_clean.columns:
            if col not in self.exclude_columns and X_clean[col].dtype == 'object':
                mask = X_clean[col].isin(self.INVALID_VALUES)
                if hasattr(X_clean[col], 'str'):
                    mask |= X_clean[col].str.strip().eq('')
                if mask.sum() > 0:
                    X_clean.loc[mask, col] = np.nan
        return X_clean


class NumericConverterTransformer(BaseEstimator, TransformerMixin):
    """Convierte todas las columnas a formato numérico."""
    
    def __init__(self, exclude_columns: Optional[List[str]] = None):
        self.exclude_columns = exclude_columns or []
    
    def fit(self, X, y=None):
        """No requiere ajuste, retorna el transformador sin cambios."""
        return self
    
    def transform(self, X):
        """Convierte columnas a tipo numérico, valores no convertibles se vuelven NaN."""
        X_numeric = X.copy()
        for col in X_numeric.columns:
            if col not in self.exclude_columns:
                X_numeric[col] = pd.to_numeric(X_numeric[col], errors='coerce')
        return X_numeric


class CustomImputerTransformer(BaseEstimator, TransformerMixin):
    """Rellena valores faltantes usando mediana, media o moda."""
    
    def __init__(self, method: str = 'median', exclude_columns: Optional[List[str]] = None):
        self.method = method
        self.exclude_columns = exclude_columns or []
        self.fill_values_ = {}
    
    def fit(self, X, y=None):
        """Calcula estadísticos de imputación para cada columna numérica."""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        cols_to_impute = [col for col in numeric_cols if col not in self.exclude_columns]
        
        for col in cols_to_impute:
            if self.method == 'median':
                self.fill_values_[col] = X[col].median()
            elif self.method == 'mean':
                self.fill_values_[col] = X[col].mean()
            else:
                mode_values = X[col].mode()
                self.fill_values_[col] = mode_values[0] if len(mode_values) > 0 else 0
            
            if pd.isna(self.fill_values_[col]):
                self.fill_values_[col] = 0
        
        return self
    
    def transform(self, X):
        """Rellena valores faltantes usando los estadísticos calculados en fit."""
        X_imputed = X.copy()
        for col in X_imputed.columns:
            if col in X_imputed.select_dtypes(include=[np.number]).columns:
                if col in self.fill_values_:
                    X_imputed[col] = X_imputed[col].fillna(self.fill_values_[col])
                else:
                    X_imputed[col] = X_imputed[col].fillna(0)
        
        return X_imputed
