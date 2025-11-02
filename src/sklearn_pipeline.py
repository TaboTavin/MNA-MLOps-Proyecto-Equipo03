"""
Pipeline de entrenamiento con preprocesamiento, modelos y seguimiento en MLflow.
Gestiona carga de datos, división train/test, transformaciones y evaluación de modelos.
"""

import pandas as pd
from typing import Dict, Tuple
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn

from src.transformers import DataCleanerTransformer, NumericConverterTransformer, CustomImputerTransformer


class SklearnMLPipeline:
    """Pipeline completo que entrena modelos y registra resultados en MLflow."""
    
    def __init__(self, data_path: str, target_column: str = 'Class', 
                 experiment_name: str = "sklearn_pipeline"):
        self.data_path = data_path
        self.target_column = target_column
        self.experiment_name = experiment_name
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def _load_and_split(self, test_size: float = 0.2) -> None:
        """Carga datos, limpia etiquetas y divide en conjuntos de entrenamiento y prueba."""
        df = pd.read_csv(self.data_path)
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column].astype(str).str.strip().str.lower()
        
        mask = ~y.isnull() & ~y.isin(['nan', 'none', ''])
        X, y = X[mask], y[mask]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
    
    def _build_pipeline(self, model: BaseEstimator) -> Pipeline:
        """Construye pipeline con limpieza, conversión numérica, imputación, escalado y clasificador."""
        return Pipeline([
            ('cleaner', DataCleanerTransformer()),
            ('numeric_converter', NumericConverterTransformer()),
            ('imputer', CustomImputerTransformer(method='median')),
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])
    
    def _train_model(self, model: BaseEstimator, model_name: str) -> Tuple:
        """Entrena modelo, calcula métricas y registra todo en MLflow."""
        mlflow.set_experiment(self.experiment_name)
        
        with mlflow.start_run(run_name=model_name):
            pipeline = self._build_pipeline(model)
            pipeline.fit(self.X_train, self.y_train)
            
            y_pred = pipeline.predict(self.X_test)
            
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
            
            mlflow.log_params(model.get_params())
            mlflow.log_metrics({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
            
            from mlflow.models import infer_signature
            signature = infer_signature(self.X_train, y_pred)
            input_example = self.X_train.iloc[:5] if hasattr(self.X_train, 'iloc') else self.X_train[:5]
            
            mlflow.sklearn.log_model(
                pipeline, 
                artifact_path="model",
                signature=signature,
                input_example=input_example
            )
            
            return pipeline, mlflow.active_run().info.run_id, accuracy
        
    def run(self, models: Dict[str, BaseEstimator], test_size: float = 0.2) -> Dict:
        """Ejecuta entrenamiento para todos los modelos y retorna resultados."""
        self._load_and_split(test_size)
        
        return {
            name: {
                'pipeline': pipeline,
                'run_id': run_id,
                'accuracy': accuracy
            }
            for name, model in models.items()
            for pipeline, run_id, accuracy in [self._train_model(model, name)]
        }
