"""
Interfaz simplificada para ejecutar el pipeline de entrenamiento.
Envuelve SklearnMLPipeline para facilitar su uso.
"""

from typing import Dict
from sklearn.base import BaseEstimator
from src.sklearn_pipeline import SklearnMLPipeline


class MLPipeline:
    """Ejecuta pipeline completo de preprocesamiento, entrenamiento y evaluación."""
    
    def __init__(self, data_path: str, target_column: str = 'Class', 
                 experiment_name: str = "sklearn_pipeline"):
        self.sklearn_pipeline = SklearnMLPipeline(data_path, target_column, experiment_name)
    
    def run(self, models: Dict[str, BaseEstimator], test_size: float = 0.2) -> Dict:
        """Entrena modelos y retorna resultados con métricas."""
        return self.sklearn_pipeline.run(models, test_size)