"""
===============================================================================
Paquete: src
Propósito:
    Este archivo convierte la carpeta /src en un paquete Python.
    Permite que se puedan importar módulos internos de manera estructurada.

Ejemplos de uso:
    >>> from src.data.make_dataset import DataSplitter
    >>> from src.features.build_features import FeatureBuilder
    >>> from src.models.train_model import ModelTrainer
    >>> from src.models.predict_model import Predictor
    >>> from src.visualization.visual import Visualizer

Notas:
    - Aquí puedes centralizar imports comunes para hacerlos más cortos.
    - Por ejemplo, si defines __all__, tus compañeros pueden importar desde src
      directamente en lugar de ir a cada submódulo.
===============================================================================
"""

__version__ = "0.1.0"
__author__ = "Equipo 03 - MNA MLOps"

# Opcional: exponer clases/funciones principales de cada módulo
from .data.make_dataset import DataSplitter, SplitConfig, DatasetPaths
from .features.build_features import FeatureBuilder, FeatureConfig, ProjectPaths as FeaturePaths
from .models.train_model import ModelTrainer, TrainConfig, ProjectPaths as TrainPaths
from .models.predict_model import Predictor, ProjectPaths as PredictPaths
from .visualization.visual import Visualizer, ProjectPaths as VisualPaths

# Definir qué se exporta si alguien hace `from src import ...`
__all__ = [
    "DataSplitter", "SplitConfig", "DatasetPaths",
    "FeatureBuilder", "FeatureConfig", "FeaturePaths",
    "ModelTrainer", "TrainConfig", "TrainPaths",
    "Predictor", "PredictPaths",
    "Visualizer", "VisualPaths",
]