"""
Configuración global de pytest.
Define fixtures compartidas para todos los tests.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil


@pytest.fixture(scope="session")
def test_data_dir():
    """Directorio temporal para datos de prueba que persiste durante toda la sesión."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_music_dataframe():
    """DataFrame de ejemplo con datos de música y emociones."""
    np.random.seed(42)
    n_samples = 100
    
    df = pd.DataFrame({
        "Tempo": np.random.uniform(60, 180, n_samples),
        "Energy": np.random.uniform(0, 1, n_samples),
        "Danceability": np.random.uniform(0, 1, n_samples),
        "Valence": np.random.uniform(0, 1, n_samples),
        "Acousticness": np.random.uniform(0, 1, n_samples),
        "Class": np.random.choice(["happy", "sad", "energetic", "calm"], n_samples)
    })
    return df


@pytest.fixture
def sample_csv_file(tmp_path, sample_music_dataframe):
    """Archivo CSV temporal con datos de música."""
    csv_path = tmp_path / "test_music_data.csv"
    sample_music_dataframe.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_csv_with_missing_values(tmp_path):
    """CSV con valores faltantes para probar preprocesamiento."""
    df = pd.DataFrame({
        "feat1": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0],
        "feat2": ["1", "2", "NULL", "4", "error", "6", "7", "8"],
        "feat3": [10, 20, 30, 40, 50, 60, 70, 80],
        "Class": ["happy", "happy", "sad", "sad", "energetic", "energetic", "calm", "calm"]
    })
    csv_path = tmp_path / "test_missing_values.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def multiclass_dataset(tmp_path):
    """Dataset con múltiples clases balanceadas."""
    n_per_class = 20
    classes = ["happy", "sad", "energetic", "calm"]
    
    data = []
    for cls in classes:
        for _ in range(n_per_class):
            data.append({
                "feat1": np.random.randn(),
                "feat2": np.random.randn(),
                "feat3": np.random.randn(),
                "Class": cls
            })
    
    df = pd.DataFrame(data)
    csv_path = tmp_path / "multiclass_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def small_training_dataset(tmp_path):
    """Dataset pequeño para pruebas rápidas de entrenamiento."""
    df = pd.DataFrame({
        "tempo": [120, 130, 140, 150, 80, 90, 100, 110],
        "energy": [0.8, 0.9, 0.85, 0.95, 0.3, 0.2, 0.25, 0.35],
        "Class": ["energetic", "energetic", "energetic", "energetic", 
                  "calm", "calm", "calm", "calm"]
    })
    csv_path = tmp_path / "small_training.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture(autouse=True)
def reset_mlflow():
    """Limpia el estado de MLflow antes de cada test."""
    import mlflow
    # Termina cualquier run activo
    if mlflow.active_run() is not None:
        mlflow.end_run()
    yield
    # Limpieza después del test
    if mlflow.active_run() is not None:
        mlflow.end_run()
