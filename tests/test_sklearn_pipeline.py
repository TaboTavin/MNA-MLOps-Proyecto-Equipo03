"""
Pruebas de sklearn_pipeline.
"""

import pandas as pd
import os
import sys

# Añadir el directorio raíz del proyecto al PYTHONPATH
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)


from sklearn.linear_model import LogisticRegression

from src.sklearn_pipeline import SklearnMLPipeline


def _create_dummy_dataset(tmp_path):
    """Crea un CSV pequeño para pruebas con una columna 'Class'."""
    df = pd.DataFrame({
        "feat1": [0.1, 0.2, 0.3, 0.4, 5.0, 6.0, 6.5, 7.0],
        "feat2": ["1", "2", "3", "4", "5", "6", "7", "8"],   # como string para probar conversión
        "Class": ["happy", "happy", "happy", "happy", "sad", "sad", "sad", "sad"],
    })
    path = tmp_path / "dummy_music.csv"
    df.to_csv(path, index=False)
    return path


def test_build_pipeline_contains_all_expected_steps(tmp_path):
    path = _create_dummy_dataset(tmp_path)
    pipeline = SklearnMLPipeline(
        data_path=str(path),
        target_column="Class",
        experiment_name="test_build_pipeline",
    )

    model = LogisticRegression(max_iter=1000)
    sklearn_pipe = pipeline._build_pipeline(model)

    step_names = [name for name, _ in sklearn_pipe.steps]

    assert step_names == [
        "cleaner",
        "numeric_converter",
        "imputer",
        "scaler",
        "classifier",
    ]


def test_train_model_logs_all_metrics(monkeypatch, tmp_path):
    """Verifica que _train_model calcule y envíe accuracy, precision, recall, f1 a mlflow."""
    path = _create_dummy_dataset(tmp_path)
    pipeline = SklearnMLPipeline(
        data_path=str(path),
        target_column="Class",
        experiment_name="test_metrics",
    )
    # split de datos
    pipeline._load_and_split(test_size=0.25)

    logged_metrics = {}

    def fake_log_metrics(metrics_dict):
        logged_metrics.update(metrics_dict)

    # Solo parcheamos log_metrics para inspeccionar el dict
    monkeypatch.setattr(
        "src.sklearn_pipeline.mlflow.log_metrics",
        fake_log_metrics,
    )

    model = LogisticRegression(max_iter=1000)

    trained_pipeline, run_id, accuracy = pipeline._train_model(model, "LR_test")

    # run_id debe existir
    assert run_id is not None
    # accuracy debe estar entre 0 y 1
    assert 0.0 <= accuracy <= 1.0

    # Se deben haber logueado las 4 métricas
    for key in ["accuracy", "precision", "recall", "f1_score"]:
        assert key in logged_metrics
        assert 0.0 <= logged_metrics[key] <= 1.0

    # El pipeline devuelto debe poder predecir
    preds = trained_pipeline.predict(pipeline.X_test)
    assert len(preds) == len(pipeline.X_test)


def test_run_end_to_end_pipeline(tmp_path):
    """
    Prueba de integración: carga CSV -> preprocesa -> entrena -> predice -> devuelve métricas.
    """
    path = _create_dummy_dataset(tmp_path)

    pipeline = SklearnMLPipeline(
        data_path=str(path),
        target_column="Class",
        experiment_name="test_end_to_end",
    )

    models = {
        "log_reg": LogisticRegression(max_iter=1000),
    }

    results = pipeline.run(models=models, test_size=0.25)

    assert "log_reg" in results

    res = results["log_reg"]
    assert "pipeline" in res
    assert "run_id" in res
    assert "accuracy" in res

    assert 0.0 <= res["accuracy"] <= 1.0

    # Verificar que el pipeline entrenado realmente predice
    trained_pipeline = res["pipeline"]
    preds = trained_pipeline.predict(pipeline.X_test)
    assert len(preds) == len(pipeline.X_test)