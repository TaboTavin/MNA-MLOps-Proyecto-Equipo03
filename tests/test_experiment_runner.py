"""
Pruebas de  ModelExperimentRunner. 
"""

import pandas as pd
import os
import sys

# Añadir el directorio raíz del proyecto al PYTHONPATH
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from sklearn.linear_model import LogisticRegression

from src.experiment_runner import ModelExperimentRunner, ExperimentConfig


def _create_dummy_dataset(tmp_path):
    df = pd.DataFrame({
        "feat1": [0.1, 0.2, 0.3, 0.4, 5.0, 6.0, 6.5, 7.0],
        "feat2": [1, 2, 3, 4, 5, 6, 7, 8],
        "Class": ["happy", "happy", "happy", "happy", "sad", "sad", "sad", "sad"],
    })
    path = tmp_path / "dummy_music_runner.csv"
    df.to_csv(path, index=False)
    return path


def test_model_experiment_runner_single_config(tmp_path):
    data_path = _create_dummy_dataset(tmp_path)

    runner = ModelExperimentRunner(
        data_path=str(data_path),
        target_column="Class",
        test_size=0.25,
    )

    configs = [
        ExperimentConfig(
            name="LR_Test",
            model_class=LogisticRegression,
            params={"max_iter": 1000},
        )
    ]

    results = runner.run_experiment("test_experiment_runner", configs)

    assert "LR_Test" in results

    res = results["LR_Test"]
    assert "pipeline" in res
    assert "run_id" in res
    assert "accuracy" in res

    assert 0.0 <= res["accuracy"] <= 1.0