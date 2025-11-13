"""
Pruebas de integración end-to-end del pipeline completo.
Validan el flujo desde carga de datos hasta predicción y métricas.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.sklearn_pipeline import SklearnMLPipeline
from src.experiment_runner import ModelExperimentRunner, ExperimentConfig, HyperparameterExperiments
from src.pipeline import MLPipeline


class TestEndToEndPipeline:
    """Pruebas de integración del flujo completo del pipeline."""
    
    def test_complete_pipeline_flow(self, sample_csv_file):
        """
        Prueba E2E: Carga → Preprocesamiento → Entrenamiento → Predicción → Métricas.
        """
        # Configuración
        pipeline = SklearnMLPipeline(
            data_path=str(sample_csv_file),
            target_column="Class",
            experiment_name="test_e2e_complete"
        )
        
        # Preparación de datos
        pipeline._load_and_split(test_size=0.3)
        
        # Validaciones de carga
        assert pipeline.X_train is not None
        assert pipeline.X_test is not None
        assert pipeline.y_train is not None
        assert pipeline.y_test is not None
        assert len(pipeline.X_train) + len(pipeline.X_test) <= 100  # Tamaño original
        
        # Validación de clases en entrenamiento y prueba
        assert len(pipeline.y_train.unique()) > 1
        assert len(pipeline.y_test.unique()) > 1
        
        # Entrenamiento de modelo
        model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        trained_pipeline, run_id, accuracy = pipeline._train_model(model, "RF_E2E")
        
        # Validaciones de entrenamiento
        assert trained_pipeline is not None
        assert run_id is not None
        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0
        
        # Predicción
        predictions = trained_pipeline.predict(pipeline.X_test)
        
        # Validaciones de predicción
        assert len(predictions) == len(pipeline.X_test)
        assert all(pred in pipeline.y_train.unique() for pred in predictions)
        
        # Validación de que el pipeline transformó correctamente los datos
        X_test_transformed = trained_pipeline.named_steps['scaler'].transform(
            trained_pipeline.named_steps['imputer'].transform(
                trained_pipeline.named_steps['numeric_converter'].transform(
                    trained_pipeline.named_steps['cleaner'].transform(pipeline.X_test)
                )
            )
        )
        assert not np.isnan(X_test_transformed).any(), "Pipeline debe eliminar todos los NaN"
    
    def test_multiple_models_training_flow(self, sample_csv_file):
        """
        Prueba E2E: Entrena múltiples modelos y compara métricas.
        """
        pipeline = SklearnMLPipeline(
            data_path=str(sample_csv_file),
            target_column="Class",
            experiment_name="test_e2e_multiple_models"
        )
        
        models = {
            "RandomForest": RandomForestClassifier(n_estimators=50, random_state=42),
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
            "GradientBoosting": GradientBoostingClassifier(n_estimators=50, random_state=42)
        }
        
        results = pipeline.run(models=models, test_size=0.25)
        
        # Validar que todos los modelos se entrenaron
        assert len(results) == len(models)
        
        for model_name, result in results.items():
            assert model_name in models
            assert "pipeline" in result
            assert "run_id" in result
            assert "accuracy" in result
            
            # Validar métricas
            assert 0.0 <= result["accuracy"] <= 1.0
            
            # Validar que el pipeline puede predecir
            predictions = result["pipeline"].predict(pipeline.X_test)
            assert len(predictions) == len(pipeline.X_test)
    
    def test_preprocessing_pipeline_handles_dirty_data(self, sample_csv_with_missing_values):
        """
        Prueba E2E: Pipeline maneja correctamente datos sucios con valores faltantes.
        """
        pipeline = SklearnMLPipeline(
            data_path=str(sample_csv_with_missing_values),
            target_column="Class",
            experiment_name="test_e2e_dirty_data"
        )
        
        pipeline._load_and_split(test_size=0.25)
        
        # Construir y aplicar pipeline de preprocesamiento
        model = LogisticRegression(max_iter=1000, random_state=42)
        full_pipeline = pipeline._build_pipeline(model)
        
        # Entrenar con datos que contienen problemas
        full_pipeline.fit(pipeline.X_train, pipeline.y_train)
        
        # Predecir
        predictions = full_pipeline.predict(pipeline.X_test)
        
        # Validaciones
        assert len(predictions) == len(pipeline.X_test)
        assert not any(pd.isna(predictions)), "No debe haber predicciones NaN"
        
        # Verificar que el preprocesamiento funcionó
        X_transformed = full_pipeline.named_steps['cleaner'].transform(pipeline.X_test)
        assert isinstance(X_transformed, pd.DataFrame)


class TestExperimentRunnerIntegration:
    """Pruebas de integración del ExperimentRunner."""
    
    def test_experiment_runner_executes_multiple_configs(self, sample_csv_file):
        """
        Prueba E2E: ExperimentRunner ejecuta múltiples configuraciones correctamente.
        """
        runner = ModelExperimentRunner(
            data_path=str(sample_csv_file),
            target_column="Class",
            test_size=0.3
        )
        
        configs = [
            ExperimentConfig("RF_Test1", RandomForestClassifier, {"n_estimators": 50, "random_state": 42}),
            ExperimentConfig("RF_Test2", RandomForestClassifier, {"n_estimators": 100, "max_depth": 5, "random_state": 42}),
            ExperimentConfig("LR_Test", LogisticRegression, {"max_iter": 1000, "random_state": 42})
        ]
        
        results = runner.run_experiment("test_multi_config_integration", configs)
        
        # Validaciones
        assert len(results) == len(configs)
        
        for config in configs:
            assert config.name in results
            result = results[config.name]
            assert "accuracy" in result
            assert 0.0 <= result["accuracy"] <= 1.0
    
    def test_hyperparameter_experiments_factory(self, small_training_dataset):
        """
        Prueba E2E: Factory de experimentos genera configuraciones válidas.
        """
        runner = ModelExperimentRunner(
            data_path=str(small_training_dataset),
            target_column="Class",
            test_size=0.25
        )
        
        # Probar Random Forest configs
        rf_configs = HyperparameterExperiments.random_forest_configs()
        assert len(rf_configs) == 5
        
        results = runner.run_experiment("test_rf_factory", rf_configs[:2])  # Solo 2 para velocidad
        assert len(results) == 2
        
        for result in results.values():
            assert "accuracy" in result
            assert result["accuracy"] >= 0.0


class TestMLPipelineWrapper:
    """Pruebas de integración de la interfaz MLPipeline."""
    
    def test_mlpipeline_wrapper_runs_successfully(self, sample_csv_file):
        """
        Prueba E2E: Wrapper MLPipeline ejecuta el flujo completo.
        """
        pipeline = MLPipeline(
            data_path=str(sample_csv_file),
            target_column="Class",
            experiment_name="test_wrapper_integration"
        )
        
        models = {
            "TestRF": RandomForestClassifier(n_estimators=50, random_state=42)
        }
        
        results = pipeline.run(models=models, test_size=0.3)
        
        assert "TestRF" in results
        assert "accuracy" in results["TestRF"]
        assert 0.0 <= results["TestRF"]["accuracy"] <= 1.0


class TestDataQualityValidation:
    """Pruebas de validación de calidad de datos en el pipeline."""
    
    def test_pipeline_rejects_invalid_target_column(self, sample_csv_file):
        """
        Valida que el pipeline falla apropiadamente con columna objetivo inválida.
        """
        pipeline = SklearnMLPipeline(
            data_path=str(sample_csv_file),
            target_column="NonExistentColumn",
            experiment_name="test_invalid_target"
        )
        
        with pytest.raises(KeyError):
            pipeline._load_and_split()
    
    def test_pipeline_handles_single_class_data(self, tmp_path):
        """
        Valida comportamiento cuando hay una sola clase (caso edge).
        """
        # Crear dataset con una sola clase
        df = pd.DataFrame({
            "feat1": [1, 2, 3, 4, 5],
            "feat2": [10, 20, 30, 40, 50],
            "Class": ["happy", "happy", "happy", "happy", "happy"]
        })
        csv_path = tmp_path / "single_class.csv"
        df.to_csv(csv_path, index=False)
        
        pipeline = SklearnMLPipeline(
            data_path=str(csv_path),
            target_column="Class",
            experiment_name="test_single_class"
        )
        
        # Debe fallar en train_test_split por stratify
        with pytest.raises(ValueError):
            pipeline._load_and_split()
    
    def test_pipeline_preserves_class_distribution(self, multiclass_dataset):
        """
        Valida que el split estratificado preserva la distribución de clases.
        """
        pipeline = SklearnMLPipeline(
            data_path=str(multiclass_dataset),
            target_column="Class",
            experiment_name="test_stratification"
        )
        
        pipeline._load_and_split(test_size=0.25)
        
        # Calcular distribuciones
        train_dist = pipeline.y_train.value_counts(normalize=True).sort_index()
        test_dist = pipeline.y_test.value_counts(normalize=True).sort_index()
        
        # Las distribuciones deben ser similares (tolerancia 0.1)
        for cls in train_dist.index:
            assert abs(train_dist[cls] - test_dist[cls]) < 0.15, \
                f"Distribución desbalanceada para clase {cls}"
