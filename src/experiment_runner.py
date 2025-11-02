"""
Ejecuta experimentos de aprendizaje automático con diferentes configuraciones de modelos.
Define hiperparámetros predefinidos para comparar variantes de cada algoritmo.
"""

from typing import Dict, List, Any
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.sklearn_pipeline import SklearnMLPipeline


class ExperimentConfig:
    """Encapsula nombre, clase y parámetros de un modelo."""
    
    def __init__(self, name: str, model_class: type, params: Dict[str, Any]):
        self.name = name
        self.model_class = model_class
        self.params = params
    
    def create_model(self) -> BaseEstimator:
        """Crea instancia del modelo con sus parámetros."""
        return self.model_class(**self.params)


class ModelExperimentRunner:
    """Ejecuta experimentos usando SklearnMLPipeline con múltiples configuraciones."""
    
    def __init__(self, data_path: str, target_column: str = "Class", test_size: float = 0.2):
        self.data_path = data_path
        self.target_column = target_column
        self.test_size = test_size
    
    def run_experiment(self, experiment_name: str, configs: List[ExperimentConfig]) -> Dict:
        """Entrena múltiples modelos y retorna resultados."""
        pipeline = SklearnMLPipeline(self.data_path, self.target_column, experiment_name)
        
        models = {
            config.name: config.create_model()
            for config in configs
        }
        
        return pipeline.run(models=models, test_size=self.test_size)


class HyperparameterExperiments:
    """Factory de configuraciones de hiperparámetros para 5 tipos de modelos."""
    
    @staticmethod
    def random_forest_configs() -> List[ExperimentConfig]:
        """Cinco configuraciones de Random Forest con diferentes profundidades y número de árboles."""
        return [
            ExperimentConfig("RF_Baseline", RandomForestClassifier, {
                "n_estimators": 100, "max_depth": None, "min_samples_split": 2,
                "min_samples_leaf": 1, "random_state": 42, "n_jobs": -1
            }),
            ExperimentConfig("RF_Shallow", RandomForestClassifier, {
                "n_estimators": 100, "max_depth": 5, "min_samples_split": 5,
                "min_samples_leaf": 2, "random_state": 42, "n_jobs": -1
            }),
            ExperimentConfig("RF_Deep", RandomForestClassifier, {
                "n_estimators": 200, "max_depth": 20, "min_samples_split": 2,
                "min_samples_leaf": 1, "random_state": 42, "n_jobs": -1
            }),
            ExperimentConfig("RF_Conservative", RandomForestClassifier, {
                "n_estimators": 150, "max_depth": 10, "min_samples_split": 10,
                "min_samples_leaf": 4, "random_state": 42, "n_jobs": -1
            }),
            ExperimentConfig("RF_MoreTrees", RandomForestClassifier, {
                "n_estimators": 300, "max_depth": 15, "min_samples_split": 5,
                "min_samples_leaf": 2, "random_state": 42, "n_jobs": -1
            })
        ]
    
    @staticmethod
    def gradient_boosting_configs() -> List[ExperimentConfig]:
        """Cinco configuraciones de Gradient Boosting variando tasa de aprendizaje y profundidad."""
        return [
            ExperimentConfig("GB_FastLearning", GradientBoostingClassifier, {
                "n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "random_state": 42
            }),
            ExperimentConfig("GB_SlowLearning", GradientBoostingClassifier, {
                "n_estimators": 200, "learning_rate": 0.05, "max_depth": 3, "random_state": 42
            }),
            ExperimentConfig("GB_DeepTrees", GradientBoostingClassifier, {
                "n_estimators": 100, "learning_rate": 0.1, "max_depth": 7, "random_state": 42
            }),
            ExperimentConfig("GB_Conservative", GradientBoostingClassifier, {
                "n_estimators": 150, "learning_rate": 0.01, "max_depth": 5, "random_state": 42
            }),
            ExperimentConfig("GB_Aggressive", GradientBoostingClassifier, {
                "n_estimators": 300, "learning_rate": 0.15, "max_depth": 5, "random_state": 42
            })
        ]
    
    @staticmethod
    def logistic_regression_configs() -> List[ExperimentConfig]:
        """Cinco configuraciones de Regresión Logística con diferentes penalizaciones y regularización."""
        return [
            ExperimentConfig("LR_L2_Weak", LogisticRegression, {
                "max_iter": 1000, "C": 0.1, "solver": 'lbfgs', "penalty": 'l2',
                "random_state": 42, "n_jobs": -1
            }),
            ExperimentConfig("LR_L2_Strong", LogisticRegression, {
                "max_iter": 1000, "C": 10.0, "solver": 'lbfgs', "penalty": 'l2',
                "random_state": 42, "n_jobs": -1
            }),
            ExperimentConfig("LR_L2_Balanced", LogisticRegression, {
                "max_iter": 1000, "C": 1.0, "solver": 'lbfgs', "penalty": 'l2',
                "random_state": 42, "n_jobs": -1
            }),
            ExperimentConfig("LR_Saga_L1", LogisticRegression, {
                "max_iter": 1000, "C": 1.0, "solver": 'saga', "penalty": 'l1',
                "random_state": 42, "n_jobs": -1
            }),
            ExperimentConfig("LR_ElasticNet", LogisticRegression, {
                "max_iter": 1000, "C": 1.0, "solver": 'saga', "penalty": 'elasticnet',
                "l1_ratio": 0.5, "random_state": 42, "n_jobs": -1
            })
        ]
    
    @staticmethod
    def svm_configs() -> List[ExperimentConfig]:
        """Cinco configuraciones de Support Vector Machine con diferentes kernels y parámetros."""
        return [
            ExperimentConfig("SVM_RBF_Soft", SVC, {
                "C": 0.1, "kernel": 'rbf', "gamma": 'scale', "random_state": 42
            }),
            ExperimentConfig("SVM_RBF_Hard", SVC, {
                "C": 10.0, "kernel": 'rbf', "gamma": 'scale', "random_state": 42
            }),
            ExperimentConfig("SVM_RBF_Balanced", SVC, {
                "C": 1.0, "kernel": 'rbf', "gamma": 'scale', "random_state": 42
            }),
            ExperimentConfig("SVM_Linear", SVC, {
                "C": 1.0, "kernel": 'linear', "random_state": 42
            }),
            ExperimentConfig("SVM_Poly", SVC, {
                "C": 1.0, "kernel": 'poly', "degree": 3, "gamma": 'scale', "random_state": 42
            })
        ]
    
    @staticmethod
    def decision_tree_configs() -> List[ExperimentConfig]:
        """Cinco configuraciones de Árbol de Decisión variando profundidad y muestras mínimas."""
        return [
            ExperimentConfig("DT_Unrestricted", DecisionTreeClassifier, {
                "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1, "random_state": 42
            }),
            ExperimentConfig("DT_Shallow", DecisionTreeClassifier, {
                "max_depth": 5, "min_samples_split": 5, "min_samples_leaf": 2, "random_state": 42
            }),
            ExperimentConfig("DT_Medium", DecisionTreeClassifier, {
                "max_depth": 10, "min_samples_split": 10, "min_samples_leaf": 4, "random_state": 42
            }),
            ExperimentConfig("DT_Deep", DecisionTreeClassifier, {
                "max_depth": 20, "min_samples_split": 2, "min_samples_leaf": 1, "random_state": 42
            }),
            ExperimentConfig("DT_Pruned", DecisionTreeClassifier, {
                "max_depth": 15, "min_samples_split": 20, "min_samples_leaf": 10, "random_state": 42
            })
        ]
    
    @staticmethod
    def baseline_configs() -> List[ExperimentConfig]:
        """Cinco modelos con configuración estándar para comparación base."""
        return [
            ExperimentConfig("RandomForest", RandomForestClassifier, {
                "n_estimators": 100, "max_depth": 10, "min_samples_split": 5,
                "min_samples_leaf": 2, "random_state": 42, "n_jobs": -1
            }),
            ExperimentConfig("LogisticRegression", LogisticRegression, {
                "max_iter": 1000, "C": 1.0, "solver": 'lbfgs', "random_state": 42, "n_jobs": -1
            }),
            ExperimentConfig("GradientBoosting", GradientBoostingClassifier, {
                "n_estimators": 100, "learning_rate": 0.1, "max_depth": 5, "random_state": 42
            }),
            ExperimentConfig("DecisionTree", DecisionTreeClassifier, {
                "max_depth": 10, "min_samples_split": 5, "min_samples_leaf": 2, "random_state": 42
            }),
            ExperimentConfig("SVM", SVC, {
                "C": 1.0, "kernel": 'rbf', "gamma": 'scale', "random_state": 42
            })
        ]
