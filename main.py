"""
Orquestador principal de experimentos de machine learning.
Ejecuta entrenamientos, registra modelos y genera visualizaciones usando MLflow.
"""

from typing import Dict, Tuple, Optional
from mlflow_manager import MLflowExperimentManager
from src.visual import ModelVisualizer
from src.experiment_runner import ModelExperimentRunner, HyperparameterExperiments


class MLExperimentOrchestrator:
    """Orquesta experimentos de machine learning con MLflow y visualización."""
    
    def __init__(self, data_path: str = "data/turkish_music_emotion_cleaned.csv",
                 target_column: str = "Class", test_size: float = 0.2):
        """Inicializa el orquestador con configuración de datos y componentes."""
        self.runner = ModelExperimentRunner(data_path, target_column, test_size)
        self.manager = MLflowExperimentManager()
        self.visualizer = ModelVisualizer()
        self.experiments = HyperparameterExperiments()
    
    def run_random_forest_tuning(self) -> Dict:
        """Ejecuta tuning de Random Forest con 5 configuraciones."""
        configs = self.experiments.random_forest_configs()
        return self.runner.run_experiment("RF_Hyperparameter_Tuning", configs)
    
    def run_gradient_boosting_tuning(self) -> Dict:
        """Ejecuta tuning de Gradient Boosting con 5 configuraciones."""
        configs = self.experiments.gradient_boosting_configs()
        return self.runner.run_experiment("GB_Hyperparameter_Tuning", configs)
    
    def run_logistic_regression_tuning(self) -> Dict:
        """Ejecuta tuning de Logistic Regression con 5 configuraciones."""
        configs = self.experiments.logistic_regression_configs()
        return self.runner.run_experiment("LR_Hyperparameter_Tuning", configs)
    
    def run_svm_tuning(self) -> Dict:
        """Ejecuta tuning de SVM con 5 configuraciones."""
        configs = self.experiments.svm_configs()
        return self.runner.run_experiment("SVM_Hyperparameter_Tuning", configs)
    
    def run_decision_tree_tuning(self) -> Dict:
        """Ejecuta tuning de Decision Tree con 5 configuraciones."""
        configs = self.experiments.decision_tree_configs()
        return self.runner.run_experiment("DT_Hyperparameter_Tuning", configs)
    
    def run_baseline_experiment(self) -> Dict:
        """Ejecuta experimento baseline con 5 modelos estándar."""
        configs = self.experiments.baseline_configs()
        return self.runner.run_experiment("music_emotion_classification", configs)
    
    def run_all_tuning_experiments(self) -> Dict[str, Dict]:
        """Ejecuta todos los experimentos de tuning para los 5 modelos."""
        return {
            'Random Forest': self.run_random_forest_tuning(),
            'Gradient Boosting': self.run_gradient_boosting_tuning(),
            'Logistic Regression': self.run_logistic_regression_tuning(),
            'SVM': self.run_svm_tuning(),
            'Decision Tree': self.run_decision_tree_tuning()
        }
    
    def run_single_tuning(self, model_type: str) -> Dict:
        """Ejecuta tuning de un modelo específico usando su código."""
        experiments_map = {
            "rf": self.run_random_forest_tuning,
            "gb": self.run_gradient_boosting_tuning,
            "lr": self.run_logistic_regression_tuning,
            "svm": self.run_svm_tuning,
            "dt": self.run_decision_tree_tuning
        }
        
        if model_type.lower() in experiments_map:
            return experiments_map[model_type.lower()]()
        else:
            raise ValueError(f"Tipo inválido: {model_type}. Use: {list(experiments_map.keys())}")
    
    def print_experiment_summary(self, experiment_name: str) -> None:
        """Imprime tabla resumen de métricas del experimento."""
        self.manager.print_metrics_table(experiment_name)
    
    def analyze_and_register_best_model(self, experiment_name: str, 
                                       model_registry_name: str = "music_emotion_classifier") -> Optional[Dict]:
        """Analiza resultados y registra el mejor modelo en MLflow Model Registry."""
        best_run = self.manager.get_best_run(experiment_name, metric='accuracy')
        
        if not best_run:
            print(f"No se encontraron runs en el experimento: {experiment_name}")
            return None
        
        best_accuracy = best_run['metrics']['accuracy']
        best_run_id = best_run['run_id']
        best_model_name = best_run['run_name']
        
        version = self.manager.register_model(
            run_id=best_run_id,
            model_name=model_registry_name,
            description=f"{best_model_name} | Accuracy: {best_accuracy:.4f}"
        )
        
        print(f"\nMejor: {best_model_name} ({best_accuracy:.4f}) - Registrado")
        
        return {
            'model_name': best_model_name,
            'accuracy': best_accuracy,
            'registered': True,
            'version': version
        }
    
    def generate_visualizations(self, experiment_name: str, output_dir: str = "reports") -> None:
        """Genera visualizaciones comparativas de los modelos del experimento."""
        try:
            experiment = self.manager._get_experiment(experiment_name)
            if not experiment:
                print(f"Experimento '{experiment_name}' no encontrado")
                return
            
            runs = self.manager.client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["metrics.accuracy DESC"]
            )
            
            if not runs:
                print(f"No hay runs en el experimento '{experiment_name}'")
                return
            
            comparison_data = [
                {
                    'metrics': run.data.metrics,
                    'tags': run.data.tags,
                    'start_time': run.info.start_time
                }
                for run in runs
            ]
            
            print("\nGenerando visualizaciones...")
            
            path1 = self.visualizer.plot_model_comparison(
                comparison_data, metric_name='accuracy', top_n=10
            )
            if path1:
                print(f"  - Comparacion accuracy: {path1}")
            
            path2 = self.visualizer.plot_experiment_evolution(
                comparison_data, metric_name='accuracy'
            )
            if path2:
                print(f"  - Evolucion temporal: {path2}")
            
            metrics_data = [
                {
                    'model': run.data.tags.get('mlflow.runName', 'Unknown'),
                    'accuracy': run.data.metrics.get('accuracy', 0),
                    'precision': run.data.metrics.get('precision', 0),
                    'recall': run.data.metrics.get('recall', 0),
                    'f1_score': run.data.metrics.get('f1_score', 0)
                }
                for run in runs[:10]
            ]
            
            path3 = self.visualizer.plot_metrics_comparison_table(metrics_data)
            if path3:
                print(f"  - Comparacion metricas: {path3}")
                
        except Exception as e:
            print(f"Error generando visualizaciones: {e}")
    
    def run_experiment_with_analysis(self, experiment_func, experiment_name: str,
                                    model_registry_name: str = "music_emotion_classifier",
                                    generate_plots: bool = True) -> Tuple[Dict, Optional[Dict]]:
        """Ejecuta un experimento completo con análisis y visualización."""
        results = experiment_func()
        self.print_experiment_summary(experiment_name)
        best_model = self.analyze_and_register_best_model(experiment_name, model_registry_name)
        
        if generate_plots:
            self.generate_visualizations(experiment_name)
        
        return results, best_model


def main():
    """Punto de entrada principal para ejecutar experimentos de clasificación de emociones."""
    orchestrator = MLExperimentOrchestrator()
    
    # Ejecuta experimento baseline con 5 modelos estándar
    results, best_model = orchestrator.run_experiment_with_analysis(
        experiment_func=orchestrator.run_baseline_experiment,
        experiment_name="music_emotion_classification",
        generate_plots=True
    )
    
    return results, best_model


if __name__ == "__main__":
    results, best_model_info = main()
