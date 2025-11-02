"""
Gestión de experimentos y modelos en MLflow.
Permite crear experimentos, registrar ejecuciones, comparar métricas y gestionar el Model Registry.
"""

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class MLflowExperimentManager:
    """Gestiona experimentos de MLflow y registro de modelos."""
    
    def __init__(self, tracking_uri: str = "./mlruns", verbose: bool = False):
        """Inicializa el gestor de experimentos con la URI de tracking especificada."""
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
        self.verbose = verbose
    
    def _get_experiment(self, exp_name: str):
        """Obtiene un experimento por nombre."""
        return self.client.get_experiment_by_name(exp_name)
    
    def create_versioned_experiment(self, base_name: str, version: str, 
                                   description: str = "") -> Tuple[str, bool]:
        """Crea un experimento con nombre versionado o recupera uno existente."""
        exp_name = f"{base_name}_v{version}"
        try:
            exp_id = self.client.create_experiment(exp_name, tags={
                "version": version, "base_name": base_name,
                "created_at": datetime.now().isoformat(), "description": description
            })
            return exp_id, True
        except:
            return self.client.get_experiment_by_name(exp_name).experiment_id, False
    
    def log_run_with_metadata(self, experiment_name: str, run_name: str, model,
                             metrics: Dict, params: Dict, tags: Dict = None, 
                             artifacts: Dict = None) -> str:
        """Registra una ejecución completa con métricas, parámetros, modelo y artefactos."""
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            if tags:
                mlflow.set_tags(tags)
            if model is not None:
                mlflow.sklearn.log_model(model, name="model")
            if artifacts:
                for name, path in artifacts.items():
                    mlflow.log_artifact(path, name)
            return run.info.run_id
    
    def compare_experiments(self, experiment_names: List[str], 
                           metric: str = "accuracy") -> pd.DataFrame:
        """Compara ejecuciones de múltiples experimentos basándose en una métrica."""
        results = [
            {
                'experiment': exp_name,
                'run_name': run.data.tags.get('mlflow.runName', 'N/A'),
                'run_id': run.info.run_id,
                metric: run.data.metrics.get(metric, 0),
                'status': run.info.status,
                'start_time': pd.to_datetime(run.info.start_time, unit='ms')
            }
            for exp_name in experiment_names
            if (experiment := self._get_experiment(exp_name))
            for run in self.client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=[f"metrics.{metric} DESC"]
            )
        ]
        
        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values(metric, ascending=False)
        return df
    
    def get_best_run(self, experiment_name: str, metric: str = "accuracy") -> Optional[Dict]:
        """Obtiene la mejor ejecución de un experimento según una métrica."""
        experiment = self._get_experiment(experiment_name)
        if not experiment:
            return None
        
        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} DESC"],
            max_results=1
        )
        
        return {
            'run_id': runs[0].info.run_id,
            'run_name': runs[0].data.tags.get('mlflow.runName', 'N/A'),
            'metrics': runs[0].data.metrics,
            'params': runs[0].data.params,
            'artifact_uri': runs[0].info.artifact_uri
        } if runs else None
    
    def register_model(self, run_id: str, model_name: str, 
                      description: str = "") -> Optional[str]:
        """Registra un modelo en el Model Registry desde una ejecución específica."""
        try:
            model_version = mlflow.register_model(f"runs:/{run_id}/model", model_name)
            self.client.update_model_version(
                name=model_name, version=model_version.version, description=description
            )
            return model_version.version
        except Exception:
            return None
    
    def list_registered_models(self) -> pd.DataFrame:
        """Lista todos los modelos registrados con sus versiones y estados."""
        models = [
            {
                'model_name': rm.name, 'version': mv.version, 'stage': mv.current_stage,
                'run_id': mv.run_id, 'status': mv.status,
                'created_at': pd.to_datetime(mv.creation_timestamp, unit='ms')
            }
            for rm in self.client.search_registered_models()
            for mv in self.client.search_model_versions(f"name='{rm.name}'")
        ]
        return pd.DataFrame(models)
    
    def delete_experiment_runs(self, experiment_name: str) -> int:
        """Elimina todas las ejecuciones de un experimento."""
        experiment = self._get_experiment(experiment_name)
        if not experiment:
            return 0
        runs = self.client.search_runs(experiment_ids=[experiment.experiment_id])
        for run in runs:
            self.client.delete_run(run.info.run_id)
        return len(runs)
    
    def print_metrics_table(self, experiment_name: str, save_csv: bool = True) -> None:
        """Genera tabla comparativa con métricas en consola y la guarda en CSV."""
        experiment = self._get_experiment(experiment_name)
        if not experiment:
            return
        
        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"]
        )
        
        if not runs:
            return
        
        seen_models = {}
        for run in runs:
            model_name = run.data.tags.get('mlflow.runName', 'Unknown')
            if model_name not in seen_models:
                seen_models[model_name] = run
        
        sorted_runs = sorted(seen_models.values(), 
                           key=lambda r: r.data.metrics.get('accuracy', 0.0), 
                           reverse=True)
        
        print(f"\nExperimento: {experiment_name}")
        print("=" * 100)
        print(f"{'Modelo':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
        print("-" * 100)
        
        data_rows = []
        
        for run in sorted_runs:
            model_name = run.data.tags.get('mlflow.runName', 'Unknown')
            accuracy = run.data.metrics.get('accuracy', 0.0)
            precision = run.data.metrics.get('precision', 0.0)
            recall = run.data.metrics.get('recall', 0.0)
            f1 = run.data.metrics.get('f1_score', 0.0)
            
            print(f"{model_name:<25} {accuracy:>10.4f} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f}")
            
            data_rows.append({
                'Modelo': model_name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1
            })
        
        print("=" * 100)
        
        if save_csv and data_rows:
            import os
            df = pd.DataFrame(data_rows)
            output_dir = "reports"
            os.makedirs(output_dir, exist_ok=True)
            csv_path = os.path.join(output_dir, f"{experiment_name}_metrics.csv")
            df.to_csv(csv_path, index=False)
            print(f"\nTabla guardada en: {csv_path}")
