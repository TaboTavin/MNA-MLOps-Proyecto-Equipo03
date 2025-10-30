"""
===============================================================================
Archivo: train_model.py
Propósito:
    - Entrenar y evaluar modelos de Machine Learning sobre los datasets procesados.
    - Comparar varios algoritmos y registrar métricas clave.
    - Guardar el mejor modelo entrenado (pipeline completo).

Entradas esperadas:
    - data/processed/X_train.csv
    - data/processed/y_train.csv
    - data/processed/X_test.csv
    - data/processed/y_test.csv
    (estos archivos se generan en build_features.py)

Salidas:
    - models/model.joblib   (mejor modelo entrenado y serializado con joblib)
    - reports/metrics.csv   (tabla con métricas por modelo probado)

Métricas calculadas:
    - Accuracy
    - F1-score macro

Cómo usar desde TERMINAL:
    # Entrenar todos los modelos predefinidos y guardar el mejor
    python -m src.models.train_model

    # Especificar modelos (ejemplo: solo logistic regression y random forest)
    python -m src.models.train_model --models logistic random_forest

Cómo usar desde NOTEBOOK:
    1. Añadir la raíz al sys.path:
        import sys, os
        repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
        if repo_root not in sys.path:
            sys.path.append(repo_root)
        %load_ext autoreload
        %autoreload 2

    2. Importar y ejecutar:
        from src.models.train_model import ProjectPaths, TrainConfig, ModelTrainer

        paths = ProjectPaths()
        cfg   = TrainConfig(models=["logistic", "random_forest"])
        trainer = ModelTrainer(paths, cfg)

        best_model, metrics_df = trainer.run()

    3. Revisar métricas:
        print(metrics_df)
        # cargar el modelo guardado
        import joblib
        pipe = joblib.load(paths.models_dir / "model.joblib")

Notas:
    - La selección de variables supervisada (ANOVA/PCA) se puede integrar aquí
      dentro de los pipelines de cada modelo (después del preprocesador).
    - Este módulo está listo para conectarse con MLflow/DVC en el Punto 4.
===============================================================================
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# -----------------------------------------------------------------------------
# Rutas y configuración
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ProjectPaths:
    root: Path = Path(__file__).resolve().parents[2]
    data_processed: Path = root / "data" / "processed"
    models_dir: Path = root / "models"
    reports_dir: Path = root / "reports"

    def ensure(self) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class TrainConfig:
    models: List[str] = None  # ["logistic", "random_forest", "svm", "mlp"]
    target_col: str = "Class"

    def __post_init__(self):
        if self.models is None:
            object.__setattr__(self, "models", ["logistic", "random_forest", "svm", "mlp"])


# -----------------------------------------------------------------------------
# Clase principal: ModelTrainer
# -----------------------------------------------------------------------------
class ModelTrainer:
    def __init__(self, paths: ProjectPaths, cfg: TrainConfig) -> None:
        self.paths = paths
        self.cfg = cfg

    def run(self) -> Tuple[object, pd.DataFrame]:
        """Ejecuta el flujo completo: carga, entrena, evalúa, guarda mejor modelo."""
        self.paths.ensure()
        X_train, y_train, X_test, y_test = self._load_data()

        # Definir modelos disponibles
        model_map = {
            "logistic": LogisticRegression(max_iter=500),
            "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
            "svm": SVC(kernel="rbf", probability=True),
            "mlp": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42),
        }

        results = []
        best_model, best_score, best_name = None, -1, None

        for name in self.cfg.models:
            if name not in model_map:
                logger.warning(f"Modelo '{name}' no está implementado, se omite.")
                continue

            logger.info(f"Entrenando modelo: {name}")
            model = model_map[name]
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")

            results.append({"model": name, "accuracy": acc, "f1_macro": f1})

            if f1 > best_score:
                best_model, best_score, best_name = model, f1, name

        # Guardar métricas
        metrics_df = pd.DataFrame(results)
        metrics_path = self.paths.reports_dir / "metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"Métricas guardadas en: {metrics_path}")
        logger.info(f"\n{metrics_df}")

        # Guardar mejor modelo
        if best_model is not None:
            model_path = self.paths.models_dir / "model.joblib"
            joblib.dump(best_model, model_path)
            logger.info(f"Mejor modelo '{best_name}' guardado en {model_path}")

        return best_model, metrics_df

    # ---------------- Métodos internos ----------------
    def _load_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Carga X_train, y_train, X_test, y_test desde data/processed."""
        X_train = pd.read_csv(self.paths.data_processed / "X_train.csv")
        y_train = pd.read_csv(self.paths.data_processed / "y_train.csv")[self.cfg.target_col]
        X_test = pd.read_csv(self.paths.data_processed / "X_test.csv")
        y_test = pd.read_csv(self.paths.data_processed / "y_test.csv")[self.cfg.target_col]
        logger.info(f"Data shapes -> X_train: {X_train.shape}, X_test: {X_test.shape}")
        return X_train, y_train, X_test, y_test


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenar y evaluar modelos ML.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logistic", "random_forest", "svm", "mlp"],
        help="Modelos a entrenar (default: todos).",
    )
    parser.add_argument("--target", type=str, default="Class", help="Columna objetivo.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths()
    cfg = TrainConfig(models=args.models, target_col=args.target)
    trainer = ModelTrainer(paths, cfg)
    trainer.run()


if __name__ == "__main__":
    main()