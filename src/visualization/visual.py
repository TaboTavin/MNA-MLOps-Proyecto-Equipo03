"""
===============================================================================
Archivo: visual.py
Propósito:
    - Proveer funciones de visualización para el proyecto de ML.
    - Incluir gráficas estándar: distribución de clases, matriz de confusión,
      comparación de métricas entre modelos.
    - Guardar automáticamente los gráficos en /reports/figures.

Entradas esperadas:
    - Archivos procesados (train/test con columna objetivo).
    - Reportes de métricas (ej. reports/metrics.csv).
    - Modelo entrenado + predicciones.

Salidas:
    - Archivos .png dentro de reports/figures/

Cómo usar desde TERMINAL:
    # Graficar distribución de clases en train
    python -m src.visualization.visual --plot class_dist --data data/processed/train.csv --target Class

    # Graficar matriz de confusión de un modelo entrenado
    python -m src.visualization.visual --plot confusion --data data/processed/X_test.csv --target Class

Cómo usar desde NOTEBOOK:
    1. Añadir la raíz al sys.path:
        import sys, os
        repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
        if repo_root not in sys.path:
            sys.path.append(repo_root)
        %load_ext autoreload
        %autoreload 2

    2. Importar y usar:
        from src.visualization.visual import ProjectPaths, Visualizer

        paths = ProjectPaths()
        viz = Visualizer(paths)

        # Distribución de clases
        viz.plot_class_distribution("data/processed/train.csv", target_col="Class")

        # Matriz de confusión (requiere modelo entrenado)
        viz.plot_confusion_matrix("data/processed/X_test.csv", "data/processed/y_test.csv")

Notas:
    - Las figuras se guardan en /reports/figures, con nombres descriptivos.
    - Este módulo NO entrena modelos, solo consume resultados.
===============================================================================
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# -----------------------------------------------------------------------------
# Rutas del proyecto
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ProjectPaths:
    root: Path = Path(__file__).resolve().parents[2]
    data_processed: Path = root / "data" / "processed"
    reports_dir: Path = root / "reports"
    figures_dir: Path = reports_dir / "figures"
    models_dir: Path = root / "models"

    def ensure(self) -> None:
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Clase principal: Visualizer
# -----------------------------------------------------------------------------
class Visualizer:
    def __init__(self, paths: ProjectPaths) -> None:
        self.paths = paths
        self.paths.ensure()

    # ------------------ Distribución de clases ------------------
    def plot_class_distribution(self, csv_path: str | Path, target_col: str = "Class", title: str = None) -> Path:
        """Genera un gráfico de barras con la distribución de clases."""
        df = pd.read_csv(csv_path)
        if target_col not in df.columns:
            raise ValueError(f"La columna '{target_col}' no está en {csv_path}")

        counts = df[target_col].value_counts().sort_index()
        plt.figure(figsize=(6, 4))
        sns.barplot(x=counts.index, y=counts.values, palette="Set2")
        plt.title(title or f"Distribución de clases ({csv_path})")
        plt.xlabel("Clase")
        plt.ylabel("Frecuencia")
        plt.xticks(rotation=30)

        out_path = self.paths.figures_dir / f"class_dist_{Path(csv_path).stem}.png"
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        logger.info(f"Gráfico de distribución guardado en {out_path}")
        return out_path

    # ------------------ Matriz de confusión ------------------
    def plot_confusion_matrix(
        self,
        X_csv: str | Path,
        y_csv: str | Path,
        model_path: Optional[str | Path] = None,
        target_col: str = "Class",
        normalize: str = "true",
    ) -> Path:
        """
        Genera matriz de confusión con un modelo entrenado.

        Parameters
        ----------
        X_csv : CSV con features de test
        y_csv : CSV con etiquetas reales
        model_path : ruta al modelo entrenado (default: models/model.joblib)
        target_col : columna con la etiqueta en y_csv
        normalize : None | 'true' | 'all' (normalización en matriz)
        """
        X_test = pd.read_csv(X_csv)
        y_test = pd.read_csv(y_csv)[target_col]

        model_path = Path(model_path) if model_path else (self.paths.models_dir / "model.joblib")
        if not model_path.exists():
            raise FileNotFoundError(f"No se encontró el modelo en {model_path}. Entrena primero con train_model.py")

        model = joblib.load(model_path)
        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred, labels=sorted(y_test.unique()), normalize=normalize)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(y_test.unique()))

        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        plt.title("Matriz de confusión")

        out_path = self.paths.figures_dir / "confusion_matrix.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Matriz de confusión guardada en {out_path}")
        return out_path

    # ------------------ Comparación de métricas ------------------
    def plot_metrics_comparison(self, metrics_csv: str | Path, metric: str = "f1_macro") -> Path:
        """Genera gráfico de barras comparando modelos por una métrica (ej. f1_macro)."""
        df = pd.read_csv(metrics_csv)
        if metric not in df.columns:
            raise ValueError(f"La métrica '{metric}' no está en {metrics_csv}")

        plt.figure(figsize=(6, 4))
        sns.barplot(x="model", y=metric, data=df, palette="Set1")
        plt.title(f"Comparación de modelos por {metric}")
        plt.xlabel("Modelo")
        plt.ylabel(metric)
        plt.xticks(rotation=30)

        out_path = self.paths.figures_dir / f"metrics_comparison_{metric}.png"
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        logger.info(f"Comparación de métricas guardada en {out_path}")
        return out_path


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualizaciones del proyecto ML.")
    parser.add_argument("--plot", type=str, required=True, choices=["class_dist", "confusion", "metrics"], help="Tipo de gráfico.")
    parser.add_argument("--data", type=str, help="Ruta al CSV (train/test o métricas).")
    parser.add_argument("--target", type=str, default="Class", help="Columna objetivo (para class_dist/confusion).")
    parser.add_argument("--y", type=str, help="CSV con y_test (para confusion).")
    parser.add_argument("--model", type=str, default=None, help="Ruta al modelo entrenado (para confusion).")
    parser.add_argument("--metric", type=str, default="f1_macro", help="Métrica para comparar (para metrics).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths()
    viz = Visualizer(paths)

    if args.plot == "class_dist":
        if not args.data:
            raise ValueError("--data requerido para plot class_dist")
        viz.plot_class_distribution(args.data, target_col=args.target)

    elif args.plot == "confusion":
        if not args.data or not args.y:
            raise ValueError("--data y --y requeridos para plot confusion")
        viz.plot_confusion_matrix(args.data, args.y, model_path=args.model, target_col=args.target)

    elif args.plot == "metrics":
        if not args.data:
            raise ValueError("--data requerido para plot metrics")
        viz.plot_metrics_comparison(args.data, metric=args.metric)


if __name__ == "__main__":
    main()