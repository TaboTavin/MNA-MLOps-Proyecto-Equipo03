"""
===============================================================================
Archivo: predict_model.py
Propósito:
    - Cargar el modelo entrenado (model.joblib) desde /models
    - Realizar predicciones sobre un nuevo CSV de entrada
    - Guardar las predicciones en un archivo CSV

Entradas esperadas:
    - models/model.joblib   (generado por train_model.py)
    - CSV con features de entrada (ejemplo: data/processed/X_test.csv)

Salida:
    - data/processed/predictions.csv (o ruta indicada)

Cómo usar desde TERMINAL:
    # Predecir sobre X_test y guardar en data/processed/predictions.csv
    python -m src.models.predict_model --input data/processed/X_test.csv

    # Especificar un archivo de salida
    python -m src.models.predict_model --input data/processed/X_test.csv --output reports/preds.csv

Cómo usar desde NOTEBOOK:
    1. Añadir la raíz al sys.path:
        import sys, os
        repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
        if repo_root not in sys.path:
            sys.path.append(repo_root)
        %load_ext autoreload
        %autoreload 2

    2. Importar y usar:
        from src.models.predict_model import ProjectPaths, Predictor

        paths = ProjectPaths()
        predictor = Predictor(paths)
        preds = predictor.run("data/processed/X_test.csv")

    3. Ver resultados:
        print(preds.head())

Notas:
    - El CSV de entrada debe contener las mismas columnas/features que se usaron
      para entrenar el modelo.
    - Este script carga el modelo entrenado como joblib y lo usa directamente.
===============================================================================
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

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
    models_dir: Path = root / "models"

    def ensure(self) -> None:
        self.data_processed.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Clase principal: Predictor
# -----------------------------------------------------------------------------
class Predictor:
    """
    Encapsula la lógica de predicción con un modelo entrenado.
    """

    def __init__(self, paths: ProjectPaths) -> None:
        self.paths = paths

    def run(self, input_csv: str | Path, output_csv: Optional[str | Path] = None) -> pd.DataFrame:
        """
        Carga el modelo y realiza predicciones sobre un CSV de entrada.

        Parameters
        ----------
        input_csv : str | Path
            Ruta al CSV de entrada (features).
        output_csv : Optional[str | Path]
            Ruta para guardar las predicciones (default: data/processed/predictions.csv).

        Returns
        -------
        pd.DataFrame
            DataFrame con las predicciones.
        """
        self.paths.ensure()

        # Paso 1: Cargar modelo
        model_path = self.paths.models_dir / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"No se encontró el modelo entrenado en {model_path}. "
                                    "Ejecuta primero train_model.py")
        model = joblib.load(model_path)
        logger.info(f"Modelo cargado desde {model_path}")

        # Paso 2: Cargar datos de entrada
        input_csv = Path(input_csv)
        if not input_csv.exists():
            raise FileNotFoundError(f"No se encontró el archivo de entrada {input_csv}")
        X_new = pd.read_csv(input_csv)
        logger.info(f"Datos de entrada: {X_new.shape} desde {input_csv}")

        # Paso 3: Generar predicciones
        preds = model.predict(X_new)
        preds_df = pd.DataFrame(preds, columns=["prediction"])

        # Paso 4: Guardar resultados
        output_csv = Path(output_csv) if output_csv else (self.paths.data_processed / "predictions.csv")
        preds_df.to_csv(output_csv, index=False)
        logger.info(f"Predicciones guardadas en {output_csv}")

        return preds_df


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realizar predicciones con un modelo entrenado.")
    parser.add_argument("--input", type=str, required=True, help="CSV con datos de entrada.")
    parser.add_argument("--output", type=str, default=None, help="Ruta de salida para predicciones.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths()
    predictor = Predictor(paths)
    predictor.run(args.input, args.output)


if __name__ == "__main__":
    main()