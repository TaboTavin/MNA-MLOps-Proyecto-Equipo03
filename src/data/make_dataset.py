"""
===============================================================================
Archivo: make_dataset.py
Propósito:
    - Cargar el dataset crudo (CSV) desde la carpeta /data
    - Normalizar la columna objetivo (por defecto "Class")
    - Realizar split estratificado en train/test
    - Guardar los archivos resultantes en /data/processed

Cómo usar este archivo desde la TERMINAL:
    1. Desde la raíz del proyecto, correr:
        python -m src.data.make_dataset

    2. Opciones disponibles:
        --raw        Ruta al CSV crudo (si no se pasa, se busca automáticamente
                     en /data: primero "turkish_music_emotion_modified.csv",
                     luego "turkis_music_emotion_original.csv").
        --target     Nombre de la columna objetivo (default: Class).
        --test-size  Proporción del conjunto de prueba (default: 0.2).
        --seed       Semilla para reproducibilidad (default: 42).
        --no-stratify  Si se incluye, desactiva el split estratificado.

    3. Ejemplos:
        # Caso normal (detecta el CSV en data/)
        python -m src.data.make_dataset

        # Indicando dataset específico y 25% test
        python -m src.data.make_dataset --raw data/turkish_music_emotion_modified.csv --test-size 0.25

        # Sin estratificación
        python -m src.data.make_dataset --no-stratify

Cómo usar este archivo dentro de un NOTEBOOK (.ipynb):
    1. Añadir la raíz del proyecto al sys.path (si estás en /notebooks):
        import sys, os
        repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
        if repo_root not in sys.path:
            sys.path.append(repo_root)

        %load_ext autoreload
        %autoreload 2

    2. Importar y ejecutar el splitter:
        from pathlib import Path
        from src.data.make_dataset import DatasetPaths, SplitConfig, DataSplitter

        paths = DatasetPaths()
        cfg   = SplitConfig(target_col="Class", test_size=0.2, random_state=42)

        splitter = DataSplitter(paths, cfg)
        raw_csv = Path(paths.data_dir) / "turkish_music_emotion_modified.csv"

        train_path, test_path = splitter.run(raw_csv)

    3. Cargar los archivos generados en pandas:
        import pandas as pd
        train = pd.read_csv(train_path)
        test  = pd.read_csv(test_path)

        display(train.head(), test.shape)

Salida:
    - data/processed/train.csv
    - data/processed/test.csv

Notas:
    - Esta etapa sólo prepara el dataset base (sin imputación de features ni escalado).
    - La ingeniería de variables y preprocesamiento adicional se hará en build_features.py.
===============================================================================
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
# Configuración de logging (para trazas claras durante la ejecución)
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# -----------------------------------------------------------------------------
# Configuración de rutas y parámetros con dataclasses (POO simple y extensible)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class DatasetPaths:
    """Define las rutas principales del proyecto siguiendo la estructura Cookiecutter."""
    root: Path = Path(__file__).resolve().parents[2]  # raíz del repo
    data_dir: Path = root / "data"
    processed_dir: Path = data_dir / "processed"

    def ensure_processed(self) -> None:
        """Crea la carpeta /data/processed si no existe."""
        self.processed_dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class SplitConfig:
    """Parámetros de división train/test."""
    target_col: str = "Class"
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True

    # Opcional: mapeo para normalizar clases del target
    canonical_map: tuple[tuple[str, str], ...] = (
        ("angry", "angry"),
        ("anger", "angry"),
        ("happy", "happy"),
        ("happiness", "happy"),
        ("relax", "relax"),
        ("relaxed", "relax"),
        ("sad", "sad"),
        ("sadness", "sad"),
    )

# -----------------------------------------------------------------------------
# Clase principal: DataSplitter
# -----------------------------------------------------------------------------
class DataSplitter:
    """
    Encapsula la lógica de preparación del dataset crudo:
    - Carga
    - Normalización de la etiqueta (target)
    - Split train/test
    - Guardado de los resultados
    """

    def __init__(self, paths: DatasetPaths, cfg: SplitConfig) -> None:
        self.paths = paths
        self.cfg = cfg

    def run(
        self,
        raw_csv_path: Path,
        out_train: Optional[Path] = None,
        out_test: Optional[Path] = None,
    ) -> Tuple[Path, Path]:
        """
        Ejecuta el flujo completo de procesamiento.

        Parameters
        ----------
        raw_csv_path : Path
            Ruta al CSV crudo.
        out_train : Optional[Path]
            Ruta de salida para train.csv.
        out_test : Optional[Path]
            Ruta de salida para test.csv.

        Returns
        -------
        Tuple[Path, Path]
            Rutas finales (train.csv, test.csv).
        """
        self.paths.ensure_processed()

        # Paso 1: Cargar datos
        df = self._load_raw(raw_csv_path)

        # Paso 2: Normalizar columna objetivo
        df = self._normalize_target(df)

        # Paso 3: Split train/test
        train_df, test_df = self._split(df)

        # Paso 4: Guardar resultados
        train_path = out_train or (self.paths.processed_dir / "train.csv")
        test_path = out_test or (self.paths.processed_dir / "test.csv")

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        logger.info("Archivos guardados en data/processed/:")
        logger.info(f"  Train -> {train_path}  (shape={train_df.shape})")
        logger.info(f"  Test  -> {test_path}   (shape={test_df.shape})")

        return train_path, test_path

    def _load_raw(self, csv_path: Path) -> pd.DataFrame:
        """Carga el CSV crudo en un DataFrame."""
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV no encontrado: {csv_path}")
        logger.info(f"Cargando CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        logger.info(f"Dimensiones iniciales: {df.shape}")
        return df

    def _normalize_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normaliza la columna objetivo (lowercase, trim, mapeo canónico)."""
        target = self.cfg.target_col
        if target not in df.columns:
            raise ValueError(f"No se encontró la columna target '{target}' en el dataset.")

        # Convertir a string, limpiar espacios, pasar a minúsculas
        df[target] = (
            df[target]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"nan": None, "none": None})
        )

        # Aplicar mapeo de clases (ej. "happiness" → "happy")
        canon_map = {src: dst for src, dst in self.cfg.canonical_map}
        df[target] = df[target].map(lambda x: canon_map.get(x, x))

        # Eliminar filas sin clase válida
        before = len(df)
        df = df.dropna(subset=[target]).reset_index(drop=True)
        after = len(df)
        if after < before:
            logger.info(f"Eliminadas {before - after} filas sin etiqueta válida.")

        # Reporte de distribución
        dist = df[target].value_counts(dropna=False).to_dict()
        logger.info(f"Distribución de clases: {dist}")

        return df

    def _split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Divide en train/test estratificado si aplica."""
        y = df[self.cfg.target_col]
        stratify = y if (self.cfg.stratify and y.nunique() > 1) else None

        train_df, test_df = train_test_split(
            df,
            test_size=self.cfg.test_size,
            random_state=self.cfg.random_state,
            stratify=stratify,
        )
        logger.info(f"Split -> train: {train_df.shape}, test: {test_df.shape}")
        return train_df, test_df

# -----------------------------------------------------------------------------
# CLI para correr desde terminal
# -----------------------------------------------------------------------------
def _guess_default_raw(paths: DatasetPaths) -> Path:
    """
    Heurística para seleccionar el CSV si no se pasa argumento:
    - Primero busca turkish_music_emotion_modified.csv
    - Luego turkis_music_emotion_original.csv
    - Si no, el primer CSV que encuentre en /data
    """
    cand1 = paths.data_dir / "turkish_music_emotion_modified.csv"
    cand2 = paths.data_dir / "turkis_music_emotion_original.csv"
    if cand1.exists():
        return cand1
    if cand2.exists():
        return cand2
    csvs = list(paths.data_dir.glob("*.csv"))
    if csvs:
        logger.warning(f"Usando el primer CSV encontrado en data/: {csvs[0].name}")
        return csvs[0]
    raise FileNotFoundError("No se encontró ningún CSV en /data/. Pasa --raw.")

def parse_args() -> argparse.Namespace:
    """Define los argumentos disponibles para CLI."""
    parser = argparse.ArgumentParser(description="Crear split train/test desde un CSV crudo.")
    parser.add_argument("--raw", type=str, default=None, help="Ruta al CSV crudo.")
    parser.add_argument("--target", type=str, default="Class", help="Columna objetivo.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proporción test.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria.")
    parser.add_argument("--no-stratify", action="store_true", help="Desactiva estratificación.")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    paths = DatasetPaths()
    raw_csv = Path(args.raw) if args.raw else _guess_default_raw(paths)

    cfg = SplitConfig(
        target_col=args.target,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=not args.no_stratify,
    )
    splitter = DataSplitter(paths, cfg)
    splitter.run(raw_csv)

if __name__ == "__main__":
    main()