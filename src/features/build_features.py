"""
===============================================================================
Archivo: build_features.py
Propósito:
    - Tomar los CSVs de train/test generados por make_dataset.py
    - Detectar columnas numéricas y categóricas
    - Limpiar textos no-numéricos -> NaN (sin tocar la columna objetivo)
    - Preprocesar con un ColumnTransformer (Imputación + Escalado + OneHot)
    - Transformar X_train / X_test sin "fuga" (fit solo con train)
    - Guardar matrices transformadas y el preprocesador ya ajustado

¿Por qué aquí y no en train_model?
    - Este módulo NO usa la y para selección de variables (ANOVA/PCA) para
      evitar fuga de información. Deja listo un preprocesador robusto y reusable.
    - La selección supervisada (si se desea) se añadirá dentro del pipeline
      de modelado en train_model.py (con y y validación).

Entradas esperadas (generadas por make_dataset.py):
    - data/processed/train.csv (incluye columna objetivo)
    - data/processed/test.csv  (incluye columna objetivo)

Salidas:
    - data/processed/X_train.csv
    - data/processed/y_train.csv
    - data/processed/X_test.csv
    - data/processed/y_test.csv
    - models/preprocessor.joblib  (ColumnTransformer ajustado)

Cómo usar desde línea de comandos:
    # Caso por defecto (target='Class', RobustScaler, imputación mediana)
    python -m src.features.build_features

    # Cambiando target y usando StandardScaler
    python -m src.features.build_features --target Class --scaler standard

    # Cambiando estrategia de imputación numérica
    python -m src.features.build_features --num-imputer mean

Cómo usar dentro de un notebook (.ipynb):
    # 1) Añadir la raíz del repo al sys.path (si estás en notebooks/)
    import sys, os
    repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
    if repo_root not in sys.path:
        sys.path.append(repo_root)
    %load_ext autoreload
    %autoreload 2

    # 2) Importar y ejecutar el builder
    from src.features.build_features import FeatureConfig, FeatureBuilder, ProjectPaths

    paths = ProjectPaths()                  # rutas estándar
    cfg   = FeatureConfig(target_col="Class", scaler="robust")  # config
    builder = FeatureBuilder(paths, cfg)
    out = builder.run()                     # transforma y guarda artefactos

    # 3) Cargar matrices listas para modelado
    import pandas as pd
    X_train = pd.read_csv(paths.data_processed / "X_train.csv")
    y_train = pd.read_csv(paths.data_processed / "y_train.csv")["Class"]
    X_test  = pd.read_csv(paths.data_processed / "X_test.csv")
    y_test  = pd.read_csv(paths.data_processed / "y_test.csv")["Class"]

Notas:
    - Este módulo asume que los CSV de train/test YA existen (make_dataset.py).
    - La columna objetivo no se transforma; solo se separa a y_train/y_test.
    - Las transformaciones que requieren y (ANOVA, etc.) deberán ir en train_model.py
      dentro del Pipeline para mantenerse libres de fuga en CV/holdout.
===============================================================================
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.impute import SimpleImputer


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


# -----------------------------------------------------------------------------
# Rutas del proyecto (compatibles con tu estructura cookiecutter)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ProjectPaths:
    root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = root / "data"
    data_processed: Path = data_dir / "processed"
    models_dir: Path = root / "models"

    def ensure(self) -> None:
        self.data_processed.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Configuración del preprocesamiento
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class FeatureConfig:
    """
    Parámetros de preprocesamiento.

    target_col: nombre de la etiqueta (no se transforma).
    num_imputer: 'median' | 'mean' | 'most_frequent'  (imputación numérica)
    cat_imputer: 'most_frequent' | 'constant'
    scaler: 'robust' | 'standard' | 'none'
    drop_unused: columnas a descartar explícitamente (si aplica)
    """
    target_col: str = "Class"
    num_imputer: str = "median"
    cat_imputer: str = "most_frequent"
    scaler: str = "robust"     # 'robust' recomendado por presencia de outliers
    drop_unused: Optional[List[str]] = None


# -----------------------------------------------------------------------------
# Utilidades internas
# -----------------------------------------------------------------------------
INVALID_NUMERIC_TOKENS = {"nan", "none", "null", "na", "inf", "-inf", "error", ""}

def _coerce_numeric(df: pd.DataFrame, num_cols: List[str]) -> pd.DataFrame:
    """
    Convierte columnas numéricas representadas como texto a números:
    - Reemplaza tokens no numéricos por NaN
    - Usa pandas.to_numeric(errors='coerce')
    """
    for c in num_cols:
        if df[c].dtype == object:
            df[c] = (
                df[c]
                .astype(str)
                .str.strip()
                .str.lower()
                .replace({tok: np.nan for tok in INVALID_NUMERIC_TOKENS})
            )
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _infer_column_types(df: pd.DataFrame, target_col: str) -> Tuple[List[str], List[str]]:
    """Devuelve listas de columnas numéricas y categóricas (excluyendo el target)."""
    features = df.drop(columns=[target_col], errors="ignore")
    num_cols = features.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = features.select_dtypes(exclude=["number"]).columns.tolist()
    return num_cols, cat_cols


def _build_column_transformer(
    num_cols: List[str],
    cat_cols: List[str],
    cfg: FeatureConfig,
) -> ColumnTransformer:
    """
    Crea el ColumnTransformer:
      Numéricas: SimpleImputer -> (Robust|Standard)Scaler? -> passthrough si 'none'
      Categóricas: SimpleImputer -> OneHotEncoder(handle_unknown='ignore')
    """
    # Pipeline numérico
    num_steps = []
    num_steps.append(("imputer", SimpleImputer(strategy=cfg.num_imputer)))
    if cfg.scaler == "robust":
        num_steps.append(("scaler", RobustScaler()))
    elif cfg.scaler == "standard":
        num_steps.append(("scaler", StandardScaler()))
    # else: 'none' => no scaler

    num_pipe = Pipeline(steps=num_steps)

    # Pipeline categórico
    cat_fill = {"strategy": cfg.cat_imputer}
    if cfg.cat_imputer == "constant":
        cat_fill.update({"fill_value": "missing"})

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(**cat_fill)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return preprocessor


def _feature_names(preprocessor: ColumnTransformer, num_cols: List[str], cat_cols: List[str]) -> List[str]:
    """Obtiene nombres de columnas tras el ColumnTransformer (OneHot incluido)."""
    try:
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        # fallback simple
        names = []
        names.extend([f"num__{c}" for c in num_cols])
        names.extend([f"cat__{c}_{i}" for c in cat_cols for i in range(9999)])
        return names


# -----------------------------------------------------------------------------
# Clase principal: FeatureBuilder
# -----------------------------------------------------------------------------
class FeatureBuilder:
    """
    Encapsula el flujo de construcción de features:
        load -> detect types -> coerce numerics -> fit preprocessor -> transform
        -> save matrices -> persist preprocessor

    Uso típico:
        paths = ProjectPaths(); cfg = FeatureConfig(target_col="Class")
        FeatureBuilder(paths, cfg).run()
    """

    def __init__(self, paths: ProjectPaths, cfg: FeatureConfig) -> None:
        self.paths = paths
        self.cfg = cfg

    # --------------------------- API pública ----------------------------
    def run(self) -> dict:
        """Ejecuta el pipeline de features de principio a fin y devuelve rutas."""
        self.paths.ensure()
        train_df, test_df = self._load_processed()
        train_df, test_df = self._drop_unused(train_df, test_df)

        # Detectar tipos
        num_cols, cat_cols = _infer_column_types(train_df, self.cfg.target_col)

        # Coerción numérica (sin tocar target)
        train_df[num_cols] = _coerce_numeric(train_df, num_cols)[num_cols]
        test_df[num_cols] = _coerce_numeric(test_df, num_cols)[num_cols]

        # Separar X/y
        y_train = train_df[self.cfg.target_col].copy()
        y_test = test_df[self.cfg.target_col].copy()
        X_train = train_df.drop(columns=[self.cfg.target_col])
        X_test = test_df.drop(columns=[self.cfg.target_col])

        # Preprocesador (fit solo con TRAIN)
        pre = _build_column_transformer(num_cols, cat_cols, self.cfg)
        logger.info("Fitting preprocessor with TRAIN only (no leakage).")
        X_train_t = pre.fit_transform(X_train)
        X_test_t = pre.transform(X_test)

        # Construir DataFrames con nombres de columnas expandidos
        feat_names = pre.get_feature_names_out()
        X_train_df = pd.DataFrame(X_train_t, columns=feat_names)
        X_test_df = pd.DataFrame(X_test_t, columns=feat_names)

        # Guardar matrices y preprocesador
        out = self._save_outputs(X_train_df, y_train, X_test_df, y_test, pre)

        logger.info("Feature building completed.")
        return out

    # ------------------------- métodos internos -------------------------
    def _load_processed(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_path = self.paths.data_processed / "train.csv"
        test_path = self.paths.data_processed / "test.csv"
        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(
                "No se encuentran train.csv/test.csv. Ejecuta primero make_dataset.py"
            )
        logger.info(f"Cargando processed: {train_path.name}, {test_path.name}")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        logger.info(f"Shapes -> train: {train_df.shape} | test: {test_df.shape}")
        return train_df, test_df

    def _drop_unused(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Elimina columnas explícitamente marcadas como no usadas (si aplica)."""
        if self.cfg.drop_unused:
            drop_cols = [c for c in self.cfg.drop_unused if c in train_df.columns]
            if drop_cols:
                logger.info(f"Descartando columnas: {drop_cols}")
                train_df = train_df.drop(columns=drop_cols)
                test_df = test_df.drop(columns=drop_cols)
        return train_df, test_df

    def _save_outputs(
        self,
        X_train_df: pd.DataFrame,
        y_train: pd.Series,
        X_test_df: pd.DataFrame,
        y_test: pd.Series,
        preprocessor: ColumnTransformer,
    ) -> dict:
        # Asegurar directorios
        self.paths.ensure()

        # Archivos de salida
        X_train_path = self.paths.data_processed / "X_train.csv"
        y_train_path = self.paths.data_processed / "y_train.csv"
        X_test_path = self.paths.data_processed / "X_test.csv"
        y_test_path = self.paths.data_processed / "y_test.csv"
        preproc_path = self.paths.models_dir / "preprocessor.joblib"

        # Guardar CSVs
        X_train_df.to_csv(X_train_path, index=False)
        X_test_df.to_csv(X_test_path, index=False)
        y_train.to_frame(self.cfg.target_col).to_csv(y_train_path, index=False)
        y_test.to_frame(self.cfg.target_col).to_csv(y_test_path, index=False)

        # Guardar preprocesador
        joblib.dump(preprocessor, preproc_path)

        logger.info("Guardados:")
        logger.info(f"  {X_train_path.name} | {X_test_path.name}")
        logger.info(f"  {y_train_path.name} | {y_test_path.name}")
        logger.info(f"  Preprocessor -> {preproc_path}")

        return {
            "X_train": X_train_path,
            "y_train": y_train_path,
            "X_test": X_test_path,
            "y_test": y_test_path,
            "preprocessor": preproc_path,
        }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construcción de features.")
    parser.add_argument("--target", type=str, default="Class", help="Columna objetivo.")
    parser.add_argument(
        "--num-imputer",
        type=str,
        default="median",
        choices=["median", "mean", "most_frequent"],
        help="Estrategia de imputación para numéricas.",
    )
    parser.add_argument(
        "--cat-imputer",
        type=str,
        default="most_frequent",
        choices=["most_frequent", "constant"],
        help="Estrategia de imputación para categóricas.",
    )
    parser.add_argument(
        "--scaler",
        type=str,
        default="robust",
        choices=["robust", "standard", "none"],
        help="Escalador para numéricas.",
    )
    parser.add_argument(
        "--drop",
        type=str,
        nargs="*",
        default=None,
        help="Columnas a descartar explícitamente.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths()
    cfg = FeatureConfig(
        target_col=args.target,
        num_imputer=args.num_imputer,
        cat_imputer=args.cat_imputer,
        scaler=args.scaler,
        drop_unused=args.drop,
    )
    builder = FeatureBuilder(paths, cfg)
    builder.run()


if __name__ == "__main__":
    main()