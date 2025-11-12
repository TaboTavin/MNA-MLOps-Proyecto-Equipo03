"""
Pruebas de transformadores.
"""

import pandas as pd
import numpy as np
import os
import sys

# Añadir el directorio raíz del proyecto al PYTHONPATH
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from src.transformers import (
    DataCleanerTransformer,
    NumericConverterTransformer,
    CustomImputerTransformer,
)


def test_data_cleaner_replaces_invalid_strings_with_nan():
    df = pd.DataFrame({
        "a": ["ok", "NULL", "   ", "error", "value"],
        "b": [1, 2, 3, 4, 5],
    })

    cleaner = DataCleanerTransformer()
    df_clean = cleaner.fit_transform(df)

    # 'a' debe tener NaN donde había valores inválidos
    assert df_clean["a"].isna().sum() == 3
    # 'b' no debe cambiar
    assert df_clean["b"].equals(df["b"])


def test_numeric_converter_casts_columns_to_numeric():
    df = pd.DataFrame({
        "num_str": ["1", "2", "3", "abc"],
        "mixed": [10, "20", "x", None],
    })

    converter = NumericConverterTransformer()
    df_num = converter.fit_transform(df)

    # Todas las columnas deben ser numéricas
    assert np.issubdtype(df_num["num_str"].dtype, np.number)
    assert np.issubdtype(df_num["mixed"].dtype, np.number)

    # Valores no convertibles deben ser NaN
    assert np.isnan(df_num.loc[3, "num_str"])
    assert np.isnan(df_num.loc[2, "mixed"])


def test_custom_imputer_median_strategy():
    df = pd.DataFrame({
        "x": [1.0, 2.0, np.nan, 4.0],
        "y": [np.nan, 10.0, 10.0, 10.0],
    })

    imputer = CustomImputerTransformer(method="median")
    imputer.fit(df)
    df_imputed = imputer.transform(df)

    # Mediana de x = 2.0 (valores: 1, 2, 4)
    assert df_imputed.loc[2, "x"] == 2.0

    # Mediana de y = 10.0 (valores: 10, 10, 10)
    assert df_imputed.loc[0, "y"] == 10.0


def test_custom_imputer_mean_strategy():
    df = pd.DataFrame({
        "x": [1.0, np.nan, 3.0],
    })

    imputer = CustomImputerTransformer(method="mean")
    imputer.fit(df)
    df_imputed = imputer.transform(df)

    # Media de x = (1 + 3) / 2 = 2.0
    assert df_imputed.loc[1, "x"] == 2.0


def test_custom_imputer_handles_all_nan_column():
    df = pd.DataFrame({
        "x": [np.nan, np.nan, np.nan],
    })

    imputer = CustomImputerTransformer(method="median")
    imputer.fit(df)
    df_imputed = imputer.transform(df)

    # Si toda la columna es NaN, definiste que se rellene con 0
    assert (df_imputed["x"] == 0).all()