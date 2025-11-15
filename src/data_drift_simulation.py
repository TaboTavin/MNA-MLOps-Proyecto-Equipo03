"""Simulación de data drift y monitoreo de performance usando Kolmogorov-Smirnov."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Rutas y constantes del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
DATA_PATH = BASE_DIR / "data" / "turkish_music_emotion_cleaned.csv"
MONITORING_DATA_PATH = BASE_DIR / "data" / "turkish_music_emotion_monitoring.csv"
REPORT_DIR = BASE_DIR / "reports"
METRICS_REPORT_PATH = REPORT_DIR / "data_drift_metrics.csv"
KS_RESULTS_PATH = REPORT_DIR / "data_drift_ks_results.csv"
ALERTS_PATH = REPORT_DIR / "data_drift_alerts.json"

TARGET_COLUMN = "Class"
MLFLOW_TRACKING_URI = "./mlruns"
MODEL_RUN_ID = "753a3a15b0b4472f90bc77e07d509dfe"

ALPHA = 0.05
PERFORMANCE_DROP_THRESHOLD = 0.05


def load_dataset(path: Path) -> pd.DataFrame:
    """Carga dataset y normaliza la columna objetivo."""
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el dataset en {path}")

    df = pd.read_csv(path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"La columna objetivo '{TARGET_COLUMN}' no está presente en el dataset")

    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str).str.strip().str.lower()
    df = df[df[TARGET_COLUMN].notna()]
    return df.reset_index(drop=True)


def generate_monitoring_dataset(features: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Genera un dataset de monitoreo aplicando cambios de distribución controlados."""
    rng = np.random.default_rng(seed)
    drifted = features.copy()

    # Incrementar energía general
    energy_columns = ["_RMSenergy_Mean", "_Lowenergy_Mean"]
    drifted[energy_columns] = drifted[energy_columns] * 1.35

    # Cambiar tempo con ruido gaussiano para simular aceleración
    drifted["_Tempo_Mean"] = drifted["_Tempo_Mean"] + rng.normal(12.0, 3.0, len(drifted))

    # Aumentar componentes espectrales
    spectral_columns = ["_Spectralcentroid_Mean", "_Spectralspread_Mean", "_Spectralflatness_Mean"]
    spectral_noise = rng.normal(1.15, 0.03, size=(len(drifted), len(spectral_columns)))
    drifted[spectral_columns] = drifted[spectral_columns] * spectral_noise

    # Reducir la estabilidad armónica
    harmonic_columns = [col for col in drifted.columns if col.startswith("_HarmonicChangeDetectionFunction")]
    harmonic_noise = rng.normal(0.85, 0.02, size=(len(drifted), len(harmonic_columns)))
    drifted[harmonic_columns] = drifted[harmonic_columns] * harmonic_noise

    # Desplazar cromagrama y asegurar límites válidos
    chroma_columns = [col for col in drifted.columns if col.startswith("_Chromagram")]
    chroma_shift = rng.normal(0.1, 0.05, size=(len(drifted), len(chroma_columns)))
    drifted[chroma_columns] = np.clip(drifted[chroma_columns] + chroma_shift, 0.0, 1.0)

    drifted["_Pulseclarity_Mean"] = np.clip(
        drifted["_Pulseclarity_Mean"] + rng.normal(-0.08, 0.04, len(drifted)),
        0.0,
        1.0,
    )

    return drifted


def evaluate_model(model, features: pd.DataFrame, target: pd.Series) -> Dict[str, float]:
    """Calcula métricas de clasificación para un modelo dado."""
    predictions = model.predict(features)
    return {
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, average="weighted", zero_division=0)),
        "recall": float(recall_score(target, predictions, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(target, predictions, average="weighted", zero_division=0)),
    }


def perform_ks_test(
    baseline: pd.DataFrame, monitoring: pd.DataFrame, alpha: float
) -> pd.DataFrame:
    """Ejecuta KS-Test para cada feature y devuelve resultados como DataFrame."""
    results: List[Dict[str, float]] = []
    for column in baseline.columns:
        baseline_values = baseline[column].astype(float)
        monitoring_values = monitoring[column].astype(float)
        statistic, p_value = ks_2samp(baseline_values, monitoring_values)
        results.append(
            {
                "feature": column,
                "ks_statistic": float(statistic),
                "p_value": float(p_value),
                "drift_detected": bool(p_value < alpha),
            }
        )

    return pd.DataFrame(results)


def build_alerts(
    baseline_metrics: Dict[str, float],
    monitoring_metrics: Dict[str, float],
    ks_results: pd.DataFrame,
) -> Dict[str, object]:
    """Construye estructura de alertas basada en caída de performance y KS-Test."""
    performance_drop = baseline_metrics["accuracy"] - monitoring_metrics["accuracy"]
    flagged_features = ks_results.loc[ks_results["drift_detected"], "feature"].tolist()

    alert_triggered = performance_drop >= PERFORMANCE_DROP_THRESHOLD or bool(flagged_features)

    recommended_actions: List[str] = []
    if performance_drop >= PERFORMANCE_DROP_THRESHOLD:
        recommended_actions.append("Re-entrenar el modelo con datos recientes." )
    if flagged_features:
        recommended_actions.append(
            "Revisar distribución de features con drift y actualizar el pipeline de preprocesamiento."
        )

    return {
        "status": "alert" if alert_triggered else "ok",
        "performance_drop": round(float(performance_drop), 6),
        "performance_drop_threshold": PERFORMANCE_DROP_THRESHOLD,
        "baseline_accuracy": round(float(baseline_metrics["accuracy"]), 6),
        "monitoring_accuracy": round(float(monitoring_metrics["accuracy"]), 6),
        "flagged_features": flagged_features,
        "ks_alpha": ALPHA,
        "recommended_actions": recommended_actions,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    logger = logging.getLogger("data_drift_simulation")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Cargando dataset base desde %s", DATA_PATH)
    dataset = load_dataset(DATA_PATH)
    features = dataset.drop(columns=[TARGET_COLUMN])
    target = dataset[TARGET_COLUMN]

    logger.info("Generando dataset de monitoreo con drift controlado")
    monitoring_features = generate_monitoring_dataset(features)
    monitoring_dataset = monitoring_features.copy()
    monitoring_dataset[TARGET_COLUMN] = target.values
    monitoring_dataset.to_csv(MONITORING_DATA_PATH, index=False)
    logger.info("Dataset de monitoreo guardado en %s", MONITORING_DATA_PATH)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"runs:/{MODEL_RUN_ID}/model"
    logger.info("Cargando modelo desde %s", model_uri)
    model = mlflow.sklearn.load_model(model_uri)

    logger.info("Calculando métricas en dataset base")
    baseline_metrics = evaluate_model(model, features, target)

    logger.info("Calculando métricas en dataset de monitoreo")
    monitoring_metrics = evaluate_model(model, monitoring_features, target)

    metrics_df = pd.DataFrame(
        [
            {"dataset": "baseline", **baseline_metrics},
            {"dataset": "monitoring", **monitoring_metrics},
        ]
    )
    metrics_df.to_csv(METRICS_REPORT_PATH, index=False)
    logger.info("Métricas guardadas en %s", METRICS_REPORT_PATH)

    logger.info("Ejecutando prueba KS para cada feature")
    ks_results = perform_ks_test(features, monitoring_features, ALPHA)
    ks_results.to_csv(KS_RESULTS_PATH, index=False)
    logger.info("Resultados KS guardados en %s", KS_RESULTS_PATH)

    alerts = build_alerts(baseline_metrics, monitoring_metrics, ks_results)
    with ALERTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(alerts, f, indent=2, ensure_ascii=True)
    logger.info("Alertas y acciones guardadas en %s", ALERTS_PATH)

    logger.info("Simulación completada. Estado: %s", alerts["status"])


if __name__ == "__main__":
    main()
