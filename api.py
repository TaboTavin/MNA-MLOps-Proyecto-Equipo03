"""
API de predicción de emociones musicales.
Servicio FastAPI para servir el modelo de clasificación de emociones en música.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List, Dict
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

# Agregar src/ al path para importar transformers
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI(
    title="Music Emotion Classification API",
    description="API para clasificar emociones en música turca tradicional usando características acústicas",
    version="1.0.0"
)

# Configuración del modelo
MODEL_NAME = "music_emotion_classifier"
MODEL_VERSION = "2"  # Versión en Production con RandomForest
MLFLOW_TRACKING_URI = "./mlruns"

# NOTA: Si el modelo no se carga automáticamente, puedes especificar un run_id directamente:
# Ejemplo: MODEL_RUN_ID = "tu_run_id_aqui"
# Para encontrar tu run_id, ejecuta: python -c "import mlflow; client = mlflow.tracking.MlflowClient(); runs = client.search_runs('721352221964493676'); print(runs[0].info.run_id if runs else 'No runs found')"
MODEL_RUN_ID = "753a3a15b0b4472f90bc77e07d509dfe"  # Run más reciente de RandomForest sin mixed_type_col

# Variable global para el modelo cargado
model = None


class MusicFeatures(BaseModel):
    """
    Schema de entrada para las características musicales.
    Características acústicas extraídas del audio de música turca.
    """
    # Energy and dynamics
    RMSenergy_Mean: float = Field(..., alias="_RMSenergy_Mean", description="Root Mean Square energy")
    Lowenergy_Mean: float = Field(..., alias="_Lowenergy_Mean", description="Low energy content")
    
    # Rhythm features
    Fluctuation_Mean: float = Field(..., alias="_Fluctuation_Mean", description="Fluctuation pattern")
    Tempo_Mean: float = Field(..., alias="_Tempo_Mean", description="Tempo in BPM")
    
    # MFCC features (13 coefficients)
    MFCC_Mean_1: float = Field(..., alias="_MFCC_Mean_1", description="MFCC coefficient 1")
    MFCC_Mean_2: float = Field(..., alias="_MFCC_Mean_2", description="MFCC coefficient 2")
    MFCC_Mean_3: float = Field(..., alias="_MFCC_Mean_3", description="MFCC coefficient 3")
    MFCC_Mean_4: float = Field(..., alias="_MFCC_Mean_4", description="MFCC coefficient 4")
    MFCC_Mean_5: float = Field(..., alias="_MFCC_Mean_5", description="MFCC coefficient 5")
    MFCC_Mean_6: float = Field(..., alias="_MFCC_Mean_6", description="MFCC coefficient 6")
    MFCC_Mean_7: float = Field(..., alias="_MFCC_Mean_7", description="MFCC coefficient 7")
    MFCC_Mean_8: float = Field(..., alias="_MFCC_Mean_8", description="MFCC coefficient 8")
    MFCC_Mean_9: float = Field(..., alias="_MFCC_Mean_9", description="MFCC coefficient 9")
    MFCC_Mean_10: float = Field(..., alias="_MFCC_Mean_10", description="MFCC coefficient 10")
    MFCC_Mean_11: float = Field(..., alias="_MFCC_Mean_11", description="MFCC coefficient 11")
    MFCC_Mean_12: float = Field(..., alias="_MFCC_Mean_12", description="MFCC coefficient 12")
    MFCC_Mean_13: float = Field(..., alias="_MFCC_Mean_13", description="MFCC coefficient 13")
    
    # Timbral features
    Roughness_Mean: float = Field(..., alias="_Roughness_Mean", description="Roughness measure")
    Roughness_Slope: float = Field(..., alias="_Roughness_Slope", description="Roughness slope")
    Zero_crossingrate_Mean: float = Field(..., alias="_Zero-crossingrate_Mean", description="Zero crossing rate")
    AttackTime_Mean: float = Field(..., alias="_AttackTime_Mean", description="Attack time")
    AttackTime_Slope: float = Field(..., alias="_AttackTime_Slope", description="Attack time slope")
    Rolloff_Mean: float = Field(..., alias="_Rolloff_Mean", description="Spectral rolloff")
    
    # Event and pulse features
    Eventdensity_Mean: float = Field(..., alias="_Eventdensity_Mean", description="Event density")
    Pulseclarity_Mean: float = Field(..., alias="_Pulseclarity_Mean", description="Pulse clarity")
    
    # Spectral features
    Brightness_Mean: float = Field(..., alias="_Brightness_Mean", description="Spectral brightness")
    Spectralcentroid_Mean: float = Field(..., alias="_Spectralcentroid_Mean", description="Spectral centroid")
    Spectralspread_Mean: float = Field(..., alias="_Spectralspread_Mean", description="Spectral spread")
    Spectralskewness_Mean: float = Field(..., alias="_Spectralskewness_Mean", description="Spectral skewness")
    Spectralkurtosis_Mean: float = Field(..., alias="_Spectralkurtosis_Mean", description="Spectral kurtosis")
    Spectralflatness_Mean: float = Field(..., alias="_Spectralflatness_Mean", description="Spectral flatness")
    EntropyofSpectrum_Mean: float = Field(..., alias="_EntropyofSpectrum_Mean", description="Entropy of spectrum")
    
    # Chromagram features (12 pitch classes)
    Chromagram_Mean_1: float = Field(..., alias="_Chromagram_Mean_1", description="Chromagram bin 1")
    Chromagram_Mean_2: float = Field(..., alias="_Chromagram_Mean_2", description="Chromagram bin 2")
    Chromagram_Mean_3: float = Field(..., alias="_Chromagram_Mean_3", description="Chromagram bin 3")
    Chromagram_Mean_4: float = Field(..., alias="_Chromagram_Mean_4", description="Chromagram bin 4")
    Chromagram_Mean_5: float = Field(..., alias="_Chromagram_Mean_5", description="Chromagram bin 5")
    Chromagram_Mean_6: float = Field(..., alias="_Chromagram_Mean_6", description="Chromagram bin 6")
    Chromagram_Mean_7: float = Field(..., alias="_Chromagram_Mean_7", description="Chromagram bin 7")
    Chromagram_Mean_8: float = Field(..., alias="_Chromagram_Mean_8", description="Chromagram bin 8")
    Chromagram_Mean_9: float = Field(..., alias="_Chromagram_Mean_9", description="Chromagram bin 9")
    Chromagram_Mean_10: float = Field(..., alias="_Chromagram_Mean_10", description="Chromagram bin 10")
    Chromagram_Mean_11: float = Field(..., alias="_Chromagram_Mean_11", description="Chromagram bin 11")
    Chromagram_Mean_12: float = Field(..., alias="_Chromagram_Mean_12", description="Chromagram bin 12")
    
    # Harmonic change detection features
    HarmonicChangeDetectionFunction_Mean: float = Field(..., alias="_HarmonicChangeDetectionFunction_Mean", description="Harmonic change mean")
    HarmonicChangeDetectionFunction_Std: float = Field(..., alias="_HarmonicChangeDetectionFunction_Std", description="Harmonic change std")
    HarmonicChangeDetectionFunction_Slope: float = Field(..., alias="_HarmonicChangeDetectionFunction_Slope", description="Harmonic change slope")
    HarmonicChangeDetectionFunction_PeriodFreq: float = Field(..., alias="_HarmonicChangeDetectionFunction_PeriodFreq", description="Harmonic period frequency")
    HarmonicChangeDetectionFunction_PeriodAmp: float = Field(..., alias="_HarmonicChangeDetectionFunction_PeriodAmp", description="Harmonic period amplitude")
    HarmonicChangeDetectionFunction_PeriodEntropy: float = Field(..., alias="_HarmonicChangeDetectionFunction_PeriodEntropy", description="Harmonic period entropy")
    
    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "_RMSenergy_Mean": 0.052,
                "_Lowenergy_Mean": 0.554,
                "_Fluctuation_Mean": 9.136,
                "_Tempo_Mean": 130.043,
                "_MFCC_Mean_1": 3.997,
                "_MFCC_Mean_2": 0.363,
                "_MFCC_Mean_3": 0.887,
                "_MFCC_Mean_4": 0.078,
                "_MFCC_Mean_5": 0.221,
                "_MFCC_Mean_6": 0.118,
                "_MFCC_Mean_7": -0.151,
                "_MFCC_Mean_8": -0.131,
                "_MFCC_Mean_9": 0.129,
                "_MFCC_Mean_10": 0.154,
                "_MFCC_Mean_11": 0.274,
                "_MFCC_Mean_12": 0.232,
                "_MFCC_Mean_13": 0.246,
                "_Roughness_Mean": 51.542,
                "_Roughness_Slope": 0.325,
                "_Zero-crossingrate_Mean": 403.129,
                "_AttackTime_Mean": 0.027,
                "_AttackTime_Slope": -0.014,
                "_Rolloff_Mean": 1844.664,
                "_Eventdensity_Mean": 1.336,
                "_Pulseclarity_Mean": 0.082,
                "_Brightness_Mean": 0.173,
                "_Spectralcentroid_Mean": 1121.368,
                "_Spectralspread_Mean": 1970.389,
                "_Spectralskewness_Mean": 3.621,
                "_Spectralkurtosis_Mean": 18.037,
                "_Spectralflatness_Mean": 0.03,
                "_EntropyofSpectrum_Mean": 0.806,
                "_Chromagram_Mean_1": 0.496,
                "_Chromagram_Mean_2": 0.0,
                "_Chromagram_Mean_3": 0.047,
                "_Chromagram_Mean_4": 0.03,
                "_Chromagram_Mean_5": 0.314,
                "_Chromagram_Mean_6": 0.038,
                "_Chromagram_Mean_7": 0.024,
                "_Chromagram_Mean_8": 0.951,
                "_Chromagram_Mean_9": 0.426,
                "_Chromagram_Mean_10": 1.0,
                "_Chromagram_Mean_11": 0.008,
                "_Chromagram_Mean_12": 0.101,
                "_HarmonicChangeDetectionFunction_Mean": 0.316,
                "_HarmonicChangeDetectionFunction_Std": 0.261,
                "_HarmonicChangeDetectionFunction_Slope": 0.018,
                "_HarmonicChangeDetectionFunction_PeriodFreq": 1.035,
                "_HarmonicChangeDetectionFunction_PeriodAmp": 0.593,
                "_HarmonicChangeDetectionFunction_PeriodEntropy": 0.97
            }
        }
    
    @validator('*', pre=True)
    def validate_numeric(cls, v):
        """Valida que todos los valores sean numéricos válidos."""
        if v is None:
            return 0.0
        try:
            return float(v)
        except (ValueError, TypeError):
            raise ValueError(f"El valor debe ser numérico, recibido: {v}")


class PredictionRequest(BaseModel):
    """Schema de entrada para predicciones batch."""
    features: List[MusicFeatures] = Field(..., description="Lista de características musicales")
    
    class Config:
        schema_extra = {
            "example": {
                "features": [
                    {
                        "Tempo": 120.5,
                        "Energy": 0.75,
                        "Danceability": 0.68,
                        "Valence": 0.82,
                        "Acousticness": 0.45,
                        "Instrumentalness": 0.0,
                        "Liveness": 0.1,
                        "Speechiness": 0.05
                    }
                ]
            }
        }


class PredictionResponse(BaseModel):
    """Schema de salida para predicciones."""
    predictions: List[str] = Field(..., description="Emociones predichas")
    model_version: str = Field(..., description="Versión del modelo utilizado")
    confidence_scores: List[Dict[str, float]] = Field(None, description="Scores de confianza por clase")
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": ["happy", "sad"],
                "model_version": "1",
                "confidence_scores": [
                    {"happy": 0.85, "sad": 0.10, "energetic": 0.03, "calm": 0.02},
                    {"happy": 0.15, "sad": 0.75, "energetic": 0.05, "calm": 0.05}
                ]
            }
        }


class HealthResponse(BaseModel):
    """Schema de respuesta para health check."""
    status: str
    model_loaded: bool
    model_name: str
    model_version: str


@app.on_event("startup")
async def load_model():
    """Carga el modelo desde MLflow al iniciar la aplicación."""
    global model
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        # Si hay un RUN_ID específico, cargar desde ahí primero
        if MODEL_RUN_ID:
            try:
                model_uri = f"runs:/{MODEL_RUN_ID}/model"
                logger.info(f"Intentando cargar modelo desde RUN: {model_uri}")
                model = mlflow.sklearn.load_model(model_uri)
                logger.info(f"Modelo cargado exitosamente desde run {MODEL_RUN_ID}")
                return
            except Exception as run_error:
                logger.warning(f"No se pudo cargar desde run {MODEL_RUN_ID}: {run_error}")
        
        # Intentar cargar desde Model Registry
        try:
            model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
            logger.info(f"Intentando cargar modelo desde Model Registry: {model_uri}")
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Modelo {MODEL_NAME} v{MODEL_VERSION} cargado exitosamente desde Model Registry")
        except Exception as registry_error:
            # Si falla, intentar cargar desde el directorio de modelos
            logger.warning(f"No se pudo cargar desde Model Registry: {registry_error}")
            model_path = f"./mlruns/models/{MODEL_NAME}"
            if Path(model_path).exists():
                logger.info(f"Intentando cargar modelo desde: {model_path}")
                # Buscar el directorio version-X
                version_dir = Path(model_path) / f"version-{MODEL_VERSION}"
                if version_dir.exists():
                    model_full_path = version_dir / "artifacts/model"
                    if model_full_path.exists():
                        logger.info(f"Cargando desde: {model_full_path}")
                        model = mlflow.sklearn.load_model(str(model_full_path))
                        logger.info(f"Modelo cargado exitosamente desde {model_full_path}")
                    else:
                        raise FileNotFoundError(f"No se encontró el modelo en {model_full_path}")
                else:
                    # Intentar buscar cualquier versión disponible
                    versions = [d for d in Path(model_path).iterdir() if d.is_dir() and d.name.startswith('version-')]
                    if versions:
                        latest_version = max(versions, key=lambda x: int(x.name.split('-')[1]))
                        model_full_path = latest_version / "artifacts/model"
                        if model_full_path.exists():
                            logger.info(f"Cargando desde: {model_full_path}")
                            model = mlflow.sklearn.load_model(str(model_full_path))
                            logger.info(f"Modelo cargado exitosamente desde {model_full_path}")
                        else:
                            raise FileNotFoundError(f"No se encontró el modelo en {model_full_path}")
                    else:
                        raise FileNotFoundError(f"No hay versiones de modelo en {model_path}")
            else:
                raise FileNotFoundError(f"No existe el directorio {model_path}")
    except Exception as e:
        logger.error(f"Error al cargar el modelo: {str(e)}")
        logger.warning("La API se iniciará sin modelo cargado. Use el endpoint /load_model para cargar uno.")


@app.get("/", tags=["General"])
async def root():
    """Endpoint raíz con información de la API."""
    return {
        "message": "Music Emotion Classification API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Verifica el estado de la API y el modelo."""
    return HealthResponse(
        status="healthy" if model is not None else "model_not_loaded",
        model_loaded=model is not None,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_emotion(request: PredictionRequest):
    """
    Predice la emoción de una o más piezas musicales.
    
    - **features**: Lista de características musicales extraídas de audio
    
    Retorna las emociones predichas y scores de confianza.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no cargado. Contacte al administrador o use /load_model"
        )
    
    try:
        # Convertir features a DataFrame usando los alias (nombres con guion bajo)
        features_list = [feature.dict(by_alias=True) for feature in request.features]
        df = pd.DataFrame(features_list)
        
        # Realizar predicción
        predictions = model.predict(df)
        
        # Obtener probabilidades si el modelo las soporta
        confidence_scores = None
        if hasattr(model, 'predict_proba'):
            try:
                probas = model.predict_proba(df)
                classes = model.classes_ if hasattr(model, 'classes_') else None
                
                if classes is not None:
                    confidence_scores = [
                        {cls: float(prob) for cls, prob in zip(classes, proba)}
                        for proba in probas
                    ]
            except Exception as e:
                logger.warning(f"No se pudieron obtener probabilidades: {str(e)}")
        
        return PredictionResponse(
            predictions=predictions.tolist(),
            model_version=MODEL_VERSION,
            confidence_scores=confidence_scores
        )
        
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al realizar la predicción: {str(e)}"
        )


@app.post("/load_model", tags=["Admin"])
async def load_model_endpoint(model_version: str = MODEL_VERSION):
    """
    Carga o recarga el modelo desde MLflow Model Registry.
    
    - **model_version**: Versión del modelo a cargar (default: 1)
    """
    global model, MODEL_VERSION
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{MODEL_NAME}/{model_version}"
        logger.info(f"Cargando modelo desde: {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)
        MODEL_VERSION = model_version
        logger.info(f"Modelo {MODEL_NAME} v{model_version} cargado exitosamente")
        return {
            "message": f"Modelo cargado exitosamente",
            "model_name": MODEL_NAME,
            "model_version": model_version
        }
    except Exception as e:
        logger.error(f"Error al cargar el modelo: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al cargar el modelo: {str(e)}"
        )


@app.get("/model_info", tags=["Admin"])
async def get_model_info():
    """Obtiene información del modelo actualmente cargado."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="No hay modelo cargado"
        )
    
    info = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_type": type(model).__name__,
    }
    
    # Información adicional si está disponible
    if hasattr(model, 'named_steps'):
        info["pipeline_steps"] = list(model.named_steps.keys())
    
    if hasattr(model, 'classes_'):
        info["classes"] = model.classes_.tolist()
    
    if hasattr(model, 'n_features_in_'):
        info["n_features"] = model.n_features_in_
    
    return info


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
