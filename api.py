"""
API de predicción de emociones musicales.
Servicio FastAPI para servir el modelo de clasificación de emociones en música.
Diseñado para correr en un entorno Docker donde el modelo se entrena al inicio.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional
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

# --- CONFIGURACIÓN SIMPLIFICADA ---
# El script 'start.sh' genera 'mlruns' localmente
MLFLOW_TRACKING_URI = "./mlruns"
MODEL_NAME = "music_emotion_classifier"
# --- FIN DE CONFIGURACIÓN ---

# Inicializar FastAPI
app = FastAPI(
    title="Music Emotion Classification API",
    description="API para clasificar emociones en música turca tradicional",
    version="1.1.0"
)

# Variable global para el modelo cargado
model = None
model_version_loaded = "N/A" # Variable para guardar la versión cargada

# Schemas de Pydantic (Sin cambios)
class MusicFeatures(BaseModel):
    # Energy and dynamics
    RMSenergy_Mean: float = Field(..., alias="_RMSenergy_Mean", description="Root Mean Square energy")
    Lowenergy_Mean: float = Field(..., alias="_Lowenergy_Mean", description="Low energy content")
    Fluctuation_Mean: float = Field(..., alias="_Fluctuation_Mean", description="Fluctuation pattern")
    Tempo_Mean: float = Field(..., alias="_Tempo_Mean", description="Tempo in BPM")
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
    Roughness_Mean: float = Field(..., alias="_Roughness_Mean", description="Roughness measure")
    Roughness_Slope: float = Field(..., alias="_Roughness_Slope", description="Roughness slope")
    Zero_crossingrate_Mean: float = Field(..., alias="_Zero-crossingrate_Mean", description="Zero crossing rate")
    AttackTime_Mean: float = Field(..., alias="_AttackTime_Mean", description="Attack time")
    AttackTime_Slope: float = Field(..., alias="_AttackTime_Slope", description="Attack time slope")
    Rolloff_Mean: float = Field(..., alias="_Rolloff_Mean", description="Spectral rolloff")
    Eventdensity_Mean: float = Field(..., alias="_Eventdensity_Mean", description="Event density")
    Pulseclarity_Mean: float = Field(..., alias="_Pulseclarity_Mean", description="Pulse clarity")
    Brightness_Mean: float = Field(..., alias="_Brightness_Mean", description="Spectral brightness")
    Spectralcentroid_Mean: float = Field(..., alias="_Spectralcentroid_Mean", description="Spectral centroid")
    Spectralspread_Mean: float = Field(..., alias="_Spectralspread_Mean", description="Spectral spread")
    Spectralskewness_Mean: float = Field(..., alias="_Spectralskewness_Mean", description="Spectral skewness")
    Spectralkurtosis_Mean: float = Field(..., alias="_Spectralkurtosis_Mean", description="Spectral kurtosis")
    Spectralflatness_Mean: float = Field(..., alias="_Spectralflatness_Mean", description="Spectral flatness")
    EntropyofSpectrum_Mean: float = Field(..., alias="_EntropyofSpectrum_Mean", description="Entropy of spectrum")
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
    HarmonicChangeDetectionFunction_Mean: float = Field(..., alias="_HarmonicChangeDetectionFunction_Mean", description="Harmonic change mean")
    HarmonicChangeDetectionFunction_Std: float = Field(..., alias="_HarmonicChangeDetectionFunction_Std", description="Harmonic change std")
    HarmonicChangeDetectionFunction_Slope: float = Field(..., alias="_HarmonicChangeDetectionFunction_Slope", description="Harmonic change slope")
    HarmonicChangeDetectionFunction_PeriodFreq: float = Field(..., alias="_HarmonicChangeDetectionFunction_PeriodFreq", description="Harmonic period frequency")
    HarmonicChangeDetectionFunction_PeriodAmp: float = Field(..., alias="_HarmonicChangeDetectionFunction_PeriodAmp", description="Harmarmonic period amplitude")
    HarmonicChangeDetectionFunction_PeriodEntropy: float = Field(..., alias="_HarmonicChangeDetectionFunction_PeriodEntropy", description="Harmonic period entropy")
    
    class Config:
        populate_by_name = True # Actualizado para Pydantic v2
        json_schema_extra = { # Actualizado para Pydantic v2
            "example": {
                "_RMSenergy_Mean": 0.052, "_Lowenergy_Mean": 0.554, "_Fluctuation_Mean": 9.136, "_Tempo_Mean": 130.043,
                "_MFCC_Mean_1": 3.997, "_MFCC_Mean_2": 0.363, "_MFCC_Mean_3": 0.887, "_MFCC_Mean_4": 0.078,
                "_MFCC_Mean_5": 0.221, "_MFCC_Mean_6": 0.118, "_MFCC_Mean_7": -0.151, "_MFCC_Mean_8": -0.131,
                "_MFCC_Mean_9": 0.129, "_MFCC_Mean_10": 0.154, "_MFCC_Mean_11": 0.274, "_MFCC_Mean_12": 0.232,
                "_MFCC_Mean_13": 0.246, "_Roughness_Mean": 51.542, "_Roughness_Slope": 0.325,
                "_Zero-crossingrate_Mean": 403.129, "_AttackTime_Mean": 0.027, "_AttackTime_Slope": -0.014,
                "_Rolloff_Mean": 1844.664, "_Eventdensity_Mean": 1.336, "_Pulseclarity_Mean": 0.082,
                "_Brightness_Mean": 0.173, "_Spectralcentroid_Mean": 1121.368, "_Spectralspread_Mean": 1970.389,
                "_Spectralskewness_Mean": 3.621, "_Spectralkurtosis_Mean": 18.037, "_Spectralflatness_Mean": 0.03,
                "_EntropyofSpectrum_Mean": 0.806, "_Chromagram_Mean_1": 0.496, "_Chromagram_Mean_2": 0.0,
                "_Chromagram_Mean_3": 0.047, "_Chromagram_Mean_4": 0.03, "_Chromagram_Mean_5": 0.314,
                "_Chromagram_Mean_6": 0.038, "_Chromagram_Mean_7": 0.024, "_Chromagram_Mean_8": 0.951,
                "_Chromagram_Mean_9": 0.426, "_Chromagram_Mean_10": 1.0, "_Chromagram_Mean_11": 0.008,
                "_Chromagram_Mean_12": 0.101, "_HarmonicChangeDetectionFunction_Mean": 0.316,
                "_HarmonicChangeDetectionFunction_Std": 0.261, "_HarmonicChangeDetectionFunction_Slope": 0.018,
                "_HarmonicChangeDetectionFunction_PeriodFreq": 1.035, "_HarmonicChangeDetectionFunction_PeriodAmp": 0.593,
                "_HarmonicChangeDetectionFunction_PeriodEntropy": 0.97
            }
        }

    @validator('*', pre=True)
    def validate_numeric(cls, v):
        if v is None: return 0.0
        try: return float(v)
        except (ValueError, TypeError): raise ValueError(f"El valor debe ser numérico, recibido: {v}")

class PredictionRequest(BaseModel):
    features: List[MusicFeatures]

class PredictionResponse(BaseModel):
    predictions: List[str]
    model_version: str
    confidence_scores: List[Dict[str, float]] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    model_version: str

# --- LÓGICA DE CARGA DE MODELO (MUY SIMPLIFICADA) ---
@app.on_event("startup")
async def load_model():
    """
    Carga el modelo desde MLflow al iniciar la aplicación.
    Esta función ASUME que 'main.py' ya corrió (controlado por start.sh)
    y registró un modelo en el Model Registry local.
    """
    global model, model_version_loaded
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        # Esta es la lógica robusta: Cargar la última versión disponible.
        # El script 'start.sh' corre 'main.py' primero, que crea la 'version-1'.
        # Esta línea la encontrará.
        model_uri = f"models:/{MODEL_NAME}/Latest"
        
        logger.info(f"Intentando cargar la ÚLTIMA versión del modelo desde: {model_uri}")
        
        # Cargamos el modelo
        model = mlflow.sklearn.load_model(model_uri)
        
        # Obtenemos la versión que se cargó
        client = mlflow.tracking.MlflowClient()
        latest_version_info = client.get_latest_versions(MODEL_NAME, stages=["None", "Staging", "Production"])
        if latest_version_info:
            # --- INICIO DE LA CORRECCIÓN ---
            # El .version de MLflow es un INT, pero el schema Pydantic espera un STR
            # Lo convertimos explícitamente a string.
            model_version_loaded = str(latest_version_info[0].version)
            # --- FIN DE LA CORRECCIÓN ---
        else:
            model_version_loaded = "1" # Fallback
            
        logger.info(f"Modelo {MODEL_NAME} v{model_version_loaded} cargado exitosamente.")
        
    except Exception as e:
        logger.error(f"Error al cargar el modelo: {str(e)}")
        logger.error("Asegúrate de que 'main.py' se ejecute primero y registre el modelo.")
        logger.warning("La API se iniciará SIN modelo. Las predicciones fallarán.")


@app.get("/", tags=["General"])
async def root():
    return {
        "message": "Music Emotion Classification API",
        "version": "1.1.0",
        "endpoints": ["/health", "/predict", "/docs", "/redoc", "/model_info", "/load_model"]
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    return HealthResponse(
        status="healthy" if model is not None else "model_not_loaded",
        model_loaded=model is not None,
        model_name=MODEL_NAME,
        model_version=model_version_loaded
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_emotion(request: PredictionRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no cargado. Revisa los logs de inicio."
        )
    
    try:
        features_list = [feature.dict(by_alias=True) for feature in request.features]
        df = pd.DataFrame(features_list)
        
        predictions = model.predict(df)
        
        confidence_scores = None
        if hasattr(model, 'predict_proba'):
            try:
                probas = model.predict_proba(df)
                classes = model.classes_
                confidence_scores = [
                    {cls: float(prob) for cls, prob in zip(classes, proba)}
                    for proba in probas
                ]
            except Exception as e:
                logger.warning(f"No se pudieron obtener probabilidades: {str(e)}")
        
        return PredictionResponse(
            predictions=predictions.tolist(),
            model_version=model_version_loaded,
            confidence_scores=confidence_scores
        )
        
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al realizar la predicción: {str(e)}"
        )

# --- ENDPOINTS DE ADMINISTRACIÓN ---

@app.post("/load_model", tags=["Admin"])
async def load_model_endpoint(model_version: str = "Latest"):
    """
    Carga o recarga el modelo desde MLflow Model Registry.
    
    - **model_version**: Versión del modelo a cargar (default: "Latest")
    """
    global model, model_version_loaded
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{MODEL_NAME}/{model_version}"
        logger.info(f"Cargando modelo desde: {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)
        
        # Si cargamos "Latest", necesitamos saber qué versión real se cargó
        if model_version.lower() == "latest":
            client = mlflow.tracking.MlflowClient()
            latest_info = client.get_latest_versions(MODEL_NAME, stages=["None", "Staging", "Production"])
            if latest_info:
                model_version_loaded = str(latest_info[0].version)
        else:
            model_version_loaded = model_version
            
        logger.info(f"Modelo {MODEL_NAME} v{model_version_loaded} cargado exitosamente")
        return {
            "message": f"Modelo cargado exitosamente",
            "model_name": MODEL_NAME,
            "model_version": model_version_loaded
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
        "model_version": model_version_loaded, # Usamos la variable global
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