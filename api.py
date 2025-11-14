import pandas as pd
import mlflow
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import os
import uvicorn

# Inicializar la aplicación FastAPI
app = FastAPI(
    title="Servicio de Predicción MLOps",
    description="API para predicciones de modelos ML usando MLflow",
    version="1.0.0"
)

# Variable global para almacenar el modelo cargado
model = None

class PredictionRequest(BaseModel):
    """Modelo de solicitud para predicciones"""
    features: Dict[str, Any]
    
class BatchPredictionRequest(BaseModel):
    """Modelo de solicitud para predicciones por lotes"""
    data: List[Dict[str, Any]]

class PredictionResponse(BaseModel):
    """Modelo de respuesta para predicciones"""
    prediction: Any
    model_version: str = None

class BatchPredictionResponse(BaseModel):
    """Modelo de respuesta para predicciones por lotes"""
    predictions: List[Any]
    model_version: str = None

def load_model():
    """Cargar el modelo MLflow"""
    global model
    try:
        # Intentar cargar primero desde variable de entorno
    
       
        return True
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Cargar modelo al iniciar"""
    success = load_model()
    if not success:
        print("Advertencia: No se pudo cargar el modelo al iniciar")

@app.get("/")
async def root():
    """Endpoint de verificación de estado"""
    return {"message": "El Servicio de Predicción MLOps está funcionando"}

@app.get("/health")
async def health_check():
    """Verificación de estado con estatus del modelo"""
    model_loaded = model is not None
    return {
        "status": "saludable" if model_loaded else "no saludable",
        "model_loaded": model_loaded
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Realizar una predicción individual"""
    global model
    
    if model is None:
        # Intentar recargar el modelo
        if not load_model():
            raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    try:
        # Convertir características a DataFrame
        input_data = pd.DataFrame([request.features])
        
        # Hacer predicción
        prediction = model.predict(input_data)
        
        # Extraer valor de predicción individual
        pred_value = prediction[0] if hasattr(prediction, '__getitem__') else prediction
        
        return PredictionResponse(
            prediction=pred_value,
            model_version=getattr(model, 'metadata', {}).get('model_version', 'desconocido')
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error de predicción: {str(e)}")

@app.post("/model/reload")
async def reload_model():
    """Recargar el modelo"""
    success = load_model()
    if success:
        return {"message": "Modelo recargado exitosamente"}
    else:
        raise HTTPException(status_code=500, detail="Falló al recargar el modelo")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
