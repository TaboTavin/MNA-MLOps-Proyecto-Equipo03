# Clasificación de Emociones en Música Turca Tradicional

**Tecnológico de Monterrey**

**Maestría en Inteligencia Artificial Aplicada**  
**Curso:** Machine Learning Operations

---

**Integrantes: Equipo 03**
- David Alejandro Velázquez Valdéz A01632648
- Christian Gustavo Martínez Ramírez A01796999
- César Gustavo Lopez Zamarripa A00967602
- Daniel Vinicio Espinosa Herrera A01796585
- Marcelo Alanis Alvarez A01796009

## Descripción del Proyecto
Este proyecto forma parte de la materia de MLOps y tiene como objetivo aplicar un flujo completo de ciencia de datos con buenas prácticas de versionado de código y datos.

Trabajaremos con el dataset Turkish Music Emotion (UCI Machine Learning Repository), el cual contiene características extraídas de piezas musicales de la tradición turca, con etiquetas de emociones humanas evocadas por la música.  

**Problema principal**:  
Clasificar piezas musicales según la emoción que transmiten (ej. *happy, sad, angry, relaxed*) a partir de atributos acústicos.

---

## Estructura del repositorio
```
MNA-MLOps-Proyecto-Equipo03/
│
├── data/                          # Datasets (DVC)
│   ├── turkis_music_emotion_original.csv
│   ├── turkish_music_emotion_cleaned.csv
│   ├── turkish_music_emotion_modified.csv
│   └── turkish_music_emotion_monitoring.csv
│
├── notebooks/                     # Notebooks de exploración y análisis
│   ├── EDA_Inicial.ipynb
│   ├── EDA_Final.ipynb
│   ├── Transformacion_de_Datos.ipynb
│   └── 01_model_construction.ipynb
│
├── src/                           # Código fuente modular
│   ├── transformers.py            # Transformadores sklearn
│   ├── sklearn_pipeline.py        # Pipeline de entrenamiento
│   ├── experiment_runner.py       # Ejecución de experimentos
│   ├── pipeline.py                # Interfaz simplificada
│   ├── data_drift_simulation.py   # Simulación de data drift
│   └── visual.py                  # Visualización
│
├── tests/                         # Pruebas automatizadas
│   ├── conftest.py                # Fixtures compartidas
│   ├── test_transformers.py       # Pruebas de transformadores
│   ├── test_sklearn_pipeline.py   # Pruebas del pipeline
│   ├── test_experiment_runner.py  # Pruebas del orquestador
│   ├── test_integration.py        # Pruebas end-to-end
│   ├── test_metrics.py            # Pruebas de métricas
│   ├── test_mlflow_integration.py # Pruebas de MLflow
│   └── TESTING_COMPLETE.md        # Documentación de pruebas
│
├── models/                        # Modelos entrenados
│
├── reports/                       # Resultados y visualizaciones
│   ├── music_emotion_classification_metrics.csv
│   ├── data_drift_metrics.csv
│   ├── data_drift_ks_results.csv
│   └── data_drift_alerts.json
│
├── api.py                         # API FastAPI para serving
├── mlflow_manager.py              # Gestión de MLflow
├── main.py                        # Orquestador principal
├── test_data.json                 # Datos de prueba para API
├── environment.yml                # Entorno conda
├── requirements.txt               # Dependencias Python
├── pytest.ini                     # Configuración de pytest
├── Makefile                       # Comandos automatizados
└── README.md
```

---

## Configuración del entorno

### 1. Clonar el repositorio
Usando GitHub Desktop:  
- `Clone repository` → https://github.com/TaboTavin/MNA-MLOps-Proyecto-Equipo03.git.

O en terminal:  
```bash
git clone https://github.com/TaboTavin/MNA-MLOps-Proyecto-Equipo03.git
cd MNA-MLOps-Proyecto-Equipo03
```

### 2. Crear entorno virtual

**Con Python venv:**
```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/macOS

# Instalar dependencias
pip install -r requirements.txt
```

**Con conda:**
```bash
# Crear entorno virtual
conda create -n music-emotion python=3.11

# Activar entorno virtual
conda activate music-emotion

# Instalar dependencias
pip install -r requirements.txt
```

### 3. Descargar datos con DVC
dvc pull 
```

## Dataset
- **Nombre:** Turkish Music Emotion Dataset
- **Fuente:** UCI Machine Learning Repository
- **Descripción:** características de audio extraídas de música turca tradicional.
- **Tipos de variables**
	- Spectral features: centroid, rolloff, flux, spread.
	- Rhythmic features: tempo, beats.
	- MFCCs: coeficientes 1–13.
	- Chroma features: energía por pitch.
	- Emotion label: clase objetivo (ej. happy, sad, angry, relaxed).

## Versionado de datos con DVC

- `data/turkis_music_emotion_original.csv` → versión original del dataset.
- `data/turkish_music_emotion_cleaned.csv` → versión limpia para modelado.
- `data/turkish_music_emotion_modified.csv` → versión con transformaciones adicionales.

---

## Arquitectura del proyecto

### Pipeline de Machine Learning
El proyecto implementa un pipeline completo con las siguientes etapas:

1. **Transformación de datos** (`src/transformers.py`)
   - `DataCleanerTransformer`: Limpieza de valores inválidos
   - `NumericConverterTransformer`: Conversión a formato numérico
   - `CustomImputerTransformer`: Imputación de valores faltantes

2. **Entrenamiento de modelos** (`src/sklearn_pipeline.py`)
   - Pipeline de scikit-learn con preprocesamiento integrado
   - Registro automático de métricas en MLflow
   - Evaluación con accuracy, precision, recall y F1-score

3. **Experimentación** (`src/experiment_runner.py`)
   - Configuraciones de modelos: Random Forest, Gradient Boosting, Logistic Regression, SVM, Decision Tree
   - Tuning de hiperparámetros con 5 configuraciones por modelo
   - Eliminación de código duplicado mediante Factory Pattern

4. **Tracking con MLflow** (`mlflow_manager.py`)
   - Registro de experimentos y ejecuciones
   - Comparación de modelos y métricas
   - Model Registry para versionado de modelos

5. **Visualización** (`src/visual.py`)
   - Gráficas comparativas de modelos
   - Radar charts para análisis multimétrico
   - Evolución temporal de experimentos

### Orquestador principal
El archivo `main.py` coordina todo el flujo mediante la clase `MLExperimentOrchestrator`.

---

## Ejecución del proyecto

### Ejecutar experimentos
El script principal entrena múltiples modelos y registra resultados en MLflow:

```bash
python main.py
```

**¿Qué hace `main.py`?**
- Carga y preprocesa el dataset de emociones musicales
- Entrena 5 modelos base (Random Forest, Gradient Boosting, Logistic Regression, SVM, Decision Tree)
- Registra métricas (accuracy, precision, recall, F1-score) en MLflow
- Genera visualizaciones comparativas en `reports/`
- Registra el mejor modelo en MLflow Model Registry
- Exporta tabla de métricas a CSV

**Resultados generados:**
- Archivo CSV con métricas en `reports/music_emotion_classification_metrics.csv`
- Gráficas PNG en `reports/` (comparación de modelos, evolución temporal, radar chart)
- Modelos registrados en `mlruns/`

---

## Monitoreo y Data Drift

El repositorio incluye una simulación de data drift para evaluar la pérdida de performance del modelo en un escenario de monitoreo.

- Ejecuta la simulación con `python src/data_drift_simulation.py`.
- Se genera un dataset con drift controlado en `data/turkish_music_emotion_monitoring.csv` para pruebas futuras.
- El archivo `reports/data_drift_metrics.csv` compara accuracy, precision, recall y F1-score entre el dataset original y el de monitoreo.
- `reports/data_drift_ks_results.csv` contiene el estadístico y p-value del Test de Kolmogorov-Smirnov por feature (drift si `p_value < 0.05`).
- Las alertas y acciones sugeridas se documentan en `reports/data_drift_alerts.json`; si la caída de accuracy supera 0.05 o hay features con drift, el estado cambia a `alert`.

Usa estos artefactos para decidir acciones de mantenimiento (reentrenamiento del modelo y ajustes de preprocesamiento).

### Instrucciones para ejecutar las pruebas de drift

1. Activa el entorno virtual del proyecto (`conda activate MNA-MLOps-Proyecto-Equipo03` o la alternativa equivalente que utilices).
2. Ubícate en la raíz del repositorio `MNA-MLOps-Proyecto-Equipo03/`.
3. Ejecuta `python src/data_drift_simulation.py` para generar el dataset de monitoreo, calcular métricas y correr las pruebas KS feature por feature.
4. Revisa los resultados en `reports/data_drift_metrics.csv`, `reports/data_drift_ks_results.csv` y `reports/data_drift_alerts.json` para determinar si existe degradación de performance y qué acciones aplicar.

---

## Pruebas automatizadas

El proyecto implementa pruebas unitarias y de integración con pytest para validar el funcionamiento del pipeline.

### Estructura de pruebas
```
tests/
├── conftest.py                    # Fixtures compartidas
├── test_transformers.py           # Pruebas de transformadores
├── test_sklearn_pipeline.py       # Pruebas del pipeline
├── test_experiment_runner.py      # Pruebas del orquestador
├── test_integration.py            # Pruebas end-to-end
├── test_metrics.py                # Pruebas de métricas
└── test_mlflow_integration.py     # Pruebas de MLflow
```

### Ejecutar pruebas

**Instalación de pytest (primera vez):**
```bash
pip install pytest pytest-cov
```

**Ejecutar todas las pruebas:**
```bash
pytest -q
```

**Ejecutar con detalle:**
```bash
pytest -v
```

**Ejecutar con cobertura de código:**
```bash
pytest --cov=src --cov-report=html --cov-report=term
```

**Ejecutar pruebas específicas:**
```bash
pytest tests/test_transformers.py -v
```

Las pruebas cubren:
- Transformadores de datos (limpieza, conversión numérica, imputación)
- Pipeline de entrenamiento completo
- Cálculo de métricas (accuracy, precision, recall, F1-score)
- Integración con MLflow (logging de parámetros, métricas y modelos)
- Flujo end-to-end (carga de datos, preprocesamiento, entrenamiento, predicción)

Para más detalles, consulta la documentación en `tests/TESTING.md`.

---

## Serving y API del Modelo

### Descripción
El proyecto incluye un servicio REST API desarrollado con FastAPI para servir el modelo de clasificación de emociones musicales.

### Modelo en Producción

**Artefacto del modelo:**
- Ruta: `models:/music_emotion_classifier/2`
- Versión: 2
- Algoritmo: RandomForest (accuracy: 0.8395)
- Registro: MLflow Model Registry

El modelo se carga automáticamente desde MLflow Model Registry al iniciar la API.

### Instalación de dependencias

```bash
pip install fastapi uvicorn pydantic
```

O actualizar el entorno:
```bash
conda env update -f environment.yml
```

### Iniciar el servidor

**Asegúrate de activar el entorno conda primero:**
```bash
conda activate MNA-MLOps-Proyecto-Equipo03
```

**Inicia el servidor:**
```bash
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

El servidor estará disponible en `http://localhost:8000`

**Nota:** El servidor cargará automáticamente el modelo versión 2 desde MLflow Model Registry al iniciar.

### Endpoints disponibles

**GET /**
- Información general de la API
- Retorna nombre, versión y lista de endpoints disponibles

**GET /health**
- Verifica el estado del servicio y modelo cargado
- Retorna: `{"status": "healthy", "model_loaded": true, "model_name": "...", "model_version": "..."}`

**POST /predict**
- Realiza predicciones de emociones musicales
- Input: Lista de características acústicas (50 features por muestra)
- Output: Emociones predichas y scores de confianza por clase
- Soporta predicciones batch (múltiples muestras en una sola request)

**GET /model_info**
- Información del modelo cargado
- Retorna: nombre, versión, tipo de pipeline, pasos y clases

**POST /load_model**
- Carga o recarga una versión específica del modelo
- Útil para actualizar el modelo sin reiniciar el servidor

### Documentación interactiva

Una vez iniciado el servidor, accede a:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Schema de entrada/salida

**Request (POST /predict):**

El API acepta las 50 características acústicas extraídas del audio. Ejemplo con algunas características:

```json
{
  "features": [
    {
      "_RMSenergy_Mean": 0.052,
      "_Lowenergy_Mean": 0.554,
      "_Fluctuation_Mean": 9.136,
      "_Tempo_Mean": 130.043,
      "_MFCC_Mean_1": 3.997,
      "_MFCC_Mean_2": 0.363,
      "_Roughness_Mean": 51.542,
      "_Brightness_Mean": 0.173,
      "_Chromagram_Mean_1": 0.496,
      "_HarmonicChangeDetectionFunction_Mean": 0.316
    }
  ]
}
```

**Nota:** El ejemplo anterior muestra solo 10 de las 50 características requeridas. Para ver la lista completa de características, consulta la documentación interactiva en `/docs` o revisa el dataset en `data/turkish_music_emotion_cleaned.csv`.

**Response:**
```json
{
  "predictions": ["relax"],
  "model_version": "2",
  "confidence_scores": [
    {
      "angry": 0.012,
      "happy": 0.003,
      "relax": 0.848,
      "sad": 0.137
    }
  ]
}
```

### Ejemplo de uso

**Verificar estado del servidor:**
```bash
curl http://localhost:8000/health
```

**Realizar predicción con datos reales:**

Puedes usar datos directamente del dataset para hacer pruebas:

```bash
# Crear payload con datos del dataset
python3 -c "
import pandas as pd
import json

df = pd.read_csv('data/turkish_music_emotion_cleaned.csv')
sample = df.iloc[0].drop('Class').to_dict()
payload = {'features': [sample]}

with open('/tmp/api_test.json', 'w') as f:
    json.dump(payload, f)

print('Payload creado en /tmp/api_test.json')
"

# Hacer predicción
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @/tmp/api_test.json
```

### Pruebas del API

Para probar el API con múltiples muestras de diferentes emociones:

```bash
# Crear payload con una muestra de cada emoción
python3 -c "
import pandas as pd
import json

df = pd.read_csv('data/turkish_music_emotion_cleaned.csv')
samples = []
emotions = []

for emotion in ['relax', 'happy', 'sad', 'angry']:
    if emotion in df['Class'].values:
        sample = df[df['Class'] == emotion].iloc[0]
        samples.append(sample.drop('Class').to_dict())
        emotions.append(emotion)

payload = {'features': samples}

with open('/tmp/api_test_multi.json', 'w') as f:
    json.dump(payload, f)

print(f'Payload con {len(samples)} muestras creado')
print(f'Emociones: {emotions}')
"

# Ejecutar predicciones
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @/tmp/api_test_multi.json | python3 -m json.tool
```

**Resultado esperado:**

El API devolverá las predicciones para cada muestra junto con los scores de confianza:

```json
{
  "predictions": ["relax", "happy", "sad", "angry"],
  "model_version": "2",
  "confidence_scores": [
    {"angry": 0.012, "happy": 0.003, "relax": 0.848, "sad": 0.137},
    {"angry": 0.052, "happy": 0.895, "relax": 0.026, "sad": 0.026},
    {"angry": 0.094, "happy": 0.077, "relax": 0.147, "sad": 0.682},
    {"angry": 0.883, "happy": 0.048, "relax": 0.030, "sad": 0.040}
  ]
}

### Validación de entrada

La API utiliza Pydantic para validar automáticamente:
- **Tipos de datos**: Todas las características deben ser valores numéricos (float)
- **Campos requeridos**: Las 50 características acústicas son obligatorias
- **Formato de entrada**: JSON válido con estructura `{"features": [...]}`
- **Aliases**: Acepta nombres con guion bajo (ej. `_RMSenergy_Mean`)

**Características requeridas (50 features):**
- Energy y dinámicas: `_RMSenergy_Mean`, `_Lowenergy_Mean`
- Ritmo: `_Fluctuation_Mean`, `_Tempo_Mean`
- MFCC (13 coeficientes): `_MFCC_Mean_1` a `_MFCC_Mean_13`
- Timbre: `_Roughness_Mean`, `_Roughness_Slope`, `_Zero-crossingrate_Mean`, etc.
- Espectrales: `_Brightness_Mean`, `_Spectralcentroid_Mean`, `_Spectralspread_Mean`, etc.
- Chromagram (12 bins): `_Chromagram_Mean_1` a `_Chromagram_Mean_12`
- Armónicos: `_HarmonicChangeDetectionFunction_Mean`, `_Std`, `_Slope`, etc.

Para la lista completa, consulta `/docs` o el código fuente en `api.py`.

### Manejo de errores

La API maneja los siguientes casos:
- Modelo no cargado (503)
- Datos de entrada inválidos (422)
- Errores de predicción (500)
- Campos faltantes o incorrectos (422)

---

## Despliegue con Docker

### Prerequisitos
- Docker instalado en tu sistema
- Git para clonar el repositorio

### Construcción de la imagen Docker

**1. Clonar el repositorio:**
```bash
git clone https://github.com/TaboTavin/MNA-MLOps-Proyecto-Equipo03.git
cd MNA-MLOps-Proyecto-Equipo03
```

**2. Construir la imagen Docker:**
```bash
docker build -t music-emotion-api .
```

### Ejecutar el contenedor

**Ejecutar en segundo plano (detached) con volumen persistente:**
```bash
docker run -d -p 8000:8000 -v mlruns_data:/app/mlruns --name music-emotion-container music-emotion-api
```

**Verificar que el contenedor está ejecutándose:**
```bash
docker ps
```

### Acceso al API containerizada

Una vez que el contenedor esté ejecutándose, el API estará disponible en:

- **Endpoint base**: `http://localhost:8000`
- **Documentación Swagger**: `http://localhost:8000/docs`
- **Documentación ReDoc**: `http://localhost:8000/redoc`
- **Health check**: `http://localhost:8000/health`

### Probar el API containerizada

**Verificar estado del servicio:**
```bash
curl http://localhost:8000/health
```

**Realizar predicción de prueba:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @test_data.json
```

**Verificar información del modelo cargado:**
```bash
curl http://localhost:8000/model_info
```

### Comandos útiles de Docker

**Ver logs del contenedor:**
```bash
docker logs music-emotion-container
```

**Acceder al contenedor (debug):**
```bash
docker exec -it music-emotion-container /bin/bash
```

**Explorar datos de MLflow dentro del contenedor:**
```bash
docker exec -it music-emotion-container ls -la /app/mlruns
```

**Detener el contenedor:**
```bash
docker stop music-emotion-container
```

**Eliminar el contenedor (los datos de MLflow se mantienen en el volumen):**
```bash
docker rm music-emotion-container
```

**Eliminar la imagen:**
```bash
docker rmi music-emotion-api
```

**Eliminar el volumen de datos (¡CUIDADO: esto borra todos los modelos y experimentos!):**
```bash
docker volume rm mlruns_data
```


---