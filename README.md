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
│   └── turkish_music_emotion_modified.csv
│
├── notebooks/                     # Notebooks de exploración y análisis
│   ├── EDA_Inicial.ipynb
│   ├── EDA_Final.ipynb
│   ├── Transformacion_de_Datos.ipynb
│   └── 01_model_construction.ipynb
│
├── src/                           # Código fuente modular
│   ├── data/
│   │   └── make_dataset.py
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   ├── train_model.py
│   │   └── predict_model.py
│   ├── visualization/
│   │   └── visual.py
│   ├── transformers.py            # Transformadores sklearn
│   ├── sklearn_pipeline.py        # Pipeline de entrenamiento
│   ├── experiment_runner.py       # Ejecución de experimentos
│   └── pipeline.py                # Interfaz simplificada
│
├── models/                        # Modelos entrenados
│
├── reports/                       # Resultados y visualizaciones
│
├── mlflow_manager.py              # Gestión de MLflow
├── main.py                        # Orquestador principal
├── environment.yml                # Entorno conda
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

### 2. Crear entorno con conda

conda env create -f environment.yml
conda activate MNA-MLOps-Proyecto-Equipo03

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

Para más detalles, consulta la documentación en `tests/testing.md`.

