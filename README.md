# Clasificación de Emociones en Música Turca Tradicional

## Descripción del Proyecto
Este proyecto forma parte de la materia de MLOps y tiene como objetivo aplicar un flujo completo de ciencia de datos con buenas prácticas de versionado de código y datos.

Trabajaremos con el dataset Turkish Music Emotion (UCI Machine Learning Repository), el cual contiene características extraídas de piezas musicales de la tradición turca, con etiquetas de emociones humanas evocadas por la música.  

**Problema principal**:  
Clasificar piezas musicales según la emoción que transmiten (ej. *happy, sad, angry, relaxed*) a partir de atributos acústicos.

---

## Estructura del repositorio

MNA-MLOps-Proyecto-Equipo03/
│
├── data/                # Datasets (DVC)
│   └── README.md
│
├── notebooks/            # Notebooks de exploración y pruebas
│
├── src/                  # Código fuente
│   ├── data_preprocessing.py
│   ├── train_model.py
│   └── utils.py
│
├── models/               # Modelos entrenados (DVC)
│
├── reports/              # Resultados (EDA, gráficas, tablas, métricas)
│
├── requirements.txt      # Dependencias con pip
├── environment.yml       # Entorno reproducible con conda
├── dvc.yaml              # Pipeline de DVC 
├── .gitignore
├── .dvcignore
└── README.md

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

---

### Dataset
	•	Nombre: Turkish Music Emotion Dataset
	•	Fuente: UCI Machine Learning Repository
	•	Descripción: características de audio extraídas de música turca tradicional.
	•	Tipos de variables:
	•	Spectral features: centroid, rolloff, flux, spread.
	•	Rhythmic features: tempo, beats.
	•	MFCCs: coeficientes 1–13.
	•	Chroma features: energía por pitch.
	•	Emotion label: clase objetivo (ej. happy, sad, angry, relaxed).

Versionado de datos con DVC:
	•	data/dataset_raw.csv → versión original.
	•	data/dataset_clean.csv → versión procesada para modelado.
