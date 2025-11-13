# Documentación de Pruebas Automatizadas

Este documento describe el sistema de pruebas unitarias y de integración implementado con pytest para validar el pipeline del proyecto de clasificación de emociones en música turca.

## Tabla de Contenidos

- [Estructura de Pruebas](#estructura-de-pruebas)
- [Instalación de Dependencias](#instalación-de-dependencias)
- [Ejecución de Pruebas](#ejecución-de-pruebas)
- [Tipos de Pruebas](#tipos-de-pruebas)
- [Cobertura de Código](#cobertura-de-código)
- [Fixtures Compartidas](#fixtures-compartidas)

---

## Estructura de Pruebas

```
tests/
├── __init__.py                      # Inicialización del paquete de tests
├── conftest.py                      # Fixtures compartidas y configuración global
├── test_transformers.py             # Pruebas unitarias de transformadores
├── test_sklearn_pipeline.py         # Pruebas unitarias del pipeline sklearn
├── test_experiment_runner.py        # Pruebas de integración del runner
├── test_integration.py              # Pruebas E2E del flujo completo
├── test_metrics.py                  # Pruebas unitarias de métricas
├── test_mlflow_integration.py       # Pruebas de integración con MLflow
└── TESTING.md                       # Documentación original
```

### Descripción de Archivos

**test_transformers.py**: Pruebas unitarias de transformadores personalizados
- DataCleanerTransformer
- NumericConverterTransformer
- CustomImputerTransformer

**test_sklearn_pipeline.py**: Pruebas del pipeline principal
- Construcción del pipeline
- Carga y división de datos
- Entrenamiento de modelos
- Logging de métricas

**test_experiment_runner.py**: Pruebas del orquestador de experimentos
- Ejecución de múltiples configuraciones
- Factory de hiperparámetros

**test_integration.py**: Pruebas end-to-end
- Flujo completo: carga, preprocesamiento, entrenamiento, predicción
- Validación de calidad de datos
- Manejo de datos sucios
- Preservación de distribución de clases

**test_metrics.py**: Pruebas de cálculo de métricas
- Accuracy, Precision, Recall, F1-Score
- Casos extremos y edge cases
- Consistencia entre métricas

**test_mlflow_integration.py**: Pruebas de integración con MLflow
- Logging de parámetros y métricas
- Guardado de modelos
- Gestión de experimentos

---

## Instalación de Dependencias

### 1. Activar el entorno Conda

```bash
conda activate MNA-MLOps-Proyecto-Equipo03
```

### 2. Instalar pytest

```bash
pip install pytest pytest-cov
```

Alternativamente, actualiza el archivo `environment.yml`:

```yaml
- pip:
    - pytest==8.3.4
    - pytest-cov==6.0.0
```

---

## Ejecución de Pruebas

### Comando básico (modo silencioso)

```bash
pytest -q
```

### Ejecución con salida detallada

```bash
pytest -v
```

### Ejecutar archivo específico

```bash
pytest tests/test_transformers.py -v
```

### Ejecutar prueba específica

```bash
pytest tests/test_transformers.py::test_data_cleaner_replaces_invalid_strings_with_nan -v
```

### Ejecutar por categoría (clase)

```bash
pytest tests/test_integration.py::TestEndToEndPipeline -v
```

### Mostrar output de print

```bash
pytest -v -s
```

### Ejecutar solo pruebas rápidas (excluir integración)

```bash
pytest tests/test_transformers.py tests/test_metrics.py -v
```

### Detener en el primer error

```bash
pytest -x
```

### Modo debugging (prints y errores detallados)

```bash
pytest -vv -s --tb=long
```

---

## Tipos de Pruebas

### 1. Pruebas Unitarias

Validan funciones y módulos individuales de forma aislada.

**Ejemplos:**
- Transformadores de datos (test_transformers.py)
- Cálculo de métricas (test_metrics.py)
- Construcción de pipelines (test_sklearn_pipeline.py)

**Características:**
- Rápidas (menos de 1 segundo por prueba)
- Sin dependencias externas
- Aisladas y determinísticas

### 2. Pruebas de Integración

Validan la interacción entre múltiples componentes.

**Ejemplos:**
- Entrenamiento completo con MLflow (test_mlflow_integration.py)
- Ejecución de experimentos con múltiples modelos (test_experiment_runner.py)
- Flujo de preprocesamiento, modelo y métricas (test_sklearn_pipeline.py)

**Características:**
- Más lentas (1-5 segundos por prueba)
- Integran múltiples módulos
- Usan datos sintéticos

### 3. Pruebas End-to-End

Validan el flujo completo del sistema desde inicio hasta fin.

**Ejemplos:**
- TestEndToEndPipeline::test_complete_pipeline_flow
- TestEndToEndPipeline::test_multiple_models_training_flow
- TestDataQualityValidation::test_pipeline_preserves_class_distribution

**Características:**
- Las más lentas (5-30 segundos)
- Simulan uso real del sistema
- Mayor confianza en la calidad del código

---

## Cobertura de Código

### Ejecutar pruebas con cobertura

```bash
pytest --cov=src --cov-report=html --cov-report=term
```

### Ver reporte HTML

```bash
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### Cobertura por archivo

```bash
pytest --cov=src --cov-report=term-missing
```

---

## Fixtures Compartidas

El archivo `conftest.py` define fixtures reutilizables para todas las pruebas.

### Fixtures Disponibles

| Fixture | Descripción | Uso |
|---------|-------------|-----|
| `test_data_dir` | Directorio temporal (sesión) | Almacenar datos temporales |
| `sample_music_dataframe` | DataFrame con 100 muestras | Pruebas de transformadores |
| `sample_csv_file` | CSV temporal con datos balanceados | Pruebas de pipelines |
| `sample_csv_with_missing_values` | CSV con valores faltantes | Pruebas de limpieza |
| `multiclass_dataset` | Dataset con 4 clases balanceadas | Validación de estratificación |
| `small_training_dataset` | Dataset pequeño (8 muestras) | Pruebas rápidas |
| `reset_mlflow` | Limpia estado de MLflow | Auto-aplicado a todos los tests |

### Ejemplo de Uso

```python
def test_my_function(sample_csv_file):
    # sample_csv_file es inyectado automáticamente
    df = pd.read_csv(sample_csv_file)
    assert len(df) == 100
```

---

## Mejores Prácticas

### 1. Nombres descriptivos

```python
# Correcto
def test_data_cleaner_replaces_invalid_strings_with_nan():

# Incorrecto
def test_cleaner():
```

### 2. Arrange-Act-Assert

```python
def test_accuracy_calculation():
    # Arrange
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]
    
    # Act
    accuracy = accuracy_score(y_true, y_pred)
    
    # Assert
    assert accuracy == 0.75
```

### 3. Un concepto por prueba

```python
# Correcto
def test_imputer_uses_median():
def test_imputer_handles_all_nan():

# Incorrecto
def test_imputer():  # Prueba demasiado grande
```

### 4. Usar fixtures para datos comunes

```python
# Correcto
def test_pipeline(sample_csv_file):

# Incorrecto
def test_pipeline():
    df = pd.DataFrame(...)  # Código duplicado
```

---

## Troubleshooting

### Error: "Import pytest could not be resolved"

**Solución:**
```bash
pip install pytest pytest-cov
```

### Error: "ModuleNotFoundError: No module named 'src'"

**Solución:** Ejecutar pytest desde la raíz del proyecto:
```bash
cd /path/to/MNA-MLOps-Proyecto-Equipo03
pytest -v
```

### Error: MLflow run no termina

**Solución:** El fixture reset_mlflow debería manejarlo, pero se puede forzar:
```python
import mlflow
mlflow.end_run()
```

### Tests muy lentos

**Solución:** Ejecutar solo pruebas unitarias:
```bash
pytest tests/test_transformers.py tests/test_metrics.py -v
```

---

## Métricas de Calidad

### Objetivos

- Cobertura de código: Mayor a 80%
- Tiempo de ejecución: Menor a 60 segundos (todas las pruebas)
- Tasa de éxito: 100%
- Pruebas por módulo: Mínimo 5 pruebas

### Estado actual

```bash
# Ejecutar para ver estadísticas
pytest --collect-only
```

---

## Resumen de Comandos

```bash
# Ejecutar todas las pruebas (silencioso)
pytest -q

# Ejecutar todas las pruebas (detallado)
pytest -v

# Ejecutar con cobertura
pytest --cov=src --cov-report=html --cov-report=term

# Ejecutar pruebas específicas
pytest tests/test_transformers.py -v

# Ejecutar con debugging
pytest -vv -s --tb=long

# Ver solo nombres de pruebas
pytest --collect-only

# Ejecutar en paralelo (requiere pytest-xdist)
pytest -n auto
```

---

## Referencias

- [Documentación de pytest](https://docs.pytest.org/)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [Best Practices for Testing ML Code](https://madewithml.com/courses/mlops/testing/)

---

## Checklist de Testing

- Pruebas unitarias de transformadores
- Pruebas unitarias de métricas
- Pruebas de integración del pipeline
- Pruebas E2E del flujo completo
- Pruebas de integración con MLflow
- Fixtures compartidas en conftest.py
- Documentación completa
- CI/CD integración (próximo paso)

---
