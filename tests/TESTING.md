## Pruebas automatizadas

Este proyecto utiliza **pytest** para las pruebas unitarias y de integración.

### Estructura de pruebas

Las pruebas se encuentran en el directorio `tests/`:

- `tests/test_transformers.py` → pruebas unitarias de los transformadores personalizados.
- `tests/test_sklearn_pipeline.py` → pruebas unitarias e integración del `SklearnMLPipeline`.
- `tests/test_experiment_runner.py` → pruebas de integración del `ModelExperimentRunner`.

### Ejecución

Desde la raíz del proyecto, activa tu entorno conda y ejecuta:

```bash
pytest -q