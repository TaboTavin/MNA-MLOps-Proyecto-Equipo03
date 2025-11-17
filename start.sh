#!/bin/bash

# Terminar el script si cualquier comando falla
set -e

echo "Iniciando proceso MLOps en Docker..."

# 1. Correr el entrenamiento
echo "Entrenando modelo (esto puede tardar unos minutos)..."
python main.py
echo "Entrenamiento finalizado."

# 2. Iniciar la API
echo "Iniciando servidor API..."
exec python -m uvicorn api:app --host 0.0.0.0 --port 8000