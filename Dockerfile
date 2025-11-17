FROM python:3.11-slim

WORKDIR /app

# Instalamos dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos todo el código (incluyendo el start.sh que acabamos de crear)
COPY . .

# Aseguramos que mlruns use rutas locales dentro del contenedor
ENV MLFLOW_TRACKING_URI="file:///app/mlruns"

# Exponemos el puerto
EXPOSE 8000

# Damos permisos de ejecución al script
RUN chmod +x start.sh

# CMD ahora ejecuta nuestro script en lugar de uvicorn directamente
CMD ["./start.sh"]