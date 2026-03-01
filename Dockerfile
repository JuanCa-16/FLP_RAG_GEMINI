FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencias necesarias para numpy, pandas y compilación
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        gcc \
        g++ \
        python3-dev \
        libffi-dev \
        libssl-dev \
        libopenblas-dev \
        libpq-dev \
        curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalarlos
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --force-reinstall -r requirements.txt

# Copiar todo el código
COPY . .

EXPOSE 10000

# Tu app está en app.py y la instancia se llama "app"
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
