FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencias necesarias para pandas, numpy y compilación
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        gcc \
        g++ \
        python3-dev \
        libatlas-base-dev \
        libopenblas-dev \
        libpq-dev \
        curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# requirements
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# copiar código
COPY . .

EXPOSE 10000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
