FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

ENV MVP_MODEL_ARTIFACT_PATH=/app/mlops/artifacts/24-nn-1

EXPOSE 8000

CMD ["gunicorn", "-b", "0.0.0.0:8000", "flask_app:create_app()"]
