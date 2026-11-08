# VisionGuard model API: champion model built in, CPU TensorFlow, Gunicorn
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 TF_CPP_MIN_LOG_LEVEL=2 PORT=8080

WORKDIR /srv
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt
RUN useradd --create-home --uid 1000 apiuser

COPY visionguard/ visionguard/
COPY api/ api/
COPY model/champion.keras model/metadata.json model/

USER apiuser
EXPOSE 8080
# shell form so ${PORT} is read at run time: cloud platforms set PORT themselves
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "4", "--timeout", "120", "--access-logfile", "-", "api.main:create_api()"]