# VisionGuard web app: Flask + Gunicorn + CPU TensorFlow
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# 1) Dependencies FIRST: this slow layer is reused from cache until requirements-serve.txt changes
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

# 2) Non-root user, and the folders the app may write to
RUN useradd --create-home --uid 1000 appuser \
 && mkdir -p /app/instance /app/model \
 && chown -R appuser:appuser /app/instance

# 3) Source code LAST: it changes often, and rebuilding these layers takes seconds
COPY visionguard/ visionguard/
COPY app/ app/

USER appuser
EXPOSE 8000

# Docker checks this every 30 s; the 60 s grace period covers TensorFlow loading the model
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# Gunicorn: a production WSGI server instead of Flask's development server
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "4", "--timeout", "120", "app:create_app()"]