# VisionGuard (22i0896_A2_Ops)

![CI](https://github.com/SikanderAtif/22i0896_A2_Ops/actions/workflows/ci.yml/badge.svg)

Classifies an uploaded image as AI-generated or real, with a confidence value.
TensorFlow/Keras model tracked with MLflow on DagsHub, served by Flask + Gunicorn,
with every prediction logged in PostgreSQL.

## Run with Docker
1. Copy `.env.example` to `.env` and fill in the values.
2. Download the champion model once: `python -m scripts.export_champion`
3. Start everything: `docker compose up -d --build`
4. Open http://localhost:8000