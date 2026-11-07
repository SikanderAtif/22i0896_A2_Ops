import io
import pytest
from PIL import Image
from app import create_app


class FakePredictor:
    """Stands in for the real model, so tests run in CI without TensorFlow weights."""
    def predict(self, data):
        return {"label": "AI-generated", "p_ai": 0.95, "confidence": 0.76, "threshold": 0.79,
                "model_name": "test-model", "model_version": "0", "latency_ms": 1.0}

    def info(self):
        return {"name": "test-model", "version": "0", "threshold": 0.79, "img_size": 224}


@pytest.fixture
def app():
    return create_app(predictor=FakePredictor(),
                      config_overrides={"SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:", "TESTING": True})


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (200, 30, 30)).save(buf, "PNG")
    return buf.getvalue()