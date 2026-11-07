import io
import pytest
from api.main import create_api
from tests.conftest import FakePredictor

KEY = "test-key"


@pytest.fixture
def api_client():
    return create_api(predictor=FakePredictor(), api_key=KEY).test_client()


def post(client, data, name, key=KEY):
    headers = {"X-API-Key": key} if key else {}
    return client.post("/predict", data={"image": (io.BytesIO(data), name)},
                       headers=headers, content_type="multipart/form-data")


def test_health_needs_no_key(api_client):
    r = api_client.get("/health")
    assert r.status_code == 200 and r.get_json()["model"]["version"] == "0"


def test_predict_requires_key(api_client, png_bytes):
    assert post(api_client, png_bytes, "a.png", key=None).status_code == 401
    assert post(api_client, png_bytes, "a.png", key="wrong").status_code == 401


def test_predict_returns_json_verdict(api_client, png_bytes):
    r = post(api_client, png_bytes, "a.png")
    body = r.get_json()
    assert r.status_code == 200 and body["label"] == "AI-generated"
    assert body["image"]["width"] == 64 and body["image"]["format"] == "PNG"


def test_bad_upload_gives_json_error(api_client):
    r = post(api_client, b"text", "notes.txt")
    assert r.status_code == 415 and r.get_json()["error"] == "Unsupported file type"


def test_refuses_to_start_without_key(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        create_api(predictor=FakePredictor())