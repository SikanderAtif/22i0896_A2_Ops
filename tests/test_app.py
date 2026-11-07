import io
from app.db import db, Prediction


def upload(client, data, name):
    return client.post("/predict", data={"image": (io.BytesIO(data), name)},
                       content_type="multipart/form-data")


def count_rows(app):
    with app.app_context():
        return db.session.scalar(db.select(db.func.count()).select_from(Prediction))


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"
    assert r.get_json()["model"]["version"] == "0"


def test_valid_upload_is_recorded_and_shown(app, client, png_bytes):
    r = upload(client, png_bytes, "cat.png")
    assert r.status_code == 302 and "/result/1" in r.headers["Location"]
    assert count_rows(app) == 1
    assert b"AI-generated" in client.get("/result/1").data
    assert b"cat.png" in client.get("/history").data


def test_wrong_file_type_rejected(app, client):
    assert upload(client, b"just text", "notes.txt").status_code == 415
    assert count_rows(app) == 0          # rejected uploads leave no record


def test_undecodable_image_rejected(client, png_bytes):
    assert upload(client, png_bytes[:40], "broken.png").status_code == 422


def test_too_large_rejected(app, client, png_bytes):
    app.config["MAX_UPLOAD_BYTES"] = 50
    assert upload(client, png_bytes, "big.png").status_code == 413


def test_filename_is_sanitized(app, client, png_bytes):
    upload(client, png_bytes, "../../evil name.png")
    with app.app_context():
        assert db.session.scalars(db.select(Prediction.filename)).first() == "evil_name.png"