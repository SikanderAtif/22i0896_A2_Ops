"""VisionGuard model API: the champion model behind a key-protected HTTP endpoint."""
import hmac
import os
from flask import Flask, jsonify, request
from visionguard.inference import Predictor
from visionguard.uploads import MemoryOnlyRequest, UploadError, read_upload


def create_api(predictor=None, api_key=None):
    app = Flask(__name__)
    app.request_class = MemoryOnlyRequest            # uploads stay in memory, never on disk

    api_key = api_key or os.environ.get("API_KEY", "")
    if not api_key:
        raise RuntimeError("API_KEY is not set - refusing to start an unprotected model API.")
    max_bytes = int(float(os.getenv("MAX_UPLOAD_MB", "5")) * 1024 * 1024)
    app.config["MAX_CONTENT_LENGTH"] = max_bytes * 4

    # model loaded ONCE at start-up
    pr = predictor or Predictor(os.getenv("MODEL_PATH", "model/champion.keras"),
                                os.getenv("MODEL_METADATA", "model/metadata.json"))

    def error(status, title, message):
        return jsonify({"status": status, "error": title, "message": message}), status

    @app.get("/")
    def index():
        return jsonify({"service": "VisionGuard model API", "project": "22i0896_A2_Ops",
                        "endpoints": {"GET /health": "model status (no key needed)",
                                      "POST /predict": "multipart field 'image' + header X-API-Key"}})

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "model": pr.info()})

    @app.post("/predict")
    def predict():
        sent = request.headers.get("X-API-Key", "")
        if not hmac.compare_digest(sent.encode(), api_key.encode()):     # constant-time comparison
            return error(401, "Unauthorized", "Missing or wrong X-API-Key header.")
        up = read_upload(request.files.get("image"), max_bytes)
        result = pr.predict(up["data"])
        result["image"] = {k: up[k] for k in ("filename", "size", "width", "height", "format", "sha256")}
        return jsonify(result)

    @app.errorhandler(UploadError)
    def upload_error(e):
        return error(e.status, e.title, e.message)

    for code, title in [(404, "Not found"), (405, "Method not allowed"),
                        (413, "File too large"), (500, "Internal error")]:
        app.register_error_handler(code, lambda e, c=code, t=title: error(c, t, "See the API documentation at /."))
    return app