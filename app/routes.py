from flask import Blueprint, current_app, jsonify, redirect, render_template, request, url_for
from sqlalchemy import text
from .db import db, Prediction
from visionguard.uploads import UploadError, read_upload

bp = Blueprint("main", __name__)


@bp.get("/")
def index():
    return render_template("index.html", max_mb=current_app.config["MAX_UPLOAD_MB"])


@bp.post("/predict")
def predict():
    up = read_upload(request.files.get("image"), current_app.config["MAX_UPLOAD_BYTES"])
    r = current_app.extensions["predictor"].predict(up["data"])
    rec = Prediction(filename=up["filename"], file_size_bytes=up["size"], image_sha256=up["sha256"],
                     image_width=up["width"], image_height=up["height"], image_format=up["format"],
                     label=r["label"], p_ai=r["p_ai"], confidence=r["confidence"],
                     threshold=r["threshold"], model_name=r["model_name"],
                     model_version=r["model_version"], latency_ms=r["latency_ms"])
    db.session.add(rec)
    db.session.commit()
    return redirect(url_for("main.result", pred_id=rec.id))      # Post/Redirect/Get


@bp.get("/result/<int:pred_id>")
def result(pred_id):
    return render_template("result.html", p=db.get_or_404(Prediction, pred_id))


@bp.get("/history")
def history():
    page = request.args.get("page", 1, type=int)
    q = db.select(Prediction).order_by(Prediction.created_at.desc())
    pagination = db.paginate(q, page=page, per_page=current_app.config["HISTORY_PAGE_SIZE"],
                             error_out=False)
    return render_template("history.html", pagination=pagination)


@bp.get("/health")
def health():
    try:
        db.session.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db.session.rollback()
        db_ok = False
    body = {"status": "ok" if db_ok else "degraded",
            "model_loaded": True,
            "model": current_app.extensions["predictor"].info(),
            "database": "ok" if db_ok else "unreachable",
            "debug": current_app.debug}
    return jsonify(body), (200 if db_ok else 503)


ERRORS = {400: ("Bad request", "The request could not be understood."),
          404: ("Page not found", "That page does not exist."),
          405: ("Method not allowed", "That action is not allowed on this page."),
          413: ("File too large", "The upload is larger than the server accepts."),
          500: ("Something went wrong", "An unexpected error occurred. Please try again.")}


def register_error_handlers(app):
    @app.errorhandler(UploadError)
    def upload_error(e):
        return render_template("error.html", status=e.status, title=e.title, message=e.message), e.status

    for code, (title, message) in ERRORS.items():
        def handler(e, code=code, title=title, message=message):
            if code == 500:
                db.session.rollback()
            return render_template("error.html", status=code, title=title, message=message), code
        app.register_error_handler(code, handler)