from datetime import datetime, timezone
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Prediction(db.Model):
    """One row per request: what was decided, when, by which model, how confidently."""
    __tablename__ = "predictions"
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, index=True,
                           default=lambda: datetime.now(timezone.utc))
    filename = db.Column(db.String(120), nullable=False)          # sanitized name only
    file_size_bytes = db.Column(db.Integer, nullable=False)
    image_sha256 = db.Column(db.String(64), nullable=False)       # identifies the image without storing it
    image_width = db.Column(db.Integer, nullable=False)
    image_height = db.Column(db.Integer, nullable=False)
    image_format = db.Column(db.String(16), nullable=False)
    label = db.Column(db.String(20), nullable=False)              # what was decided
    p_ai = db.Column(db.Float, nullable=False)
    confidence = db.Column(db.Float, nullable=False)              # how confidently
    threshold = db.Column(db.Float, nullable=False)
    model_name = db.Column(db.String(100), nullable=False)        # by which model
    model_version = db.Column(db.String(20), nullable=False)
    latency_ms = db.Column(db.Float, nullable=False)