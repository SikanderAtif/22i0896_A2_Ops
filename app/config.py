import os


class Config:
    """Every setting comes from an environment variable, with a safe default."""
    def __init__(self):
        self.MODEL_PATH = os.getenv("MODEL_PATH", "model/champion.keras")
        self.MODEL_METADATA = os.getenv("MODEL_METADATA", "model/metadata.json")
        t = os.getenv("DECISION_THRESHOLD")                 # optional override of the model's threshold
        self.DECISION_THRESHOLD = float(t) if t else None
        self.SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", "sqlite:///visionguard.db")
        self.MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "5"))
        self.MAX_UPLOAD_BYTES = int(self.MAX_UPLOAD_MB * 1024 * 1024)
        # transport limit is larger, so the app itself can answer with a friendly 413 page
        self.MAX_CONTENT_LENGTH = self.MAX_UPLOAD_BYTES * 4
        self.SECRET_KEY = os.getenv("SECRET_KEY", "dev-only-change-me")
        self.HISTORY_PAGE_SIZE = int(os.getenv("HISTORY_PAGE_SIZE", "25"))