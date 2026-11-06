import os
from dotenv import load_dotenv
from flask import Flask
from visionguard.inference import Predictor
from .config import Config
from .db import db
from .routes import bp, register_error_handlers
from .uploads import MemoryOnlyRequest


def create_app():
    load_dotenv()
    app = Flask(__name__, instance_relative_config=True)
    app.request_class = MemoryOnlyRequest
    cfg = Config()
    app.config.from_object(cfg)
    os.makedirs(app.instance_path, exist_ok=True)       # SQLite file lives in instance/

    db.init_app(app)
    with app.app_context():
        db.create_all()

    # the model is loaded ONCE here, at start-up
    app.extensions["predictor"] = Predictor(cfg.MODEL_PATH, cfg.MODEL_METADATA, cfg.DECISION_THRESHOLD)

    @app.context_processor
    def inject_model():
        return {"model_info": app.extensions["predictor"].info()}

    app.register_blueprint(bp)
    register_error_handlers(app)
    return app