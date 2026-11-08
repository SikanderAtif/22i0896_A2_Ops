import os
from dotenv import load_dotenv
from flask import Flask
from .config import Config
from .db import db
from .routes import bp, register_error_handlers
from visionguard.uploads import MemoryOnlyRequest

def create_app(predictor=None, config_overrides=None):
    load_dotenv()
    app = Flask(__name__, instance_relative_config=True)
    app.request_class = MemoryOnlyRequest
    cfg = Config()
    app.config.from_object(cfg)
    if config_overrides:
        app.config.update(config_overrides)
    os.makedirs(app.instance_path, exist_ok=True)

    db.init_app(app)
    with app.app_context():
        db.create_all()

    if predictor is None:
        if app.config["INFERENCE_MODE"] == "remote":
            from visionguard.remote import RemotePredictor
            predictor = RemotePredictor(app.config["MODEL_API_URL"], app.config["MODEL_API_KEY"],
                                        app.config["MODEL_API_TIMEOUT"])
        else:
            from visionguard.inference import Predictor        # imports TensorFlow only in local mode
            predictor = Predictor(app.config["MODEL_PATH"], app.config["MODEL_METADATA"],
                                  app.config["DECISION_THRESHOLD"])
    app.extensions["predictor"] = predictor

    @app.context_processor
    def inject_model():
        return {"model_info": app.extensions["predictor"].info()}

    app.register_blueprint(bp)
    register_error_handlers(app)
    return app