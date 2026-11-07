"""MLflow setup, reproducibility tags and the manual per-epoch logger."""
import os
import subprocess
from pathlib import Path
import yaml
import mlflow
import keras
from dotenv import load_dotenv

EXPERIMENT = "22i0896_A2_Ops"
REGISTERED_MODEL = "22i0896_A2_Ops"


def setup(target="dagshub", experiment=EXPERIMENT):
    if target == "dagshub":
        load_dotenv()
        uri = os.getenv("MLFLOW_TRACKING_URI")
        if not uri:
            raise SystemExit("MLFLOW_TRACKING_URI is not set - create .env (see .env.example)")
        mlflow.set_tracking_uri(uri)
    else:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment)
    print("MLflow tracking to:", mlflow.get_tracking_uri())


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return "unknown"


def dvc_md5(dvc_file):
    """The data version: the hash stored in a DVC pointer file."""
    try:
        return yaml.safe_load(Path(dvc_file).read_text())["outs"][0]["md5"]
    except Exception:
        return "unknown"


def standard_tags(seed, mode):
    return {"git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": str(_git("status", "--porcelain") != ""),
            "dvc_clean_md5": dvc_md5("data/clean.dvc"),
            "dvc_raw_md5": dvc_md5("data/raw.dvc"),
            "seed": str(seed), "tracking_mode": mode, "student": "22i0896"}


class MLflowEpochLogger(keras.callbacks.Callback):
    """Manual logging: every Keras metric, every epoch."""
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({k: float(v) for k, v in (logs or {}).items()}, step=epoch + 1)