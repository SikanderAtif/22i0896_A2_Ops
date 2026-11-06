"""Download the @champion model from the DagsHub registry into model/ (run once, not per request)."""
import json
from datetime import datetime, timezone
from pathlib import Path
import mlflow
from mlflow import MlflowClient
from visionguard import tracking

tracking.setup("dagshub")
c = MlflowClient()
mv = c.get_model_version_by_alias(tracking.REGISTERED_MODEL, "champion")
run = c.get_run(mv.run_id)
model = mlflow.keras.load_model(f"models:/{tracking.REGISTERED_MODEL}@champion")

out = Path("model"); out.mkdir(exist_ok=True)
model.save(out / "champion.keras")
meta = {"model_name": tracking.REGISTERED_MODEL, "version": mv.version, "alias": "champion",
        "run_id": mv.run_id, "run_name": run.info.run_name,
        "threshold": run.data.metrics["decision_threshold"],
        "img_size": int(model.input_shape[1]),
        "val_roc_auc": run.data.metrics.get("final_val_roc_auc"),
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
(out / "metadata.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))