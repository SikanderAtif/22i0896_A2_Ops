import argparse
from mlflow import MlflowClient
from visionguard import tracking

a = argparse.ArgumentParser(); a.add_argument("--version", required=True); a = a.parse_args()
tracking.setup("dagshub")
c = MlflowClient()
for v in sorted(c.search_model_versions(f"name='{tracking.REGISTERED_MODEL}'"), key=lambda v: int(v.version)):
    run = c.get_run(v.run_id)
    print(f"version {v.version:>2}  {run.info.run_name:28s} "
          f"val_auc={run.data.metrics.get('final_val_roc_auc', float('nan')):.4f}")
c.set_registered_model_alias(tracking.REGISTERED_MODEL, "champion", a.version)
print(f"\nAlias 'champion' -> version {a.version}")