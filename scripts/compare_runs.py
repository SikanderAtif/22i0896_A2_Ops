import mlflow, pandas as pd, matplotlib.pyplot as plt
from visionguard import tracking

tracking.setup("dagshub")
runs = mlflow.search_runs(experiment_names=[tracking.EXPERIMENT])
runs = runs.dropna(subset=["metrics.final_val_roc_auc"])
cols = {"tags.mlflow.runName": "run", "tags.note": "change",
        "metrics.final_val_roc_auc": "val_auc", "metrics.final_val_precision": "val_precision",
        "metrics.final_val_f1": "val_f1", "metrics.best_val_loss": "best_val_loss",
        "metrics.test_roc_auc": "test_auc", "metrics.test_precision": "test_precision",
        "metrics.test_recall": "test_recall", "metrics.test_f1": "test_f1",
        "metrics.test_accuracy": "test_accuracy"}
t = runs[list(cols)].rename(columns=cols).sort_values("val_auc", ascending=False)
t = t.round(4); t.to_csv("reports/run_comparison.csv", index=False)
print(t.to_string(index=False))

ax = t.set_index("run")[["val_auc", "test_auc"]].plot.bar(figsize=(10, 4), rot=30)
ax.set_ylim(0.5, 1.0); ax.set_ylabel("ROC-AUC"); ax.set_title("VisionGuard runs: validation vs test AUC")
plt.tight_layout(); plt.savefig("reports/figures/run_comparison.png", dpi=150)