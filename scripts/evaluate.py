import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import keras
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             precision_recall_curve, ConfusionMatrixDisplay, RocCurveDisplay, roc_curve)
from visionguard.data import make_dataset

a = argparse.ArgumentParser()
a.add_argument("--run", required=True)
a.add_argument("--beta", type=float, default=0.5, help="<1 favours precision, >1 favours recall")
a.add_argument("--force", action="store_true")
a = a.parse_args()

run = Path("models") / a.run
if (run / "metrics.json").exists() and not a.force:
    raise SystemExit("Test set already evaluated for this run. The test set is used once.")
size = json.loads((run / "config.json").read_text())["img_size"]
model = keras.models.load_model(run / "best.keras")


def predict(split):
    df = pd.read_csv(f"data/splits/{split}.csv")
    return df, model.predict(make_dataset(f"data/splits/{split}.csv", size), verbose=0).ravel()


def scores(y, p, t):
    pred = (p >= t).astype(int)
    return dict(threshold=round(float(t), 4), accuracy=accuracy_score(y, pred),
                precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred, zero_division=0),
                f1=f1_score(y, pred, zero_division=0), roc_auc=roc_auc_score(y, p))


# ---- 1. threshold from VALIDATION only ----
val_df, val_p = predict("val")
prec, rec, thr = precision_recall_curve(val_df.y, val_p)
b2 = a.beta ** 2
fb = (1 + b2) * prec[:-1] * rec[:-1] / (b2 * prec[:-1] + rec[:-1] + 1e-12)
best_t = float(thr[int(np.argmax(fb))])
print(f"Chosen threshold (max F{a.beta} on val): {best_t:.4f}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(thr, prec[:-1], label="precision"); ax.plot(thr, rec[:-1], label="recall")
ax.plot(thr, fb, label=f"F{a.beta}"); ax.axvline(best_t, ls="--", c="red", label=f"chosen = {best_t:.2f}")
ax.set_xlabel("threshold"); ax.set_title(f"{a.run}: threshold sweep on VALIDATION"); ax.legend()
fig.tight_layout(); fig.savefig(run / "threshold_sweep.png", dpi=150); plt.close(fig)

# ---- 2. TEST set, once ----
test_df, test_p = predict("test")
y = test_df.y.values
m = scores(y, test_p, best_t); m05 = scores(y, test_p, 0.5)
pred = (test_p >= best_t).astype(int)

ConfusionMatrixDisplay.from_predictions(y, pred, display_labels=["Real", "AI"], cmap="Blues")
plt.title(f"{a.run}: test confusion matrix (t = {best_t:.2f})"); plt.tight_layout()
plt.savefig(run / "confusion_matrix.png", dpi=150); plt.close()

RocCurveDisplay.from_predictions(y, test_p, name=a.run)
fpr, tpr, t_roc = roc_curve(y, test_p); k = int(np.argmin(np.abs(t_roc - best_t)))
plt.scatter(fpr[k], tpr[k], c="red", zorder=3, label="chosen threshold")
plt.plot([0, 1], [0, 1], "k:"); plt.title(f"{a.run}: test ROC"); plt.legend(); plt.tight_layout()
plt.savefig(run / "roc_curve.png", dpi=150); plt.close()

test_df.assign(prob=test_p, pred=pred).to_csv(run / "test_predictions.csv", index=False)
(run / "metrics.json").write_text(json.dumps({"beta": a.beta, "chosen_threshold": m, "at_0.5": m05}, indent=2))

# ---- 3. comparison table across models ----
row = {"model": a.run, **{k: round(v, 4) for k, v in m.items()}}
table = Path("reports/metrics_table.csv")
old = pd.read_csv(table) if table.exists() else pd.DataFrame(columns=list(row))
pd.concat([old[old.model != a.run], pd.DataFrame([row])], ignore_index=True).to_csv(table, index=False)
print(pd.read_csv(table).to_string(index=False))
print("\nAt default 0.5 threshold:", {k: round(v, 4) for k, v in m05.items()})