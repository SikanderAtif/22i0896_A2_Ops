import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             precision_recall_curve, roc_curve, ConfusionMatrixDisplay, RocCurveDisplay)
from visionguard.data import make_dataset


def predict_split(model, split, img_size):
    csv = f"data/splits/{split}.csv"
    return pd.read_csv(csv), model.predict(make_dataset(csv, img_size), verbose=0).ravel()


def scores(y, p, t, prefix):
    pred = (p >= t).astype(int)
    return {f"{prefix}_accuracy": accuracy_score(y, pred),
            f"{prefix}_precision": precision_score(y, pred, zero_division=0),
            f"{prefix}_recall": recall_score(y, pred, zero_division=0),
            f"{prefix}_f1": f1_score(y, pred, zero_division=0),
            f"{prefix}_roc_auc": roc_auc_score(y, p)}


def full_evaluation(model, img_size, out, name, beta=0.5):
    # threshold from VALIDATION only (max F-beta, beta<1 favours precision)
    val_df, val_p = predict_split(model, "val", img_size)
    prec, rec, thr = precision_recall_curve(val_df.y, val_p)
    b2 = beta ** 2
    fb = (1 + b2) * prec[:-1] * rec[:-1] / (b2 * prec[:-1] + rec[:-1] + 1e-12)
    t = float(thr[int(np.argmax(fb))])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thr, prec[:-1], label="precision"); ax.plot(thr, rec[:-1], label="recall")
    ax.plot(thr, fb, label=f"F{beta}"); ax.axvline(t, ls="--", c="red", label=f"chosen = {t:.2f}")
    ax.set_xlabel("threshold"); ax.set_title(f"{name}: threshold sweep on VALIDATION"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "threshold_sweep.png", dpi=150); plt.close(fig)

    # TEST set, once per run
    test_df, test_p = predict_split(model, "test", img_size)
    y, pred = test_df.y.values, (test_p >= t).astype(int)

    ConfusionMatrixDisplay.from_predictions(y, pred, display_labels=["Real", "AI"], cmap="Blues")
    plt.title(f"{name}: test confusion matrix (t = {t:.2f})"); plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=150); plt.close()

    RocCurveDisplay.from_predictions(y, test_p, name=name)
    fpr, tpr, t_roc = roc_curve(y, test_p); k = int(np.argmin(np.abs(t_roc - t)))
    plt.scatter(fpr[k], tpr[k], c="red", zorder=3, label="chosen threshold")
    plt.plot([0, 1], [0, 1], "k:"); plt.title(f"{name}: test ROC"); plt.legend(); plt.tight_layout()
    plt.savefig(out / "roc_curve.png", dpi=150); plt.close()

    test_df.assign(prob=test_p, pred=pred).to_csv(out / "test_predictions.csv", index=False)
    metrics = {"decision_threshold": t,
               **scores(val_df.y.values, val_p, t, "final_val"),
               **scores(y, test_p, t, "test")}
    figs = [out / "threshold_sweep.png", out / "confusion_matrix.png", out / "roc_curve.png"]
    return metrics, figs