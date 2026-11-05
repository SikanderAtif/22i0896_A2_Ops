import argparse, json
from pathlib import Path
import pandas as pd, matplotlib.pyplot as plt
from PIL import Image

a = argparse.ArgumentParser(); a.add_argument("--run", required=True); a = a.parse_args()
run = Path("models") / a.run
thr = json.loads((run / "metrics.json").read_text())["chosen_threshold"]["threshold"]
df = pd.read_csv(run / "test_predictions.csv")
audit = (pd.read_csv("reports/audit/audit.csv")
         .rename(columns={"path": "source"})[["source", "width", "height", "format", "mode", "file_size_kb"]])
df = df.merge(audit, on="source", how="left")

wrong = df[df.pred != df.y].copy()
print(f"Misclassified: {len(wrong)} of {len(df)}")
wrong["margin"] = (wrong.prob - thr).abs()
wrong = wrong.sort_values("margin", ascending=False).head(10).reset_index(drop=True)

name = lambda v: "AI" if v == 1 else "Real"
fig, axes = plt.subplots(2, 5, figsize=(17, 8))
for ax in axes.flat: ax.axis("off")
for i, r in wrong.iterrows():
    ax = axes.flat[i]
    with Image.open(r.filepath) as im: ax.imshow(im.convert("RGB"))
    ax.set_title(f"#{i+1} true={name(r.y)} pred={name(r.pred)}\n"
                 f"p(AI)={r.prob:.2f}  orig {r.width}×{r.height} {r.format}", fontsize=9)
fig.suptitle(f"{a.run}: most confident test errors (threshold {thr:.2f})"); fig.tight_layout()
fig.savefig(run / "misclassified.png", dpi=150)
wrong.drop(columns=["margin"]).to_csv(run / "misclassified.csv", index=False)
print(wrong[["filepath", "y", "prob", "width", "height", "format", "file_size_kb"]].to_string())