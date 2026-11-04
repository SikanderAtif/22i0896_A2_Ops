import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
FIG = Path("reports/figures"); FIG.mkdir(parents=True, exist_ok=True)
ok = pd.read_csv("reports/audit/audit.csv").query("readable")
ok["aspect"] = ok.width / ok.height
classes = sorted(ok.label.unique())

# 1) sample grid
n = 8
fig, axes = plt.subplots(len(classes), n, figsize=(2.2 * n, 2.7 * len(classes)))
for i, c in enumerate(classes):
    for j, (_, r) in enumerate(ok[ok.label == c].sample(n, random_state=42).iterrows()):
        with Image.open(r.path) as im:
            im = im.convert("RGB"); im.thumbnail((256, 256))
        axes[i, j].imshow(im); axes[i, j].axis("off")
        axes[i, j].set_title(f"{c}\n{r.width}×{r.height} {r.format}", fontsize=8)
plt.suptitle("Random samples per class"); plt.tight_layout()
plt.savefig(FIG / "sample_grid.png", dpi=150); plt.close()

# 2) histograms
specs = [("width", "Width (px)", False), ("height", "Height (px)", False),
         ("file_size_kb", "File size (KB, log scale)", True), ("aspect", "Aspect ratio (w/h)", False)]
for col, xlabel, log in specs:
    fig, ax = plt.subplots(figsize=(7, 4))
    vals = ok[col]
    bins = np.logspace(np.log10(vals.min()), np.log10(vals.max()), 40) if log else 40
    for c in classes:
        ax.hist(ok.loc[ok.label == c, col], bins=bins, alpha=0.55, label=c)
    if log: ax.set_xscale("log")
    ax.set_xlabel(xlabel); ax.set_ylabel("Number of images"); ax.legend()
    ax.set_title(f"{xlabel.split(' (')[0]} distribution per class")
    plt.tight_layout(); plt.savefig(FIG / f"hist_{col}.png", dpi=150); plt.close()