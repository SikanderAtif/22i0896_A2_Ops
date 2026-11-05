import numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from visionguard.data import make_dataset

ds = make_dataset("data/splits/train.csv", batch_size=16, training=True)
x, y = next(iter(ds))
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(np.clip(x[i].numpy() / 255.0, 0, 1))
    ax.set_title("AI" if y[i, 0] == 1 else "Real"); ax.axis("off")
plt.suptitle("One augmented training batch (224×224)"); plt.tight_layout()
Path("reports/figures").mkdir(parents=True, exist_ok=True)
plt.savefig("reports/figures/augmented_batch.png", dpi=150)