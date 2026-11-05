import pandas as pd
from pathlib import Path
from visionguard.models import build_baseline_cnn

cnn = build_baseline_cnn()
cnn.summary()
rows = [dict(layer=l.name, type=type(l).__name__,
             output_shape=str(tuple(l.output.shape)), params=l.count_params())
        for l in cnn.layers]
df = pd.DataFrame(rows)
print(df.to_string(index=False)); print("Total params:", df.params.sum())
Path("reports").mkdir(exist_ok=True); df.to_csv("reports/cnn_layers.csv", index=False)