import keras, keras_tuner as kt, pandas as pd, numpy as np
from scipy.stats import spearmanr
from visionguard.data import make_dataset
from visionguard.models import build_transfer, compile_model, unfreeze_last, last_block_size

keras.utils.set_random_seed(42)
train_ds = make_dataset("data/splits/train.csv", training=True)
val_ds = make_dataset("data/splits/val.csv")

LAST = last_block_size(keras.applications.MobileNetV2(include_top=False, weights=None,
                                                      input_shape=(224, 224, 3)), "mobilenetv2")
UNFREEZE_CHOICES = [0, LAST, 30, 60]   # 0 = frozen backbone, LAST = only the last block


def build(hp):
    model, base = build_transfer("mobilenetv2", 224, hp.Float("dropout", 0.1, 0.5, step=0.1))
    unfreeze_last(base, hp.Choice("unfreeze", UNFREEZE_CHOICES))
    compile_model(model, hp.Float("lr", 1e-5, 1e-3, sampling="log"))
    return model


tuner = kt.RandomSearch(build, objective="val_loss", max_trials=10, seed=42,
                        directory="kt", project_name="visionguard", overwrite=True)
tuner.search(train_ds, validation_data=val_ds, epochs=8,
             callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])

rows = [dict(trial=t.trial_id, **t.hyperparameters.values, val_loss=round(t.score, 4))
        for t in tuner.oracle.get_best_trials(num_trials=100) if t.score is not None]
df = pd.DataFrame(rows).sort_values("val_loss")
df.to_csv("reports/tuner_visionguard.csv", index=False)
print(df.to_string(index=False))

print("\n== Which hyperparameter mattered most? ==")
print("Mean val_loss per 'unfreeze' value:\n", df.groupby("unfreeze").val_loss.agg(["mean", "count"]))
for hp in ["lr", "dropout"]:
    rho, _ = spearmanr(df[hp], df.val_loss)
    print(f"Spearman correlation of {hp} with val_loss: {rho:+.2f}")
print("\nBEST:", tuner.get_best_hyperparameters(1)[0].values)