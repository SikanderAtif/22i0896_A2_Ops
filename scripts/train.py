import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import keras
from visionguard.data import make_dataset
from visionguard.models import (build_baseline_cnn, build_transfer, compile_model,
                                unfreeze_last, last_block_size)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["cnn", "mobilenetv2", "efficientnetb0"], default="cnn")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ft-epochs", type=int, default=15)
    p.add_argument("--ft-lr", type=float, default=1e-5)
    p.add_argument("--unfreeze", type=int, default=-1, help="-1 = last block, 0 = no fine-tuning")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", default=None)
    return p.parse_args()


def make_callbacks(ckpt, initial_best=None):
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True,
                                        initial_value_threshold=initial_best, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2,
                                          min_lr=1e-7, verbose=1),
    ]


def class_weights(csv):
    y = pd.read_csv(csv)["y"]
    return {int(c): len(y) / (2 * n) for c, n in y.value_counts().items()}


def trainable_report(model, stage, out):
    names = []
    for layer in model.layers:
        if isinstance(layer, keras.Model):          # the backbone
            names += [f"{layer.name}/{s.name}" for s in layer.layers if s.trainable_weights]
        elif layer.trainable_weights:
            names.append(layer.name)
    n = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    text = f"== {stage}: {len(names)} trainable layers, {n:,} trainable params ==\n" + "\n".join(names)
    print(text); (out / f"trainable_{stage.replace(' ', '_').lower()}.txt").write_text(text)


def plot_history(h, title, path, first_epoch=1, boundary=None):
    epochs = range(first_epoch, first_epoch + len(h["loss"]))
    best = first_epoch + int(np.argmin(h["val_loss"]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, m in zip(axes, ["loss", "accuracy"]):
        ax.plot(epochs, h[m], label=f"train {m}")
        ax.plot(epochs, h[f"val_{m}"], label=f"val {m}")
        ax.axvline(best, ls="--", c="red", label=f"best epoch = {best}")
        if boundary: ax.axvline(boundary + 0.5, ls=":", c="gray", label="fine-tuning starts")
        ax.set_xlabel("epoch"); ax.set_title(m); ax.legend()
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return best


def main():
    a = parse()
    keras.utils.set_random_seed(a.seed)               # Python, NumPy and TF seeds in one call
    name = a.run_name or f"{a.model}_{a.img_size}_{time.strftime('%Y%m%d-%H%M%S')}"
    out = Path("models") / name; out.mkdir(parents=True, exist_ok=True)
    ckpt = str(out / "best.keras")

    train_ds = make_dataset("data/splits/train.csv", a.img_size, a.batch_size,
                            training=True, augment=not a.no_augment, seed=a.seed)
    val_ds = make_dataset("data/splits/val.csv", a.img_size, a.batch_size)
    cw = class_weights("data/splits/train.csv"); print("class weights:", cw)

    boundary = None
    if a.model == "cnn":
        model = build_baseline_cnn(a.img_size, a.dropout)
        compile_model(model, a.lr); model.summary()
        h = model.fit(train_ds, validation_data=val_ds, epochs=a.epochs,
                      class_weight=cw, callbacks=make_callbacks(ckpt)).history
    else:
        model, base = build_transfer(a.model, a.img_size, a.dropout)
        compile_model(model, a.lr); model.summary()
        trainable_report(model, "Stage 1 frozen", out)
        h1 = model.fit(train_ds, validation_data=val_ds, epochs=a.epochs,
                       class_weight=cw, callbacks=make_callbacks(ckpt)).history
        plot_history(h1, f"{name} - stage 1 (frozen backbone)", out / "curves_stage1.png")
        h = h1
        n = last_block_size(base, a.model) if a.unfreeze < 0 else a.unfreeze
        if n > 0:
            boundary = len(h1["loss"])
            unfreeze_last(base, n)
            compile_model(model, a.ft_lr)            # must recompile after changing trainable
            trainable_report(model, "Stage 2 fine-tune", out)
            h2 = model.fit(train_ds, validation_data=val_ds, epochs=boundary + a.ft_epochs,
                           initial_epoch=boundary, class_weight=cw,
                           callbacks=make_callbacks(ckpt, min(h1["val_loss"]))).history
            plot_history(h2, f"{name} - stage 2 (last {n} layers unfrozen)",
                         out / "curves_stage2.png", first_epoch=boundary + 1)
            h = {k: h1[k] + h2.get(k, []) for k in h1}

    h = {k: [float(v) for v in vals] for k, vals in h.items()}
    best = plot_history(h, f"{name} - all epochs", out / "curves.png", boundary=boundary)
    cfg = vars(a) | {"run_name": name, "best_epoch": best, "best_val_loss": min(h["val_loss"])}
    (out / "history.json").write_text(json.dumps(h, indent=2))
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"\nBest epoch: {best}  (val_loss = {min(h['val_loss']):.4f})  -> {ckpt}")


if __name__ == "__main__":
    main()