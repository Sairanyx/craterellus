"""Custom binary CNN (M2): the tuning journey.

Version 1 (runs/cnn):
  1. baseline_whole    whole photos at 224, no regularization, fixed epochs
  2. baseline_tiles    480 px grid tiles at 224, no regularization            -> did not learn (loss stuck at ln 2)
  3. final_tiles       tiles + Dropout + L2 + augmentation + EarlyStopping/ModelCheckpoint
Version 2 (runs/cnn_v2): mushroom-centred patches (make_patches.py) instead of grid tiles
  1. baseline_patches  patches, no regularization, fixed epochs, lr 1e-4 (1e-3 did not converge in v1)
  2. final_patches     patches + BatchNorm + Dropout + L2 + augmentation + EarlyStopping/ModelCheckpoint
Version 2.1 (runs/cnn_v2_1): final_patches stopped too early on a noisy val curve, so
  1. final_patches     same model, EarlyStopping patience 15 + ReduceLROnPlateau

uv run src/cnn_train.py --work DL_project/work --version v2

Writes <work>/runs/<cnn|cnn_v2>/<experiment>/  model .h5, history.csv, curves.png
       <work>/runs/<cnn|cnn_v2>/tuning_table.csv
"""
import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")  # Keras 3 on PyTorch -> runs on the GPU (CUDA torch is installed for YOLO)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import keras  # load before TensorFlow, so PyTorch gets the GPU's CUDA libraries first
from keras import layers, regularizers
import tensorflow as tf  # only for the image data pipeline

SIZE = 224
BATCH = 32


def load(folder, split, shuffle):
    return keras.utils.image_dataset_from_directory(
        folder / split, labels="inferred", label_mode="binary",
        class_names=["no_mushroom", "mushroom"], image_size=(SIZE, SIZE),
        batch_size=BATCH, shuffle=shuffle, seed=0)


def build(regularized, batchnorm=False, lr=1e-3):
    """4 conv blocks (32-64-128-128) -> dense -> sigmoid. Same network in every experiment."""
    l2 = regularizers.l2(1e-4) if regularized else None
    model = keras.Sequential([keras.Input((SIZE, SIZE, 3)), layers.Rescaling(1 / 255)])
    for filters in (32, 64, 128, 128):
        if batchnorm:
            model.add(layers.Conv2D(filters, 3, padding="same", use_bias=False))
            model.add(layers.BatchNormalization())  # keeps activations in range, so training actually moves
            model.add(layers.ReLU())
        else:
            model.add(layers.Conv2D(filters, 3, padding="same", activation="relu"))
        model.add(layers.MaxPooling2D())
    model.add(layers.GlobalAveragePooling2D() if regularized else layers.Flatten())
    if regularized:
        model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation="relu", kernel_regularizer=l2))
    if regularized:
        model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation="sigmoid", kernel_regularizer=l2))
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy",
                  metrics=["accuracy", keras.metrics.Precision(name="precision"),
                           keras.metrics.Recall(name="recall")])
    return model


def augment(x, y):
    """Random flips, 90-degree turns, brightness and contrast (top-down photos: any rotation is realistic).
    Runs in the data pipeline, not inside the model, so the model saves cleanly as .h5."""
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    x = tf.image.rot90(x, tf.random.uniform([], 0, 4, dtype=tf.int32))
    x = tf.image.random_brightness(x, 40)
    x = tf.image.random_contrast(x, 0.8, 1.2)
    return tf.clip_by_value(x, 0, 255), y


def class_weights(folder):
    n_pos = len(list((folder / "train" / "mushroom").glob("*.jpg")))
    n_neg = len(list((folder / "train" / "no_mushroom").glob("*.jpg")))
    total = n_pos + n_neg
    return {0: total / (2 * n_neg), 1: total / (2 * n_pos)}


def run(name, data_dir, out, regularized, epochs, batchnorm=False, lr=1e-3, patience=8, reduce_lr=False):
    out.mkdir(parents=True, exist_ok=True)
    train, val, test = load(data_dir, "train", True), load(data_dir, "val", False), load(data_dir, "test", False)
    if regularized:
        train = train.map(augment)
    model = build(regularized, batchnorm, lr)
    callbacks = []
    if regularized:
        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(str(out / "cnn_best.h5"), monitor="val_loss", save_best_only=True),
        ]
        if reduce_lr:  # halve the learning rate when val loss stalls for 4 epochs
            callbacks.append(keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5))
    hist = model.fit(train, validation_data=val, epochs=epochs, class_weight=class_weights(data_dir),
                     callbacks=callbacks, verbose=2)
    if not regularized:
        model.save(out / f"{name}.h5")

    h = hist.history
    with open(out / "history.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch"] + list(h))
        for i in range(len(h["loss"])):
            w.writerow([i + 1] + [h[k][i] for k in h])
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for k, a in (("accuracy", ax[0]), ("loss", ax[1])):
        a.plot(h[k], label="train")
        a.plot(h[f"val_{k}"], label="val")
        a.set_title(f"{name}: {k}")
        a.set_xlabel("epoch")
        a.legend()
    fig.tight_layout()
    fig.savefig(out / "curves.png")
    plt.close(fig)

    best = int(np.argmin(h["val_loss"])) if regularized else len(h["loss"]) - 1
    _, acc, prec, rec = model.evaluate(test, verbose=0)
    return {"experiment": name, "epochs_run": len(h["loss"]),
            "train_acc": round(h["accuracy"][best], 3), "val_acc": round(h["val_accuracy"][best], 3),
            "gap": round(h["accuracy"][best] - h["val_accuracy"][best], 3),
            "test_acc": round(acc, 3), "test_precision": round(prec, 3), "test_recall": round(rec, 3)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--version", choices=["v1", "v2", "v2.1"], default="v1")
    args = p.parse_args()
    keras.utils.set_random_seed(0)
    print('Keras backend:', keras.backend.backend())

    if args.version == "v1":
        runs = args.work / "runs" / "cnn"
        table = [
            run("baseline_whole", args.work / "whole", runs / "baseline_whole", False, args.epochs),
            run("baseline_tiles", args.work / "tiles", runs / "baseline_tiles", False, args.epochs),
            run("final_tiles", args.work / "tiles", runs / "final_tiles", True, args.epochs * 2),
        ]
    elif args.version == "v2.1":
        runs = args.work / "runs" / "cnn_v2_1"
        table = [
            run("final_patches", args.work / "patches", runs / "final_patches", True, args.epochs * 3,
                batchnorm=True, patience=15, reduce_lr=True),
        ]
    else:
        runs = args.work / "runs" / "cnn_v2"
        table = [
            run("baseline_patches", args.work / "patches", runs / "baseline_patches", False, args.epochs, lr=1e-4),
            run("final_patches", args.work / "patches", runs / "final_patches", True, args.epochs * 2,
                batchnorm=True),
        ]
    with open(runs / "tuning_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=table[0].keys())
        w.writeheader()
        w.writerows(table)
    for row in table:
        print(row)


if __name__ == "__main__":
    main()
