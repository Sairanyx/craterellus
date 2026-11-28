"""
Run 1: baseline CNN on whole photos (224x224).
4 conv blocks -> Flatten -> Dense(128) -> sigmoid, 30 fixed epochs.
No dropout, no L2, no augmentation and no class weights.
Saves model, training curves, confusion matrix, and tuning table row.
"""


import sys

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

sys.path.append("src/split")
from nice_printing import section

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

# Hyperparameters

SEED = 0         # same randomness every time for starting weights and shuffle order
SIZE = 224       # photos are shrunk to 224 x 224 , suspecting too small
BATCH = 24       # photos per training step
LR = 1e-3        # Adam learning rate of 0.001
EPOCHS = 30      # fixed rate number of epochs for this run 1 baseline
RUN_NAME = "run1_baseline"
OUT = Path("results/cnn") / RUN_NAME
OUT.mkdir(parents=True, exist_ok=True)


keras.utils.set_random_seed(SEED)

section("0. Versions")
print("Python:    ", sys.version.split()[0])
print("TensorFlow:", tf.__version__)
print("GPUs:      ", tf.config.list_physical_devices("GPU"))


# Data

section("1. Data")
train = keras.utils.image_dataset_from_directory(           # using the Target_Absent = 0 and the Target_Present = 1
    "dataset/cnn/train",
    label_mode="binary",
    image_size=(SIZE, SIZE),
    batch_size=BATCH, 
    shuffle=True, seed=SEED,
)
val = keras.utils.image_dataset_from_directory(
    "dataset/cnn/val",
    label_mode="binary",
    image_size=(SIZE, SIZE),
    batch_size=BATCH,
    shuffle=False,
)

print("Classes:", train.class_names)


# Model 

section("2. Model")
model = keras.Sequential([
    layers.Input(shape=(SIZE, SIZE, 3)),
    layers.Rescaling(1.0 / 255),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()
with open(OUT / "summary.txt", "w") as f:
    model.summary(print_fn=lambda line: f.write(line + "\n"))


# Train

section("3. Train")
history = model.fit(train, validation_data=val, epochs=EPOCHS)
model.save(OUT / "model.keras")

hist_df = pd.DataFrame(history.history)
hist_df.to_csv(OUT / "history.csv", index_label="epoch")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(hist_df["loss"], label="train")
ax1.plot(hist_df["val_loss"], label="val")
ax1.set_title("Loss")
ax1.set_xlabel("epoch")
ax1.legend()

ax2.plot(hist_df["accuracy"], label="train")
ax2.plot(hist_df["val_accuracy"], label="val")
ax2.set_title("Accuracy")
ax2.set_xlabel("epoch")
ax2.legend()

fig.tight_layout()
fig.savefig(OUT / "curves.png", dpi=150)
plt.close(fig)


# Validation

section("4. Validation")
y_true = np.concatenate([y.numpy() for _, y in val]).ravel().astype(int)
y_prob = model.predict(val, verbose=0).ravel()
y_pred = (y_prob >= 0.5).astype(int)

train_acc = hist_df["accuracy"].iloc[-1]
val_acc = accuracy_score(y_true, y_pred)
val_prec = precision_score(y_true, y_pred, zero_division=0)
val_rec = recall_score(y_true, y_pred, zero_division=0)
val_f1 = f1_score(y_true, y_pred, zero_division=0)
cm = confusion_matrix(y_true, y_pred)

print(f"Train acc:      {train_acc:.3f}")
print(f"Val acc:       {val_acc:.3f}")
print(f"Val precision: {val_prec:.3f}")
print(f"Val recall:    {val_rec:.3f}")
print(f"Val F1:        {val_f1:.3f}")
print("Confusion matrix (rows=true [0,1], cols=pred [0,1]):")
print(cm)

disp = ConfusionMatrixDisplay(cm, display_labels=train.class_names)
disp.plot(cmap="Blues", colorbar=False)
plt.title(f"{RUN_NAME} (val)")
plt.tight_layout()
plt.savefig(OUT / "confusion_matrix.png", dpi=150)
plt.close()

row = {
    "run": RUN_NAME,
    "what_changed": "whole photos 224. plain baseline",
    "train_acc": round(train_acc, 3),
    "val_acc": round(val_acc, 3),
    "gap": round(train_acc - val_acc, 3),
    "val_precision": round(val_prec, 3),
    "val_recall": round(val_rec, 3),
    "val_f1": round(val_f1, 3),
    "epochs": len(hist_df),
    "learnings": "Overfits (train 1.000 vs val 0.912, val_loss 0.20->0.82); predicts mushroom for 5/7 empty photos.",
}

table_path = Path("results/cnn/tuning_table.csv")
if table_path.exists():
    table = pd.read_csv(table_path)
    table = table[table["run"] != RUN_NAME]
    table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
else:
    table = pd.DataFrame([row])
table.to_csv(table_path, index=False)
print("\nSaved:", OUT)
