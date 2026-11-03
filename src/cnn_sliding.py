"""Use the patch CNN on whole photos: slide it over the photo at several window sizes.

The CNN was trained on mushroom-centred patches, so on a full photo we cut overlapping
windows (several sizes, because mushrooms are near or far), score each one, and mark
the windows it calls "mushroom".

uv run src/cnn_sliding.py --work DL_project/work

Reports on the test photos:
  photo level   a photo is "mushroom" if any window scores >= threshold
  box recall    share of labelled mushrooms whose centre lies in a "mushroom" window
  window prec.  share of "mushroom" windows that contain a labelled mushroom centre
  speed         windows and seconds per photo
Writes <work>/runs/cnn_sliding/*.jpg   heatmap examples
"""
import argparse
import csv
import os
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

import cv2
import keras
import numpy as np

from make_tiles import boxes_px

SIZE = 224
WINDOWS = (160, 256, 384)   # window sides in px on the 1920 px photo (far, mid, near mushrooms)
STRIDE = 0.5                # step = half a window


def windows(w, h):
    out = []
    for side in WINDOWS:
        step = int(side * STRIDE)
        for y in range(0, h - side + 1, step):
            for x in range(0, w - side + 1, step):
                out.append((x, y, side))
    return out


def score(model, img, wins, batch=256):
    crops = np.stack([cv2.resize(img[y:y + s, x:x + s], (SIZE, SIZE)) for x, y, s in wins]).astype("float32")
    return np.concatenate([model.predict(crops[i:i + batch], verbose=0)[:, 0] for i in range(0, len(crops), batch)])


def heatmap(img, wins, probs, threshold):
    heat = np.zeros(img.shape[:2], "float32")
    for (x, y, s), p in zip(wins, probs):
        if p >= threshold:
            heat[y:y + s, x:x + s] = np.maximum(heat[y:y + s, x:x + s], p)
    color = cv2.cvtColor(cv2.applyColorMap((heat * 255).astype("uint8"), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    mask = (heat > 0)[..., None]
    return np.where(mask, (0.55 * img + 0.45 * color).astype("uint8"), img)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--model", default="runs/cnn_v2_1/final_patches/cnn_best.h5")
    args = p.parse_args()

    model = keras.models.load_model(args.work / args.model)
    out = args.work / "runs" / "cnn_sliding"
    out.mkdir(parents=True, exist_ok=True)
    with open(args.work / "metadata.csv") as f:
        test = [r for r in csv.DictReader(f) if r["split"] == "test"]

    results, seconds, n_windows = [], [], []
    for r in test:
        bgr = cv2.imread(str(args.work / "images" / r["filename"]))
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        boxes = boxes_px(args.work / "labels" / (Path(r["filename"]).stem + ".txt"), w, h)
        wins = windows(w, h)
        t = time.time()
        probs = score(model, img, wins)
        seconds.append(time.time() - t)
        n_windows.append(len(wins))
        results.append((r, img, boxes, wins, probs))

    print(f"Test photos: {len(test)}  ({sum(r['has_mushroom'] == '1' for r in test)} with mushrooms)")
    print(f"Speed: {np.mean(n_windows):.0f} windows per photo, {np.mean(seconds):.2f} s per photo "
          f"(~{1 / np.mean(seconds):.1f} photos/s)\n")
    print("threshold | photos: mushroom found / empty photos flagged | box recall | window precision")
    for th in (0.5, 0.8, 0.95):
        hit_pos = hit_neg = found = total = good = flagged = 0
        for r, img, boxes, wins, probs in results:
            on = [(x, y, s) for (x, y, s), pr in zip(wins, probs) if pr >= th]
            if r["has_mushroom"] == "1":
                hit_pos += bool(on)
            else:
                hit_neg += bool(on)
            centres = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]
            total += len(centres)
            found += sum(any(x <= cx < x + s and y <= cy < y + s for x, y, s in on) for cx, cy in centres)
            flagged += len(on)
            good += sum(any(x <= cx < x + s and y <= cy < y + s for cx, cy in centres) for x, y, s in on)
        n_pos = sum(r["has_mushroom"] == "1" for r in test)
        print(f"   {th:.2f}   | {hit_pos}/{n_pos} found, {hit_neg}/{len(test) - n_pos} empty flagged"
              f"      | {found / max(total, 1):.2f}       | {good / max(flagged, 1):.2f}")

    for r, img, boxes, wins, probs in results[::9]:
        vis = heatmap(img, wins, probs, 0.8)
        for b in boxes:
            cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255), 3)
        cv2.imwrite(str(out / r["filename"]), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"\nHeatmaps -> {out}")


if __name__ == "__main__":
    main()
