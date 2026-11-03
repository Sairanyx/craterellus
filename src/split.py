"""Split the dataset 70/15/15 by patch and build the YOLO dataset folder.

A patch is a run of photos by one person, each taken < 60 s and < 15 m after the
previous one. Whole patches go to one split, so near-identical shots never end up
in both train and test.

uv run src/split.py --work DL_project/work

Reads  <work>/images, <work>/labels, <work>/metadata.csv
Writes <work>/metadata.csv     (+ patch_id, split)
       <work>/yolo/{train,val,test}/{images,labels}
       <work>/yolo/data.yaml
"""
import argparse
import csv
import math
import random
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

GAP_SECONDS, GAP_METRES = 60, 15
# Empty photos were taken walking (one every few seconds), so they need tighter breaks to form patches
NEG_GAP_SECONDS, NEG_GAP_METRES = 15, 8
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 0


def metres(a, b):
    if not (a["lat"] and b["lat"]):
        return 0
    la1, lo1, la2, lo2 = (float(a["lat"]), float(a["lon"]), float(b["lat"]), float(b["lon"]))
    return math.hypot((la1 - la2) * 111_000, (lo1 - lo2) * 111_000 * math.cos(math.radians(la1)))


def assign_patches(rows):
    rows.sort(key=lambda r: (r["author"], r["time"]))
    patch = 0
    for i, r in enumerate(rows):
        if i:
            prev = rows[i - 1]
            gap = (datetime.fromisoformat(r["time"]) - datetime.fromisoformat(prev["time"])).total_seconds()
            secs, mts = (NEG_GAP_SECONDS, NEG_GAP_METRES) if r["has_mushroom"] == "0" else (GAP_SECONDS, GAP_METRES)
            if r["author"] != prev["author"] or r["has_mushroom"] != prev["has_mushroom"] \
                    or gap > secs or metres(prev, r) > mts:
                patch += 1
        r["patch_id"] = patch


def assign_splits(rows):
    """Split positives and negatives separately, so every split gets some of each."""
    for has in ("1", "0"):
        patches = defaultdict(list)
        for r in rows:
            if r["has_mushroom"] == has:
                patches[r["patch_id"]].append(r)
        ids = sorted(patches)
        random.Random(SEED).shuffle(ids)
        total = sum(len(v) for v in patches.values())
        done = 0
        for pid in ids:
            share = done / total
            split = "train" if share < SPLITS["train"] else "val" if share < SPLITS["train"] + SPLITS["val"] else "test"
            for r in patches[pid]:
                r["split"] = split
            done += len(patches[pid])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", type=Path, required=True)
    args = p.parse_args()

    with open(args.work / "metadata.csv") as f:
        rows = list(csv.DictReader(f))
    assign_patches(rows)
    assign_splits(rows)

    yolo = args.work / "yolo"
    shutil.rmtree(yolo, ignore_errors=True)
    for r in rows:
        for kind, ext in (("images", ".jpg"), ("labels", ".txt")):
            dst = yolo / r["split"] / kind
            dst.mkdir(parents=True, exist_ok=True)
            src = args.work / kind / (Path(r["filename"]).stem + ext)
            if src.exists():
                shutil.copy(src, dst / src.name)

    (yolo / "data.yaml").write_text(
        f"path: {yolo}\ntrain: train/images\nval: val/images\ntest: test/images\n"
        "names:\n  0: suppilovahvero\n")

    rows.sort(key=lambda r: r["filename"])
    with open(args.work / "metadata.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print(f"{len({r['patch_id'] for r in rows})} patches")
    for s in SPLITS:
        part = [r for r in rows if r["split"] == s]
        neg = sum(r["has_mushroom"] == "0" for r in part)
        print(f"{s:5s}: {len(part):3d} images ({len(part) - neg} with mushrooms, {neg} without)")


if __name__ == "__main__":
    main()
