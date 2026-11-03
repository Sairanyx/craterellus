"""Cut mushroom-centred patches for the CNN (M2, version 2).

Version 1 (make_tiles.py) cut photos into a fixed 480 px grid. At 224 px most mushrooms
became 10-20 px and some "empty" tiles held unlabelled mushrooms, so the CNN could not learn.
Here every positive patch is centred on a labelled mushroom, so it fills a good part of the
image, and negative patches come only from places known to be empty.

uv run src/make_patches.py --work DL_project/work

Positive: one patch per box, ~2.5x the box size (min 128 px), randomly shifted a little.
Negative: random patches of the same sizes from empty photos, and from mushroom photos
          far away from every box (each box grown by 50 %, since auto-labels miss some).
Writes <work>/patches/{train,val,test}/{mushroom,no_mushroom}/*.jpg   (224 x 224)
"""
import argparse
import csv
import random
import shutil
from pathlib import Path

from PIL import Image

from make_tiles import boxes_px

SIZE = 224
MIN_CROP = 128
CONTEXT = 2.5        # patch side = 2.5 x the larger box side
NEG_PER_POS = 2      # empty patches per mushroom patch
SEED = 0


def overlaps(crop, boxes, grow=0.5):
    x1, y1, x2, y2 = crop
    for bx1, by1, bx2, by2 in boxes:
        gw, gh = (bx2 - bx1) * grow, (by2 - by1) * grow
        if x1 < bx2 + gw and x2 > bx1 - gw and y1 < by2 + gh and y2 > by1 - gh:
            return True
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", type=Path, required=True)
    args = p.parse_args()
    rng = random.Random(SEED)

    out = args.work / "patches"
    shutil.rmtree(out, ignore_errors=True)
    with open(args.work / "metadata.csv") as f:
        rows = list(csv.DictReader(f))

    photos, sizes = [], []
    for r in rows:
        im = Image.open(args.work / "images" / r["filename"]).convert("RGB")
        boxes = boxes_px(args.work / "labels" / (Path(r["filename"]).stem + ".txt"), *im.size)
        photos.append((r, im, boxes))

    counts = {}

    def save(im, crop, split, cls, name):
        d = out / split / cls
        d.mkdir(parents=True, exist_ok=True)
        im.crop(crop).resize((SIZE, SIZE)).save(d / name, quality=90)
        counts[(split, cls)] = counts.get((split, cls), 0) + 1

    # positives: one patch per box
    for r, im, boxes in photos:
        w, h = im.size
        for i, (bx1, by1, bx2, by2) in enumerate(boxes):
            side = int(min(max(MIN_CROP, CONTEXT * max(bx2 - bx1, by2 - by1)), w, h))
            cx = (bx1 + bx2) / 2 + rng.uniform(-0.15, 0.15) * side
            cy = (by1 + by2) / 2 + rng.uniform(-0.15, 0.15) * side
            x1 = int(min(max(0, cx - side / 2), w - side))
            y1 = int(min(max(0, cy - side / 2), h - side))
            save(im, (x1, y1, x1 + side, y1 + side), r["split"], "mushroom", f"{Path(r['filename']).stem}_{i}.jpg")
            sizes.append(side)

    # negatives: same size distribution, from empty photos and empty areas, per split
    for split in ("train", "val", "test"):
        target = NEG_PER_POS * counts.get((split, "mushroom"), 0)
        pool = [ph for ph in photos if ph[0]["split"] == split]
        made, tries = 0, 0
        while made < target and tries < target * 50:
            tries += 1
            r, im, boxes = rng.choice(pool)
            w, h = im.size
            side = min(rng.choice(sizes), w, h)
            x1, y1 = rng.randint(0, w - side), rng.randint(0, h - side)
            crop = (x1, y1, x1 + side, y1 + side)
            if boxes and overlaps(crop, boxes):
                continue
            save(im, crop, split, "no_mushroom", f"{Path(r['filename']).stem}_neg{made}.jpg")
            made += 1

    for split in ("train", "val", "test"):
        print(f"{split:5s}: {counts.get((split, 'mushroom'), 0):5d} mushroom patches, "
              f"{counts.get((split, 'no_mushroom'), 0):5d} empty patches")


if __name__ == "__main__":
    main()
