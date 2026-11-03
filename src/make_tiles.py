"""Cut every photo into square tiles for the CNN (M2).

At 224 px a whole photo shrinks mushrooms to 2-3 px. A 480 px tile resized to 224
keeps them about half their original size, so the small CNN can see them.

uv run src/make_tiles.py --work DL_project/work

Tile label: 1 if a box sits mostly (>= 50 %) inside the tile, 0 if no box touches it.
Tiles that only cut a small corner of a box are skipped (ambiguous).

Writes <work>/tiles/{train,val,test}/{mushroom,no_mushroom}/*.jpg
       <work>/whole/{train,val,test}/{mushroom,no_mushroom}/*.jpg   (whole photos, for the baseline)
"""
import argparse
import csv
import shutil
from pathlib import Path

from PIL import Image

TILE = 480
SIZE = 224


def boxes_px(label_file, w, h):
    if not label_file.exists():
        return []
    out = []
    for line in label_file.read_text().split("\n"):
        if line.strip():
            _, cx, cy, bw, bh = map(float, line.split())
            out.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out


def tile_label(tile, boxes):
    x1, y1, x2, y2 = tile
    best = 0.0
    for bx1, by1, bx2, by2 in boxes:
        ix = max(0, min(x2, bx2) - max(x1, bx1))
        iy = max(0, min(y2, by2) - max(y1, by1))
        if ix * iy:
            best = max(best, ix * iy / ((bx2 - bx1) * (by2 - by1)))
    if best >= 0.5:
        return 1
    return 0 if best == 0 else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", type=Path, required=True)
    args = p.parse_args()

    for d in ("tiles", "whole"):
        shutil.rmtree(args.work / d, ignore_errors=True)
    with open(args.work / "metadata.csv") as f:
        rows = list(csv.DictReader(f))

    counts = {}
    for r in rows:
        im = Image.open(args.work / "images" / r["filename"]).convert("RGB")
        w, h = im.size
        boxes = boxes_px(args.work / "labels" / (Path(r["filename"]).stem + ".txt"), w, h)
        stem = Path(r["filename"]).stem

        cls = "mushroom" if r["has_mushroom"] == "1" else "no_mushroom"
        dst = args.work / "whole" / r["split"] / cls
        dst.mkdir(parents=True, exist_ok=True)
        im.resize((SIZE, SIZE)).save(dst / r["filename"], quality=90)

        for y in range(0, h - TILE + 1, TILE):
            for x in range(0, w - TILE + 1, TILE):
                label = tile_label((x, y, x + TILE, y + TILE), boxes)
                if label is None:
                    continue
                cls = "mushroom" if label else "no_mushroom"
                dst = args.work / "tiles" / r["split"] / cls
                dst.mkdir(parents=True, exist_ok=True)
                im.crop((x, y, x + TILE, y + TILE)).resize((SIZE, SIZE)).save(
                    dst / f"{stem}_{x}_{y}.jpg", quality=90)
                counts[(r["split"], cls)] = counts.get((r["split"], cls), 0) + 1

    for split in ("train", "val", "test"):
        print(f"{split:5s}: {counts.get((split, 'mushroom'), 0):5d} mushroom tiles, "
              f"{counts.get((split, 'no_mushroom'), 0):5d} empty tiles")


if __name__ == "__main__":
    main()
