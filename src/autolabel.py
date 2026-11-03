"""Draft mushroom boxes with Grounding DINO (zero-shot), saved as YOLO labels.

The boxes are drafts: check them in the previews (or fix them in CVAT) before trusting them.

HF_HOME=DL_project/hf_cache uv run --with transformers src/autolabel.py --work DL_project/work

Reads  <work>/images/*.jpg and <work>/metadata.csv
Writes <work>/labels/*.txt     YOLO: class cx cy w h (normalised), class 0 = suppilovahvero
       <work>/autolabel.csv    boxes and scores per image
       <work>/previews/*.jpg   contact sheets with the boxes drawn
"""
import argparse
import csv
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision.ops import nms
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

MODEL = "IDEA-Research/grounding-dino-base"
PROMPT = "mushroom. pine cone. dead leaf. bark. stick."   # look-alikes as extra classes...
KEEP = "mushroom"                                         # ...but keep only mushroom boxes
TILE, STRIDE = 960, 640     # mushrooms are small: look at overlapping zoomed-in tiles
BOX_THRESHOLD = 0.35
MAX_AREA = 0.25             # drop boxes bigger than this share of the image (whole-scene boxes)


def positions(size):
    """Tile start positions that cover 0..size."""
    if size <= TILE:
        return [0]
    starts = list(range(0, size - TILE, STRIDE))
    return starts + [size - TILE]


def detect(model, proc, im):
    """Boxes (x1, y1, x2, y2) and scores for one PIL image, from tiles + the whole image."""
    crops = [(x, y, im.crop((x, y, x + TILE, y + TILE)))
             for y in positions(im.height) for x in positions(im.width)]
    crops.append((0, 0, im))  # whole image catches big mushrooms cut by tile edges

    boxes, scores = [], []
    for x, y, crop in crops:
        inp = proc(images=crop, text=PROMPT, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inp)
        r = proc.post_process_grounded_object_detection(
            out, inp.input_ids, threshold=BOX_THRESHOLD, text_threshold=0.25,
            target_sizes=[crop.size[::-1]])[0]
        names = r.get("text_labels", r.get("labels"))
        for b, s, name in zip(r["boxes"].tolist(), r["scores"].tolist(), names):
            if KEEP not in str(name):
                continue
            boxes.append([b[0] + x, b[1] + y, b[2] + x, b[3] + y])
            scores.append(s)
    if not boxes:
        return [], []

    b, s = torch.tensor(boxes), torch.tensor(scores)
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    keep = area < MAX_AREA * im.width * im.height
    b, s = b[keep], s[keep]
    keep = nms(b, s, 0.5)
    b, s = b[keep], s[keep]

    # drop a box that mostly sits inside a bigger kept box (partial duplicate from a tile edge)
    final = []
    for i in range(len(b)):
        inside = False
        for j in range(len(b)):
            if i == j or (b[j, 2] - b[j, 0]) * (b[j, 3] - b[j, 1]) <= (b[i, 2] - b[i, 0]) * (b[i, 3] - b[i, 1]):
                continue
            ix = max(0, min(b[i, 2], b[j, 2]) - max(b[i, 0], b[j, 0]))
            iy = max(0, min(b[i, 3], b[j, 3]) - max(b[i, 1], b[j, 1]))
            if ix * iy > 0.7 * (b[i, 2] - b[i, 0]) * (b[i, 3] - b[i, 1]):
                inside = True
                break
        if not inside:
            final.append(i)
    return b[final].tolist(), s[final].tolist()


def contact_sheets(work, results, per_sheet=24, thumb=360):
    out = work / "previews"
    out.mkdir(exist_ok=True)
    cols = 6
    for k in range(0, len(results), per_sheet):
        chunk = results[k:k + per_sheet]
        rows = (len(chunk) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * thumb, rows * (thumb + 18)), "white")
        d = ImageDraw.Draw(sheet)
        for i, (name, boxes, _) in enumerate(chunk):
            im = Image.open(work / "images" / name)
            dr = ImageDraw.Draw(im)
            for bx in boxes:
                dr.rectangle(bx, outline="red", width=6)
            im.thumbnail((thumb, thumb))
            x, y = (i % cols) * thumb, (i // cols) * (thumb + 18)
            sheet.paste(im, (x, y + 18))
            d.text((x + 3, y + 3), f"{name}  ({len(boxes)})", fill="black")
        sheet.save(out / f"sheet_{k // per_sheet:02d}.jpg", quality=85)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", type=Path, required=True)
    args = p.parse_args()

    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL).to("cuda").eval()

    with open(args.work / "metadata.csv") as f:
        meta = list(csv.DictReader(f))
    label_dir = args.work / "labels"
    label_dir.mkdir(exist_ok=True)

    results, log = [], []
    for i, row in enumerate(meta, 1):
        name = row["filename"]
        im = Image.open(args.work / "images" / name).convert("RGB")
        boxes, scores = detect(model, proc, im)
        found = len(boxes)
        if row["has_mushroom"] == "0":
            boxes, scores = [], []  # known empty photo: any box is a false positive

        with open(label_dir / (Path(name).stem + ".txt"), "w") as f:
            for x1, y1, x2, y2 in boxes:
                f.write(f"0 {(x1 + x2) / 2 / im.width:.6f} {(y1 + y2) / 2 / im.height:.6f} "
                        f"{(x2 - x1) / im.width:.6f} {(y2 - y1) / im.height:.6f}\n")
        results.append((name, boxes, scores))
        log.append({"filename": name, "has_mushroom": row["has_mushroom"], "raw_boxes": found,
                    "kept_boxes": len(boxes),
                    "scores": " ".join(f"{s:.2f}" for s in scores)})
        print(f"{i:4d}/{len(meta)} {name}: {found} found, {len(boxes)} kept")

    with open(args.work / "autolabel.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=log[0].keys())
        w.writeheader()
        w.writerows(log)
    contact_sheets(args.work, results)
    print("Done.")


if __name__ == "__main__":
    main()
