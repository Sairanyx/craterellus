"""Convert the raw HEIC photos to JPG and write metadata.csv.

uv run src/preprocess.py --src DL_project/data/images_all --mapping DL_project/data/images_all_mapping.csv --out DL_project/work

Output:
  <out>/images/*.jpg   EXIF rotation applied, longest side <= 1920 px
  <out>/metadata.csv   filename, original_name, author, time, lat, lon, has_mushroom
"""
import argparse
import csv
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()

MAX_SIDE = 1920


def gps(exif):
    """(lat, lon) in degrees from EXIF, or (None, None)."""
    g = exif.get_ifd(0x8825)
    if not g or 2 not in g:
        return None, None

    def deg(v, ref):
        d = float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600
        return -d if ref in ("S", "W") else d

    return round(deg(g[2], g[1]), 6), round(deg(g[4], g[3]), 6)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--mapping", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    with open(args.mapping) as f:
        mapping = {r["new_name"]: r for r in csv.DictReader(f)}

    img_dir = args.out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(args.src.iterdir()):
        im = Image.open(path)
        exif = im.getexif()
        taken = datetime.strptime(exif[306], "%Y:%m:%d %H:%M:%S") if 306 in exif else None
        lat, lon = gps(exif)

        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((MAX_SIDE, MAX_SIDE))
        name = path.stem + ".jpg"
        im.save(img_dir / name, quality=92)

        info = mapping.get(path.name, {})
        rows.append({
            "filename": name,
            "original_name": info.get("original_name", path.name),
            "author": info.get("author", ""),
            "time": taken.isoformat() if taken else "",
            "lat": lat, "lon": lon,
            "has_mushroom": 0 if path.stem.startswith("no_mushroom") else 1,
        })
        print(f"{len(rows):4d} {name}  {im.size}")

    with open(args.out / "metadata.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"Done: {len(rows)} images -> {img_dir}")


if __name__ == "__main__":
    main()
