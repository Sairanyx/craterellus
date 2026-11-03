"""Fine-tune a pretrained YOLOv5 on our mushroom boxes (M3).

uv run src/yolo_train.py --work DL_project/work

The pretrained backbone already knows edges, textures and shapes from COCO.
Ultralytics swaps the 80-class head for a 1-class head (suppilovahvero) from data.yaml.

Writes <work>/runs/yolo/train/   weights/best.pt, results.png, confusion matrix, ...
       <work>/runs/yolo/test/    metrics + prediction images on the test split
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--model", default="yolov5su.pt")
    p.add_argument("--imgsz", type=int, default=960)    # high resolution for small mushrooms
    p.add_argument("--epochs", type=int, default=80)
    args = p.parse_args()

    data = args.work / "yolo" / "data.yaml"
    runs = args.work / "runs" / "yolo"

    model = YOLO(args.work / args.model if (args.work / args.model).exists() else args.model)
    model.train(
        data=str(data), imgsz=args.imgsz, epochs=args.epochs, patience=20,
        batch=-1,                       # pick the biggest batch that fits the GPU
        project=str(runs), name="train", exist_ok=True,
        degrees=10, flipud=0.5,         # top-down photos: any rotation is realistic
    )

    best = YOLO(runs / "train" / "weights" / "best.pt")
    m = best.val(data=str(data), split="test", imgsz=args.imgsz,
                 project=str(runs), name="test", exist_ok=True)
    print(f"TEST  mAP50={m.box.map50:.3f}  mAP50-95={m.box.map:.3f}  "
          f"P={m.box.mp:.3f}  R={m.box.mr:.3f}")

    best.predict(source=str(args.work / "yolo" / "test" / "images"), imgsz=args.imgsz, conf=0.25,
                 save=True, project=str(runs), name="test_predictions", exist_ok=True)


if __name__ == "__main__":
    main()
