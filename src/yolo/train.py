"""
Fine-tune a pretrained Ultralytics YOLO on the suppilovahvero dataset.

The mushrooms are small (most boxes < 1 % of the image) and camouflaged, so we
train at a large image size and fine-tune a COCO-pretrained model rather than
training from scratch.

Run build_folders.py first so dataset/yolo/ and its data.yaml exist.

Examples (run from the repo root):
    uv run src/yolo/train.py                                  # yolo11s @ 960, the default
    uv run src/yolo/train.py --model yolo11n.pt --name n_960  # fast baseline
    uv run src/yolo/train.py --model yolo11m.pt --batch 8     # bigger, on a cloud GPU
    uv run src/yolo/train.py --model yolov5su.pt --name v5s   # the older v5 baseline

Every run writes to results/yolo/<name>/:
  - weights/best.pt      the best model (by val), to upload to R2 models/yolo/
  - results.png, *.png   training curves and plots (a deliverable)
  - run_info.json        exact settings + versions, for reproducibility

The comparison table for the report is just the val/test rows of several runs.
"""

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path


# Defaults chosen for this task: small camouflaged objects, small dataset.
DATA = "dataset/yolo/data.yaml"
PROJECT = "results/yolo"


def git_commit() -> str:
    """Short commit hash, so a run can be tied to the exact code. '' if unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="yolo11s.pt",
                   help="pretrained weights to fine-tune (e.g. yolo11n.pt, yolo11s.pt, yolo11m.pt, yolov5su.pt)")
    p.add_argument("--imgsz", type=int, default=960,
                   help="training image size; large because the mushrooms are tiny")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=-1,
                   help="batch size; -1 lets Ultralytics auto-pick for the GPU (keep low on 4 GB cards)")
    p.add_argument("--device", default="0",
                   help="'0' for the first GPU, 'cpu' to force CPU")
    p.add_argument("--patience", type=int, default=30,
                   help="early stopping: stop if val hasn't improved for this many epochs")
    p.add_argument("--seed", type=int, default=0,
                   help="fixed seed for reproducibility")
    p.add_argument("--data", default=DATA)
    p.add_argument("--project", default=PROJECT)
    p.add_argument("--name", default=None,
                   help="run folder name under results/yolo/; defaults to the model stem")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not Path(args.data).exists():
        raise SystemExit(f"{args.data} not found — run: uv run src/split/build_folders.py")

    # Default the run name to the model stem (e.g. 'yolo11s') so runs are easy to tell apart.
    name = args.name or Path(args.model).stem

    # Import here so --help works without loading the heavy stack.
    import torch
    import ultralytics
    from ultralytics import YOLO

    device = args.device
    if device != "cpu" and not torch.cuda.is_available():
        print("CUDA not available — falling back to CPU (this will be slow).")
        device = "cpu"

    print("=" * 60)
    print("Fine-tuning YOLO on suppilovahvero")
    print("=" * 60)
    print(f"model      {args.model}")
    print(f"imgsz      {args.imgsz}")
    print(f"epochs     {args.epochs}  (patience {args.patience})")
    print(f"batch      {args.batch}  (-1 = auto)")
    print(f"device     {device}" + (f"  [{torch.cuda.get_device_name(0)}]" if device != "cpu" else ""))
    print(f"seed       {args.seed}")
    print(f"data       {args.data}")
    print(f"output     {args.project}/{name}/")
    print("=" * 60)

    model = YOLO(args.model)  # downloads the pretrained weights on first use

    results = model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=device,
        patience=args.patience,
        seed=args.seed,
        project=args.project,
        name=name,
        exist_ok=True,       # overwrite a previous run of the same name
        plots=True,          # save curves / PR / confusion matrix
    )

    run_dir = Path(results.save_dir)
    best = run_dir / "weights" / "best.pt"

    # Reproducibility stamp — exactly the fields Milestone 3 asks us to state.
    info = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "model": args.model,
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device != "cpu" else "cpu",
        "patience": args.patience,
        "seed": args.seed,
        "data": args.data,
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "best_weights": str(best),
    }
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2))

    print("\nDone.")
    print(f"best weights : {best}")
    print(f"run info     : {run_dir / 'run_info.json'}")
    print(f"plots        : {run_dir}/  (results.png, confusion_matrix.png, ...)")
    print("\nNext: evaluate on the held-out test split, then upload best.pt to R2:")
    print(f"  rclone copy {best} r2:craterellus/models/yolo -P")


if __name__ == "__main__":
    main()
