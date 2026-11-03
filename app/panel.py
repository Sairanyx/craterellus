"""Craterellus demo app.

  Photo    CNN sliding heatmap (is there a mushroom here?) + YOLO boxes (where exactly?)
  Video    YOLO + ByteTrack on a GoPro clip -> annotated video + unique mushroom count
  Live     Tello stream -> YOLO boxes in real time (laptop on the Tello's Wi-Fi)
  Metrics  CNN tuning table + training curves, YOLO results

CRATERELLUS_WORK=/path/to/DL_project/work uv run app/panel.py
"""
import csv
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

import cv2
import gradio as gr
import numpy as np
import keras
from ultralytics import YOLO

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from cnn_sliding import heatmap, score, windows  # noqa: E402

WORK = Path(os.environ.get("CRATERELLUS_WORK", "/run/media/sairanyx/E476-F3F3/DL_project/work"))
YOLO_WEIGHTS = WORK / "runs/yolo/train/weights/best.pt"
CNN_WEIGHTS = WORK / "runs/cnn_v2_1/final_patches/cnn_best.h5"
IMGSZ = 960
CNN_THRESHOLD = 0.8

yolo = YOLO(YOLO_WEIGHTS) if YOLO_WEIGHTS.exists() else None
cnn = keras.models.load_model(CNN_WEIGHTS) if CNN_WEIGHTS.exists() else None


def fit(img, longest=1920):
    """Resize an RGB image so its longest side is at most `longest`, like the training photos."""
    h, w = img.shape[:2]
    s = longest / max(h, w)
    return cv2.resize(img, (round(w * s), round(h * s))) if s < 1 else img


def cnn_map(img):
    """Slide the patch CNN over the photo; colour the windows it calls "mushroom"."""
    h, w = img.shape[:2]
    wins = windows(w, h)
    probs = score(cnn, img, wins)
    return heatmap(img, wins, probs, CNN_THRESHOLD), float(probs.max())


def analyse_photo(img, conf):
    if img is None:
        return None, None, ""
    img = fit(img)
    lines = []
    grid = None
    if cnn:
        grid, top = cnn_map(img)
        lines.append(f"CNN: {'mushroom' if top >= CNN_THRESHOLD else 'no mushroom'} (highest window {top:.2f})")
    boxes = None
    if yolo:
        r = yolo.predict(img[:, :, ::-1], imgsz=IMGSZ, conf=conf, verbose=False)[0]  # YOLO wants BGR
        boxes = r.plot(line_width=3)[:, :, ::-1]
        lines.append(f"YOLO: {len(r.boxes)} mushroom(s) found")
    return grid, boxes, "\n".join(lines) or "No trained models found yet."


def analyse_video(path, conf, progress=gr.Progress()):
    if not path or not yolo:
        return None, "Upload a video (and train YOLO first)."
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_path = str(WORK / "video_out.mp4")
    writer, ids, step = None, set(), 2 if fps > 40 else 1   # 60 fps GoPro: every 2nd frame
    for i in progress.tqdm(range(total)):
        ok, frame = cap.read()
        if not ok:
            break
        if i % step:
            continue
        frame = fit(frame, 1280)
        r = yolo.track(frame, imgsz=IMGSZ, conf=conf, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
        if r.boxes.id is not None:
            ids.update(r.boxes.id.int().tolist())
        plotted = r.plot(line_width=2)
        cv2.putText(plotted, f"Unique mushrooms: {len(ids)}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)
        if writer is None:
            h, w = plotted.shape[:2]
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps / step, (w, h))
        writer.write(plotted)
    cap.release()
    if writer:
        writer.release()
    return out_path, f"{len(ids)} unique mushrooms found"


def live_tello(conf):
    """Stream the Tello camera through YOLO. Connect the laptop to the Tello's Wi-Fi first."""
    from djitellopy import Tello
    tello = Tello()
    tello.connect()
    tello.streamon()
    frames = tello.get_frame_read()
    try:
        while True:
            frame = frames.frame  # RGB
            if yolo:
                r = yolo.predict(frame[:, :, ::-1], imgsz=640, conf=conf, verbose=False)[0]
                frame = r.plot(line_width=2)[:, :, ::-1]
            yield frame
    finally:
        tello.streamoff()


def metrics():
    rows = []
    for version in ("cnn", "cnn_v2", "cnn_v2_1"):
        table_file = WORK / "runs" / version / "tuning_table.csv"
        if table_file.exists():
            rows += [{"version": version, **r} for r in csv.DictReader(open(table_file))]
    images = [str(p) for p in [
        *sorted((WORK / "runs").glob("cnn*/*/curves.png")),
        WORK / "runs/yolo/train/results.png",
        WORK / "runs/yolo/test/confusion_matrix.png",
    ] if Path(p).exists()]
    return [list(r.values()) for r in rows], (list(rows[0].keys()) if rows else None), images


with gr.Blocks(title="Craterellus") as demo:
    gr.Markdown("# 🍄 Craterellus: finding suppilovahvero")
    conf = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="YOLO confidence threshold")

    with gr.Tab("Photo"):
        photo = gr.Image(label="Forest photo", type="numpy")
        with gr.Row():
            grid_out = gr.Image(label="CNN: mushroom heatmap (sliding window)")
            box_out = gr.Image(label="YOLO: boxes")
        summary = gr.Textbox(label="Result")
        photo.change(analyse_photo, [photo, conf], [grid_out, box_out, summary])
        examples = sorted((WORK / "yolo/test/images").glob("*.jpg"))[:12]
        if examples:
            gr.Examples([[str(p)] for p in examples], [photo])

    with gr.Tab("Video"):
        video = gr.Video(label="GoPro clip")
        run_video = gr.Button("Find mushrooms")
        video_out = gr.Video(label="Result")
        count = gr.Textbox(label="Count")
        run_video.click(analyse_video, [video, conf], [video_out, count])

    with gr.Tab("Live drone"):
        gr.Markdown("Connect the laptop to the **TELLO-XXXXXX** Wi-Fi, then press Start.")
        start = gr.Button("Start stream")
        live = gr.Image(label="Tello camera")
        start.click(live_tello, [conf], live)

    with gr.Tab("Metrics"):
        refresh = gr.Button("Load metrics")
        table = gr.Dataframe(label="CNN tuning journey")
        gallery = gr.Gallery(label="Training curves and YOLO results", columns=2)

        def show():
            values, headers, imgs = metrics()
            return gr.Dataframe(value=values, headers=headers), imgs

        refresh.click(show, None, [table, gallery])

if __name__ == "__main__":
    demo.launch()
