"""
Craterellus demo app to show: photo, video and live drone using our custom
made CNN and YOLO.
"""

import random
import subprocess
from pathlib import Path

import cv2
import gradio as gr
from ultralytics import YOLO


LOGO = Path("assets/name_mushroom.svg").read_text()
TEST_PHOTOS = Path("dataset/yolo/images/test")
YOLO_MODEL = Path ("models/yolo/best.pt")
TEST_VIDEOS = Path ("dataset/clean/videos")
OUT_VIDEOS = Path("results/videos")
CHUNK_SECONDS = 2                     

def process_photo(image, mode):
    if image is None:
        return None

    if mode == "CNN":                                     # CNN heatmap
        return image

    if mode == "YOLO" and yolo is not None:
        results = yolo(image[:, :, ::-1], imgsz=960, verbose=False)  # turning from RGB → BGR for YOLO
        return results[0].plot()[:, :, ::-1]

    return image                                          # comparison will be added here ...


def shuffle_photo():
    photos = list(TEST_PHOTOS.glob("*.jpg"))
    return str(random.choice(photos))


def pick_video():
    videos = list(TEST_VIDEOS.rglob("*.mp4"))
    return str(random.choice(videos))


def start_chunk(path, fps):
    return subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "2560x720", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20", "-pix_fmt", "yuv420p", str(path)],
        stdin=subprocess.PIPE)


def process_video(path):
    if path is None or yolo is None:
        return
    OUT_VIDEOS.mkdir(parents=True, exist_ok=True)
    for old in OUT_VIDEOS.glob("chunk_*.ts"):        # remove the pieces from the last video
        old.unlink()
    fps = cv2.VideoCapture(path).get(cv2.CAP_PROP_FPS) / 2
    results = yolo.predict(source=path, stream=True, vid_stride=2, imgsz=960, verbose=False)

    for i, r in enumerate(results):
        if i % int(fps * CHUNK_SECONDS) == 0:                 # every 2 s: start a new piece
            if i > 0:
                encoder.stdin.close(); encoder.wait(); yield str(piece)
            piece = OUT_VIDEOS / f"chunk_{i}.ts"
            encoder = start_chunk(piece, fps)
        both = cv2.hconcat([cv2.resize(r.orig_img, (1280, 720)), cv2.resize(r.plot(), (1280, 720))])
        cv2.line(both, (1280, 0), (1280, 720), (200, 220, 232), 6) 
        encoder.stdin.write(both.tobytes())

    encoder.stdin.close(); encoder.wait(); yield str(piece)   # the last piece


def play_test_video():
    videos = list(TEST_VIDEOS.rglob("*.mp4"))
    yield from process_video(str(random.choice(videos)))


def launch_drone():
    subprocess.Popen(["uv", "run", "app/drone/fly.py"])     # opens the live drone window
    return "Drone window opened: fly with the PS5 controller, close the window to stop."


if YOLO_MODEL.exists():
    yolo = YOLO(YOLO_MODEL)
    print("YOLO loaded:", YOLO_MODEL)
else:
    yolo = None
    print("YOLO not loaded, showing clean photo")


with gr.Blocks(title="Craterellus") as app:
    gr.HTML(LOGO, elem_id="logo")

    with gr.Tab("Photo"):
        with gr.Row():
            mode = gr.Radio(["CNN", "YOLO", "Compare"], value="CNN", label="Model")
            shuffle = gr.Button("Shuffle test photo")
        with gr.Row():
            photo = gr.Image(label="Photo")
            result = gr.Image(label="Result")
        photo.change(process_photo, [photo, mode], result)
        mode.change(process_photo, [photo, mode], result)
        shuffle.click(shuffle_photo, None, photo)
    with gr.Tab("Video"):
        with gr.Row():
            pick = gr.Button("Load test video")
            upload = gr.Video(label="Or upload your own", height=260, width=400)
        result_video = gr.Video(label="Original  |  YOLO", streaming=True, autoplay=True)
        pick.click(play_test_video, None, result_video)
        upload.upload(process_video, upload, result_video)

    with gr.Tab("Stream"):
        launch = gr.Button("Launch drone")
        status = gr.Markdown()
        launch.click(launch_drone, None, status)

JADE_BG = "#111c18"        # page background
JADE_BLOCK = "#16241e"     # boxes
JADE_BORDER = "#23372b"    # box edges
TEXT = "#c1c497"           # text
BROWN = "#5b4636"          # buttons and labels
BROWN_HOVER = "#6e5c44"
CREAM = "#e8dcc8"          # text on brown boxes

theme = gr.themes.Soft(
    primary_hue="amber",
    font=gr.themes.GoogleFont("Space Grotesk"),
).set(
    body_background_fill_dark=JADE_BG,
    background_fill_primary_dark=JADE_BG,
    background_fill_secondary_dark=JADE_BLOCK,
    block_background_fill_dark=JADE_BLOCK,
    block_border_color_dark=JADE_BORDER,
    input_background_fill_dark=JADE_BG,
    body_text_color_dark=TEXT,
    button_primary_background_fill_dark=BROWN,
    button_primary_background_fill_hover_dark=BROWN_HOVER,
    button_primary_text_color_dark=CREAM,
    button_secondary_background_fill_dark=BROWN,
    button_secondary_background_fill_hover_dark=BROWN_HOVER,
    button_secondary_text_color_dark=CREAM,
    block_label_background_fill_dark=BROWN,
    block_label_text_color_dark=CREAM,
    block_title_background_fill_dark=BROWN,
    block_title_text_color_dark=CREAM,
    checkbox_label_background_fill_selected_dark=BROWN,
    checkbox_label_text_color_selected_dark=CREAM,
)

css = "#logo svg { height: 90px; width: auto; }"

app.launch(theme=theme, css=css, favicon_path="assets/logo.svg")