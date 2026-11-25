"""
Craterellus demo app to show: photo, video and live drone using our custom
made CNN and YOLO.
"""
import random
from pathlib import Path

import gradio as gr
from ultralytics import YOLO
import cv2


LOGO = Path("assets/name_mushroom.svg").read_text()
TEST_PHOTOS = Path("dataset/yolo/images/test")
YOLO_MODEL = Path ("models/yolo/best.pt")
TEST_VIDEOS = Path ("dataset/clean/videos")

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


def process_video(path):
    if path is None or yolo is None:
        return
    video = cv2.VideoCapture(path)
    ok, frame = video.read()
    while ok:
        results = yolo(frame, imgsz=960, verbose=False)
        yield results[0].plot()[:, :, ::-1]
        ok, frame = video.read()
    video.release()


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
        pick = gr.Button("Load test video")
        with gr.Row():
            video = gr.Video(label="Video")
            video_result = gr.Image(label="YOLO")
        pick.click(pick_video, None, video)
        video.change(process_video, video, video_result)

    with gr.Tab("Stream"):
        start = gr.Button("Start stream")
        stream = gr.Image(label="Drone + YOLO")


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