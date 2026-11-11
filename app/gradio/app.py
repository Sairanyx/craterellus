"""
Craterellus demo app to show: photo, video and live drone using our custom
made CNN and YOLO.
"""

import gradio as gr


def process_photo(image, mode):
    return image                 # just demo for now till models are done


with gr.Blocks(title="Craterellus") as app:
    gr.Markdown("# 🍄 Craterellus")

    with gr.Tab("Photo"):
        with gr.Row():
            mode = gr.Radio(["CNN", "YOLO", "Compare"], value="YOLO", label="Model")
            shuffle = gr.Button("Shuffle test photo")
        with gr.Row():
            photo = gr.Image(label="Photo")
            result = gr.Image(label="Result")
        photo.change(process_photo, [photo, mode], result)
        mode.change(process_photo, [photo, mode], result)

    with gr.Tab("Video"):
        with gr.Row():
            video = gr.Video(label="Video")
            video_result = gr.Image(label="YOLO")

    with gr.Tab("Stream"):
        start = gr.Button("Start stream")
        stream = gr.Image(label="Drone + YOLO")

app.launch()