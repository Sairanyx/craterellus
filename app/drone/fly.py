"""
Fly the Tello drone with the PS5 controller and watch the camera.
Finds mushrooms with YOLO if the model is there, if not, it just streams.

Run from the repo root (laptop on the TELLO Wi-Fi): uv run app/drone/fly.py

Make sure you have the controller close by, you are familiar with the controls
(see docs/drone.md) and there is space!
"""

import os
os.environ["YOLO_OFFLINE"] = "1"    # Tello Wi-Fi has no internet: skip Ultralytics' online check

import time
from pathlib import Path

import cv2
from ultralytics import YOLO

import controller
import drone

MODEL = Path("models/yolo/best.pt")

if MODEL.exists():
    model = YOLO(MODEL)
    print(f"YOLO lock and loaded!: {MODEL}")
else:
    model = None
    print("YOLO not locked and not loaded, just streaming for now")

drone.connect()
drone.start_video()

try:
    while True:
        presses = controller.new_presses()
        if controller.OPTION in presses:
            drone.emergency()
            break
        if controller.X in presses:
            drone.take_off()
        if controller.O in presses:
            drone.land()

        drone.move(*controller.movement())

        frame = cv2.cvtColor(drone.get_frame(), cv2.COLOR_RGB2BGR)
        if model is not None:
            results = model(frame, verbose=False)
            frame = results[0].plot()
        cv2.imshow("Tello live", frame)
        cv2.waitKey(1)

        time.sleep(0.05)

finally:
    drone.stop()
