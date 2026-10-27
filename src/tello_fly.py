"""Fly the Tello from the laptop with the keyboard and see / record its video.

Connect the laptop to the TELLO-XXXXXX Wi-Fi first, then: python src/tello_fly.py

Keys (click the video window first):
  t takeoff   g land   x EMERGENCY (motors off)
  w/s forward/back   a/d left/right   r/f up/down   q/e turn
  v start/stop recording to recordings/   Esc land and quit
"""
import time
from pathlib import Path

import cv2
from djitellopy import Tello

SPEED = 30          # 0-100, keep low indoors
HOLD = 0.3          # seconds a key press keeps moving the drone
RC_EVERY = 0.1      # send movement 10x per second, more floods the drone
MIN_BATTERY = 20    # % needed to take off

MOVES = {           # key: (left/right, forward/back, up/down, turn)
    "w": (0, SPEED, 0, 0), "s": (0, -SPEED, 0, 0),
    "a": (-SPEED, 0, 0, 0), "d": (SPEED, 0, 0, 0),
    "r": (0, 0, SPEED, 0), "f": (0, 0, -SPEED, 0),
    "q": (0, 0, 0, -SPEED), "e": (0, 0, 0, SPEED),
}

tello = Tello()
tello.connect()
print("Battery:", tello.get_battery(), "%")
tello.streamon()
frames = tello.get_frame_read()

def safe_land():
    """Land, ignoring the 'error' the Tello gives when it is already down."""
    try:
        tello.land()
    except Exception as err:
        print("Land:", err)


writer = None
move, last_key, last_rc = (0, 0, 0, 0), 0.0, 0.0
flying = False

try:
    while True:
        frame = cv2.cvtColor(frames.frame, cv2.COLOR_RGB2BGR)
        if writer:
            writer.write(frame)

        battery, height = tello.get_battery(), tello.get_height()
        info = f"bat {battery}%  h {height}cm  temp {tello.get_highest_temperature()}C  {'REC' if writer else ''}"
        cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Tello", frame)

        key = cv2.waitKey(1) & 0xFF
        ch = chr(key) if key != 255 else ""

        if key == 27:                       # Esc
            break
        elif ch == "x":
            tello.emergency()
            flying = False
        elif ch == "t" and not flying:
            if battery < MIN_BATTERY:
                print(f"Battery {battery}% too low to take off")
            else:
                tello.takeoff()
                flying = True
        elif ch == "g" and flying:
            safe_land()
            flying = False
        elif ch == "v":
            if writer:
                writer.release()
                writer = None
            else:
                Path("recordings").mkdir(exist_ok=True)
                name = f"recordings/tello_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
                print("Recording to", name)
        elif ch in MOVES:
            move, last_key = MOVES[ch], time.time()

        if flying and height == 0 and time.time() - last_key > 3:
            print("Drone landed by itself (battery or temperature?)")
            flying = False

        if time.time() - last_key > HOLD:
            move = (0, 0, 0, 0)
        if flying and time.time() - last_rc > RC_EVERY:
            tello.send_rc_control(*move)
            last_rc = time.time()
finally:
    if flying:
        safe_land()
    if writer:
        writer.release()
    tello.streamoff()
    cv2.destroyAllWindows()
