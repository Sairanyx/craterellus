"""
Connection to Tello drone, connects, takes off, lands, moves and streams video by default.
"""

from djitellopy import Tello

MIN_BATTERY = 20   # Official is 10% but we set 20% for warning

tello = Tello()
frame_read = None


def connect():
    tello.connect()
    print(f"Battery: {tello.get_battery()}%")


def take_off():
    if tello.get_battery() < MIN_BATTERY:
        print("Battery too low, insert higher charged battery or charge this one")
        return
    tello.takeoff()

def land():
    if tello.is_flying:
        tello.land()


def emergency():
    tello.emergency()


def move(side, front, height, turn):
    if tello.is_flying:
        tello.send_rc_control(side, front, height, turn)


def start_video():
    global frame_read
    tello.streamon()
    frame_read = tello.get_frame_read()


def get_frame():
    return frame_read.frame


def stop():
    land()
    tello.end()


if __name__ == "__main__":
    connect()
    stop()