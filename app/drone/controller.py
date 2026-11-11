"""PS5 controller for the Tello.

movement()     -> (side, front, height, turn), each -SPEED to SPEED, 0 when let go
new_presses()  -> buttons just pressed: X (take off), O (land), OPTION (emergency but will drop be carefull)

Test alone: uv run app/drone/controller.py once controller is bluetooth connected or by wire.
"""

import pygame

X = 0
O = 1
OPTION = 9

# Stick axes

# Movement
LEFT_STICK_SIDE = 0        # drone slides left / right
LEFT_STICK_UP_DOWN = 1     # drone flies forward / back

# Turn and height
RIGHT_STICK_SIDE = 3       # drone turns
RIGHT_STICK_UP_DOWN = 4    # drone climbs / descends

# Speed and joystick sensitivity
SPEED = 50                 # Max speed is 100, so be careful when setting it over 80
DEADZONE = 0.15            # Sensitivity of the stick, from 0 to 1. Also be careful when setting this

pygame.init()

controller = pygame.joystick.Joystick(0)


# Sticks and speed

def stick(axis):
    value = controller.get_axis(axis)    # -1.0 - 1.0
    if abs(value) < DEADZONE:
        return 0
    return int(value * SPEED)            # -50 - 50 when speed is set to 50


def movement():
    pygame.event.pump()                  # gets the newest stick position
    side = stick(LEFT_STICK_SIDE)
    front = -stick(LEFT_STICK_UP_DOWN)
    height = -stick(RIGHT_STICK_UP_DOWN)
    turn = stick(RIGHT_STICK_SIDE)
    return side, front, height, turn


held_before = []


def new_presses():
    global held_before
    pygame.event.pump()

    # 1. Buttons held down right now
    held_now = []
    for button in (X, O, OPTION):
        if controller.get_button(button):
            held_now.append(button)

    # 2. Buttons that were not held last time: those are new presses
    new = []
    for button in held_now:
        if button not in held_before:
            new.append(button)

    # 3. Store it for next time
    held_before = held_now
    return new


if __name__ == "__main__":
    import time
    while True:
        presses = new_presses()
        moves = movement()
        if presses or moves != (0, 0, 0, 0):
            print(presses, moves)
        time.sleep(0.2)