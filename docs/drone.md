# Drone

**Model:** Ryze Tello (TLW004)

## Now

- Flew it and recorded test videos over a mushroom-sized object just to test.
- **Control:** the Tello app on a phone.
- **Video:** saved on the phone. Getting it to the laptop is manual: save it in the app, then send it with LocalSend.
- **Camera:** points forward and can't tilt down. So gotta do it from afar a bit.

## Next: live processing

To process video live, the drone has to connect to the **laptop**, not the phone.

```
PS5 controller ──Bluetooth──► Laptop ──Wi-Fi──► Tello
                               ▲                  │
                               └──── live video ──┘
                                     → YOLO → Gradio
```

1. Connect the laptop to the Tello's Wi-Fi.
2. Pair a PS5 controller with the laptop over Bluetooth.
3. A Python script (`djitellopy` + `pygame`) flies the drone with the controller and receives the live video.
4. YOLO runs on each frame and the result shows in Gradio.

The phone and the laptop can't be connected to the drone at the same time.
