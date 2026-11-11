# Drone

**Ryze Tello (TLW004).** It flies indoors only, over the terrarium.

## Camera

- **Direction:** points forward, can't tilt down. Fly back and up to see the terrarium.
- **Live stream:** 960×720, H.264, over Wi-Fi.
- **Phone app recordings:** 1280×720, 30 fps MP4.

## Two ways to use it

| | Phone (Tello app) | Laptop (live demo) |
|---|---|---|
| Control | the app | keyboard or PS5 controller (Python) |
| Video | saved on the phone | streamed live to the laptop → YOLO → Gradio |
| Use for | test flights, recordings | the live demo |

The phone and the laptop can't connect to the drone at the same time.

## Live demo setup

```
Controller ──Bluetooth──► Laptop ──Wi-Fi──► Tello
                            ▲                 │
                            └── live video ───┘
                                → YOLO (laptop GPU) → Gradio
```

1. Connect the laptop to the **TELLO-XXXXXX** Wi-Fi.
2. **Linux firewall:** allow the Tello's packets, or the video and status never arrive:
   ```bash
   sudo ufw allow from 192.168.10.1 to any port 8890,11111 proto udp
   ```
3. A Python script (`djitellopy`) flies the drone and receives the stream.
4. YOLO runs on each frame on the laptop, and Gradio shows the result.

## Safety

- **Battery:** land below 20%. The Tello also lands by itself when the battery runs low.
- **Emergency key:** have one that stops the motors.
- **Terrarium:** press the moss down, so the propeller wind doesn't move it.
- **Backup:** record a flight in advance, in case the live demo fails.
