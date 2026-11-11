# Drone

**Ryze Tello (TLW004)** and for this project was only flown indoors in safe open classroom environment. No licence needed indoors and drone is under 120grams in weigth. Flawn for testing and also for demo day over the terrarium with a PS5 controller. The laptop runs YOLO on the live video.

```
PS5 controller ──Bluetooth──► Laptop ──Wi-Fi──► Tello
                                ▲                 │
                                └── live video ───┘
```

## Fly

1. **Controller:** hold **PS + Create (small button top left)** until the light flashes, then connect to it in the laptop's Bluetooth settings.
2. **Drone:** add a battery that has charge and turn the Tello on and connect the laptop to its **TELLO-XXXXXX** Wi-Fi.
3. **Fly** (from the repo root):
   ```bash
   uv run app/drone/fly.py
   ```

| Control | Action |
|---|---|
| ✕ | take off |
| ○ | land |
| Options | **emergency: motors off so it will crash so be careful** |
| Left stick | forward / back, left / right |
| Right stick | up / down, turn |

**Before flying:** battery above 20%, enough space, and know where land button is, so O or worst case press Option button but drone might break.

## If it doesn't connect

- **Laptop jumps back to another Wi-Fi:** turn off auto-connect for your other networks.
- **No answer from the drone on Linux:** try allowing it through the firewall, ip address below is universal for all Tello in production:
  ```bash
  sudo ufw allow from 192.168.10.1 to any port 8890,11111 proto udp
  ```
