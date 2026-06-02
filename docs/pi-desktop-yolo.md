# Pi camera + desktop YOLO (recommended setup)

The Pi stays at the ceiling with USB camera + projector HDMI. Your Mac/PC does vision (YOLO) over **Ethernet**.

```
  [USB cam] → [Pi 4] ──MJPEG (~5–15 Mbps)──→ [Desktop: YOLO + ghost-ball]
                    ↑                                    │
                    └──── overlay JSON (UDP, tiny) ──────┘
```

This is **usable** for shot lining (balls stationary, ~5–15 updates/s). It is **not** for tracking a fast break in real time.

## Expected latency (720p, wired Ethernet)

| Stage | Typical |
|-------|---------|
| Pi capture + JPEG encode | 15–40 ms |
| Network | 5–20 ms |
| YOLOv8n on desktop GPU | 10–40 ms |
| YOLOv8n on desktop CPU only | 80–250 ms |
| Overlay UDP → Pi | &lt; 5 ms |

**Total:** ~100–200 ms on a GPU desktop, ~200–400 ms on CPU-only — fine when balls are still and you are aiming.

Use **Ethernet**, not Wi‑Fi, and **720p** in config (not 1080p).

## On the Pi (at the table)

```bash
cd ~/Pool_Fool
source .venv/bin/activate
pip install -e .

# Stream only (camera + MJPEG; projector overlay optional)
pool-fool-edge --config config/default.yaml --mode combined
```

`combined` = MJPEG on port **8080** + listen for overlay lines on UDP **8765**.

Set Pi IP in `config/default.yaml` on **both** machines under `network.overlay_udp_host` (desktop sends to Pi).

## On the desktop (Mac)

1. Install YOLO extras:

```bash
pip install -e ".[yolo]"
```

2. Edit `config/default.yaml`:

```yaml
vision:
  detector: yolo

network:
  overlay_udp_host: "192.168.1.XXX"   # Pi IP address
```

3. Calibrate **once** using the Pi stream (same view as production):

```bash
pool-fool-calibrate capture --camera "http://192.168.1.XXX:8080/stream.mjpg"
pool-fool-calibrate table --image config/calibration/snapshot.jpg
pool-fool-calibrate play-region --image config/calibration/snapshot.jpg
```

4. Run processing:

```bash
pool-fool-app --config config/default.yaml \
  --camera "http://192.168.1.XXX:8080/stream.mjpg" \
  --send-overlay
```

Debug window on the Mac; projector on the Pi shows lines when balls are **STATIONARY**.

## If it feels slow

- Use `yolov8n.pt` (default), not larger models
- Lower resolution: `cameras.width/height: 1280x720` or `640x480` on Pi
- Desktop: close other GPU apps
- Classical detector is faster but jitterier; YOLO trades latency for stability

## Why not YOLO on the Pi?

Pi 4 can run tiny YOLO slowly (~1–5 FPS). Your Mac does 10–30+ FPS. Splitting roles keeps the Pi as a simple, reliable **camera + display appliance**.
