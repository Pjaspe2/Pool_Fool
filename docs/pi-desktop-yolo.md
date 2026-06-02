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

### Stream URL from your Mac

The Pi prints `http://0.0.0.0:8080/...` — that means “listen on all interfaces” **on the Pi only**. From the Mac, use the Pi’s hostname via **mDNS**:

```text
http://pool.local:8080/stream.mjpg
```

(`pool` is the hostname set in Raspberry Pi Imager; if yours differs, use `http://<hostname>.local:8080/stream.mjpg`.)

You can also use the Pi’s IP from `hostname -I` on the Pi, e.g. `http://192.168.12.115:8080/stream.mjpg`.

Verify in a browser or:

```bash
curl -I http://pool.local:8080/stream.mjpg
```

Set `network.overlay_udp_host` on the **Mac** to the same Pi (`pool.local` or the IP) so overlay lines reach the Pi.

## On the desktop (Mac)

### 1. Install YOLO (Mac Terminal, in the project venv)

```bash
cd ~/CursorCode/Pool_Fool   # or wherever you cloned it
source .venv/bin/activate
pip install -e ".[yolo]"
```

First run downloads `yolov8n.pt` (~6 MB). This can take several minutes.

Quick check:

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); print('ok')"
```

### 2. Config

`config/default.yaml` should already have:

```yaml
vision:
  detector: yolo
  yolo_model: yolov8n.pt
  yolo_confidence: 0.35
  yolo_imgsz: 640

network:
  overlay_udp_host: "pool.local"
```

Ball detection is **YOLO-only** (no Hough/felt pipeline).

### 3. Calibrate **once** using the Pi stream (same view as production)

Pi must be streaming (`pool-fool-edge --mode stream` or `combined`):

```bash
pool-fool-calibrate capture --camera "http://pool.local:8080/stream.mjpg"
pool-fool-calibrate table --image config/calibration/snapshot.jpg
pool-fool-calibrate play-region --image config/calibration/snapshot.jpg
```

### 4. Run YOLO + ghost-ball

```bash
pool-fool-app --config config/default.yaml \
  --camera "http://pool.local:8080/stream.mjpg" \
  --send-overlay
```

You should see `Ball detector: yolo` in the terminal.

### What you should see (no projector yet)

| Where | What |
|-------|------|
| **Browser** `http://pool.local:8080/stream.mjpg` | Raw camera only — **no** YOLO, **no** aim lines |
| **Mac app window** `pool_fool_debug` | Pi video + **magenta/green YOLO boxes** + ghost-ball lines when aiming works |
| **Pi projector** | Nothing until you run `--mode combined` + projector calibration |

**How to tell YOLO is active:** top-left HUD says `Balls: YOLO yolov8n.pt ...`, balls are drawn as **rectangles** (magenta = object, green = cue). Classical mode uses **white/orange circles** instead.

**Orange quad** = play-region mask from `play_region.npz` — saved calibration, **not** the ball detector. It persists across classical/YOLO and is correct only if the camera view still matches when you ran `play-region` (re-do after moving camera Mac → Pi).

Cue-stick Hough is **off** by default (`use_cue_line: false`). With two balls, `aim_mode: cue_to_object` draws a line cue→nearest object for testing.

The app does **not** draw on the browser stream. If no window appears, check the Terminal for `Stream connected` or `Stream error`, and look behind other windows for `pool_fool_debug`.

First YOLO frame can take **10–30 s** after the stream connects (model warmup).

**Keys:** `q` quit · `r` lock which ball is the cue ball (if YOLO picks the wrong one)

### YOLO tuning

| Symptom | Fix |
|--------|-----|
| No balls detected | Lower `yolo_confidence` (try `0.25`) |
| Too many false balls (pockets, reflections) | Raise `yolo_confidence` (try `0.45`); run `play-region` calibration |
| Mac feels sluggish | Set `yolo_imgsz: 416` or `yolo_frame_stride: 3` |
| Wrong cue ball | Press `r` while cue ball is visible |
| MOVING when ball is still | Lower `ball_tracker_alpha` (e.g. `0.08`); raise `stationary_velocity_mm_s` (e.g. `60`) |
| One white ball, weak detect | Lower `yolo_confidence` to `0.18`; keep `yolo_clahe: true` |

COCO class **32** is generic “sports ball” — a custom pool-ball model would help most long-term.

## Cannot open the stream from the Mac?

1. **Do not use `http://0.0.0.0:8080/...` on the Mac** — use `http://pool.local:8080/stream.mjpg` or the Pi’s IP.
2. **On the Pi** (while `pool-fool-edge` is running):
   ```bash
   curl -I http://127.0.0.1:8080/stream.mjpg
   ss -tlnp | grep 8080
   hostname    # e.g. pool → Mac URL is http://pool.local:8080/stream.mjpg
   ```
   You should see `HTTP/1.0 200` and something listening on `0.0.0.0:8080`.
3. **On the Mac**:
   ```bash
   ping -c 2 pool.local
   curl -I --max-time 5 http://pool.local:8080/stream.mjpg
   ```
   If mDNS fails, use `hostname -I` on the Pi and ping/curl that IP instead.
4. If (2) works but (3) fails: different Wi‑Fi/VLAN, **AP/client isolation**, or firewall — use **Ethernet on both** or disable guest-network isolation.
5. Safari can be picky with MJPEG; try **Chrome** or `pool-fool-app --camera "http://pool.local:8080/stream.mjpg"`.

## If it feels slow

- Use `yolov8n.pt` (default), not larger models
- Lower resolution: `cameras.width/height: 1280x720` or `640x480` on Pi
- Desktop: close other GPU apps
- Classical detector is faster but jitterier; YOLO trades latency for stability

## Why not YOLO on the Pi?

Pi 4 can run tiny YOLO slowly (~1–5 FPS). Your Mac does 10–30+ FPS. Splitting roles keeps the Pi as a simple, reliable **camera + display appliance**.
