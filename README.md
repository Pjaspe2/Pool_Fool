# Pool Fool

Ghost-ball pool assist: cameras at the table, vision on a desktop, projected aim lines on the felt.

## Architecture

- **Edge (Raspberry Pi 4):** USB camera capture, H.264 stream to desktop, receives overlay lines, HDMI to projector.
- **Desktop:** Table/camera calibration, ball and cue detection, ghost-ball geometry, overlay composition.

**Pi + Mac over Ethernet:** [docs/pi-desktop-yolo.md](docs/pi-desktop-yolo.md) — Pi streams MJPEG, desktop runs YOLO, UDP overlay back to projector.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### Red felt / lens (optional)

```bash
# Hover mouse on felt to read HSV; tune config vision.felt_hsv_red
pool-fool-calibrate felt-sample --config config/default.yaml --camera 0

# No printer: ChArUco on phone (measure one square on screen with a ruler)
pool-fool-calibrate lens-aruco --square-mm 40 --camera 0

# Or rough focal length from mount height (does not fix strong fisheye)
pool-fool-calibrate lens-estimate --camera-height-mm 1500

# Re-run table calibration after lens
```

See [docs/calibration-without-printer.md](docs/calibration-without-printer.md) for Xbox boxes, skipping lens, etc.

### Ball detection (YOLO)

```bash
pip install -e ".[yolo]"
```

Uses YOLOv8n COCO class 32 (“sports ball”). Tune in `config/default.yaml` under `vision:` (`yolo_confidence`, `ball_tracker_alpha`, etc.). See [docs/pi-desktop-yolo.md](docs/pi-desktop-yolo.md).

### Play-area mask (ignore pockets / rails)

```bash
pool-fool-calibrate play-region --config config/default.yaml --camera 0
# or from snapshot: --image config/calibration/snapshot.jpg
```

Click **inside the rails** (TL → TR → BR → BL), excluding pockets. Orange outline in the app shows the mask.

### 1. Calibrate overhead camera → table plane

```bash
pool-fool-calibrate table --config config/default.yaml --camera 0
```

Click four table corners (rail inside edges), then `s` to save homography to `config/calibration/table_homography.npz`.

### 2. Run desktop debug (webcam or video file)

```bash
pool-fool-app --config config/default.yaml --camera 0
```

Keys: `q` quit, `r` reset cue-ball hint, `c` re-run corner calibration.

### 3. Edge (on Pi)

```bash
# Stream MJPEG to desktop + listen for overlay + HDMI projector
pool-fool-edge --config config/default.yaml --mode combined

# Stream only (test camera + Mac browser)
pool-fool-edge --config config/default.yaml --mode stream
```

From the Mac, open the stream at **`http://<pi-hostname>.local:8080/stream.mjpg`** (e.g. `http://pool.local:8080/stream.mjpg` if the Pi hostname is `pool`). Do not use `0.0.0.0` in the browser — that address is only meaningful on the Pi.

On the desktop, point at the Pi stream and send overlays:

```bash
# config/default.yaml: overlay_udp_host = pool.local (or Pi IP)
pool-fool-app --config config/default.yaml \
  --camera http://pool.local:8080/stream.mjpg \
  --send-overlay --projector-preview
```

See [docs/pi-desktop-yolo.md](docs/pi-desktop-yolo.md) for the full Pi + Mac workflow.

### 4. Projector calibration

```bash
pool-fool-calibrate projector --config config/default.yaml --camera 0
```

## Tests

```bash
pytest
```

## Config

Edit [`config/default.yaml`](config/default.yaml) for table size, ball radius, network addresses, and vision thresholds.
