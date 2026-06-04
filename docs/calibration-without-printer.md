# Calibration without a printer

## What you already have (no extra gear)

**Table calibration** (`pool-fool-calibrate table`) only needs four clicks on the felt. It uses your table size from `config/default.yaml`. No paper.

That fixes **perspective** (overhead trapezoid → table mm). It does **not** fully fix **fisheye** barrel bend.

---

## Option A — ChArUco on laptop screen (best without printer)

**Pi overhead + bent rails:** see [lens-calibration-pi.md](lens-calibration-pi.md) (15+ samples, reprojection error, `undistort_alpha`).

Works with the **Pi overhead stream** (pattern on the MacBook screen under the camera) or a local webcam (`--camera 0`).

### 1. Measure the pattern

```bash
open config/calibration/aruco_charuco.png
```

In **Preview**, View → Enter Full Screen. Measure **one black/white square** edge-to-edge with a ruler (mm). Example: 38 mm → use `--square-mm 38`.

### 2. Physical setup (Pi)

- Pi running: `pool-fool-edge --mode stream`
- **Close** any browser tab on `http://pool.local:8080/stream.mjpg`
- Place the **laptop on the table** under the Pi camera (screen up, pattern fullscreen)
- Room lights on; avoid glare on the screen (tilt screen slightly if needed)

### 3. Run lens calibration

```bash
pool-fool-calibrate lens-aruco \
  --config config/yolo_preview.yaml \
  --square-mm 38 \
  --camera "http://pool.local:8080/stream.mjpg" \
  --no-fullscreen
```

`--no-fullscreen` keeps the pattern in Preview only (do not use the tool’s built-in fullscreen pattern).

### 4. In the capture window

| Key | Action |
|-----|--------|
| **w** | Click 4 corners of the **laptop screen** on the **LEFT** (Pi) panel: TL → TR → BR → BL |
| **SPACE** / **s** | Save a sample when corners turn **green** and you see `>>> Press SPACE or s to capture <<<` |
| **c** | Finish after **5+** samples (different positions/tilts of the laptop) |
| **q** | Quit without saving |

**c** = finish only. **SPACE** = capture. Move the laptop to new positions under the camera between captures.

### 5. Enable undistort and re-calibrate table

In `config/yolo_preview.yaml` (or `default.yaml`):

```yaml
cameras:
  undistort: true
```

```bash
pool-fool-calibrate verify-lens \
  --config config/yolo_preview.yaml \
  --camera "http://pool.local:8080/stream.mjpg"

pool-fool-calibrate table \
  --config config/yolo_preview.yaml \
  --camera "http://pool.local:8080/stream.mjpg"

pool-fool-calibrate play-region \
  --config config/yolo_preview.yaml \
  --camera "http://pool.local:8080/stream.mjpg"
```

Then run `pool-fool-app` with the same config and stream URL.

### Local webcam (no Pi)

```bash
pool-fool-calibrate lens-aruco --config config/default.yaml --square-mm 40 --camera 0 --no-fullscreen
```

Same keys; LEFT panel is the Mac webcam.

---

## Option B — Camera height + table (rough, no pattern)

If the camera is roughly straight above the table center:

```bash
pool-fool-calibrate lens-estimate --camera-height-mm 1500
```

Use your real height in mm (lens to felt). This sets focal length only; **distortion stays zero**. OK for mild wide-angle, not for strong fisheye.

---

## Option C — Xbox / game boxes on the felt

Use **identical** flat boxes (same game case size):

1. Place them in a **grid** on the red felt (e.g. 3×4), faces up, edges aligned.
2. Measure **one box length and width** in mm with a tape measure.
3. We do not auto-detect boxes yet; for lens calibration prefer Option A.

Boxes work well as a **mental check** of scale after table calibration, not as a full lens model unless you build a custom grid detector.

---

## Option D — Skip lens calibration (recommended if undistort looks worse)

Set in `config/default.yaml`:

```yaml
cameras:
  undistort: false
```

Table homography alone is enough to test ball detection and ghost-ball lines.

**Do not “invert” distortion coefficients** — negating them is not a valid undo. Disable `undistort` or rename `lens.npz` (e.g. to `lens.npz.bad`) and use the raw camera image.

If `lens_verify.jpg` looks more fisheye on the right, the cal failed (often wrong `--square-mm` on the phone or too few angles). Ignore lens until you re-try later.

---

## Recommended order

1. `felt-sample` — tune red felt HSV  
2. `table` — four corner clicks  
3. `lens-aruco` **or** `lens-estimate` **or** skip lens  
4. Re-`table` if you ran lens  
5. `pool-fool-app` — press **r** on cue ball when needed  
