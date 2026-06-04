# Lens calibration (Pi camera + laptop ChArUco)

Use this when **rails look curved** in the overhead view after `undistort: true`. Lens calibration fixes **camera fisheye/barrel** distortion; you still need **table** + **play-region** homography afterward.

---

## Before you start

1. **Pi streaming:** `pool-fool-edge` on the Pi; close browser tabs on `/stream.mjpg`.
2. **Backup** old lens file (optional):
   ```bash
   cp config/calibration/lens.npz config/calibration/lens.npz.backup
   ```
3. **Measure** one square on the fullscreen pattern (Preview) with a ruler → `--square-mm`.
4. **Regenerate pattern** (happens automatically when you run `lens-aruco`):
   ```bash
   open config/calibration/aruco_charuco.png
   ```
   View → Enter Full Screen on the **laptop screen under the Pi** (not on the Mac UI monitor if you use two displays).

---

## Run calibration (recommended flags)

```bash
cd /path/to/Pool_Fool
source .venv/bin/activate

pool-fool-calibrate lens-aruco \
  --config config/yolo_preview.yaml \
  --square-mm 36 \
  --camera "http://pool.local:8080/stream.mjpg" \
  --no-fullscreen \
  --min-images 15 \
  --min-corners 18
```

Use **your** measured `--square-mm`. For a difficult Pi stream, try `--min-images 18`.

---

## Capture strategy (fixes “bent top rail”)

The Pi sees only part of the table at a time during lens cal—you show a **pattern on the laptop**, not the pool table. You need **variety across the image**:

| Pose | Why |
|------|-----|
| Pattern centered | Baseline |
| Pattern shifted so it fills the **top** of the Pi image | Corrects distortion at top of frame |
| Pattern shifted to **bottom** | Corrects bottom |
| **Left / right** shift | Side distortion |
| **Tilt** laptop slightly (5–15°) | Different viewing angles |
| **Closer / farther** under camera | Scale change |

Press **SPACE** only when:

- Corner count ≥ `min_corners` (green status / prompt).
- Terminal prints `>>> captured sample N (XX points) <<<` with **XX ≥ 15**.

Press **z** to undo the last sample. Press **c** when you have enough samples (≥ `min-images`).

**Do not use Ctrl+C to finish** — samples live in memory until `c` runs (or Ctrl+C with enough samples now auto-saves). Ctrl+C before that loses everything.

If you see `[mpjpeg] Expected boundary '--' not found`, update the repo (calibration uses the Pi-safe MJPEG reader, not OpenCV `VideoCapture`).

**Optional `w`:** outline the laptop screen on the LEFT panel if detection is weak. If you move the laptop, press **w** again or skip warp and rely on varied poses.

---

## After `c` — read the error

Terminal prints **Mean reprojection error**:

| Error (px) | Meaning |
|------------|---------|
| **< 0.5** | Excellent — proceed |
| **0.5 – 1.0** | OK — verify on real table |
| **> 1.0** | Recalibrate with more poses or fix `--square-mm` |

---

## Verify and tune

```bash
pool-fool-calibrate verify-lens \
  --config config/yolo_preview.yaml \
  --camera "http://pool.local:8080/stream.mjpg"
```

Open `config/calibration/lens_verify.jpg`. Straight lines on the **right** should look straighter than the left.

If the **top rail** is still curved on the real table view:

1. In `config/yolo_preview.yaml` try:
   ```yaml
   cameras:
     undistort_alpha: 0.15
   ```
   (then `0.25` if needed; range 0.0–0.5).

2. Re-run **verify-lens** and `pool-fool-app`.

---

## Re-calibrate geometry (required after new lens)

Lens changes pixel mapping. **Redo from Pi stream:**

```bash
pool-fool-calibrate table \
  --config config/yolo_preview.yaml \
  --camera "http://pool.local:8080/stream.mjpg"

pool-fool-calibrate play-region \
  --config config/yolo_preview.yaml \
  --camera "http://pool.local:8080/stream.mjpg"
```

Then run `pool-fool-app` and check orange play region + cyan pockets + YOLO boxes.

---

## Still curved?

- **Perspective**, not lens: table homography wrong (re-click table corners on **felt** TL→TR→BR→BL).
- **Partial table in frame:** only the visible felt region is accurate; center may be better than top/bottom edges.
- **MJPEG compression:** try lower Pi resolution for cal, or more light on the pattern.
- **Skip lens:** set `undistort: false` and rely on table homography only (often good enough for ball centers).
