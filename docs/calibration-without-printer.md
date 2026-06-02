# Calibration without a printer

## What you already have (no extra gear)

**Table calibration** (`pool-fool-calibrate table`) only needs four clicks on the felt. It uses your table size from `config/default.yaml`. No paper.

That fixes **perspective** (overhead trapezoid → table mm). It does **not** fully fix **fisheye** barrel bend.

---

## Option A — ChArUco on phone/tablet (best without printer)

1. Measure one **square** on screen with a ruler after opening the pattern.
2. Run:

```bash
pool-fool-calibrate lens-aruco --square-mm 40 --camera 0
```

3. Open `config/calibration/aruco_charuco.png` **fullscreen in Preview** on the laptop (not the capture window).
4. One window: **left** = camera, **right** = squared pattern (after **w**).
5. Press **w**, click 4 corners on the **left** view only (TL→TR→BR→BL).
6. When you see **>>> Press SPACE or s to capture <<<**, press **SPACE** (not **c**). Terminal should print `captured sample 1`.
7. Move phone/camera to new angles; capture **5+** samples; then **c** to finish.

**c** = finish only. **SPACE** / **s** = save a sample. If you press **c** with 0 samples, it will tell you to use SPACE first.
5. Re-run **table** calibration, then `pool-fool-app`.

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
