# Table measurements and coordinates

Pool Fool maps the camera image to a **2D table plane in millimeters**. Ball centers and pocket targets live in that plane. You do **not** need YOLO to find pockets for the first shot-assist version—pockets come from your measured table size plus a small inset you can tune.

---

## Coordinate system (matches calibration)

When you run `pool-fool-calibrate table`, you click **four corners of the playing surface** in this order:

**TL → TR → BR → BL** (on the **felt**, inside the cushions—not the outer wood).

Those map to:

| Click | Table mm | Meaning |
|-------|----------|---------|
| 1 (TL) | `(0, 0)` | Origin corner |
| 2 (TR) | `(length_mm, 0)` | Along the **long** rail from TL |
| 3 (BR) | `(length_mm, width_mm)` | Diagonal corner |
| 4 (BL) | `(0, width_mm)` | Along the **short** rail from TL |

- **x** = `length_mm` direction (long side of the table; side pockets are centered on this axis at `y ≈ width/2`).
- **y** = `width_mm` direction (short side).

Your overhead view must use the **same** corner order when clicking. If the app’s gray table outline is rotated 90° vs reality, swap `length_mm` and `width_mm` in config **or** re-click corners in the correct order.

---

## What to measure with a tape measure

All dimensions are **playing surface** (inside the cushion noses), not the outer cabinet.

### 1. Playing length (`length_mm`)

Measure the **long** side of the felt rectangle, cushion nose to cushion nose.

- Typical **9-foot** table: **2540 mm** (100 in).
- **8-foot**: ~2240 mm (88 in).

### 2. Playing width (`width_mm`)

Measure the **short** side, cushion nose to cushion nose.

- Typical **9-foot**: **1270 mm** (50 in).
- **8-foot**: ~1120 mm (44 in).

Put these in `config/yolo_preview.yaml` (or `default.yaml`):

```yaml
table:
  length_mm: 2540.0   # your measured long side
  width_mm: 1270.0    # your measured short side
  ball_radius_mm: 28.575   # 57.15 mm diameter ÷ 2 (standard); measure one ball if unsure
```

### 3. Ball diameter (optional check)

Standard pool ball: **57.15 mm** (2¼ in) → `ball_radius_mm: 28.575`.

### 4. Pocket center inset (`pockets.center_inset_mm`)

Pocket **centers** are not exactly at `(0,0)`. In software they sit **inset** from each playing corner along both rails.

- Measure from the **playing corner** (felt/cushion junction) along each rail to where you want the “center” of the pocket (often roughly the middle of the pocket opening).
- Typical range: **50–70 mm** (about 2–2.5 in). Default in config: **57 mm**.

**Default (recommended):** with `pockets_from_play_region: true`, pockets are placed on your **play-region quad** (same as the orange calibration). Tune:

```yaml
table:
  pockets_from_play_region: true
  pockets:
    inset_fraction: 0.06   # 0.04–0.10: move cyan dots toward center from each corner/edge
```

Re-run `pool-fool-app` after changing `inset_fraction` until cyan circles sit on real pockets.

**Legacy:** `pockets_from_play_region: false` uses the full config rectangle and `center_inset_mm` instead.

---

## What you do **not** need for PoC

| Skip for now | Why |
|--------------|-----|
| Cabinet / slate size | Only playing surface matters for ball mm |
| Diamond spots on wood | Not used in v1 layout |
| YOLO “pocket” class | Fixed geometry is enough |
| Rail height | 2D model first |

---

## Calibration checklist (Pi + undistort)

1. `cameras.undistort: true` if lens verify looked good.
2. `pool-fool-calibrate table` — four felt corners, TL→TR→BR→BL.
3. `pool-fool-calibrate play-region` — quad **inside** rails (balls only).
4. Set `length_mm` / `width_mm` from tape measure.
5. Run `pool-fool-app` — check **gray** table border and **cyan** pocket markers.

---

## Click pocket positions (recommended)

Same idea as play-region: one frozen frame from the Pi, click each pocket center.

```bash
pool-fool-calibrate pockets \
  --config config/yolo_preview.yaml \
  --camera "http://pool.local:8080/stream.mjpg"
```

Click in order (prompt on screen):

1. corner_tl — near your table **TL** corner pocket  
2. corner_tr, corner_br, corner_bl  
3. side_left, side_right — middle of the **long** rails  

Press **s** when all 6 are placed. Saves `config/calibration/pockets.npz`.

In config:

```yaml
table:
  use_calibrated_pockets: true
```

Run `pool-fool-app` — cyan dots use your clicks. Orange play-region is still drawn for reference.

To fall back to auto-estimated pockets, set `use_calibrated_pockets: false` or delete `pockets.npz`.

---

## Pocket IDs (debug overlay)

| ID | Location |
|----|----------|
| `corner_tl` | Near origin corner |
| `corner_tr` | Long rail, y = 0 |
| `corner_br` | Opposite origin diagonally |
| `corner_bl` | Short rail from origin |
| `side_left` | Mid of long rail at small x inset |
| `side_right` | Mid of long rail at large x |

---

## Next step (shot assist)

Once pockets line up visually:

- Ball positions come from YOLO + homography.
- Pocket targets come from `TableLayout`.
- PoC shot logic: pick object ball → pick easiest pocket → draw aim line (straight, no cushions yet).

See `pool_fool/shared/table_layout.py`.
