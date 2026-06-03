# Roadmap: pockets, cue stick, numbered balls

## Done in this step

1. **Removed** placeholder aim line (cue → nearest object). Ghost-ball lines only when `use_cue_line: true` (side camera / Hough).
2. **Play region expand** — `play_region_expand_scale` in config (e.g. `1.2` = 20% larger than saved orange quad). Best fix: re-calibrate play region from the **Pi stream**.
3. **Class schema v2** — `config/annotation_classes_v2.yaml` adds `Pocket` and `Cue_Stick` for the **next** training run.

Your current `pool_combined_best.pt` was trained on **12 classes** only — it will **not** detect pockets or cue until you label and retrain.

---

## 1. Expand sensing area (orange box)

**Quick fix (try first):** in `config/yolo_preview.yaml`:

```yaml
play_region_expand_scale: 1.25   # 25% larger than saved quad
```

**Proper fix:** re-draw play region from Pi view (matches production camera):

```bash
pool-fool-calibrate capture --camera "http://pool.local:8080/stream.mjpg"
pool-fool-calibrate play-region --image config/calibration/snapshot.jpg
```

Click **inside the rails** (TL → TR → BR → BL), well toward the cushions — not a small center patch.

Set `play_region_expand_scale: 1.0` after a good re-calibration.

---

## 2. Annotate for v2 (balls by number + pocket + cue)

Use the **full** class list (not `--simple`):

```bash
pool-fool-annotate \
  --output Datasets/my_table_v2 \
  --stream "http://pool.local:8080/stream.mjpg" \
  --classes-yaml config/annotation_classes_v2.yaml
```

### Class IDs (press 0–9 for first ten)

| Key | Class |
|-----|--------|
| 0 | Break |
| 1 | **Cue_Ball** |
| 2 | Eight |
| 3 | Five |
| 4 | Four |
| 5 | Nine |
| 6 | Object_Ball (unknown solid) |
| 7 | One |
| 8 | Seven |
| 9 | Six |

Tab through classes for **Three**, **Two**, **Pocket**, **Cue_Stick** (or add digit keys later).

### What to label per frame

- Every **visible ball** with the correct **number class** (or `Object_Ball` if unsure).
- Each **pocket opening** (one box per pocket in view) — class `Pocket`.
- **Cue stick** shaft visible in frame — class `Cue_Stick` (one box along the stick).

You can batch work:

1. **Phase A** — 30 frames, balls only (numbered + cue).
2. **Phase B** — same frames, add pockets.
3. **Phase C** — 20 frames with cue stick in view, label stick + balls.

---

## 3. Train v2 model (when you have ~50+ v2 labels)

Merge Roboflow + your v2 folder (update `merge_yolo_datasets.py` to use v2 `data.yaml` names — or export one combined Roboflow project).

```bash
python scripts/merge_yolo_datasets.py \
  --roboflow "Datasets/Pool Billiard.yolov8" \
  --custom "Datasets/my_table_v2" \
  --output "Datasets/pool_combined_v2" \
  --custom-repeat 4 \
  --classes-yaml config/annotation_classes_v2.yaml

yolo detect train \
  model=config/models/pool_combined_best.pt \
  data="Datasets/pool_combined_v2/data.yaml" \
  epochs=80 \
  imgsz=640 \
  batch=8 \
  name=pool_combined_v2 \
  project=.
```

Note: adding 2 new classes changes `nc` from 12 → 14 — you **must** retrain; the old `.pt` head will not match.

---

## 4. App behavior after v2

Config already excludes non-balls from tracking:

```yaml
yolo_exclude_class_names: [break, pocket, bag, rack, table, flag, cue_stick, cue stick, stick]
```

Pockets and cue stick are detected but **not** treated as balls. Next step (later): use `Cue_Stick` for aim direction instead of Hough.

---

## Suggested order

1. Set `play_region_expand_scale` or re-calibrate play region → verify full table coverage.
2. Run app with `aim_mode: none` — balls only, no misleading line.
3. Annotate v2 in phases when ready; keep using current `pool_combined_best.pt` until v2 train finishes.
