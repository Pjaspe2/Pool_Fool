# Custom pool YOLO model (Roboflow / your own data)

## What Pool Fool uses today (default)

| Setting | Value |
|---------|--------|
| **Weights** | `yolov8n.pt` — pretrained on **COCO** (not billiards) |
| **Class filter** | COCO class **32** = generic “sports ball” |
| **Dataset** | None — zero pool-specific training |

That is why overhead pool on **red felt** is unreliable: the model was never trained on your camera angle, lighting, or table.

Config (`config/default.yaml`):

```yaml
yolo_model: yolov8n.pt
yolo_class_ids: [32]
```

---

## Your Roboflow dataset

[Pool Billiard on Roboflow Universe](https://universe.roboflow.com/nidacorian-protonmail-com/pool-billiard) — billiard-focused boxes (balls, pockets, etc., depending on how it was labeled).

### Will blue-felt training work on red felt?

**Often yes, well enough to try** — especially for **balls** (round, high contrast vs felt).

- YOLO learns shape and context (table, rails, ball size), not only hue.
- **Red vs blue felt** is a domain shift; expect some drop in accuracy until you add **your own images**.
- Best path: train on Roboflow data **plus** 50–200 frames from your Pi stream (`pool-fool-calibrate capture` snapshots).

`yolo_clahe: true` in config already helps contrast on red felt at inference time.

---

## Train from Roboflow (YOLOv8)

1. Create a free [Roboflow](https://roboflow.com) account.
2. Open the dataset → **Download** → format **YOLOv8** → unzip.
3. On your Mac (in the project venv):

```bash
source .venv/bin/activate
pip install -e ".[yolo]"

yolo detect train \
  model=yolov8n.pt \
  data=/path/to/download/data.yaml \
  epochs=80 \
  imgsz=640 \
  batch=8 \
  name=pool_billiard
```

4. Best weights are usually:

```text
runs/detect/pool_billiard/weights/best.pt
```

5. Copy into the repo and point config at it:

```yaml
yolo_model: config/models/pool_billiard_best.pt
yolo_class_ids: all          # use every ball class in the .pt
yolo_confidence: 0.35
yolo_cue_class_ids: [0]      # set to cue-ball class id from data.yaml (see names)
yolo_exclude_class_names: [pocket, bag, rack, table, flag]
```

6. Find class IDs after training:

```bash
python -c "from ultralytics import YOLO; m=YOLO('config/models/pool_billiard_best.pt'); print(m.names)"
```

Set `yolo_cue_class_ids` to the index of `cue ball` / `cue_ball` if the dataset has it.

---

## Improve accuracy on *your* table (recommended)

1. Export 100+ frames from the Pi: `http://pool.local:8080/stream.mjpg` or `pool-fool-calibrate capture`.
2. Upload to Roboflow (or label locally with CVAT / Roboflow).
3. Label **balls only** first (simplest) — one class `ball` or separate `cue_ball` + `ball`.
4. **Train** again starting from `yolov8n.pt` or from your first `best.pt` (short “fine-tune” run, ~30 epochs).

Red-felt images in the mix matter more than perfect blue-felt diversity.

---

## Similar public datasets

If [pool-billiard](https://universe.roboflow.com/nidacorian-protonmail-com/pool-billiard) is small, search Roboflow Universe for **billiard**, **pool ball**, **pool table** — e.g. [billiard-pool](https://universe.roboflow.com/xiong-pwuvd/billiard-pool-qugr8), [RF100 billiard balls](https://universe.roboflow.com/rf100-vl/ball-qgqhv-2mtfk-ch2i9-ejgb). Same workflow: download YOLOv8 → train → `best.pt`.

---

## After you have `best.pt`

```bash
pool-fool-app --config config/default.yaml \
  --camera "http://pool.local:8080/stream.mjpg"
```

HUD should still show `Balls: YOLO pool_billiard_best.pt ...`. Tune `yolo_confidence` and `ball_tracker_alpha` as in [pi-desktop-yolo.md](pi-desktop-yolo.md).
