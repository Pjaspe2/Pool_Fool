# Custom pool YOLO — step by step

## Two different things (don’t mix them up)

| Step | What it does | Did you already do it? |
|------|----------------|------------------------|
| **A. Install YOLO on the Mac** | Lets `pool-fool-app` *run* detection | **Yes** — `pip install -e ".[yolo]"` |
| **B. Train a pool model** | Produces `best.pt` that knows billiard balls | **No** — still using generic COCO `yolov8n.pt` |

Installing YOLO does **not** train anything. Training is a separate one-time job on your Mac (30–90 minutes depending on dataset size).

**Is a downloaded labeled dataset enough?**  
**Yes.** Roboflow’s export already includes images + bounding boxes. You do **not** need to label again unless you add your own photos later.

---

## What you use today (before training)

- **Weights:** `yolov8n.pt` (COCO pretrained)
- **Filter:** class 32 = “sports ball”
- **Not** trained on pool, red felt, or your Pi camera

---

## Test the 3-epoch preview model (before 80-epoch training)

A short training run produced `config/models/pool_billiard_preview.pt`. Compare to COCO:

```bash
# Pi streaming, then on Mac:
pool-fool-app --config config/yolo_preview.yaml \
  --camera "http://pool.local:8080/stream.mjpg"
```

HUD should say `(custom)` and use magenta/green boxes. Quality will be mediocre — this is only a sanity check.

---

## Step-by-step: train on [pool-billiard](https://universe.roboflow.com/nidacorian-protonmail-com/pool-billiard)

### 0. Keep the Pi streaming (for later testing)

On the Pi:

```bash
pool-fool-edge --config config/default.yaml --mode stream
```

You don’t need the Pi for training — only for testing afterward.

---

### 1. Confirm YOLO is installed (same as before)

On the Mac, in the project folder:

```bash
cd ~/CursorCode/Pool_Fool
source .venv/bin/activate
pip install -e ".[yolo]"
yolo version
```

If that prints a version, you’re done with install. **No second YOLO install** for training.

---

### 2. Get the labeled dataset from Roboflow

1. Sign in at [roboflow.com](https://roboflow.com) (free tier is fine).
2. Open [Pool Billiard](https://universe.roboflow.com/nidacorian-protonmail-com/pool-billiard).
3. Click **Download Dataset**.
4. Format: **YOLOv8**.
5. Choose a split (e.g. train/valid/test) → **Continue** → download the **zip**.
6. Unzip somewhere simple, e.g.:

```text
~/Downloads/Pool-Billiard-1/
```

Inside you should see:

- `data.yaml`  ← path you pass to training
- `train/images/`, `train/labels/`
- `valid/images/`, `valid/labels/`

That zip **is** the labeled dataset — good enough to train as-is.

---

### 3. Train on the Mac

**Important:** Roboflow’s `data.yaml` often has wrong paths (`../train/images`). They must be:

```yaml
train: train/images
val: valid/images
test: test/images
```

(Already fixed if your dataset is at `Datasets/Pool Billiard.yolov8/`.)

```bash
cd ~/CursorCode/Pool_Fool
source .venv/bin/activate

yolo detect train \
  model=yolov8n.pt \
  data="Datasets/Pool Billiard.yolov8/data.yaml" \
  epochs=80 \
  imgsz=640 \
  batch=8 \
  name=pool_billiard \
  project=runs
```

Weights end up at: `runs/detect/pool_billiard/weights/best.pt`  
(~1–2 hours on Mac CPU for 80 epochs; a 3-epoch test run takes ~5 min.)

- First run may download `yolov8n.pt` again (normal).
- Training writes to `runs/detect/pool_billiard/`.
- When finished, your model is:

```text
runs/detect/pool_billiard/weights/best.pt
```

If `batch=8` runs out of memory, try `batch=4`.

---

### 4. See what classes the model learned

```bash
python -c "from ultralytics import YOLO; m=YOLO('runs/detect/pool_billiard/weights/best.pt'); print(m.names)"
```

Example output (yours may differ):

For **Pool Billiard** (your export):

```text
{0: 'Break', 1: 'Cue_Ball', 2: 'Eight', ... 6: 'Object_Ball', ...}
```

Use **`yolo_cue_class_ids: [1]`** for `Cue_Ball`. Optionally exclude **`Break`** (rack layout, not a rolling ball): add `break` to `yolo_exclude_class_names`.

---

### 5. Copy weights into the project

```bash
mkdir -p config/models
cp runs/detect/pool_billiard/weights/best.pt config/models/pool_billiard_best.pt
```

---

### 6. Point Pool Fool at the new model

Edit `config/default.yaml`:

```yaml
vision:
  yolo_model: config/models/pool_billiard_best.pt
  yolo_class_ids: all
  yolo_confidence: 0.35
  yolo_cue_class_ids: [1]   # Cue_Ball in Pool Billiard dataset
  yolo_exclude_class_names: [break]
```

Remove or comment out the old COCO-only line `yolo_class_ids: [32]`.

---

### 7. Test with the Pi stream

Pi streaming, then on the Mac:

```bash
pool-fool-app --config config/default.yaml \
  --camera "http://pool.local:8080/stream.mjpg"
```

HUD should show something like `YOLO config/models/pool_billiard_best.pt conf=0.35 (custom)`.

Tune if needed:

- Missed balls → lower `yolo_confidence` (e.g. `0.25`)
- False boxes on pockets → add names to `yolo_exclude_class_names`

---

## Blue felt dataset on your red table

- **Good enough to try** without extra work.
- **Better:** label 50–100 Pi frames with [`pool-fool-annotate`](annotate-your-images.md) or Roboflow, then fine-tune (see [annotate-your-images.md](annotate-your-images.md)).

---

## Quick FAQ

**Do I need a GPU?**  
Apple Silicon Mac uses MPS automatically in many setups; CPU training works but is slower.

**Do I train on the Pi?**  
No. Train on the Mac; Pi only streams video.

**Can I skip training and only tune config?**  
You can tune `yolo_confidence` / tracker, but COCO will stay weak on overhead pool. Training is the real fix.

**Roboflow “Download” vs “Clone”**  
Download YOLOv8 zip is what you want. Clone into your workspace is optional if you plan to add your own images in Roboflow.

---

## Reference: default vs custom

| | Default (now) | After training |
|--|---------------|----------------|
| File | `yolov8n.pt` | `config/models/pool_billiard_best.pt` |
| Trained on | COCO | pool-billiard (+ optional your images) |
| `yolo_class_ids` | `[32]` | `all` |

See also [pi-desktop-yolo.md](pi-desktop-yolo.md) for Pi + Mac runtime.
