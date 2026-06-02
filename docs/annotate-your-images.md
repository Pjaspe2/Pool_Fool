# Annotate your own table images

Three ways to add labels, from fastest (local app) to most polished (Roboflow web).

---

## Option A — `pool-fool-annotate` (built-in, recommended)

Grabs frames from your Pi or labels existing photos. Saves **YOLO format** ready for training.

### Capture workflow (best practice)

Your loop is right. Refined version:

1. **Lighting** — Bright, even light on the felt (avoid harsh shadows on balls). Match how you’ll play (same room lights / exposure as `pool-fool-edge`).
2. **Pi streaming** — Fixed camera; don’t move the mount between capture and production.
3. **Arrange balls** — Random-ish spread; vary **count** (1 ball, 3, 8, full rack) and **positions** (rails, center, clusters). Include **only cue ball** in several frames.
4. **Grab** — `g` when balls are **still** (motion blur hurts labels).
5. **Label** — Tight box on **every visible ball** in that frame. Use **`--simple`** first (`Cue_Ball` + `Object_Ball`); use full 12 classes only if you need numbered balls later.
6. **Save** — `s` saves labels (and copies image into `train/images/` if needed).
7. **Scramble** — Move balls, repeat 4–6 for **~50–100 images** (50 minimum for a noticeable boost; 100+ is better).

Tips:

- Skip frames where balls overlap heavily until you’re comfortable drawing small boxes.
- A few **empty table** frames are optional (teaches the model “no ball” — only if you add a class or use them sparingly).
- Don’t label pockets/rails — only balls.
- After labeling, fine-tune from `pool_billiard_preview.pt` before a full 80-epoch run (see bottom of this doc).

### Quick start (Pi stream → your red felt)

Pi streaming:

```bash
pool-fool-edge --config config/default.yaml --mode stream
```

Mac:

```bash
cd ~/CursorCode/Pool_Fool
source .venv/bin/activate
pip install -e .

pool-fool-annotate \
  --output Datasets/my_red_felt \
  --stream "http://pool.local:8080/stream.mjpg" \
  --classes-yaml "Datasets/Pool Billiard.yolov8/data.yaml"
```

**Faster (2 classes only):**

```bash
pool-fool-annotate \
  --output Datasets/my_red_felt \
  --stream "http://pool.local:8080/stream.mjpg" \
  --simple
```

### Keys

| Key | Action |
|-----|--------|
| **Drag** mouse | Draw box around a ball |
| **Tab** / **,** / **.** | Next / previous class |
| **0–9** | Jump to class id |
| **g** | Grab frame from Pi → new `capture_XXXX.jpg` (balls should be still) |
| **s** | Save labels for this image (required before next grab) |
| **u** | Undo last box |
| **n** / **p** | Next / previous image |
| **q** | Save and quit |

Output layout:

```text
Datasets/my_red_felt/
  data.yaml
  train/images/capture_0001.jpg
  train/labels/capture_0001.txt
```

Each `.txt` line: `class_id center_x center_y width height` (normalized 0–1).

### Label existing photos

Put `.jpg` files in `Datasets/my_red_felt/train/images/` or pass a folder:

```bash
pool-fool-annotate \
  --output Datasets/my_red_felt \
  --images-dir ~/Pictures/pool_frames
```

---

## Option B — Roboflow (web UI)

Good if you want team review, augmentations, and one-click export.

1. [roboflow.com](https://roboflow.com) → **Create Project** → Object Detection.
2. **Upload** images (export from Pi or use `pool-fool-calibrate capture`).
3. **Annotate** with bounding boxes — use the same class names as your main dataset when possible (`Cue_Ball`, `Eight`, …).
4. **Generate** a new dataset version (train/valid split).
5. **Export** → **YOLOv8** → download zip.
6. **Merge** with Pool Billiard:
   - Copy your images into `train/images/`
   - Copy labels into `train/labels/`
   - Or upload everything into one Roboflow project and export once.

Fix `data.yaml` paths if needed (`train: train/images`, not `../train/images`).

---

## Option C — LabelImg / CVAT

- [LabelImg](https://github.com/heartexlabs/labelImg): desktop app, export YOLO.
- [CVAT](https://www.cvat.ai): browser, export YOLO 1.1.

Same idea: boxes per ball → export → place under `train/images` + `train/labels`.

---

## Train after adding your images

### Only your new folder (small set)

```bash
yolo detect train \
  model=yolov8n.pt \
  data="Datasets/my_red_felt/data.yaml" \
  epochs=50 \
  imgsz=640 \
  batch=8 \
  name=my_red_felt \
  project=runs
```

### Fine-tune from preview or full pool model (better)

After you have `config/models/pool_billiard_best.pt` (or `pool_billiard_preview.pt`):

```bash
yolo detect train \
  model=config/models/pool_billiard_preview.pt \
  data="Datasets/my_red_felt/data.yaml" \
  epochs=30 \
  imgsz=640 \
  batch=8 \
  name=my_red_felt_ft \
  project=runs
```

### Merge datasets in Roboflow (cleanest)

Upload Pi captures into the same project as Pool Billiard → export one YOLOv8 zip → train once on the combined set.

---

## Tips for good labels

- Draw boxes **tight** on each visible ball (not the whole cluster).
- Include varied shots: 1 ball, many balls, cue ball only, different table areas.
- **50–100+ images** from your real Pi camera beats hundreds of blue-felt-only images.
- Use **`--simple`** while learning; switch to full class list when you care which ball is which.
- Re-export and **re-train** after each batch of new labels.

See [pool-yolo-custom-model.md](pool-yolo-custom-model.md) for full training steps.
