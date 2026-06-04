# Vision roadmap: stick false positives, physics sim, table learning

## 1. Cue stick false detections (now)

The stick rarely appears as class `Cue_Stick` in the running model — the problem is **wood/tip boxes tagged as balls**.

**Layers in `yolo_preview.yaml`:**

| Layer | Setting |
|--------|---------|
| Class allow-list | `yolo_allow_class_names` — only real ball classes |
| Class block-list | `yolo_exclude_class_names` — pocket, stick, rack, … |
| Shape | `yolo_min_bbox_aspect: 0.72` — balls are round in overhead view |
| Color | `yolo_reject_wood_hue: true` — dark brown shaft on red felt |
| Region | play-region mask — outside felt ignored |
| Confidence | `yolo_confidence: 0.35` — tune up if still noisy |

If ghosts remain on the **tip**, try `yolo_confidence: 0.40` or add more labeled stick negatives in a future train.

---

## 2. Path A — Godot (or similar) physics sim

**Good for:** understanding cut angles, cushions, friction parameters, testing aim math offline.

**Suggested role:** reference simulator, not the live Pi pipeline.

| Pros | Cons |
|------|------|
| Repeatable shots | Must match your real table (cloth, rails, ball set) |
| Fast iteration on geometry | Overhead camera still needs real CV |
| Visual debug of ghost-ball + banks | Godot 2D physics ≠ pool physics out of the box |

**Practical sequence:**

1. Keep Python PoC lines (current `poc_pocket`).
2. Export table + pocket layout as mm (already have `pockets.npz`, homography).
3. Small Godot scene: balls as circles, cushions as segments, simple restitution.
4. Compare Godot aim line vs real video for a few staged shots.
5. Port tuned parameters back into Python (`table_physics.yaml`).

**Effort:** medium project (1–2 weeks for a useful prototype).

---

## 3. Path B — Session logging (low annotation)

**You do not need lots of bounding-box labels for B1.**

| What | Annotation? |
|------|----------------|
| Auto log ball positions | **None** — YOLO + homography already do this |
| Optional `m` / `x` key = made / missed | **One key per shot** (not Roboflow) |
| Retrain YOLO on stick frames | **Later**, only for failure cases |

**Session logger (built — `session.enabled` in config):**

```text
logs/session_YYYYMMDD_HHMMSS/
  meta.json       # table size, config path
  tracks.jsonl    # ball centers mm (automatic, throttled when stationary)
  events.jsonl    # m = made, x = missed + pocket + predicted line
```

In `pool-fool-app`: keys **1–6** pick pocket, **`[` / `]`** cycle, **`m`** made, **`x`** miss. Trajectory uses nearest object → selected pocket (same cut limits as PoC).

~20–50 logged outcomes can start tuning cushion friction. Hundreds help, but **not thousands of box labels**.

B1 **complements** physics: logs tell you if aim lines match reality; it does not replace cut-angle rules or a sim.

---

## 3b. Simple physics in Python (before Godot)

The PoC aim line now rejects **impossible cuts** (`poc_max_cut_angle_deg` in config, default 45°).

That is geometry, not cushions. Godot (path A) is still the right place for **banks** and multiple rails.

Config:

```yaml
vision:
  poc_max_cut_angle_deg: 45
```

---

## 4. Recommendation

| Order | Work |
|-------|------|
| **Now** | Session logging: pocket **1–6**, **m** made, **x** miss (`session.enabled`) |
| **Next** | Tune `poc_max_cut_angle_deg`; verify lines on real layouts |
| **Then** | **Python** cushion bounces OR small **Godot** if you need banks |
| **Later** | **B3** label only bad YOLO frames (stick), not whole dataset |

**A and B complement each other:** Godot finds equations; logs tell you if those equations match your room.

---

## 5. Quick tuning commands

```bash
pool-fool-app --config config/yolo_preview.yaml \
  --camera "http://pool.local:8080/stream.mjpg"
```

Session keys in the debug window: **1–6** pocket, **`[` / `]`** cycle, **m** made, **x** miss. Logs under `logs/session_*`.

If stick boxes persist, edit `config/yolo_preview.yaml`:

- `yolo_confidence: 0.40`
- `yolo_min_bbox_aspect: 0.78`
- `yolo_reject_wood_hue: true`

List model classes:

```bash
python -c "from ultralytics import YOLO; print(YOLO('config/models/pool_combined_best.pt').names)"
```

Adjust `yolo_allow_class_names` to match printed names exactly.
