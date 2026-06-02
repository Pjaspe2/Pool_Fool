# Pi camera + desktop YOLO (recommended setup)

The Pi stays at the ceiling with USB camera + projector HDMI. Your Mac/PC does vision (YOLO) over **Ethernet**.

```
  [USB cam] → [Pi 4] ──MJPEG (~5–15 Mbps)──→ [Desktop: YOLO + ghost-ball]
                    ↑                                    │
                    └──── overlay JSON (UDP, tiny) ──────┘
```

This is **usable** for shot lining (balls stationary, ~5–15 updates/s). It is **not** for tracking a fast break in real time.

## Expected latency (720p, wired Ethernet)

| Stage | Typical |
|-------|---------|
| Pi capture + JPEG encode | 15–40 ms |
| Network | 5–20 ms |
| YOLOv8n on desktop GPU | 10–40 ms |
| YOLOv8n on desktop CPU only | 80–250 ms |
| Overlay UDP → Pi | &lt; 5 ms |

**Total:** ~100–200 ms on a GPU desktop, ~200–400 ms on CPU-only — fine when balls are still and you are aiming.

Use **Ethernet**, not Wi‑Fi, and **720p** in config (not 1080p).

## On the Pi (at the table)

```bash
cd ~/Pool_Fool
source .venv/bin/activate
pip install -e .

# Stream only (camera + MJPEG; projector overlay optional)
pool-fool-edge --config config/default.yaml --mode combined
```

`combined` = MJPEG on port **8080** + listen for overlay lines on UDP **8765**.

### Stream URL from your Mac

The Pi prints `http://0.0.0.0:8080/...` — that means “listen on all interfaces” **on the Pi only**. From the Mac, use the Pi’s hostname via **mDNS**:

```text
http://pool.local:8080/stream.mjpg
```

(`pool` is the hostname set in Raspberry Pi Imager; if yours differs, use `http://<hostname>.local:8080/stream.mjpg`.)

You can also use the Pi’s IP from `hostname -I` on the Pi, e.g. `http://192.168.12.115:8080/stream.mjpg`.

Verify in a browser or:

```bash
curl -I http://pool.local:8080/stream.mjpg
```

Set `network.overlay_udp_host` on the **Mac** to the same Pi (`pool.local` or the IP) so overlay lines reach the Pi.

## On the desktop (Mac)

1. Install YOLO extras:

```bash
pip install -e ".[yolo]"
```

2. Edit `config/default.yaml`:

```yaml
vision:
  detector: yolo

network:
  overlay_udp_host: "pool.local"   # or Pi IP from `hostname -I`
```

3. Calibrate **once** using the Pi stream (same view as production):

```bash
pool-fool-calibrate capture --camera "http://pool.local:8080/stream.mjpg"
pool-fool-calibrate table --image config/calibration/snapshot.jpg
pool-fool-calibrate play-region --image config/calibration/snapshot.jpg
```

4. Run processing:

```bash
pool-fool-app --config config/default.yaml \
  --camera "http://pool.local:8080/stream.mjpg" \
  --send-overlay
```

Debug window on the Mac; projector on the Pi shows lines when balls are **STATIONARY**.

## Cannot open the stream from the Mac?

1. **Do not use `http://0.0.0.0:8080/...` on the Mac** — use `http://pool.local:8080/stream.mjpg` or the Pi’s IP.
2. **On the Pi** (while `pool-fool-edge` is running):
   ```bash
   curl -I http://127.0.0.1:8080/stream.mjpg
   ss -tlnp | grep 8080
   hostname    # e.g. pool → Mac URL is http://pool.local:8080/stream.mjpg
   ```
   You should see `HTTP/1.0 200` and something listening on `0.0.0.0:8080`.
3. **On the Mac**:
   ```bash
   ping -c 2 pool.local
   curl -I --max-time 5 http://pool.local:8080/stream.mjpg
   ```
   If mDNS fails, use `hostname -I` on the Pi and ping/curl that IP instead.
4. If (2) works but (3) fails: different Wi‑Fi/VLAN, **AP/client isolation**, or firewall — use **Ethernet on both** or disable guest-network isolation.
5. Safari can be picky with MJPEG; try **Chrome** or `pool-fool-app --camera "http://pool.local:8080/stream.mjpg"`.

## If it feels slow

- Use `yolov8n.pt` (default), not larger models
- Lower resolution: `cameras.width/height: 1280x720` or `640x480` on Pi
- Desktop: close other GPU apps
- Classical detector is faster but jitterier; YOLO trades latency for stability

## Why not YOLO on the Pi?

Pi 4 can run tiny YOLO slowly (~1–5 FPS). Your Mac does 10–30+ FPS. Splitting roles keeps the Pi as a simple, reliable **camera + display appliance**.
