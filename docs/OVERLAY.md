# Detection overlays

`vibrotrack plot` renders selected raw frames with the fitted particle outlines,
centres and orientation ticks. It is a **visual check**, not a new data product:
it reads `chunk_*.npz` and never changes detections, tracks or masks.

```bash
vibrotrack plot \
  --raw frames \
  --case-dir batch/sequences/case01 \
  --config apparatus.yaml \
  --frames 0,50,99 \
  --out overlays \
  --max-size 1200
```

- `--frames` uses raw frame indices. If omitted, the first, middle and last frames
  present in the chunk are used.
- `--max-size 0` keeps the native resolution. The default is 1200 px on the long side.
- Output filenames are `<case>_frame<index>_detections.jpg`.
- Green circle: fitted particle outline.
- Yellow dot: fitted centre.
- Orange tick: marker direction, drawn from `theta_rad`.

## The angle convention (important)

`theta_rad` is stored as:

```python
theta = atan2(-dy, dx)
```

where `(dx, dy)` is the offset from the fitted centre to the dark marker in
**image pixel coordinates** (x right, y down). The stored theta is therefore a
**y-up mathematical angle**. To draw it in an image, use:

```python
x_end = cx + r * cos(theta)
y_end = cy - r * sin(theta)
```

Do **not** use `cy + r*sin(theta)`: that mirrors every orientation across the
horizontal axis. The public helper is:

```python
from vibrotrack.visualize import theta_endpoint
x_end, y_end = theta_endpoint(cx, cy, r, theta)
```

The regression test `tests/test_visualize.py` checks both the helper and the
rendered tick, so this convention cannot silently regress.
