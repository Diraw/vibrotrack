# 自动边界 mask 提议

`vibrotrack mask` 从录像本身估计容器边界，生成：

```text
<out>/
  interior_mask.png   白=检测区域，黑=不检测
  mask_overlay.png    真图 + 半透明绿色区域 + 边界线
  mask.json           方法、参数、中心/半径、哈希、status=proposed
```

它不会自动把 mask 写进 `dataset.json`，也不会覆盖已有 batch。**overlay 必须人工看**；
自动边界只是提议。

## 用法

```bash
vibrotrack mask \
  --raw frames \
  --out masks/case01 \
  --config apparatus.yaml \
  --frames 12
```

然后打开 `masks/case01/mask_overlay.png`，确认绿色区域：

- 完整覆盖颗粒活动的容器内部；
- 没有把大块容器外暗区包进来；
- 边界线没有明显切掉贴壁颗粒。

确认后：

```bash
vibrotrack extract \
  --raw frames \
  --out batch \
  --config apparatus.yaml \
  --case-id case01 \
  --mask masks/case01/interior_mask.png
```

如果提议不对，可以重跑并覆盖：

```bash
vibrotrack mask ... --shape polygon --margin-px 4 --force
```

也可以完全绕过自动提议，继续使用手工/旧的 mask。

## 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--frames` | 12 | 从整段录像均匀抽多少帧做 temporal max |
| `--shape` | auto | `auto` / `circle` / `polygon` |
| `--margin-px` | 2 | mask 外扩像素；贴壁颗粒中心被切掉时加大 |
| `--threshold` | Otsu | 前景阈值；光照特殊时手动指定 |
| `--preview-frame` | 中间采样帧 | overlay 使用哪一帧原图 |
| `--force` | false | 覆盖已有提议文件 |

## 算法

1. 在录像中均匀抽帧，取逐像素最大值（temporal max）。颗粒会移动，max 能把颗粒之间
   的空隙填上，得到完整的“颗粒活动区域”。
2. 对 temporal max 做高斯模糊和 Otsu 阈值，得到前景。
3. 圆形容器优先用 Hough 圆；候选必须靠近画面中心，半径在画面尺寸的合理范围内。
   取不到圆时，回到前景最大连通域的外轮廓，用最小外接圆。
4. `--shape polygon` 或 Hough 圆覆盖不了前景大部分面积时，用外轮廓凸包，适合六角/
   矩形 ROI。
5. 圆半径或凸包外扩 `--margin-px`，避免贴壁颗粒中心落在 mask 外。
6. 用采样帧生成 overlay；mask 和 overlay 一起写盘。

## `mask.json`

关键字段：

| 字段 | 含义 |
|---|---|
| `schema` | `mask-proposal/v1` |
| `status` | 固定为 `proposed`；人工确认是流程动作，不是文件自动改写 |
| `method` | `hough_circle` / `contour_circle` / `contour_polygon` |
| `shape` | `circle` / `polygon` |
| `center_px` | 估计中心，`[x, y]` |
| `radius_px` | 圆形 mask 半径（含 margin）；多边形为 `null` |
| `margin_px` | 外扩量 |
| `threshold` | 实际前景阈值；Hough 分支可能为 `null` |
| `image_shape` | `[height, width]` |
| `frame_count` | 原始录像帧数 |
| `sample_indices` | 实际抽到的帧号 |
| `preview_index` | overlay 用的帧号 |
| `mask_coverage` | mask 占全图比例，仅作 sanity check |
| `mask_sha256` / `overlay_sha256` | 产物哈希 |
| `raw` | 只写来源目录，不复制原始帧 |

## 已知限制

- 容器必须大致居中；Hough 候选被限制在画面中心附近。
- 假设容器内部整体比外部亮。严重不均匀照明、强反光、容器贴边时可能失败。
- 非凸边界不能用凸包表示；这种情况应手工画 mask。
- 自动提议不会判断“哪颗是活性颗粒”，只给检测区域。
- 自动提议不会改变 `px_per_mm` 或 `N`；标定和装载量仍由用户提供。
