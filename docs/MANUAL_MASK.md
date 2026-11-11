# 不规则边界拟合与手动 mask

自动提议分三档：

1. `--shape circle`：圆形容器，Hough 圆 + 静态环内缘；
2. `--shape irregular`：非圆、带凹角/缺口的容器，使用前景最大连通域的**非凸轮廓**；
3. `--shape polygon`：保留旧行为，用前景轮廓的**凸包**，适合矩形/六角等凸形容器。

`--shape auto` 会先试圆；圆覆盖前景不足 80% 时自动退回 `irregular`。无论哪种方式，
都只写出 `status=proposed` 的提议，必须看 `mask_overlay.png`。

## 不规则边界

```bash
vibrotrack mask \
  --raw frames \
  --out masks/case01-irregular \
  --config apparatus.yaml \
  --shape irregular \
  --frames 12
```

算法：

- 对采样帧取 temporal max，得到颗粒活动区域；
- Otsu 阈值后取最大连通域；
- 取该连通域的**原始外轮廓**，只做轻微 `approxPolyDP` 平滑；
- **不做 convex hull**，所以 L 形、缺口、非凸边不会被填平；
- 用 `--margin-px` 向内收（负值向外扩）。

这适合“容器本身不是圆”的情况。它拟合的是**前景颗粒区域的外轮廓**，不是冻结颗粒
的内侧包络；如果边界是“圆环 + 离散冻结颗粒”，仍然应该用 CAD 或人工 mask。

## 手动画边界

### 方法 A：网页编辑器

打开：

```text
tools/mask_editor.html
```

先导出一帧浏览器能显示的图像（浏览器通常不能直接打开 TIFF）：

```bash
vibrotrack frame \
  --raw frames \
  --config apparatus.yaml \
  --index 4500 \
  --max-size 1600 \
  --out preview.png
```

然后点“载入图像”，选择 `preview.png`（PNG/JPEG；浏览器会本地读取，不上传）。
这里不要用 `--max-size` 缩小图像：网页导出的 mask 与载入图同尺寸；缩放后会和原始帧
尺寸不一致。

1. 沿容器或冻结颗粒的**内侧**依次点击边界点；
2. 点“闭合并预览”；
3. 点“导出 interior_mask.png”，得到与原图同尺寸的黑白 mask；
4. 或点“导出 boundary_points.json”，交给 `vibrotrack mask` 生成带 overlay 的提议。

### 方法 B：用 points JSON

```json
{
  "image_shape": [2400, 2400],
  "points_px": [[1200, 80], [1800, 300], [1850, 900], [1200, 2320]]
}
```

支持可选的洞/排除区：

```json
{
  "points_px": [[...]],
  "holes_px": [[[x1, y1], [x2, y2], [x3, y3]]]
}
```

生成提议：

```bash
vibrotrack mask \
  --raw frames \
  --out masks/case01-manual \
  --config apparatus.yaml \
  --points boundary_points.json \
  --preview-frame 4500 \
  --margin-px 0
```

`status` 会写成 `manual`，`method` 为 `manual_polygon`；仍然会生成
`mask_overlay.png` 供复核。

### 方法 C：直接使用导出的 mask

如果网页编辑器已经导出了 `interior_mask.png`，可以直接：

```bash
vibrotrack extract ... --mask interior_mask.png
```

mask 尺寸必须与原帧一致；不一致会直接报错，不会静默缩放。

## 选择建议

| 边界 | 建议 |
|---|---|
| 圆形容器，边界近似轴对称 | `circle` |
| 非圆、有凹角/缺口 | `irregular` |
| 矩形/六角等凸形 | `polygon` |
| 圆环 + 离散冻结颗粒 | 当前优先人工 mask；CAD 配准后再自动化 |
| 无法自动判断 | 网页编辑器手画，或导出的 mask 直接用 |

自动拟合和手动绘制都只是**检测区域**，不是物理壁面。需要壁面碰撞/几何时，仍应单独
记录固定盘中心或 `wall_disks.npy`。
