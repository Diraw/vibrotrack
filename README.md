# vibrotrack

Dense particle tracking and dataset archiving for quasi-2D vibrated granular monolayers.

`vibrotrack` turns a directory of high-speed TIFF frames into a self-describing,
hash-checked particle dataset: per-frame centres, radii, orientations, track ids, a
geometry product, and batch-level admission gates. It is deliberately independent of any
lab-specific physics package — the only inputs are frames, a calibration file, and a
boundary mask.

> 中文快速开始见下一节。这个仓库只放**提取与归档工具**，不放原始数据、不放具体课题结论。

## 它做什么

```text
TIFF frames + apparatus.yaml + boundary mask
        │
        ▼
   detect (v1/v2)  →  link (v1/v2)  →  chunk_*.npz
        │
        ▼
   geometry.json  →  dataset.json  →  index.json / acceptance.json
```

- **检测**：模板匹配 + 固定半径圆心拟合（v2），支持贴壁搜索带和冻结壁盘剔除。
- **链接**：运动预测 + 门内匈牙利分配 + ≤N 帧补链（v2）。
- **几何**：外缘包络圆拟合容器中心，给出壁半径、等 access 半径、`phi`。
- **归档**：每个 case 写 `dataset.json`；每个批次写 `index.json`、`acceptance.json`；
  `verify` 重算所有哈希。`archive` 复制到只读目录并写 `FROZEN.json`。

## 安装

```bash
python -m pip install -e .
# 或者从 GitHub 安装
python -m pip install "vibrotrack @ git+https://github.com/Diraw/vibrotrack.git@v0.1.0"
```

依赖：`numpy`、`scipy`、`opencv-python`、`pillow`、`pyyaml`。Linux 服务器如果报
`libGL.so.1`，安装 `libgl1`，或改用 `opencv-python-headless`。

## 快速开始

### 1. 准备三样东西

```text
my_project/
  frames/                  # 16-bit TIFF（.tiff/.tif），文件名按帧号排序
  masks/interior_mask.png  # 圆形/矩形内部区域，白=可检测；没有就省略
  apparatus.yaml           # 复制 examples/apparatus.example.yaml 后修改
```

`apparatus.yaml` 最少要改：

```yaml
fps: 10.0
px_per_mm: 5.1558
particle:
  diameter_mm: 14.0
  nominal_n: 485
boundary:
  mask: masks/interior_mask.png
```

`px_per_mm` 必须来自标尺/CAD/已知尺寸，并在 `calibration.source` 写清来源。**不要**
用检测出的颗粒半径反推 ppm 再声称是计量结果。

### 2. 提取一个 case

```bash
vibrotrack extract \
  --raw frames \
  --out batch \
  --config apparatus.yaml \
  --case-id case01
```

产物在 `batch/sequences/case01/`：

```text
chunk_00000_17999.npz   逐检测数组 + 帧索引 + 轨迹 id
dets_*/                 检测缓存（改配置会换目录，不会误用旧缓存）
run_config.json         这次运行的完整配置 + git commit
full_summary.json       帧数、检出中位数、有效率、链接率
dataset.json            自描述元数据（fps、px/mm、mask 哈希、文件哈希）
```

### 3. 几何产品

```bash
vibrotrack geometry \
  --case-dir batch/sequences/case01 \
  --config apparatus.yaml \
  --n-particles 485
```

`--n-particles` 是**名义装载量**，不是检出中位数；它只用于 `phi`。中心默认从 mask
质心或颗粒位置中位数出发，再由外缘包络拟合修正。`envelope_resid_rms_px > 30` 时
`geometry_warning` 会提示拟合不可靠。

### 4. 批次门禁

```bash
vibrotrack index  --batch-dir batch
vibrotrack accept --batch-dir batch
vibrotrack verify --batch-dir batch
```

`accept` 检查字段完整性、帧连续性、`center_valid`/`angle_valid` 比率、轨迹帧序、
帧数覆盖和 npz 哈希；`verify` 重新计算所有产品哈希。只有 `verify` 通过的数据才建议
进入定量分析。

### 5. 冻结归档

```bash
vibrotrack archive --batch-dir batch --archive-root archive
```

复制到 `archive/<batch>/` 并写 `FROZEN.json`。`dets_*` 检测缓存和 `progress.json` 默认不复制；归档只保留可复现的最终产品。之后只读；要改就新建批次。

## 输出格式

### `chunk_*.npz`

每个数组的第一维是“检测行”，`frame_offsets` 把行分成帧：

| 字段 | 含义 |
|---|---|
| `xy_px` | `(N,2)` 圆心像素坐标，x 向右、y 向下 |
| `radius_px` | 每颗粒拟合半径 |
| `theta_rad` | 标记方向，`atan2(-dy, dx)`；无效为 NaN |
| `center_valid` / `angle_valid` | 该行是否可用于定量分析 |
| `confidence` | 模板匹配分数 |
| `edge_rmse_px` | 径向边缘拟合残差 |
| `marker_offset_px` / `marker_area_px` | 标记偏移与面积 |
| `neighbor_distance_px` / `boundary_distance_px` | 最近邻/边界距离 |
| `quality_bits` | 位标志（角度无效、近邻过近、贴壁） |
| `segment_id` | 轨迹 id；`-1` 表示未链接 |
| `previous_detection` | 上一帧对应检测的全局行号；`-1` 表示无 |
| `link_valid` | 是否为直接链接（gap=1） |
| `step_px` / `dtheta_rad` | 帧间位移/转角 |
| `frame_ids` | 本 chunk 覆盖的帧号 |
| `frame_offsets` | 长度 `n_frames+1` 的行分割 |

### `geometry.json`

`center_px`、`wall_radius_px`、`envelope_radius_px`、`equal_access_radius_px`、
`px_per_mm`、`fps`、`particle_radius_px`、`n_particles`、`phi`、`centre_quality`。
分析端不得硬编码中心或半径，必须读这个文件。

### `dataset.json` / `index.json` / `acceptance.json`

`dataset.json` 是单个 case 的身份证；`index.json` 汇总一个批次并记录每个文件的
sha256；`acceptance.json` 是门禁报告。字段见 [`docs/SCHEMA.md`](docs/SCHEMA.md)。

## 配置参考

| 键 | 默认 | 说明 |
|---|---|---|
| `fps` | 10 | 实际采集帧率 |
| `px_per_mm` | 1 | 空间标定；必须写来源 |
| `particle.diameter_mm` | 1 | 颗粒直径 |
| `particle.nominal_n` | null | 名义装载数，用于 `phi` |
| `scale_levels` | `[128,28224]` | 16-bit 到 8-bit 的显示范围；其他相机要改 |
| `boundary.mask` | null | 边界 mask；null = 全帧 |
| `boundary.model` | auto | `auto`/`geometry`/`empirical`/`off` |
| `boundary.band` | 1 | 是否搜索 mask 外 40 px 的贴壁颗粒 |
| `boundary.dilate_px` | 40 | 贴壁搜索带宽度 |
| `detector.name` | v2 | v1 冻结归档行为，v2 固定半径圆心拟合 |
| `detector.params` | {} | 透传给 v2：`thresh`、`min_rays`、`min_sep` 等 |
| `linker.name` | v2 | v2 支持预测、匈牙利分配、补链 |
| `linker.max_gap` | 5 | 最多补几帧 |
| `run.workers` | 8 | 并行进程数；HDD 建议 1–4，SSD 可 8–12 |
| `run.shard` | 500 | 每个检测分片的帧数 |

## 边界 mask 怎么做

- 白（>0）= 检测区域，黑 = 不检测。
- 圆形容器：画**内部圆盘**，不要画冻结颗粒环；半径取亮环内缘再减 5–10 px。
- 矩形/方形 ROI：可以不传 mask（全帧），或把壁面画黑。
- mask 尺寸必须与帧一致；不一致会直接报错，不会静默裁剪。

## 性能

原装置 2500×2496、N≈485、12 workers：9000 帧约 190–260 s（取决于磁盘）。检测是
CPU 密集，链接是单线程但可与检测重叠；HDD 上不要开 12 workers，会互相抢 I/O。

## 已知限制（v0.1）

- 面向**稠密**数据（每帧数百颗粒、10 fps）。100 fps 稀疏片段请用单独的 legacy 提取器，
  不在本包内。
- 检测器假设“亮圆盘 + 暗标记”；其他外观需要新增 detector profile。
- `boundary.model=auto` 只在 mask 同目录存在 `wall_disks.npy` 时走几何壁分支；
  没有就退回经验方差分支。
- 没有 GUI；批处理请用 CLI 或 Python API。

## 开发与测试

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

测试全部用合成帧，不依赖实验室数据。真实数据回归请在你的项目仓里用固定 case 做，
不要把原始 TIFF 提交到本仓库。

## 许可与引用

MIT。见 [`LICENSE`](LICENSE) 与 [`CITATION.cff`](CITATION.cff)。

---

**维护原则**：这个仓库只放“提取 + 归档”工具。物理观测量（`|c|/R`、平台判据等）
在私有契约包里；具体课题的配置、结论和原始数据在项目仓/数据根，不在本仓库。
