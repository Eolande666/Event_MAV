> **2026-09-28 显著性过滤试验**：当前工作分支为 `codex/saliency-small-regions-20260928`。新增 `--saliency-min-area 9` 可清除小显著区域；默认值 0 仍运行下述原版。使用方法、变量及对比协议见 [显著性过滤说明](docs/SALIENCY_AREA_FILTER.md)。完整原版分支仍保留。本轮完整对比的保留测试集中心 F1 为 63.31%，低于原版 65.29%，因此默认仍关闭新过滤。详见 [本轮结果](docs/SALIENCY_AREA9_RESULTS.md)。

# 当前保留版本（2026-09-28）

原版持续性检测算法对应 `d9bf804`，保存在 `codex/best-persistence-20260928` 分支。
使用 `configs/persistence_original.json` 运行原版“邻域匹配＋轨迹持续性”；使用 `configs/baseline.json` 可独立运行 EvDetMAV baseline。
本次只保留较新版的 ECF 解码库加载与构建工具，未移入 V1/V2 的算法微调。

在本地保留测试序列 51、65、93、114 上，以**中心匹配**评价旋翼候选：Precision **62.42%**、Recall **68.43%**、F1 **65.29%**。
这是已有完整实验中综合 F1 最好的版本；不以整机标注框的 IoU 排名。数据与比较范围详见 [版本选择记录](docs/BEST_VERSION.md)。

```sh
python -m pip install -r requirements-persistence.txt
python tools/build_ecf.py
python evdetmav_persistence.py --config configs/persistence_original.json --input FRED/8.zip --out results/run_8 --width 1280 --height 720 --time-unit us --window-ms 30 --step-ms 30
```

完整检测视频：`results/persistence_mvp/fred_v1/videos/Persistence_all.mp4`。
左右对比视频：`results/persistence_mvp/fred_v1/videos/Comparison_all.mp4`。
原始 RGB 和红蓝事件完整视频：`output/FRED_current/combined/`。
单段重复视频已清理，按原顺序拼接的完整版、原始数据、实验记录、论文与真实数据图保留。
历史报告中的 `tmp/fred_figures/venv/bin/python` 是已清理的临时环境；重新运行时使用安装好依赖的 Python 3.10+。

---

# EvDetMAV reproduction

This folder contains a standalone reproduction of the algorithm flow from
`EvDetMAV: Generalized MAV Detection from Moving Event Cameras`.

Pipeline implemented in `evdetmav_detector.py`:

1. Density-aware saliency map generation
   - Split one event period `Delta T` into `n` slices.
   - For each slice, build positive and negative binary event images.
   - Accumulate the intersection `I_s^t = I_p^t intersection I_n^t`.
   - Scale the accumulated saliency map to 0-255 and threshold with the paper
     default `tau_s = 50`.

2. Spatio-temporal feature extraction
   - Rank saliency clusters by saliency score and keep the paper default
     `K = 4`.
   - Split each local event stream into `m` slices.
   - Compute the paper features:
     - `f_d`: positive-event density per slice.
     - `f_s`: cosine similarity of consecutive normalized slice images.
     - `f_p`: principal-direction similarity from consecutive slice point sets.
   - Apply moving-average filtering and score whether each feature contains
     peaks and valleys. The maximum periodicity score is 6. The paper default
     threshold `tau_p = 3` is used.

3. Clustering-based MAV detection
   - Initialize potential areas from connected saliency components.
   - Merge nearby rectangles using the paper's minimum rectangle-distance idea.
   - Refine accepted areas with connected saliency regions and a fitted 2D
     Gaussian consistency check.
   - Keep refined cyan boxes as the final detections by default. Use
     `--merge-propellers` only when you want one merged MAV box.

The paper does not specify all engineering thresholds, so unspecified values
are command-line parameters. The reported `tau_s`, `tau_p`, and `K` defaults
are preserved.

## Examples

Per-period H5 directory, matching the EventMAV dataset style:

```powershell
python scripts\EventMAV\evdetmav_detector.py `
  --input inputs\170\Event\h5 `
  --out outputs\evdetmav_170 `
  --limit-files 5
```

Continuous NPZ stream with 30 ms sliding windows:

```powershell
python scripts\EventMAV\evdetmav_detector.py `
  --input inputs\170\Event\output_events.npz `
  --out outputs\evdetmav_170_npz `
  --window-ms 30 `
  --step-ms 30 `
  --max-windows 20
```

Use the strict saliency intersection described in the paper:

```powershell
python scripts\EventMAV\evdetmav_detector.py `
  --input inputs\170\Event\h5 `
  --out outputs\evdetmav_strict `
  --intersection-radius 0
```

## Outputs

```text
evdetmav_detections_batch.csv
source_manifest.csv
per_file/*_detections.csv
saliency/*.png
event_boxes_evdetmav/*.png
segmentation/*.png
```

CSV boxes use exclusive `bbox_x1` and `bbox_y1` coordinates.

By default, `evdetmav_detections_batch.csv` stores the refined `propeller_*`
boxes. These are the cyan boxes in the visualization. Passing
`--merge-propellers` restores the earlier single green `mav` box behavior.

## Repository setup

Install the detector dependencies and inspect its options:

```sh
python -m pip install -r requirements.txt
python main.py --help
```

The repository tracks the detector, figure-generation scripts, and experiment
scripts/documentation. Local FRED datasets, generated videos, Python environments,
caches, and locally installed fonts are intentionally excluded from Git; they
remain available in the local project folder.

FRED data: https://huggingface.co/datasets/GabrieleMagrini/FRED

Scripts in `output/FRED_current/source/` document the local six-sequence experiment.
They need additional video/plotting dependencies and local resources as noted in
`output/FRED_current/README.md`; installing the core requirements alone does not
set up the FRED ECF decoder or local fonts. The figure PDFs included here are
illustrations, not an accuracy benchmark.
