# 显著性小区域过滤：第一版

本轮只优化显著图里的小连通区域，使用原版持续性配置，不改周期性/持续性的评分公式、参数或原始事件。
默认 `--saliency-min-area 0` 保持原版；本轮试验指定 `--saliency-min-area 9`。

## 小区域的定义

1. 先按原算法得到密度感知显著图（包括原有正负事件空间容差，`intersection_radius=1`）。
2. 用原阈值 `tau_s=50` 得到有效前景，在 **8 邻域**下标记连通区域。
3. 数实际前景像素，不数外接框面积。面积 1～8 像素的区域清零，恰好 9 像素及以上保留。
4. 低于显著性阈值的背景也清零，所以它们不再影响候选显著性累计分数；保留下来的显著值不重新归一化。
5. 过滤发生在候选初始化膨胀、区域合并、排序和周期性判定之前。精细裁剪后再按同一面积门限去掉碎片，避免回退路径重新使用裁剪形成的小片段。

“小像素”不是单个像素物理尺寸，也不是删除原始事件：这里专指阈值化显著图中的小连通区域。仍然保留原版的正负事件空间容差，因此该面积不是原始事件坐标去重后的面积。
本轮是“阈值前景＋面积”的统一显著图清理，不应把全部性能差异都归因于面积门限一个因素。
仅靠面积无法区分小旋翼和小噪声，真实远距离旋翼也可能被删除，因此需同时看 Precision 和 Recall。

## 代码分块与变量

- `evdetmav/saliency.py::remove_small_saliency_regions`：纯函数，输入显著图，输出过滤后的显著图和统计。
  - `saliency`：原显著值二维数组。
  - `threshold`：前景显著值下限；本轮 50。
  - `min_area`：最小区域像素数；0 关闭，试验为 9。
  - `foreground`：阈值以上像素的布尔图。
  - `labels`：连通区域编号图，0 是背景。
  - `count`：连通区域总数。
  - `areas`：每个编号包含的真实像素数。
  - `keep`：哪些编号达到面积要求，背景永不保留。
  - `filtered`：不合格位置清零后的显著图。
  - `removed`：被删除的前景区域编号对应的布尔向量。
- `evdetmav/pipeline.py`：在生成候选前应用过滤，`WindowResult` 同时保存原图、过滤图和统计。
- `evdetmav/clustering.py`：在精细裁剪区域上使用同一门限去除小碎片。
- `persistence/baseline_adapter.py`：正常周期性路径和显式关闭周期性的消融路径使用相同显著性过滤。
- `persistence/runner.py`、`evdetmav/cli.py`：保存逐窗过滤统计；无需改变原版持续性配置。
- `experiments/compare_saliency_area.py`：原 H5 对齐、原版冻结结果核验、双独立持续性轨迹、视频渲染、中心匹配评价与按编号拼接。

## 运行

在项目根目录，安装好 `requirements-persistence.txt` 中的依赖后：

```sh
# 单段：新显著性过滤＋原版持续性
python evdetmav_persistence.py --config configs/persistence_original.json --input FRED/8.zip --out results/area9_run_8 --width 1280 --height 720 --time-unit us --window-ms 30 --step-ms 30 --saliency-min-area 9

# 对照：关闭新过滤，其余不变
python evdetmav_persistence.py --config configs/persistence_original.json --input FRED/8.zip --out results/original_run_8 --width 1280 --height 720 --time-unit us --window-ms 30 --step-ms 30 --saliency-min-area 0

# 全部六段对比，输出目录必须尚不存在
python experiments/compare_saliency_area.py --out results/area9_new_comparison --area 9 --workers 4

python -m unittest discover -s tests -v
```

## 本次对比协议

- 素材：8、20、51、65、93、114 的原 ZIP/H5，无 RGB 读取。
- 所有时间窗保留旧版闭区间边界、30 ms 窗长与步长；每段从第 0 窗开始，不跳过历史。
- 左侧原版：冻结原版 CSV，每段前 12 窗重新检测并逐字段核验（仅忽略来源路径和耗时），然后用独立原版 Tracker 重放完整序列。
- 右侧新版：重新计算所有时间窗的显著性、周期性、最终候选，再进入另一独立 Tracker。
- 两侧持续性配置完全相同；参数门限在运行前固定，没有根据保留测试集评价进行调参。
- 中心匹配按原代码执行一对一匹配：预测框中心位于 GT 目标框内。不用整机 IoU 排名，也不把这一指标声称为逐旋翼标注的检测精度。
- 8、20 为已有本地校准素材，51、65、93、114 为已有本地保留测试素材；以保留测试汇总判断本轮效果。
- 原版使用缓存，因此此次耗时不构成两算法公平速度对比。
- 新版视频每侧维持 1280×720 原生画面，框线直接绘制，不缩放后回放；单独标题区使用宋体和 Times New Roman。

## 日志与结果

本次完整结果目录为 `results/saliency_area9_20260928/`，运行状态以 `status.json` 为准。
只有 `status=complete` 后才将 `metrics_aggregate.csv` 和最终合并视频视为完整结果。
逐窗过滤量见 `saliency_windows.csv`，周期性结果见 `filtered_detections.csv`，持续性详细记录见 `persistence_debug.jsonl`。
`preview_saliency.npz` 保存预览所在真实时间窗的原始/过滤显著图，预览按删除前景像素最多选择，不按 GT 或检测好坏挑选。
实验开始时保存代码快照及 SHA-256，避免未来修改代码后混淆本次结果。
