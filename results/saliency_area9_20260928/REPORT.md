# saliency_area9_20260928：当前保留素材结果

更新日期：2026-09-29。保留素材为 8、20、51、65、93；8、20 为本地校准，51、65、93 为本地留出。根据保存的逐段 TP/FP/FN 重新汇总；未重跑算法、未调参。
评价采用一对一中心匹配；表中不使用整机 IoU 排名。素材范围发生变化，不能将本表与此前不同素材范围的结果直接比较。

| 范围 | 方法 | Precision | Recall | F1 | TP | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|
| calibration | original | 94.75% | 79.45% | 86.43% | 4617 | 256 | 1194 |
| calibration | filtered | 94.13% | 70.59% | 80.68% | 4102 | 256 | 1709 |
| calibration | filtered_before_persistence | 78.28% | 76.34% | 77.30% | 4436 | 1231 | 1375 |
| holdout | original | 80.53% | 70.17% | 75.00% | 10499 | 2538 | 4463 |
| holdout | filtered | 83.53% | 63.42% | 72.10% | 9489 | 1871 | 5473 |
| holdout | filtered_before_persistence | 74.51% | 68.80% | 71.54% | 10294 | 3521 | 4668 |
| all | original | 84.40% | 72.77% | 78.15% | 15116 | 2794 | 5657 |
| all | filtered | 86.47% | 65.43% | 74.49% | 13591 | 2127 | 7182 |
| all | filtered_before_persistence | 75.61% | 70.91% | 73.18% | 14730 | 4752 | 6043 |

## 逐段中心匹配结果

| 素材 | 方法 | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| 8 | original | 98.89% | 79.03% | 87.85% |
| 8 | filtered | 99.34% | 69.13% | 81.52% |
| 8 | filtered_before_persistence | 92.71% | 73.99% | 82.30% |
| 20 | original | 91.19% | 79.85% | 85.14% |
| 20 | filtered | 89.86% | 71.97% | 79.93% |
| 20 | filtered_before_persistence | 68.78% | 78.55% | 73.34% |
| 51 | original | 77.11% | 65.37% | 70.76% |
| 51 | filtered | 79.79% | 55.18% | 65.24% |
| 51 | filtered_before_persistence | 73.86% | 61.86% | 67.33% |
| 65 | original | 93.48% | 61.92% | 74.50% |
| 65 | filtered | 95.70% | 55.72% | 70.43% |
| 65 | filtered_before_persistence | 88.98% | 61.01% | 72.39% |
| 93 | original | 72.45% | 94.06% | 81.85% |
| 93 | filtered | 76.65% | 92.56% | 83.86% |
| 93 | filtered_before_persistence | 63.15% | 95.74% | 76.10% |

检测配置、源码快照及其原始哈希保留；源码快照中的旧素材清单属于历史运行证据，不是当前待处理清单。活动脚本默认只处理当前五段。
逐段检测、评分与评估明细保留于各素材目录；视频清理核验见 `results/retained_videos_verification.json`。
