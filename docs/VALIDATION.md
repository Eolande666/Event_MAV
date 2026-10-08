# 交接验证记录（2026-10-08）

在新建交接目录的独立 Python 3.12 虚拟环境中执行，未借用原项目虚拟环境。

| 检查 | 结果 |
|---|---|
| 根据依赖清单安装 | 成功，版本见 `requirements-handoff.lock.txt` |
| C++17 构建 ECF 解码器 | 成功 |
| `python -m unittest discover -s tests -v` | 48 项通过，见 `handoff_tests.log` |
| 原始 `FRED/8.zip` 前 12 窗持续性检测 | 成功，见 `handoff_smoke.log` |
| 五个原始 ZIP 全成员 CRC 检查 | 全部通过，见 `dataset_validation.json` |
| 四方案、五段数据冻结参数重放 | 20 项逐帧计数与历史结果一致 |
| 重放汇总 CSV 与 `results/reevaluation_20260929/metrics_aggregate.csv` | 逐行一致 |

重放日志见 `handoff_replay.log`。本次没有重新运行所有序列的前两级完整检测，也未重新编码或逐帧检查全部视频。历史视频检查记录仍保留在 results 内。

本次仅修复可视化字体的跨机器导入问题，检测、关联、评分算法未改动。运行环境和新生成的复核输出不放入交接 ZIP，避免重复；保留原始处理结果和本次验证日志。
