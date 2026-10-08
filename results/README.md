# 处理结果索引

优先阅读 `reevaluation_20260929/REPORT.md`，该目录按当前五段数据重新读取标注、重放持续性并统一汇总四种方案。

| 目录 | 内容 |
|---|---|
| `reevaluation_20260929/` | 当前指标、逐帧匹配、评分、冻结配置及一致性检查 |
| `persistence_mvp/fred_v1/` | 完整持续性、消融、诊断和视频 |
| `raw_pixel3_20260928/` | 扩张前真实像素数过滤实验 |
| `saliency_area9_20260928/` | 显著面积过滤实验 |
| `script_comparison_20260924/` | 独立 script 对照结果 |
| `verification_20260928/` | 历史集成和兼容性检查 |

视频入口：`persistence_mvp/fred_v1/videos/`、`../comparison/Detection_all.mp4`、`../output/FRED_current/combined/`。
基线逐窗 CSV 位于 `../comparison/{8,20,51,65,93}/`，是重新评估必需输入。不要只拷贝汇总表而丢弃这些 CSV。
嵌套的 `source/` 仅是实验时的源码快照，请在根目录的代码包中开发。
