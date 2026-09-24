# Persistence Verification MVP

独立的 baseline 后处理模块；原 evdetmav/ 不变。最新实验、视频和限制见 `../results/persistence_mvp/fred_v1/REPORT.md`。

关闭模块：原 CLI 参数加 `--config configs/baseline.json`，由 `evdetmav_persistence.py` 原样转交原检测器。开启模块：使用本次校准生成的 `results/persistence_mvp/fred_v1/combined_config.json`，输入可为原始 FRED ZIP 或 structured H5，须指定 `--time-unit us --width 1280 --height 720`。

本次实验先校准 8/20，再评估 51/65/93/114。六段都是 FRED 官方 test 的子集，不能把本次本地划分称为官方完整测试。主报告同时展示严格 IoU 指标和明确标注的候选中心定位诊断。

配置字段全部见 config.py。配置中的数值是工程实验值，不是 EvDetMAV 论文默认值。候选历史包括有效 baseline 输出中被第三级拒绝的候选；不会用预测生成当前框。联合模式缺 St 时按文档回退 Sn；trajectory-only 的前两次观测保持 tentative、无最终输出。

`use_periodicity=False` 只可用于原始 H5 重算，真正绕过周期性调用；不可从已经过周期性筛选的缓存 CSV 重建。`use_neighborhood` / `use_trajectory` 分别控制两项评分。阶段一只支持 rejection。

依赖见 `../requirements-persistence.txt`。FRED 的 ECF 压缩另依赖现有 `tmp/fred_figures/libecf_decode.dylib`，不能删除该解码器；普通未压缩 structured H5 不依赖它。可复现实验脚本位于 `../experiments/`，测试命令 `python -m unittest discover -s tests -v`。
