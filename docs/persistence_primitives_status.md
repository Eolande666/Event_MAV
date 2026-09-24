> 后续完整 MVP 已实现。最新结果见 [完整实验报告](../results/persistence_mvp/fred_v1/REPORT.md)。下文保留基础阶段的历史记录。

# Persistence Verification 基础模块实施记录

本次在完成论文、技术文档和全项目审查后，落实不依赖待确认决策的基础模块。原 evdetmav/、main.py、evdetmav_detector.py 均未修改。

## 已实现

- persistence/types.py：独立不可变 Detection，有限数值、正面积框和事件数校验；原始 Ss、Sp 不做归一化。
- persistence/neighborhood.py：文档 §6、§8 普通与指数加权评分；显式选择 fixed/observed 冷启动；空历史返回 None。
- persistence/trajectory.py：文档 §10–13 真实时间差速度、仅历史 CV 预测、上一框尺度归一化、低速屏蔽方向、加速度；第一观测无速度，第二观测仅速度，第三观测开始残差评分。
- persistence/scoring.py：文档 §12 的指数软评分，按有效项重新归一化权重；§1 双项融合，缺失保留 None；不推测 sigma。
- persistence/config.py：总开关及待确认参数校验。当前为基础配置，关联/消融完整配置尚未完成。
- evdetmav_persistence.py、configs/baseline.json：关闭模块直接转交原 CLI；开启模块明确报未实现，不能误认为完整 MVP 已接通。
- persistence/diagnostics.py：可重现合成轨迹图、CSV、JSONL、中间有效权重和实验参数快照。

## 验证命令与结果

在项目根目录运行：

```sh
tmp/fred_figures/venv/bin/python -m unittest discover -s tests -v
tmp/fred_figures/venv/bin/python -m persistence.diagnostics
tmp/fred_figures/venv/bin/python evdetmav_persistence.py --config configs/baseline.json --help
```

11 项单元测试通过。覆盖冷启动、悬停、非等间隔匀速、预测无当前点泄漏、转弯软惩罚、重复/回退时间、NaN/Inf 和小框、邻域漏检及衰减、尺度缺失、融合缺失值、关闭模块配置。

结果目录：results/persistence_mvp/primitives/，包含 tests.log、diagnostic_config.json、debug.jsonl、diagnostics.csv、diagnostics.svg、diagnostics.png 和 baseline_help.txt。

已目视检查诊断图。英文为 Times New Roman；图中无中文。测试和诊断使用的 sigma=1、固定分母及合成轨迹只为明确公式计算，不是默认部署参数，也未在 FRED 标定。邻域图仅验证已给 hit 序列的评分，不证明 Track 漏检恢复功能已实现。

## 未完成与边界

完整 MVP 尚未完成。待确认的接入范围、冷启动分母、尺度/阈值标定见审查计划和当前对话问题。未实现一对一关联、Track 状态机、真实视频调试输出和四组消融；没有运行正式持续性接受/拒绝实验。baseline 入口仅验证了帮助参数转交，尚未做同一 H5 的结果等价回归。

后续在规则明确后完成上述模块及测试；既有 comparison 六序列 baseline 已全部生成，不会用合成诊断冒充真实实验。
