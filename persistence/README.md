# 持续性处理

在 EvDetMAV 检测候选后执行邻域匹配和轨迹持续性评分。
推荐配置为 `configs/persistence_original.json`，字段定义见 `config.py`。
关闭配置为 `configs/baseline.json`，此时入口转交原 baseline CLI，仅支持其可读取的事件格式。

候选历史保留通过前两级但被第三级拒绝的候选，不使用预测框替代当前观测。
`use_periodicity=False` 需要从原始事件重算，无法从已经过周期性筛选的 CSV 恢复候选。

数据、安装、运行命令见根目录 README。该版本不包含可视化模块、数据及历史结果。
