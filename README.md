# EventMAV 处理代码

只包含事件数据读取、无人机检测、持续性筛选和离线评估代码。数据集目录 `FRED/` 为空（仅 `.gitkeep` 占位），不包含数据、结果、绘图或视频生成代码。当前版本不需要 Git LFS。

## 安装

建议 Python 3.12，macOS / Linux。Windows 建议 WSL2，原生 Windows 的解码库构建未验证。

```sh
git clone --depth 1 https://github.com/Eolande666/Event_MAV.git
cd Event_MAV
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-persistence.txt
python -m unittest discover -s tests -v
```

测试使用自行构造的小规模数据，不需要下载数据集。

## 目录

| 路径 | 内容 |
|---|---|
| `evdetmav/` | 显著性、周期性、聚类、精细定位及 CSV 输出 |
| `persistence/` | 事件读取、候选关联、邻域和轨迹持续性、评估与重放 |
| `configs/` | 推荐持续性配置与关闭持续性的 baseline 配置 |
| `experiments/` | 历史校准与冻结参数重新评估脚本 |
| `tests/` | 处理算法测试 |
| `third_party/ecf/` | ECF 解码源码及许可证 |
| `tools/build_ecf.py` | 编译 ECF 解码库 |
| `FRED/` | 后续放入数据，当前为空 |

## 数据准备

数据来源：https://huggingface.co/datasets/GabrieleMagrini/FRED

后续可将原始 `8.zip`、`20.zip`、`51.zip`、`65.zip`、`93.zip` 放入 `FRED/`，保持 ZIP 内结构。历史本地划分为 8、20 校准，51、65、93 留出，不是官方完整测试集。
ECF 压缩事件需要支持 C++17 的编译器，先执行：

```sh
python tools/build_ecf.py
```

## 检测入口

推荐使用原版检测＋持续性配置，两个可选小区域过滤默认关闭。数据准备后可先运行 12 个窗口：

```sh
python evdetmav_persistence.py --config configs/persistence_original.json --input FRED/8.zip --out runs/smoke_8 --width 1280 --height 720 --time-unit us --window-ms 30 --step-ms 30 --max-windows 12 --saliency-min-area 0 --min-raw-component-pixels 0
```

移除 `--max-windows 12` 可处理整段，输出检测 CSV、轨迹、关联与评分 JSONL 等结构化结果。纯处理版已移除 `--save-saliency`、`--save-event-frames`、`--save-segmentation` 等图像输出参数。

独立 baseline 入口处理普通、未压缩事件 H5：

```sh
python evdetmav_detector.py --input /path/to/events.h5 --out runs/baseline --width 1280 --height 720 --time-unit us --window-ms 30 --step-ms 30
```

原始 ZIP 不可直接交给 baseline CLI。ECF 压缩 ZIP/H5 请通过开启持续性的入口读取。

## 评估与继续开发

- 配置定义：`persistence/config.py`；评分实现：`neighborhood.py`、`trajectory.py`、`scoring.py`。
- 通用评估与重放接口：`persistence/evaluation.py`、`persistence/replay.py`。
- `experiments/` 中的历史批量脚本需要额外准备原 ZIP 及 `comparison/`、`results/` 中已保存的检测 CSV。仓库未附带这些输入，刚克隆时不能直接重放历史实验。
- 校准脚本会写入历史结果路径，运行前请确认输出位置。新检测建议统一输出到 `runs/`。
- `.gitignore` 已排除数据、结果、虚拟环境和编译产物，避免再次上传大文件。
