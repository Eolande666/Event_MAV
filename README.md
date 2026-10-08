# EventMAV 同事交接版

整理日期：2026-10-08。包含可运行代码、FRED 五段原始数据、保存的处理结果与复现说明。
从现有工作区复制整理，原项目未移动或删除。算法代码以本次实际文件为准，基础 Git 提交为 `72c3254`，并包含原工作区尚未提交的五段素材更新。

## 先看这里

1. 查看 [交接说明](docs/HANDOFF.md)，了解当前结论及复现边界。
2. 查看 [最新五段重新评估报告](results/reevaluation_20260929/REPORT.md)。
3. 播放 [持续性检测视频](results/persistence_mvp/fred_v1/videos/Persistence_all.mp4) 和 [对比视频](results/persistence_mvp/fred_v1/videos/Comparison_all.mp4)。
4. 按下方命令安装、运行测试和小样本检测。

## 目录导航

保留代码依赖的相对路径，解压后不必移动数据或修改脚本。

| 类别 | 目录 / 文件 | 用途 |
|---|---|---|
| 核心代码 | `evdetmav/`、`evdetmav_detector.py`、`main.py` | 显著性、周期性、聚类和精细定位 |
| 核心代码 | `persistence/`、`evdetmav_persistence.py` | 邻域关联和轨迹持续性后处理 |
| 配置与测试 | `configs/`、`tests/` | 推荐配置、关闭持续性配置、单元测试 |
| 实验代码 | `experiments/` | 评估、消融、结果核验和视频生成 |
| 对照代码 | `script/` | 独立对照实现，不是推荐主入口 |
| 编译依赖 | `tools/`、`third_party/ecf/` | ECF 解码器构建与第三方许可证 |
| **数据集** | **`FRED/`** | **8、20、51、65、93 五个原始 ZIP，含事件数据、图像和标注** |
| **最新评估结果** | **`results/reevaluation_20260929/`** | **四种方案的汇总、逐帧匹配与持续性重放结果** |
| 完整实验结果 | `results/persistence_mvp/fred_v1/` | 原版持续性实验、评分、配置、视频、审计 |
| 过滤对比结果 | `results/raw_pixel3_20260928/`、`results/saliency_area9_20260928/` | 两种过滤方案的输出 |
| 基线结果 | `comparison/` | 独立检测入口产生的 CSV、时间窗及合并视频 |
| 原始可视化 | `output/FRED_current/` | RGB、事件合并视频及时间戳等处理结果 |
| 说明 | `docs/`、`FRED/README.md`、`results/README.md` | 接手路线、数据及结果索引 |

已排除临时环境、缓存、重复的 `script.zip`、论文改稿与作图工作目录。结果中的 `source/` 是对应实验的源码快照，仅供追溯，不是日常开发入口。

## 推荐版本

使用 `configs/persistence_original.json`，保持 `--saliency-min-area 0 --min-raw-component-pixels 0`。
在本地留出序列 51、65、93 上，中心一对一匹配的 Precision / Recall / F1 为 **80.53% / 70.17% / 75.00%**。
8、20 用于校准。这是已查看过的五段本地数据，不能当作新的未知场景泛化验证，也不是整机框 IoU/AP 指标。

## 安装与运行

建议 Python 3.10+，macOS 或 Linux。Windows 建议 WSL2；原生 Windows 解码库构建未验证。

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-handoff.lock.txt
python tools/build_ecf.py
python -m unittest discover -s tests -v
```

ECF 构建需要 C++17 编译器（macOS Command Line Tools 或 Linux g++/clang++）。不附带本机编译的动态库，避免架构不匹配。
锁定依赖记录本次 Python 3.12 的验证环境；其他 Python 版本可使用 `requirements-persistence.txt` 安装兼容版本后重新测试。

### 先跑 12 个时间窗

```sh
python evdetmav_persistence.py --config configs/persistence_original.json --input FRED/8.zip --out runs/smoke_8 --width 1280 --height 720 --time-unit us --window-ms 30 --step-ms 30 --max-windows 12 --saliency-min-area 0 --min-raw-component-pixels 0
```

去掉 `--max-windows 12` 可运行整段。输出写到 `runs/`，避免覆盖已保存的实验。
原始 ZIP 中使用 ECF 压缩的 H5 由持续性入口流式读取；不要将 ZIP 直接传给只接受普通事件文件的 baseline CLI。

### 复核保存的四种方案

```sh
python experiments/reevaluate_retained.py --out runs/reevaluation
```

该命令读取原 ZIP 的标注，使用已保存的前两级检测 CSV，从第 0 窗重新计算持续性及评估，**不重跑前两级，也不调参**。输出目录必须不存在。完整检测重跑与已有结果重放需区分。

## 获取方式和完整性

ZIP 已包含实际数据和视频，解压即可使用，无需 Git LFS。
从 GitHub 获取请先安装 Git LFS，再执行：

```sh
git clone --branch handoff/20261008 https://github.com/Eolande666/Event_MAV.git
cd Event_MAV
git lfs pull
python tools/verify_handoff.py
```

`MANIFEST.sha256` 记录交接文件校验值。完整包约数 GB，请预留至少 10 GB 空间用于解压、环境及临时 H5。
