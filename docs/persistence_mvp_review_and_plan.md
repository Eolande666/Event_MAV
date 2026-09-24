# EvDetMAV 持续性检验 MVP 审查与实施计划

本轮完成文档、论文及项目代码审查，交付实施计划，不实现新增算法，不修改原检测器。建议将当前仓库实现冻结为 **repository baseline**，在其最终 Detection 之后附加可关闭的 Persistence Verification。当前实现不能未经核对就称为与原论文逐项一致的官方 baseline。

审查基准：Git `3864488d88ba80188aa0e2b03453779d3183b4ba`。阅读范围包括全部自有 Python 源码、入口、README、依赖、制图脚本、FRED 运行脚本和未提交的 comparison 重跑脚本；数据集 ZIP、生成视频和第三方虚拟环境不作为自有源码。现有 comparison 重跑继续执行，本计划不改变其参数或代码。

资料：

- 用户技术文档：`/Users/eolande/Downloads/EvDetMAV_持续性检验模块技术实现文档.docx`，完整 26 节及附注；读取 OOXML 并渲染核对全部内容。
- 用户原论文：`/Users/eolande/resources/original_version/EvDetMAV_Generalized_MAV_Detection_From_Moving_Event_Cameras.pdf`，完整 8 页（IEEE 页码 8236–8243），重点核对 IV-A/B/C、Algorithm 1、V-A、VI 前的讨论。
- 论文与技术文档未明确的数值和策略，在下文明确标为待确认；本轮不把它们作为既定实现。

## 1 当前实际调用链

```text
main.py / evdetmav_detector.py
  → evdetmav.cli.main
  → io.iter_input_files
  → cli.process_file
      → io.load_events / load_h5_events
      → io.prepare_events
          坐标 int64、时间转秒、极性 ±1、稳定排序、边界过滤
      → cli.build_windows
      → 每个窗口选取事件，事件数不足时跳过
      → pipeline.process_window
          → saliency.build_density_saliency
          → clustering.initialize_candidates
              连通域 → 矩形间距离合并 → 外扩 → 显著性排序 → Top K
          → periodicity.evaluate_periodicity
              正事件数量 / 相邻片结构 / 相邻片主方向 → 峰谷六项计分
          → score >= tau_p 的候选进入 clustering.refine_candidate
              连通域 → 高斯一致性 → 可能回退粗区域
          → make_detection → WindowResult.detections
      → CSV / 显著图 / 事件框图 / 分割图
```

| 文件 | 已有职责 | MVP 接入考虑 |
|---|---|---|
| `evdetmav/io.py` | H5/NPZ/CSV/TXT 读取和时间、极性处理 | 保持不变；FRED ECF 无损解码在适配层完成 |
| `evdetmav/cli.py` | 窗口生成、调用、输出开关 | 保留旧入口语义；新 runner 必须另外记录被跳过窗口 |
| `evdetmav/saliency.py` | 正负事件交集累积 | 不改 |
| `evdetmav/periodicity.py` | 三类局部特征及峰谷评分 | 不改；不得把 Sp 与持续性 Sc 混用 |
| `evdetmav/clustering.py` | 初始聚类、高斯细化和回退 | 不改 |
| `evdetmav/pipeline.py` | 组织单窗口算法，输出最终检测 | 以返回的 detections 为新增模块输入 |
| `evdetmav/models.py` | Events、Candidate、Detection、WindowResult | 保留字段和旧 CSV 契约；使用独立适配对象 |
| `evdetmav/output.py`、`visualization.py` | 旧 CSV 和 PNG | 不覆盖旧输出；另建持续性日志/视频 |
| `comparison/source/rerun_detector.py` | ZIP 内 H5 解码，真实子进程调用旧入口 | 可用完成的结果和窗口时间线做回放验证，不复写 comparison |
| `output/FRED_current/source/` | 先前视频运行和重绘脚本 | 非独立评测器，不作为原论文效果证据 |
| `figures/processed_video/source/` | 图解和原第 129 段示例 | 不属于 baseline；部分输入已被清理，不能作为现在可复现的默认测试 |

项目目前没有 Track、跨窗关联、持续性配置、单元测试目录或可复用的 Precision/Recall/mAP 评测器。视频成功解码和已有周期性记录检查不等于准确率评测。依赖中的第三方测试不算本项目测试。

## 2 论文与代码差异

分类：**明确差异**是可以从论文公式和代码直接核对的不同；**论文歧义**是论文内部或正文/算法表之间不一致；**未规定的工程选择**不能直接宣布为实现错误。

| ID | Paper behavior | Current code behavior | Difference | Recommended handling |
|---|---|---|---|---|
| P1 | IV-A，PDF 第 3–4 页：每片正负区域直接求交，然后累积 | `saliency.py:46` 起默认先分别做一次 3×3 膨胀再求交；CLI 半径默认 1 | 明确增加空间容差，会扩大响应；参数 0 才是严格逐像素交集 | 冻结当前设置，不暗改为 0；未来如需论文严格分支，单独命名并做对照 |
| P2 | IV-B，第 5 页：称“cosine similarity”，但写出 z-score 后直接内积 `x̄_iᵀx̄_j` | `periodicity.py:45–57,125–128`：z-score 后再除两个向量范数 | 论文术语与所写公式不完全一致；非退化 N 像素情况下直接内积是当前结果的 N 倍 | 报告歧义，不自行选择作者本意；保留当前实现，若要修 baseline 先确认 |
| P3 | IV-C 第 6 页正文称选择周期性得分最高且通过阈值的区域；第 5 页 Algorithm 1 又在 Top K 循环中对所有 `sp >= tau_p` 更新 Pi | `pipeline.py:90–96` 将所有通过阈值的候选送入精细阶段 | 论文正文与伪代码存在选择数量歧义；当前代码不是显式只保留最高分 | 不擅自切换为 argmax；以当前多候选行为冻结，保留多目标/多旋翼关联风险 |
| P4 | IV-C/Algorithm 1 只说明计算 Gaussian shape 并检查一致性 | `clustering.py:84–106` 使用显著值加权高斯与余弦匹配；默认阈值 0.25 | 具体拟合、正则项、度量和阈值是工程补充，论文未充分规定 | 标注为复现实现选择，不能宣称论文给出了 0.25 |
| P5 | 论文没有明确给出“全部精细区域未通过时仍输出粗区域”的规则 | `clustering.py:131–136` 默认非空粗掩码回退，即使得分低于精细阈值仍可输出 | 额外保留路径，会影响最终框与误检；不等于绕过周期性，因为前面已筛过 Sp | 记录 fallback 情况，不擅自禁用；未来变更独立消融 |
| P6 | 论文说明滑动平均后检查峰谷，但未列出这里的切片数、边界处理、突出度、幅度阈值、空片策略等全部细节 | `periodicity.py:87–99` 先删除非有限值，再三点零填充 same 平滑、幅度判断、min-max、突出度 0.15；n=10、m=12 为本地默认 | 多项工程补充；删除无效片会改变特征序列时间结构；幅度判断在归一化前，而 CLI 帮助文本写 normalized-feature amplitude | 先记录事实，不用“论文默认”称呼这些数值，不顺手重写峰谷逻辑 |
| P7 | EventMAV：640×480、每 period 一架 MAV、标签覆盖旋翼区域，使用 10/15/20/30 ms；V-A 的 IoU 阈值为 0.4 | 当前实验使用 FRED 1280×720、连续 H5 30 ms 窗口；默认逐候选 propeller_*，FRED annotations 是无人机标签 | 数据域、标签语义和输出粒度不同，不是直接等价复现 | 新模块先按当前框粒度验证；评估前确认标注与输出框定义，不借用论文指标 |
| P8 | 论文主方向推导的印刷特征值关系写成 `λmax C = ξ C` | 代码 `np.linalg.eigh(C)` 取最大特征值对应向量 | 所印等式在一般维度上不构成标准特征值方程；代码符合邻近文字的“主特征向量”描述 | 按“论文公式疑点”记录；不据此改代码，也不假称确认了作者更正 |

已确认的一致点：密度是每片正事件数量；主方向相似度用相邻主特征向量绝对余弦；Sp 是三类特征的有峰/有谷六个布尔项之和；论文明确给出 tau_s=50、tau_p=3、K=4；没有 FFT 转速模块，也没有原生跨窗口 persistence。

只读数值核对（非新增算法、非完整单元测试）：

1. 两个相同的 4 像素 z-score 向量，直接内积为 4，当前 cosine 为 1，证实 P2 的尺度区别。该尺度可能在 min-max 时抵消，但归一化前的幅度阈值仍可能受影响，不能直接说整体等价。
2. 12 个全为 1 的特征值，经当前 same 三点平均后两端变为 2/3；现有峰谷检查返回 `(True, False)`。这说明常数序列也可能因边界得到一个峰分，但不代表总周期分必然通过，更不单凭这一例断言全部检测错误。
3. CLI 在相邻窗口中用左端包含和右端包含的 searchsorted，边界时刻事件可能重复使用。旧 `run_current.py` 是右端排除；这是两套运行驱动之间的差异，不能用两套结果混作同一 baseline。

本轮核对过程未改上述行为。

## 3 技术文档中需要确认的定义

| ID | 文档位置与问题 | 建议的处理方案（尚未作为实现决定） |
|---|---|---|
| D1 | §7/§10/§17 要求自适应邻域和尺度归一化；§24 又把它们列为第二阶段 | 将文档已经明确写出公式的简单半径和位置归一化作为可开关项，第一阶段不加入额外自适应方法；需要确认该划分 |
| D2 | §14 说至少三种 decision mode，同时要求第一阶段优先 Rejection；另两种没有完整判定/归一化公式 | 第一阶段只交付可运行 Rejection；Cascade、Weighted Fusion 显式报“未实现”，不伪装成同一种方法；待确认后列入后续阶段 |
| D3 | §17 中三个 sigma 和 persistence_threshold 为 null；关联门限、距离归一化分母等也未全定义 | 配置项必须显式校验；启用对应评分/判定时缺失参数应拒绝运行，禁止替换成猜测数值。单元测试可用有解析预期的显式参数，不能当论文参数 |
| D4 | history 不满 L 时的分母、是否计当前窗口、h 的“有效”是否指最终 accepted 均未规定 | 建议固定 L、含当前窗、初始化缺位记 0；valid 指 baseline 当前有效检测，拒绝的 persistence 候选仍可进入观测历史，避免永远无法冷启动；需要确认 |
| D5 | §13 说两次观测启用可计算运动项，但严格 CV 预测需要此前两次观测 | 建议第二次仅记录速度/速率；若无独立前验则 St=None，第三次开始计算预测残差/方向/加速度。禁止用当前点反算速度后再“预测”当前点造成恒为 0 的残差 |
| D6 | 文档示例 Track 默认 trajectory_score=0.0；§13/§21 要求不可用为 None | 建议采用 Optional[float] 和有效项 mask，以 None 表示缺失；不得用 0 冒充可计算低分 |
| D7 | §21“所有可用 score 在 [0,1]”与 baseline Sp∈[0,6]、未归一化 Ss 不兼容 | 新增 Sn/St/Sc 限于 [0,1]；原始 Sp/Ss 原样保留，若新增归一化视图单列命名，不重写 baseline 字段 |
| D8 | 文档说 MAV Candidate，现有默认为多个精细旋翼候选；启用 merge_propellers 会把全部候选合成一个框 | 建议 MVP 按每个现有最终候选建立 Track，不自动合成整机；这不是整机身份跟踪，需确认是否符合实验目标 |
| D9 | Tentative→Confirmed 条件未说明是否独立于 accept；轨迹不可用且 neighborhood 关闭时最终决策未规定 | 建议状态与当前拒绝分别记录；全无有效评分用 undecidable/None，不输出伪分。严格过滤还是冷启动放行必须配置并先确认 |
| D10 | §19 要求可以关闭 periodicity，但列出的四组消融其实均仍包含 baseline 的 periodicity | 四组 persistence 对照均固定 Sp 开启；额外的 periodicity-off 必须独立命名为 ablation，不能称 baseline，不能用 tau_p=0 冒充真正关闭计算 |
| D11 | 当前六段 FRED 没有在本项目声明 validation/test split，且框标签语义有差异 | 先确认验证集清单、GT 解释及匹配规则，再确定 sigma/threshold；不能在同一批测试结果上调参后宣称提升 |

与常规实现细节不同，D1–D5、D8–D11 会改变哪些目标被接受，必须先说明约定。所有“建议”均不等同于用户已经批准。

## 4 MVP 插入位置与 baseline 隔离

```text
不变的原 baseline process_window → 当前窗口最终 Detection[]
                                      │
                           persistence.enabled=false
                                      ├── 原样输出 baseline
                                      │ true
                                      ↓
                              Detection adapter
                                      ↓
                 仅用历史真实观测预测 → 一对一最近邻关联
                                      ↓
                      有界历史、逐窗口 hit/miss 时间线
                                      ↓
                      Sn / St / mask → Sc → Rejection
                                      ↓
                 accepted 当前框 + 完整 tracks.csv/debug video
```

优先采用独立 `evdetmav_persistence.py` 和 `persistence/` 包，不替换旧 `Detection`，不改 saliency/periodicity/clustering 核心。可以先回放本轮 comparison 的 baseline CSV 和完整窗口清单，完成 MVP 确定性验证，再接原始 H5 runner。回放只能使用同一次 baseline 运行的文件，记录代码 hash、数据来源和参数，不能把旧自定义切窗结果混入。

重要约束：

- 持续性只验证当前 baseline 真正输出的候选。接受集合必须是当前 Detection 集合的子集，框坐标不变。Lost 轨迹的预测中心只作调试标记，绝不输出为检测框。
- 每个真实窗口调用一次 tracker，包括零检测、低事件数和漏检窗口；不能仅迭代 CSV 中有框的行，否则连续漏检会凭空消失。
- 同一窗口的多个候选共同关联；每个候选/Track 最多匹配一次，平局顺序确定。`propeller_1` 是当前窗口标签，不能用作历史 Track ID。
- 每个源序列单独 reset；不能让 8.zip 末尾 Track 接到 20.zip。跨文件同一序列只有显式 sequence_id 与连续时间证据时才允许延续。
- 先使用历史数据完成预测和匹配，再用当前观测算残差，最后更新历史；避免当前观测泄漏进自己的预测。
- 建议 timestamp 统一取 t_end_sec（检测完成时刻）；保留 t_start/end 供审计，使用秒计算真实 Δt。框沿用 x1/y1 exclusive 的现有约定。
- 原 CSV 完整保留；新增 tracks.csv/decisions.csv 等另存。禁用模块时，除非单独明确比较计时，否则所有原始检测字段保持一致；延迟字段本来就受运行时间影响，不要求新跑数值逐位相同。
- baseline 模式不验证未启用 persistence 的 null sigma/threshold，不创建 Track，不因没有持久性参数而阻止原程序独立执行。

## 5 拟新增和修改的文件

以下是计划清单，不表示这些算法文件已经创建。

| 文件 | 计划职责 |
|---|---|
| `evdetmav_persistence.py` | 新入口；config 选择 baseline 或独立 persistence runner |
| `persistence/__init__.py`、`types.py` | 独立 Detection adapter、Track、TrackState、评分及有效项结构 |
| `persistence/config.py` | YAML/JSON 配置读取、范围/单位/缺失值校验、effective config 导出 |
| `persistence/association.py` | 一对一 nearest-neighbor/greedy 关联，门控与确定性平局处理 |
| `persistence/neighborhood.py` | binary/weighted Sn、有界窗口命中记录、邻域半径 |
| `persistence/trajectory.py` | CV 预测、真实 Δt、误差/速度/方向/加速度、有效项 mask 和 St |
| `persistence/scoring.py` | 有效项权重归一化、Sn/St 融合、Rejection 判定 |
| `persistence/tracker.py` | 生命周期、miss/恢复/reset、有界历史；不生成预测检测框 |
| `persistence/baseline_adapter.py` | 原 Detection/CSV 转换、完整窗口清单、旧入口委托；不重写原核心 |
| `persistence/runner.py` | 按 source/window 驱动，输出 baseline 与增强结果，低事件窗口 tick |
| `persistence/logging.py`、`visualization.py` | 全部候选评分、关联和生命周期日志；文字化 debug video |
| `persistence/ablations.py` | 若确认第一阶段包含 periodicity-off：复用原单模块函数构成独立变体；默认 baseline 路径不经过它 |
| `configs/baseline.yaml` | persistence.enabled=false；原参数不变 |
| `configs/persistence_mvp.yaml` | Rejection 配置；待标定参数保持显式待填写 |
| `configs/ablations/*.yaml` | N-only、T-only、N+T；额外 periodicity-off 单列 |
| `tests/test_config.py`、`test_association.py`、`test_neighborhood.py` | 配置、关联、邻域公式与边界 |
| `tests/test_trajectory.py`、`test_scoring.py`、`test_tracker.py` | 运动模型、融合、状态与数值稳定性 |
| `tests/test_baseline_compat.py`、`test_pipeline_integration.py`、`test_outputs.py` | 关闭模块等价性、无预测框、窗口完整性、CSV/video 对齐 |
| `tests/fixtures/` | 小型确定性事件/检测序列与已冻结 baseline 期望结果 |
| `requirements-dev.txt`、`requirements-persistence.txt` | 测试和新增可选依赖；不把新增模块依赖强加到旧入口 |
| `README.md`、`docs/persistence_mvp.md` | 清楚区分原算法、扩展、命令、限制和失败案例 |

原则上不修改 `evdetmav_detector.py`、`main.py`、`evdetmav/{saliency,periodicity,clustering,pipeline,models,output}.py`。若确认必须在同一 CLI 中新增开关，先提交只涉及入口包装的变更说明，并以 baseline 回归用例验证默认路径不变。不会擅自推送或重写当前 Git 历史。

## 6 公式与计划实现的对应关系

| 公式或规则 | 计划实现 | 数值与可用性约束 |
|---|---|---|
| `d = ||c - ĉ||₂`；可选文档的 distance+IoU cost | `association.py` | 只使用以前的真实观测预测；尺度/距离门限可配置，保证一对一 |
| `Sn = Σh/L` | `neighborhood.py` | 命中和未命中均入时间线，history_length 限制窗口数 |
| `Sn = Σρʲh(t-j)/Σρʲ` | `neighborhood.py` | 0<ρ≤1；ρ=1 应与相同分母的 binary 一致 |
| `v_prev=(c_prev-c_prevprev)/Δt_prev`；`ĉ=c_prev+v_prev·Δt` | `trajectory.py` | 真实秒；首观测不可估速度；不使用当前 c 来构造其预测 |
| `ep=||c-ĉ||₂`；`ep_norm=ep/(sqrt(w_prev h_prev)+ε)` | `trajectory.py` | 分别保留 px 和无量纲值；小框需正尺度/epsilon 策略 |
| `Δθ=acos(clip(dot(v,v_prev)/(norm(v)norm(v_prev)+ε),-1,1))` | `trajectory.py` | 弧度；任一相关速度低于 min_speed 时关闭方向项 |
| `a=(v-v_prev)/Δt` | `trajectory.py` | px/s²；缺历史或非正 Δt 时 None |
| `St=exp(-Σ λ'_q (error_q/σ_q)²)` | `trajectory.py`、`scoring.py` | λ' 仅对有效、启用项重新归一化；sigma 必须>0；无有效项时 None |
| `Sc=αSn+(1-α)St` | `scoring.py` | 按可用且启用的分量重分配权重；St 不可用且 Sn 可用时 Sc=Sn；其他缺失组合需明确策略 |
| `Sc<threshold` 拒绝，否则接受 | `scoring.py` | 仅过滤当前真实候选；None 不当作 0；边界等于 threshold 的行为有测试 |

新增 Sn/St/Sc 限于 [0,1]。原始 Sp 仍为 0–6，Ss 仍为原显著性总和。时间单位 s，速度 px/s，加速度 px/s²，方向 rad；归一化位置 sigma 为无量纲，未归一化时为 px。模板的 `min_speed=0.5` 若按此约定为 0.5 px/s，仍只是文档示例值，不是已验证阈值。

文档示例 L=8、max_missed=2、ρ=0.85、α=0.5 和运动权重 0.5/0.25/0.25 可保留在示例配置中并注明出处；null sigma/threshold 不会被悄悄填入“经验值”。

## 7 实施顺序与每步验收

1. **冻结并验收 baseline。** 保存代码/输入 hash、参数、窗口策略、小型真实 H5 fixture 与参考输出。两条入口相同性；新模块关闭时等价；不触动当前 comparison 任务。
2. **实现 config、types 和基础公式。** 独立可测函数，先跑各模块单元测试；拒绝非法参数。此时不接长视频，也不改变原检测器。
3. **实现关联与 Track 生命周期。** 最近邻一对一、每窗口 tick、漏检恢复、有界 deque、source reset、无预测检测框不变量。
4. **实现融合与 Rejection。** 明确 warm-up/mask/缺失策略后实现；输出保留与被拒绝候选，避免只记录成功样本。
5. **接入原始 H5 或已冻结 baseline 回放。** baseline 与增强结果使用完全相同事件窗口；先一个短片段，再完整序列。日志和 debug video 可追溯到 CSV 行。
6. **运行四组对照并交付。** baseline、N-only、T-only、N+T，周期性保持开启；periodicity-off 另行清晰命名。先报告工程正确性、拒绝率、轨迹长度、warm-up 延迟和计算开销；只有 GT/划分/评估协议确认后再报告 P/R/F1 等效果。
7. **第一阶段到此停止。** 汇报新增/修改文件、公式映射、有效配置、命令、测试结果、真实例子及失败案例。没有证据和新授权，不加入 Hungarian、Kalman、DeepSORT、ByteTrack、神经网络、光流或第二阶段扩展。

## 8 单元测试和集成测试清单

| 类别 | 必须覆盖 | 判定依据 |
|---|---|---|
| 悬停 | 固定位置、零速度、微小噪声 | 方向项关闭，不能得到虚假的 π/2 惩罚；有足够历史时残差为 0 的 St 接近 1 |
| 匀速 | 等间隔及不等间隔 timestamp | 已知速度和 CV 预测正确，三观测后 ep≈0、a≈0 |
| 转弯/加速 | 连续小转角、骤转、不同 Δt | 连续软评分，无单一角度硬删除；同参数下更大误差不提高评分 |
| 噪声/跳变 | 同轨迹内跳变、门限外随机目标 | 可匹配跳变降低 St；门外目标成为新 tentative，不强行串成一轨后宣称轨迹评分有效 |
| 漏检 | hit-hit-hit-miss-hit-hit、长时间消失 | miss 增加；≤max_missed 不删除，超过才 Deleted；漏检期间无新增检测框 |
| 冷启动 | 1/2/3 次观测 | 不可算字段 None，mask 正确；当前观测不泄漏；被拒绝 warm-up 候选仍按已确认策略维护历史 |
| 关联 | 两目标争同 Track、相同成本、输入重排 | 一对一、不依赖偶然容器次序；不以 P1/P2 当稳定身份 |
| 时间与序列 | 同窗多个目标、重复窗口、回退时间、跨 ZIP | 区分“多候选同 timestamp”与重复处理；非正 Δt 不除零；序列 reset |
| 几何与有限值 | 极小/退化框、NaN/Inf、空输入 | 无效输入显式拒绝/记录原因；不传播 NaN；合法分数范围正确 |
| 配置/消融 | 总开关、N/T 开关、rho=1、零权重、null sigma | 关闭项不参加融合；rho=1 退化一致；缺必需参数给明确错误 |
| baseline 契约 | 原入口与增强入口关闭模块对照 | 框、次序、事件数、Sp/Ss、高斯得分等一致；计时单独比较 |
| 输出 | 接受/拒绝/丢失全覆盖 | CSV 禁止 NaN/Inf；空值配合 valid mask；视频帧时间和 CSV 可对应 |

每新增一个算法模块都运行对应测试，保存命令、环境、日志和失败样例。不得为了“测试通过”删除难例，也不得以生成视频代替单元测试。

## 9 中间日志和可视化规范

每个 baseline 当前候选，无论最终接受或拒绝，`tracks.csv` 至少记录技术文档 §18 的全部字段：timestamp、track_id、center、bbox、saliency_score、periodicity_score、neighborhood_score、trajectory_score、persistence_score、velocity、speed、prediction_error、direction_change、acceleration、track_state、accepted。

额外计划记录 source_id、window_id、t_start/end、baseline_detection_id、prediction_center、normalized_error、association_distance、association_cost、gate_radius、hit、有效项 mask、有效权重、age/hit_count/miss_count、判定原因和 baseline code hash。无候选窗另存 `windows.csv`；Track 状态变化另存 `track_events.jsonl`；关联代价和中间公式量存 `debug.jsonl`。CSV 的不可用浮点留空并配可用性字段，JSON 用 null，禁止字符串 NaN 或 Inf。

debug video 保持原始分辨率，显示当前真实检测框、稳定 Track ID、有限历史轨迹、带 `PRED` 字样的预测点，以及 `Sp=4/6 Sn=… St=N/A Sc=…`、`ACCEPT/REJECT/LOST` 文字。颜色不能是唯一编码。不存在当前 detection 时只允许画预测点/历史调试标记，不能画成最终检测框。中文使用宋体、英文使用 Times New Roman，保留用户此前字体要求；图中文字不得盖住主要目标。

结果建议隔离保存到 `results/persistence_mvp/<run_id>/`，分别保留 baseline、各消融、完整配置、输入清单、CSV、视频、测试日志和失败案例。不会覆盖 `comparison/` 的原 baseline 结果。

## 10 计划命令与交付边界

原 baseline 命令现已存在：

```sh
python evdetmav_detector.py --input /path/to/decoded_events.h5 \
  --out /path/to/baseline --width 1280 --height 720 \
  --time-unit us --window-ms 30 --step-ms 30
```

下列为计划接口，尚未实现，不可当作已经可运行的命令：

```sh
python evdetmav_persistence.py --config configs/baseline.yaml --input /path/to/decoded_events.h5
python evdetmav_persistence.py --config configs/persistence_mvp.yaml --input /path/to/decoded_events.h5
python -m pytest tests/
```

进入实现前需要确认：

1. 是否按本报告建议冻结当前 repository baseline，不在 MVP 中修论文差异，并按现有最终候选粒度关联？
2. 是否接受“第一阶段 Rejection + 文档已给出的简单半径/位置归一化可选项”，把未定义的其他判定模式留到后续？冷启动及第二观测处理是否采用 D4/D5 的建议？
3. validation set 使用哪些序列或独立数据，sigma/threshold 怎样标定？若暂不定义，是否先交付单元测试和诊断评分模块，把真实数据上的正式接受/拒绝实验保持待标定状态？

本轮交付的是审查和计划；没有新增持续性算法，没有运行持续性单元测试，没有生成新增模块实验结果，也没有宣称提升检测性能。
