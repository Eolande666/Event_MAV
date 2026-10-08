"""Generate auditable evaluation report and scientific figures from saved counts."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import json,csv,hashlib,platform,subprocess,shutil
from dataclasses import replace
from collections import defaultdict
import numpy as np
from persistence.config import Config
from persistence.replay import replay,boxes_by_window,dump_csv
from persistence.evaluation import evaluate,metrics
from experiments.run_persistence_fred import dataset,OUT,HOLDOUT,SEQUENCES


def percent(value):return 'N/A' if value is None else f'{100*value:.2f}%'


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    plt.rcParams['svg.fonttype']='path'
    latin=FontProperties(fname='/System/Library/Fonts/Supplemental/Times New Roman.ttf')
    chinese=FontProperties(fname=str(ROOT/'figures/fonts/SongtiSC-Regular.ttf'))
    aggregate=json.loads((OUT/'metrics_aggregate.json').read_text())
    sequence=list(csv.DictReader((OUT/'metrics_by_sequence.csv').open()))
    cfg=Config.load(OUT/'combined_config.json')
    controlled=[]
    for seq in HOLDOUT:
        windows,baseline,aligned=dataset(seq)
        for mode,flags in [('neighborhood',dict(use_neighborhood=True,use_trajectory=False)),('trajectory',dict(use_neighborhood=False,use_trajectory=True))]:
            config=replace(cfg,**flags)
            rows,_=replay(windows,baseline,config,OUT/seq/('controlled_'+mode))
            for protocol in ('iou40','iou50','center'):
                result,_=evaluate(aligned,boxes_by_window(baseline,rows),protocol)
                controlled.append(dict(sequence=seq,mode=mode,protocol=protocol,**result))
    dump_csv(OUT/'controlled_ablation_by_sequence.csv',controlled)
    ctrl=[]
    for mode in ('neighborhood','trajectory'):
        for protocol in ('iou40','iou50','center'):
            group=[r for r in controlled if r['mode']==mode and r['protocol']==protocol]
            ctrl.append(dict(mode=mode,protocol=protocol,**metrics(*(sum(r[k] for r in group) for k in ('tp','fp','fn')))))
    dump_csv(OUT/'controlled_ablation_aggregate.csv',ctrl)
    def get(mode,protocol):return next(r for r in aggregate if r['split']=='holdout' and r['mode']==mode and r['protocol']==protocol)
    figures=OUT/'figures';figures.mkdir(exist_ok=True)
    modes=['baseline','neighborhood','trajectory','combined']
    labels=['Baseline','+ Neighborhood','+ Trajectory','+ N + T']
    colors=['#526d82','#257d98','#b78732']
    for protocol,title in [('iou50','严格框检测指标'),('center','候选中心定位诊断指标')]:
        fig,ax=plt.subplots(figsize=(9,4.5))
        x=np.arange(4)
        for j,(key,label) in enumerate([('precision','Precision'),('recall','Recall'),('f1','F1')]):
            vals=[100*get(m,protocol)[key] for m in modes]
            bars=ax.bar(x+(j-1)*.24,vals,.24,label=label,color=colors[j])
            for bar,val in zip(bars,vals):
                ax.text(bar.get_x()+bar.get_width()/2,bar.get_height(),f'{val:.2f}',ha='center',va='bottom',fontproperties=latin,fontsize=9)
        ax.set_title(title,fontproperties=chinese,fontsize=16,pad=16)
        ax.set_xticks(x,labels,fontproperties=latin);ax.set_ylabel('Percent (%)',fontproperties=latin)
        ax.set_ylim(0,1 if protocol=='iou50' else 100)
        ax.text(.01,1.015,'IoU >= 0.50 | Local holdout: 51, 65, 93, 114' if protocol=='iou50' else 'Center containment, one-to-one | NOT box detection AP',transform=ax.transAxes,fontproperties=latin,fontsize=10)
        ax.legend(prop=latin,loc='upper right',ncol=3);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        for tick in ax.get_yticklabels():tick.set_fontproperties(latin)
        fig.tight_layout();fig.savefig(figures/(protocol+'_metrics.png'),dpi=180);fig.savefig(figures/(protocol+'_metrics.svg'));plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    x=np.arange(4)
    for j,(mode,label) in enumerate([('baseline','Baseline'),('combined','+ N + T')]):
        vals=[100*float(next(r['f1'] for r in sequence if r['sequence']==s and r['mode']==mode and r['protocol']=='center')) for s in HOLDOUT]
        ax.bar(x+(j-.5)*.34,vals,.34,label=label)
    ax.set_xticks(x,HOLDOUT,fontproperties=latin);ax.set_ylim(0,100)
    ax.set_title('各序列的定位诊断对照',fontproperties=chinese,fontsize=16)
    ax.set_ylabel('Center-diagnostic F1 (%)',fontproperties=latin);ax.set_xlabel('Sequence',fontproperties=latin)
    for t in ax.get_yticklabels():t.set_fontproperties(latin)
    ax.legend(prop=latin);fig.tight_layout();fig.savefig(figures/'per_sequence.png',dpi=180);fig.savefig(figures/'per_sequence.svg');plt.close(fig)
    # Hash the precise experiment implementation and inputs; preserve source copy.
    source=OUT/'source';source.mkdir(exist_ok=True)
    source_hashes={}
    for pattern in ('persistence/*.py','experiments/*persistence*.py','tests/test_persistence*.py'):
        for path in ROOT.glob(pattern):
            relative=path.relative_to(ROOT);target=source/relative;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target);source_hashes[str(relative)]=hashlib.sha256(path.read_bytes()).hexdigest()
    for path in (ROOT/'evdetmav_persistence.py',ROOT/'configs/baseline.json'):
        relative=path.relative_to(ROOT);target=source/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        source_hashes[str(relative)]=hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT/'source_hashes.json').write_text(json.dumps(source_hashes,indent=2))
    input_hashes={}
    for seq in SEQUENCES:
        for name in ('evdetmav_detections_batch.csv','video_windows.csv'):
            p=ROOT/'comparison'/seq/name;input_hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    (OUT/'input_hashes.json').write_text(json.dumps(input_hashes,indent=2))
    environment=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,baseline_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    (OUT/'environment.json').write_text(json.dumps(environment,indent=2))
    def table(protocol,source_rows=None):
        lines=['| 方法 | TP | FP | FN | Precision | Recall | F1 |','|---|---:|---:|---:|---:|---:|---:|']
        for mode,label in zip(modes,labels):
            r=get(mode,protocol)
            lines.append(f"| {label} | {r['tp']} | {r['fp']} | {r['fn']} | {percent(r['precision'])} | {percent(r['recall'])} | {percent(r['f1'])} |")
        return '\n'.join(lines)
    b=get('baseline','center');c=get('combined','center')
    strictb=get('baseline','iou50');strictc=get('combined','iou50')
    runtimes=list(csv.DictReader((OUT/'runtime.csv').open()))
    times=[r for r in runtimes if r['sequence'] in HOLDOUT and r['mode']=='combined']
    mean=sum(float(r['mean_ms'])*int(r['windows']) for r in times)/sum(int(r['windows']) for r in times)
    controlled_table=['| 方法（阈值统一为 0.4） | Precision | Recall | F1 |','|---|---:|---:|---:|']
    for r in ctrl:
        if r['protocol']=='center':controlled_table.append(f"| {r['mode']} | {percent(r['precision'])} | {percent(r['recall'])} | {percent(r['f1'])} |")
    controlled_table.append(f"| combined | {percent(c['precision'])} | {percent(c['recall'])} | {percent(c['f1'])} |")
    perseq=['| 序列 | Baseline P / R / F1 | 第三级 P / R / F1 |','|---|---|---|']
    for s in HOLDOUT:
        row=[]
        for mode in ('baseline','combined'):
            r=next(r for r in sequence if r['sequence']==s and r['mode']==mode and r['protocol']=='center')
            row.append(' / '.join(percent(float(r[k])) for k in ('precision','recall','f1')))
        perseq.append(f"| {s} | {row[0]} | {row[1]} |")
    lines=f'''# EvDetMAV 第三级持续性检验：完整流程与本地留出实验

## 结论先行

第三级作为拒绝过滤器，减少误报但损失部分召回。严格框检测仍然很弱，不能声称已经实现高精度整机检测。

- **严格 IoU≥0.5**：Precision {percent(strictb['precision'])} → {percent(strictc['precision'])}，Recall {percent(strictb['recall'])} → {percent(strictc['recall'])}，F1 {percent(strictb['f1'])} → {percent(strictc['f1'])}。
- **候选中心定位诊断（不是标准框检测指标）**：Precision {percent(b['precision'])} → {percent(c['precision'])}，Recall {percent(b['recall'])} → {percent(c['recall'])}，F1 {percent(b['f1'])} → {percent(c['f1'])}。
- 定位诊断 FP 从 {b['fp']} 降至 {c['fp']}，减少 {(b['fp']-c['fp'])/b['fp']*100:.2f}%；TP 从 {b['tp']} 降至 {c['tp']}，减少 {b['tp']-c['tp']}，FN 相应增加。过滤器不生成框，因此不能恢复 baseline 没有发现的目标。

## 实验范围与真实性

当前 FRED 文件夹保留的 8、20、51、65、93、114 六段均完成持续性处理。第一、二级使用 comparison/ 中此前由真实 evdetmav_detector.py 从原始 ZIP/H5 生成的完整检测结果，核验源代码 SHA-256 后冻结复用；本轮没有重复计算全部序列的前两级，也没有使用旧 RGB 检测或 GT 生成候选。第三级对每一个窗口重新运行，包括没有候选的窗口。

另从原始 8.zip 直接运行新入口 12 个窗口：43 个 baseline 检测与缓存逐字段一致（源路径、计时除外），持续性逐行结果也一致。关闭模块后用同一 H5 fixture 调用原入口与新入口，结果一致。证据见 integration_contract.json。

全部视频重新读取原始 ZIP 内的 H5 事件流绘制，逐窗核对事件数和时间范围与 baseline 一致。没有读取 RGB。完整视频按文件编号升序拼接；不同文件是独立录制片段，不假称跨文件连续时间线。

## 数据划分与评估口径

官方 canonical split 表明本地六段全部属于官方 test。此次从中取 8、20 作本地校准，51、65、93、114 作本地留出；这不是官方完整 benchmark，也不是论文 EventMAV 数据集结果。校准完成并保存配置后才载入留出标注，未用留出成绩选择参数。数据量只有四个留出序列，未证明统计显著性和跨数据集泛化。

GT 使用 ZIP 内 coordinates.txt（整架无人机），按官方 Event/Frames 文件名中的微秒时间戳建立帧网格。无标注的已有帧按官方 loader 语义视为无目标。扩展标注裁到 1280×720 可见范围；完全出界框不计。每个 GT 时刻使用此前最近已完成的 30 ms baseline 窗口，最大允许年龄 30.001 ms；不拿未来检测匹配过去标注。两种算法完全相同的匹配时刻；开头没有已完成窗口的帧不参与比较。留出合计 {b['frames']} 帧、{b['tp']+b['fn']} 个 GT 目标实例。

- 严格框指标：原候选框对整机 GT，IoU≥0.5；另报 IoU≥0.4。未扩大或合并原框。
- 定位诊断：候选中心落入 GT 框即有资格匹配，但仍是一对一；同机重复框仍算 FP。它只回答候选是否落在标注目标上，不能取代 IoU 检测指标，更不能称为 mAP。
- 两个口径均用最大匹配数计算 TP；FP=输出框数−TP，FN=GT数−TP。Precision=TP/(TP+FP)，Recall=TP/(TP+FN)，F1=2TP/(2TP+FP+FN)。没有输出时 Precision 为 N/A，未定义量不伪造为 100%。按所有帧计数汇总（micro），不是逐序列百分比平均。
- 原框常常覆盖旋翼/局部事件区域，FRED 框覆盖整机，严格 IoU 很低是实际测量结果；不能据此单独断定新增持续性模块无效，也不能用中心口径掩盖框定位失败。

## 严格框检测结果：IoU≥0.5

{table('iou50')}

![严格指标](figures/iou50_metrics.png)

## 严格框检测结果：IoU≥0.4

{table('iou40')}

## 候选中心定位诊断

{table('center')}

![定位诊断](figures/center_metrics.png)

这些三种增强方法各自使用校准集上选定的阈值；不是完全相同参数的因果消融。仅邻域阈值 0.5，仅轨迹阈值 0，联合阈值 0.4；关联半径均为 40 px。

**仅轨迹的阈值为 0：其筛选作用来自前两次观测无法评分时暂不输出。** 已有可用 St 的候选不因分数低而被拒绝。因此上表不能单独证明方向/加速度软评分具有增益，也不能把这项改善全部归因于运动合理性。

为区分阈值变化，补充固定联合配置、仅切换 N/T 开关、所有阈值固定 0.4 的诊断对照：

{chr(10).join(controlled_table)}

完整严格指标也保存在 controlled_ablation_aggregate.csv。轨迹单独使用时 St 不可用即保持 tentative 并暂不输出；联合模式则依文档回退到 Sn。

## 分序列与失败分析

{chr(10).join(perseq)}

![分序列诊断](figures/per_sequence.png)

- 51：联合 F1 基本不变，Precision 的增加由 Recall 下降抵消。不能宣称所有场景都明显改善。
- 65：联合优于 baseline，但仅邻域/仅轨迹的 F1 反而更高，融合不是处处最优。
- 93：baseline 召回已经很高，第三级主要减少额外候选，同时仍丢失部分真候选。
- 114：联合定位 Precision 仍只有约 26.5%，保留大量持续出现的背景误检，说明“持续存在”并不等于“无人机”。
- 拒绝阶段只能删框；冷启动、目标突然出现、关联门外快速移动、短轨迹均可能产生额外漏检。当前逐旋翼候选关联还可能在多旋翼或近邻无人机之间切换，不等于已验证稳定 UAV 身份跟踪。
- IoU≥0.5 的 TP 由 125 降为 123，主要改善来自删除更多 FP，不是框定位能力提高。

## 参数与实现决策

基于用户要求继续跑完整实验，本次将未定义策略明确作为可复现的工程实验配置，而非论文规定：固定 L=8 冷启动分母，包含当前有效 baseline hit，尚未积累的位置记 0；历史可保留被 persistence 拒绝的 baseline 候选，防止冷启动自锁。max_missed_windows=2，超过才删除；只维护有界历史，不用预测框代替检测。

第一阶段采用固定半径的 greedy nearest-neighbor，一对一匹配；关联半径在校准集上从 10/20/40 px 中选取。未启用速度自适应半径（velocity_scale=0）。St 第三次有效观测起计算；按真实秒时间差计算速度、方向和加速度，低速关闭方向，缺失项重新归一化。Sc 缺少 St 时按技术文档 §14 回退到 Sn，修正了上一轮基础函数遗漏此规则的问题。

位置采用文档 §10/§17 已给出的归一化开关（开启），这在审查中与 §24 阶段划分的重叠已报告；未修改原 baseline。σ 来自校准集中候选中心匹配 GT 的正残差 90% 分位数，不是旋翼真实运动标签或论文常数：

- sigma_position = {cfg.sigma_position:.9g}（归一化位置误差）
- sigma_direction = {cfg.sigma_direction:.9g} rad
- sigma_acceleration = {cfg.sigma_acceleration:.9g} px/s²
- λp/λθ/λa = 0.5/0.25/0.25；有效项权重再归一化。
- α=0.5，时间衰减 ρ=0.85，min_speed=0.5 px/s，ε=1e−9。
- 联合拒绝阈值=0.4，通过 >= 判定；校准目标是中心诊断 F1，阈值网格 0..1，步长 0.05。

校准搜索、样本数、参数、每条候选 Sc/Sn/St、匹配代价和状态事件均保留，不只保存成功案例。periodicity-off 接口支持从原始 H5 真正绕过周期性，不能从已过滤 CSV 反推；本次所有四组正式对照始终开启原周期性。

## 运行开销与验证

联合模块在留出集全部窗口上的平均额外耗时约 {mean:.4f} ms/窗口。这仅是 Python tracker.step 的实测耗时，不含 CSV 写入、解码、原检测和视频编码，也不是端到端 FPS；不能用它宣称整个检测器实时。详情见 runtime.csv。

单元测试 29 项通过，涵盖冷启动、悬停、非均匀时间匀速、转弯、跳变、重复时间戳、无效数值、小框、短漏检、长消失、关联竞争、输入重排、有界历史、开关、真实周期性旁路、时间窗口与评估匹配。原 baseline 源码哈希保持一致。

## 视频与复现命令

- videos/Comparison_all.mp4：完整左右对照，左原 baseline、右加入第三级后的接受框。
- videos/Persistence_all.mp4：完整第三级最终检测视频。
- videos/Debug_all.mp4：接受/拒绝、Track ID、历史、PRED、Sp/Sn/St/Sc 中间结果。
- 每个编号/videos/ 下也保存对应三版单段视频和预览图。原 baseline 完整视频仍为 comparison/Detection_all.mp4。
- 图表 PNG 与矢量 SVG 位于 figures/；中文宋体，英文 Times New Roman；原检测框以原像素坐标绘制，不拉伸视频内容。

在 /Users/eolande/EventMAV 运行：

```sh
# 完整本地实验（校准 + 留出 + 四组对照）
tmp/fred_figures/venv/bin/python experiments/run_persistence_fred.py
# 原始 H5 事件素材重新绘制视频
tmp/fred_figures/venv/bin/python experiments/render_persistence_fred.py
# 报告、固定阈值对照、图表
tmp/fred_figures/venv/bin/python experiments/report_persistence_fred.py
# 直接从原始 ZIP 跑完整三级（去掉 max-windows 即全段）
tmp/fred_figures/venv/bin/python evdetmav_persistence.py --config results/persistence_mvp/fred_v1/combined_config.json --input FRED/8.zip --out results/persistence_mvp/direct_8 --width 1280 --height 720 --time-unit us --window-ms 30 --step-ms 30 --max-windows 12
# 测试
tmp/fred_figures/venv/bin/python -m unittest discover -s tests -v
```

入口关闭时使用 configs/baseline.json，原 CLI 参数原样转交（原入口只接受已解码 H5，不接受 ZIP）。本次 direct runner 按 FRED structured x/y/t/p 的微秒 H5 读取；ECF 解码依赖项目已有 tmp/fred_figures/libecf_decode.dylib，视频依赖 imageio_ffmpeg，不能只复制新 Python 文件到无依赖机器就保证运行。

## 文件与公式映射

- persistence/types.py/config.py：数据结构、配置、有限值检查。
- association.py/tracker.py：文档 §5/§15，一对一关联、Tentative/Confirmed/Lost/Deleted。
- neighborhood.py：§6/§8，Sn 的普通/时间衰减平均。
- trajectory.py/scoring.py：§10–14，CV 预测、位置/方向/加速度、St、Sc。
- replay.py/runner.py/events.py/baseline_adapter.py：冻结结果回放、原始 H5 入口、精确 baseline 窗口、消融适配。
- evaluation.py：严格 IoU 和独立中心诊断、时间对齐、一对一 TP 计数。
- visualization.py：原像素检测框、调试轨迹和文字。
- experiments/：校准评估、视频、契约验证、报告生成；tests/：测试。

未改 evdetmav/、evdetmav_detector.py、main.py；未增加 Kalman、Hungarian tracker、深度跟踪器或光流。完整原论文/代码差异仍见 docs/persistence_mvp_review_and_plan.md。本次完成第一阶段后停止；下一阶段应先解决整机标注与旋翼输出不匹配、补独立训练/验证素材和身份标注，再决定是否优化关联，而不是直接堆更复杂算法。

## FRED 引用（GB/T 7714 格式）

[1] MAGRINI G, MARINI N, BECATTINI F, et al. FRED: The Florence RGB-Event Drone Dataset[C]//Proceedings of the 33rd ACM International Conference on Multimedia. 2025. DOI: 10.1145/3746027.3758271.

官方资料：[FRED repository](https://github.com/miccunifi/FRED)、[canonical splits](https://github.com/miccunifi/FRED/tree/main/dataset_splits/canonical)、[data loader](https://github.com/miccunifi/FRED/blob/main/src/data/data.py)。使用的 split 文本与 loader 副本见 provenance/。本报告未虚构会议页码或出版地。
'''
    (OUT/'REPORT.md').write_text(lines)
    print('REPORT COMPLETE',flush=True)


if __name__=='__main__':main()
