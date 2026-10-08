"""Aggregate the common-window experiment; never feeds labels to a detector."""
from pathlib import Path
import sys,json,csv,subprocess,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from persistence.evaluation import metrics
from persistence.replay import dump_csv
import imageio_ffmpeg
OUT=ROOT/'results/script_comparison_20260924'
SEQS=['8','20','51','65','93','114'];METHODS=['baseline','persistence','script']
NAMES={'baseline':'原 EvDetMAV','persistence':'EvDetMAV + 持续性','script':'script（30 ms）'}

def pct(v):return 'N/A' if v is None else f'{100*v:.2f}%'
def main():
 rows=[];summaries=[]
 for seq in SEQS:
  summary=json.loads((OUT/seq/'summary.json').read_text());assert summary['status']=='complete';summaries.append(summary)
  config_path=OUT/seq/'configuration.json';config=json.loads(config_path.read_text());config.update(status='complete',windows=summary['windows']);config_path.write_text(json.dumps(config,ensure_ascii=False,indent=2))
  rows.extend(json.loads((OUT/seq/'metrics.json').read_text()))
 aggregate=[]
 for split,seqs in [('all',SEQS),('holdout',SEQS[2:]),('prior_calibration',SEQS[:2])]:
  for protocol in ['iou40','iou50','center']:
   for method in METHODS:
    group=[r for r in rows if r['sequence'] in seqs and r['protocol']==protocol and r['method']==method]
    aggregate.append(dict(split=split,method=method,protocol=protocol,frames=sum(r['frames'] for r in group),**metrics(*(sum(r[k] for r in group) for k in ['tp','fp','fn']))))
 dump_csv(OUT/'metrics_by_sequence.csv',rows);dump_csv(OUT/'metrics_aggregate.csv',aggregate)
 (OUT/'metrics_aggregate.json').write_text(json.dumps(aggregate,indent=2))
 lines=['# script 与 EvDetMAV 同数据比较','', '全部使用原始 FRED ZIP 中的 H5；没有读取 RGB，也没有把标注送入检测器。三种方法使用相同的 30 ms 半开事件窗口，全部重新计算。script 的旋转窗口保留默认末尾 9 ms，联合阈值保持 0.34，未根据本次结果调参。','', 'script 原默认窗长为 33.333 ms；本次统一为 30 ms，故是同窗比较，不是原默认窗长的复测。原历史 EvDetMAV 输出采用包含右端点的窗口，本次统一为 [start,end)，不能将历史数字直接当作本轮数字。','', '之前持续性参数使用 8/20 校准，因此主表报告未参与该校准的 51/65/93/114，另列六段全量。所有六段都属于 FRED 官方 test，本报告的留出是本地划分。','']
 for split,title in [('holdout','四段留出结果'),('all','六段全量结果')]:
  for protocol,desc in [('iou50','严格框检测 IoU ≥ 0.5'),('center','中心落入标注框（定位诊断，不等同于框检测）')]:
   lines += [f'## {title}：{desc}','','| 方法 | TP | FP | FN | Precision | Recall | F1 |','|---|---:|---:|---:|---:|---:|---:|']
   for r in aggregate:
    if r['split']==split and r['protocol']==protocol:
     lines.append(f"| {NAMES[r['method']]} | {r['tp']} | {r['fp']} | {r['fn']} | {pct(r['precision'])} | {pct(r['recall'])} | {pct(r['f1'])} |")
   lines.append('')
 lines += ['## 留出集差异','']
 for protocol in ('iou50','center'):
  selected={r['method']:r for r in aggregate if r['split']=='holdout' and r['protocol']==protocol}
  for ref in ('baseline','persistence'):
   deltas=', '.join(f"{k} {(selected['script'][k]-selected[ref][k])*100:+.2f} 个百分点" for k in ('precision','recall','f1') if selected['script'][k] is not None and selected[ref][k] is not None)
   lines.append(f"- {protocol}，script 相比 {NAMES[ref]}：{deltas}。")
 lines += ['','## 分序列严格 IoU ≥ 0.5','','| 序列 | 方法 | Precision | Recall | F1 |','|---|---|---:|---:|---:|']
 for r in rows:
  if r['protocol']=='iou50':lines.append(f"| {r['sequence']} | {NAMES[r['method']]} | {pct(r['precision'])} | {pct(r['recall'])} | {pct(r['f1'])} |")
 lines += ['','## 耗时与覆盖','','耗时来自本机同轮运行的检测调用，不包含视频编码和H5解码；两种实现的数据转换位置不完全相同，且渲染/编码共享机器资源，所以是本机观测值，不作为严格硬件基准或实时性结论。','','| 序列 | 窗口 | 原算法平均 ms | 持续性附加 ms | script 平均 ms | script P95 ms |','|---|---:|---:|---:|---:|---:|']
 for s in summaries:lines.append(f"| {s['sequence']} | {s['windows']} | {s['baseline_ms']['mean']:.2f} | {s['persistence_ms']['mean']:.2f} | {s['script_ms']['mean']:.2f} | {s['script_ms']['p95']:.2f} |")
 total_windows=sum(s['windows'] for s in summaries)
 weighted={key:sum(s[key]['mean']*s['windows'] for s in summaries)/total_windows for key in ('baseline_ms','persistence_ms','script_ms')}
 lines += ['',f"按窗口数加权的平均调用耗时：baseline {weighted['baseline_ms']:.2f} ms，持续性附加 {weighted['persistence_ms']:.3f} ms，script {weighted['script_ms']:.2f} ms；baseline/script 约 {weighted['baseline_ms']/weighted['script_ms']:.2f} 倍。"]
 total=sum(s['script_detections'] for s in summaries);obs=sum(s['accepted_observable'] for s in summaries)
 lines += ['',f'处理 {sum(s["windows"] for s in summaries)} 个窗口，视频总时长约 {sum(s["video_duration_s"] for s in summaries):.2f} 秒。script 输出 {total} 个窗口候选，其中 {total-obs} 个的旋转项不可观测。候选总数 {sum(s["candidates"] for s in summaries)}，可观测旋转候选 {sum(s["observable_candidates"] for s in summaries)}，IOC预算耗尽候选 {sum(s["budget_deferred"] for s in summaries)}。', '', '## script 算法如何实现','', '1. 同一像素的事件按时间排序，统计正负极性切换次数，以及切换覆盖了8个时间格中的多少格，生成显著图。','2. 对显著图做邻域最大值聚合和连通域提取，筛除太小、太大和高度重叠区域，最多保留24个候选。','3. 计算重复活动 P（重复切换、时间覆盖、同像素持续活动、背景对比）和结构 C（有效像素的填充程度与连通程度）。','4. 每窗最多4个候选计算一次IOC：用末尾9 ms的正负事件分别构造最新时间表面，由局部梯度估计速度。两极性共同有效像素以5×5块取均值，至少3个不共线块才能评价旋转。','5. 共同拟合一个角速度，两极性的平移分别吸收；正负速度同向/反向两种情况择残差小者。旋转评分 Rot = S × F × G，分别表示加入旋转项的F统计累计量、超越平移模型的解释率、旋转相对形变的能量比例。它不是已校准的真实旋转概率。','6. Q = 0.55 Rot + 0.30 P + 0.15 C，Q ≥ 0.34即输出；检测框由当前候选内事件坐标2%–98%分位数确定。最近中心与面积比关联只维持ID，不增加跨窗持续性评分，也不延迟确认。','', '**实现细节：**旋转不可观测时 Rot 取0，P与C最高仍能贡献0.45，因此可能达到0.34阈值并输出；不能称为“每个输出都通过旋转验证”。IOC预算耗尽则明确拒绝。本实现也没有原算法的三条时序特征峰谷评分。','','## 结果解读边界','','FRED标注对应完整无人机，而两种算法都可能给局部旋翼/事件区域框；严格IoU和中心命中回答不同问题，不能用中心指标或README的IoA替代框检测Precision/Recall。script自带28/209序列的结果来自另一组数据，本报告不与其直接比较。固定机位假设也不能直接视为已经适用于所有移动相机场景。','','## 文件','','- `videos/Comparison_all.mp4`：左原算法、中原算法加持续性、右script，逐序列8→20→51→65→93→114，各段内部保持时间顺序。','- `videos/Script_all.mp4`：script完整检测视频。','- 各序列目录：独立视频、逐候选JSONL、逐窗耗时、逐帧评测及检测CSV。','- `code_sha256.json`、`source/`、`protocol.json`：代码快照和协议。','- script现有31项单元测试全部通过。']
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
 videos=OUT/'videos';videos.mkdir(exist_ok=True)
 ff=imageio_ffmpeg.get_ffmpeg_exe()
 verification={}
 for kind in ('Script','Comparison'):
  listing=videos/(kind+'_concat.txt');listing.write_text('\n'.join("file '"+str(OUT/seq/(kind+'.mp4'))+"'" for seq in SEQS))
  with (videos/(kind+'_concat.log')).open('w') as log:subprocess.run([ff,'-y','-f','concat','-safe','0','-i',str(listing),'-c','copy','-movflags','+faststart',str(videos/(kind+'_all.mp4'))],stdout=log,stderr=log,check=True)
  checked=subprocess.run([ff,'-v','error','-i',str(videos/(kind+'_all.mp4')),'-progress','pipe:1','-f','null','-'],capture_output=True,text=True,check=True)
  (videos/(kind+'_verify.log')).write_text(checked.stderr)
  assert not checked.stderr
  fields=dict(line.split('=',1) for line in checked.stdout.splitlines() if '=' in line)
  assert int(fields['frame'])==total_windows
  verification[kind]=dict(frames=int(fields['frame']),duration_s=int(fields['out_time_us'])/1e6)
 (videos/'verification.json').write_text(json.dumps(verification,indent=2))
 hashes=json.loads((OUT/'code_sha256.json').read_text())
 assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h for name,h in hashes.items())
 (OUT/'status.json').write_text(json.dumps(dict(status='complete',sequences=SEQS,windows=sum(s['windows'] for s in summaries),algorithms_unchanged=True,videos_decoded_without_errors=True),indent=2))
 print('\n'.join(lines[:35]))
if __name__=='__main__':main()
