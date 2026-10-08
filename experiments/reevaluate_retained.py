"""Re-evaluate retained recordings against ZIP annotations with frozen parameters.

Stages 1/2 use their saved detections. Stage 3 is replayed from window zero,
including empty windows. Ground truth is used only by the offline evaluator.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run_persistence_fred import dataset, SEQUENCES, CALIBRATION, HOLDOUT
from persistence.config import Config
from persistence.evaluation import evaluate, metrics
from persistence.replay import replay, boxes_by_window, dump_csv

LABELS = {
    'baseline': '原 EvDetMAV（显著性＋周期性）',
    'persistence': '原 EvDetMAV＋持续性',
    'area9': '显著面积 ≥9＋持续性',
    'raw3': '扩张前真实像素 ≥3＋持续性',
}


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    out = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=False)
    config_path = ROOT / 'configs/persistence_original.json'
    config = Config.load(config_path)
    config.save(out / 'frozen_config.json')
    summary, inputs, consistency = [], {}, []
    sources = {'area9': ROOT / 'results/saliency_area9_20260928',
               'raw3': ROOT / 'results/raw_pixel3_20260928'}

    def record(path):
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                digest.update(chunk)
        inputs[str(path.relative_to(ROOT))] = digest.hexdigest()

    record(config_path)
    for folder in ['persistence', 'experiments']:
        files = (ROOT / folder).glob('*.py') if folder == 'persistence' else [Path(__file__), ROOT / 'experiments/run_persistence_fred.py']
        for path in files:
            record(path)
    for seq in SEQUENCES:
        print('REEVALUATING', seq, flush=True)
        dest = out / seq
        dest.mkdir()
        windows, baseline, aligned = dataset(seq)  # Fresh annotations read from ZIP.
        record(ROOT / 'FRED' / (seq + '.zip'))
        for filename in ['video_windows.csv', 'evdetmav_detections_batch.csv']:
            record(ROOT / 'comparison' / seq / filename)
        dump_csv(dest / 'alignment.csv', aligned)
        for method in LABELS:
            detections = baseline
            if method in sources:
                path = sources[method] / seq / 'filtered_detections.csv'
                record(path)
                detections = defaultdict(list)
                for row in read_csv(path):
                    detections[int(row['window_id'])].append(row)
            if method == 'baseline':
                predictions = boxes_by_window(detections)
            else:
                scores, _ = replay(windows, detections, config, dest / method)
                predictions = boxes_by_window(detections, scores)
            result, frames = evaluate(aligned, predictions, 'center')
            dump_csv(dest / (method + '_frames.csv'), frames)
            summary.append(dict(sequence=seq, split='calibration' if seq in CALIBRATION else 'holdout',
                                method=method, protocol='center', **result))
            if method in sources:
                old_name = 'filtered' if method == 'area9' else 'raw_support'
                old_path = sources[method] / seq / (old_name + '_center_frames.csv')
            else:
                old_name = 'baseline' if method == 'baseline' else 'combined'
                old_path = ROOT / 'results/persistence_mvp/fred_v1' / seq / (old_name + '_center_frames.csv')
            old = read_csv(old_path)
            identical = len(old) == len(frames) and all(
                all(int(a[k]) == b[k] for k in ['window_id', 'tp', 'fp', 'fn'])
                for a, b in zip(old, frames))
            consistency.append(dict(sequence=seq, method=method, saved_frame_counts_reproduced=identical))
            # A mismatch is reported, never silently substituted with old totals.
        print('COMPLETE', seq, len(aligned), 'evaluation frames', flush=True)
    aggregate = []
    for split, seqs in [('calibration', CALIBRATION), ('holdout', HOLDOUT), ('all', SEQUENCES)]:
        for method in LABELS:
            selected = [r for r in summary if r['sequence'] in seqs and r['method'] == method]
            counts = metrics(*(sum(r[k] for r in selected) for k in ['tp', 'fp', 'fn']))
            aggregate.append(dict(split=split, method=method, frames=sum(r['frames'] for r in selected), **counts))
    dump_csv(out / 'metrics_by_sequence.csv', summary)
    dump_csv(out / 'metrics_aggregate.csv', aggregate)
    write_json(out / 'input_sha256.json', inputs)
    write_json(out / 'consistency.json', consistency)
    write_json(out / 'protocol.json', dict(sequences=SEQUENCES, calibration=CALIBRATION, holdout=HOLDOUT,
        stages_1_2='saved detections, not rerun', stage_3='fresh replay from window zero; frozen config',
        annotations='fresh ZIP coordinates and official Event frame timestamps',
        matching='one-to-one maximum-cardinality center containment; no IoU acceptance threshold',
        aggregation='micro: sum TP/FP/FN before computing P/R/F1',
        tuning=False, limitation='previously inspected local holdout; not a new unseen test set; whole-drone GT, not individual rotor GT'))
    lines = ['# 当前五段素材重新评估', '',
        '校准：8、20；本地留出：51、65、93。重新从原 ZIP 读取标注和事件帧时间戳；第一、二级使用保存的检测结果，第三级按冻结配置从第 0 窗重新运行，包括空窗。未重跑第一、二级，未训练模型或重新调参。',
        '中心落入真值框才可匹配，一对一分配：每个目标最多计一个 TP，其余未匹配候选计 FP，未检出目标计 FN。不使用 IoU 门限评价旋翼候选。按计数加总计算指标，不平均各段百分比。', '']
    for split, title in [('holdout', '本地留出结果'), ('calibration', '校准素材结果'), ('all', '五段合计')]:
        lines += ['## ' + title, '', '| 方法 | Precision | Recall | F1 | TP | FP | FN |', '|---|---:|---:|---:|---:|---:|---:|']
        for r in aggregate:
            if r['split'] == split:
                lines.append(f"| {LABELS[r['method']]} | {r['precision']:.2%} | {r['recall']:.2%} | {r['f1']:.2%} | {r['tp']} | {r['fp']} | {r['fn']} |")
        lines.append('')
    lines += ['## 留出素材逐段对比', '', '| 素材 | 方法 | Precision | Recall | F1 |', '|---|---|---:|---:|---:|']
    for r in summary:
        if r['split'] == 'holdout':
            lines.append(f"| {r['sequence']} | {LABELS[r['method']]} | {r['precision']:.2%} | {r['recall']:.2%} | {r['f1']:.2%} |")
    lines += ['', '## 解释边界', '',
        '当前最优仅指这批已多次查看的本地素材，不能据此证明泛化到新的录制序列。删除一段素材改变了评价集合，不能将旧集合与当前集合的指标变化解释为算法提升。',
        '标注为整机框，因此中心匹配反映候选定位效果，不等于逐旋翼识别精度；同一整机内多个旋翼候选可能被记作重复 FP。',
        f"本轮 {sum(r['saved_frame_counts_reproduced'] for r in consistency)}/{len(consistency)} 项逐段逐方法检查与已保存逐帧 TP/FP/FN 一致。", 
        '仅重新运行持续性与离线评估，不能用本轮耗时比较各算法完整运行速度。全部逐帧匹配、中间评分、参数和输入哈希保存于本目录。']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    write_json(out / 'status.json', dict(status='complete', sequences=SEQUENCES,
        all_frame_counts_reproduced=all(r['saved_frame_counts_reproduced'] for r in consistency)))
    print(json.dumps(aggregate, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
