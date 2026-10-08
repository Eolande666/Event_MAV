"""Final artifact audit, manifests and source snapshot (run after video rendering)."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import json,hashlib,shutil,statistics,re
import imageio_ffmpeg
from experiments.render_persistence_fred import OUT,SEQUENCES,render_signature


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main():
    if json.loads((OUT/'video_progress.json').read_text())['stage']!='complete':
        raise RuntimeError('video rendering still in progress')
    video_manifest=[]
    for kind in ('Persistence','Debug','Comparison'):
        path=OUT/'videos'/f'{kind}_all.mp4'
        frames,seconds=imageio_ffmpeg.count_frames_and_secs(str(path))
        assert frames==22088
        reader=imageio_ffmpeg.read_frames(str(path));metadata=next(reader);reader.close()
        expected=(2560,784) if kind=='Comparison' else ((1280,928) if kind=='Debug' else (1280,784))
        assert tuple(metadata['size'])==expected,(kind,metadata['size'])
        assert (OUT/'videos'/f'{kind}_verify.log').stat().st_size==0
        video_manifest.append(dict(file=str(path.relative_to(ROOT)),frames=frames,duration_sec=seconds,
                                   dimensions=expected,bytes=path.stat().st_size))
    write(OUT/'video_manifest.json',video_manifest)
    audit={};delays=[]
    for seq in SEQUENCES:
        births={};confirmed={};count=accepted=0
        config=json.loads((OUT/seq/'combined/config.json').read_text())
        for line in (OUT/seq/'combined/debug.jsonl').open():
            row=json.loads(line);count+=1
            for key in ('neighborhood_score','trajectory_score','persistence_score'):
                value=row[key];assert value is None or 0<=value<=1
            assert row['accepted']==(row['persistence_score'] is not None and row['persistence_score']>=config['threshold'])
            accepted+=row['accepted'];i=row['track_id'];births.setdefault(i,row['timestamp'])
            if row['accepted']:confirmed.setdefault(i,row['timestamp'])
        audit[seq]=dict(candidates=count,accepted=accepted,all_scores_valid=True)
        if seq not in ('8','20'):
            d=[(stamp-births[i])*1000 for i,stamp in confirmed.items()]
            delays.append(dict(sequence=seq,created_tracks=len(births),ever_accepted_tracks=len(confirmed),
                               first_accept_delay_median_ms=statistics.median(d),first_accept_delay_max_ms=max(d)))
        path=OUT/seq/'videos/complete.json';info=json.loads(path.read_text());info['render_signature']=render_signature(seq);write(path,info)
    write(OUT/'final_score_audit.json',audit);write(OUT/'confirmation_delay.json',delays)
    from experiments.run_persistence_fred import dataset,HOLDOUT
    from persistence.replay import boxes_by_window
    from persistence.evaluation import match_boxes
    breakdown=[]
    for seq in HOLDOUT:
        _,baseline,aligned=dataset(seq)
        score_rows=[json.loads(line) for line in (OUT/seq/'combined/debug.jsonl').open()]
        for mode,boxes in [('baseline',boxes_by_window(baseline)),('combined',boxes_by_window(baseline,score_rows))]:
            outside=duplicate=0
            for frame in aligned:
                pred=boxes.get(frame['window_id'],[]);gt=[g['bbox'] for g in frame['gt']]
                matched={i for i,j in match_boxes(pred,gt,'center')}
                for i,box in enumerate(pred):
                    if i in matched:continue
                    x,y=(box[0]+box[2])/2,(box[1]+box[3])/2
                    if any(g[0]<=x<g[2] and g[1]<=y<g[3] for g in gt):duplicate+=1
                    else:outside+=1
            breakdown.append(dict(sequence=seq,mode=mode,outside_annotation_fp=outside,inside_annotation_unmatched_fp=duplicate))
    write(OUT/'false_positive_breakdown.json',breakdown)

    for name,digest in json.loads((OUT/'baseline_verified_hashes.json').read_text()).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    testlog=(OUT/'tests.log').read_text();assert testlog.rstrip().endswith('OK')
    tests=int(re.search(r'Ran (\d+) tests',testlog)[1])
    source=OUT/'source';source_hashes={}
    paths=[]
    for pattern in ('persistence/*.py','experiments/*persistence*.py','tests/test_persistence*.py'):paths.extend(ROOT.glob(pattern))
    paths.extend([ROOT/'evdetmav_persistence.py',ROOT/'configs/baseline.json',ROOT/'requirements-persistence.txt'])
    for path in paths:
        relative=path.relative_to(ROOT);target=source/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target);source_hashes[str(relative)]=hashlib.sha256(path.read_bytes()).hexdigest()
    write(OUT/'source_hashes.json',source_hashes)
    write(OUT/'status.json',dict(stage='complete',sequences=SEQUENCES,calibration=['8','20'],holdout=['51','65','93','114'],
                               unit_tests_passed=tests,total_windows=22088,video_duration_sec=662.64,
                               baseline_core_unchanged=True,report='REPORT.md',videos=[v['file'] for v in video_manifest]))
    report=OUT/'REPORT.md';text=report.read_text()
    marker='## 候选确认延迟与最终视频布局'
    if marker in text:text=text.split(marker)[0].rstrip()+'\n'
    text+='\n'+marker+'\n\n联合模块在留出序列 51/65/93/114 上，候选轨迹首次被接受的延迟中位数依次为 90/60/90/60 ms（相对同一 Track 的第一次 baseline 观测）。只统计最终曾被接受的轨迹，不代表真实目标首次可见到检出的端到端延迟；未被接受的轨迹数量另存 confirmation_delay.json。\n\n调试版评分表已移至原事件画面外，尺寸 1280×928；最终检测版为 1280×784，左右对照版为 2560×784。三个视频的 1280×720 原始传感器图像均未缩放。三版均 22,088 帧，662.64 秒，全片解码及尺寸/帧数核验通过。\n'
    text+='\n定位诊断的 FP 另拆分为标注框外候选和框内未获一对一匹配的重复候选，见 false_positive_breakdown.json。114 的保留 FP 全部属于标注框外候选。30 ms 检测窗口与约 33.333 ms 标注帧率不同，因果对齐仍有最多约 30 ms 的结果年龄，快速运动时会影响重叠/中心匹配；两组严格采用相同规则，但不能与官方逐帧同步模型分数直接比较。\n'
    report.write_text(text)
    print(json.dumps(dict(tests=tests,video_manifest=video_manifest),indent=2))


if __name__=='__main__':main()
