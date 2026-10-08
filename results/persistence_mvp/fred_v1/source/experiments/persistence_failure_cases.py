"""Keep reproducible successes and failures; GT overlay is for evaluation only."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import csv,json,subprocess
from collections import defaultdict
import imageio_ffmpeg
from PIL import Image,ImageDraw
from persistence.visualization import LATIN
from experiments.run_persistence_fred import OUT,HOLDOUT,dataset


def main():
    cases=[]
    candidates=[]
    for seq in HOLDOUT:
        _,_,aligned=dataset(seq)
        frames={int(r['window_id']):r for r in aligned}
        before={int(r['window_id']):r for r in csv.DictReader((OUT/seq/'baseline_center_frames.csv').open())}
        after={int(r['window_id']):r for r in csv.DictReader((OUT/seq/'combined_center_frames.csv').open())}
        for wid,a in after.items():
            b=before[wid]
            if wid<100:continue
            candidates.append(dict(sequence=seq,window_id=wid,before={k:int(b[k]) for k in ('tp','fp','fn')},
                after={k:int(a[k]) for k in ('tp','fp','fn')},gt=frames[wid]['gt'],timestamp=frames[wid]['timestamp']))
    choices=[('true_candidate_rejected','largest increase in FN, then earliest sequence/window',
              lambda r:r['after']['fn']-r['before']['fn']),
             ('persistent_false_positive','largest retained FP, then earliest sequence/window',lambda r:r['after']['fp']),
             ('false_positive_removed','largest FP decrease with no TP decrease, then earliest sequence/window',
              lambda r:r['before']['fp']-r['after']['fp'] if r['before']['tp']==r['after']['tp'] else -1)]
    dest=OUT/'failure_cases';dest.mkdir(exist_ok=True)
    for name,rule,key in choices:
        selected=max(candidates,key=key)
        seq=selected['sequence'];wid=selected['window_id'];target=dest/(name+'.png')
        video=OUT/seq/'videos/Comparison.mp4'
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-v','error','-i',str(video),'-vf',f'select=eq(n\\,{wid})',
                        '-frames:v','1',str(target)],check=True)
        image=Image.open(target).convert('RGB');draw=ImageDraw.Draw(image)
        for shift in (0,1280):
            for gt in selected['gt']:
                x0,y0,x1,y1=gt['bbox'];box=(shift+x0,64+y0,shift+x1,64+y1)
                draw.rectangle(box,outline=(255,80,240),width=2)
                draw.text((max(shift,min(shift+1150,shift+x0)),max(65,64+y0-23)),f"GT {gt['identity']}",font=LATIN,fill=(255,80,240))
        image.save(target)
        selected.update(case=name,selection_rule=rule,image=str(target.relative_to(OUT)),note='Magenta GT is evaluation-only; never used for detection or video tracking')
        cases.append(selected)
    (dest/'cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2)+'\n')
    report=OUT/'REPORT.md'
    with report.open('a') as f:
        f.write('\n## 可复查的成功与失败帧\n\n洋红框为评估标注，仅在这些分析图中叠加，未送入检测器。选择规则和原始 TP/FP/FN 见 failure_cases/cases.json；不用于替代全帧统计。\n\n')
        for c in cases:
            f.write(f"- {c['case']}：序列 {c['sequence']}，窗口 {c['window_id']}；[查看图片]({c['image']})。\n")
    print(json.dumps(cases,indent=2))


if __name__=='__main__':main()
