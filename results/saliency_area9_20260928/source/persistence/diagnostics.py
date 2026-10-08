"""Reproducible synthetic diagnostics, not FRED performance evaluation."""
import csv
import json
from pathlib import Path
from .types import Detection
from .trajectory import motion_features
from .scoring import trajectory_score
from .neighborhood import neighborhood_score


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    output = Path('results/persistence_mvp/primitives')
    output.mkdir(parents=True, exist_ok=True)
    font = FontProperties(fname='/System/Library/Fonts/Supplemental/Times New Roman.ttf')
    parameters = dict(weights=dict(position=.5,direction=.25,acceleration=.25),
                      sigmas=dict(position=1.,direction=1.,acceleration=1.),normalize_position=True)
    config = dict(purpose='Synthetic formula checks only; scales are not calibrated on FRED',
                  trajectory=parameters, min_speed=.5, epsilon=1e-9,
                  neighborhood=dict(length=8, mode='weighted', decay=.85, cold_start='fixed'))
    (output/'diagnostic_config.json').write_text(json.dumps(config, indent=2)+'\n')
    scenarios = {
        'Hover': [(0,0)]*12,
        'Constant velocity': [(i,0) for i in range(12)],
        'Smooth turn': [(i, .05*i*i) for i in range(12)],
        'Position jumps': [(i, 3*(-1)**i) for i in range(12)],
    }
    fig, axes = plt.subplots(1,2, figsize=(11,4))
    records = []
    for name, points in scenarios.items():
        history, values = [], []
        for t, (x,y) in enumerate(points):
            d = Detection(float(t),(x-1,y-1,x+1,y+1),500,6,10)
            m = motion_features(history, d,min_speed=.5,epsilon=1e-9)
            st, weights = trajectory_score(m,**parameters)
            records.append(dict(scenario=name,timestamp=t,center=d.center,trajectory_score=st,
                                effective_weights=weights,**m.to_dict()))
            values.append(st)
            history.append(d)
        axes[0].plot(range(12), values, label=name)
    for name, hits in {'Continuous':[1]*12,'Short miss':[1,1,1,0,1,1,1,1,1,1,1,1],
                       'Disappear':[1,1,1,0,0,0,0,0,0,0,0,0]}.items():
        scores = [neighborhood_score(hits[:i+1],**config['neighborhood']) for i in range(12)]
        axes[1].plot(range(12),scores,label=name)
        records.extend(dict(scenario=name,timestamp=i,hit=hits[i],neighborhood_score=s) for i,s in enumerate(scores))
    for ax, title in zip(axes, ['Trajectory formula diagnostics','Neighborhood formula diagnostics']):
        ax.set_title(title,fontproperties=font)
        ax.set_xlabel('Window index (synthetic)',fontproperties=font)
        ax.set_ylabel('Score',fontproperties=font)
        ax.set_ylim(-.03,1.05)
        ax.legend(prop=font)
        for tick in ax.get_xticklabels()+ax.get_yticklabels():
            tick.set_fontproperties(font)
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output/'diagnostics.svg')
    fig.savefig(output/'diagnostics.png',dpi=160)
    plt.close(fig)
    with (output/'debug.jsonl').open('w') as f:
        for row in records:
            f.write(json.dumps(row,allow_nan=False)+'\n')
    fields = sorted(set().union(*(r.keys() for r in records)))
    with (output/'diagnostics.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow({k:json.dumps(v,allow_nan=False) if isinstance(v,(dict,tuple)) else v for k,v in r.items()})


if __name__ == '__main__':
    main()
