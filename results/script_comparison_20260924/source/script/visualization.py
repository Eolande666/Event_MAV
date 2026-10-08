"""Optional PNG rendering; imported only when --save-frames is enabled.

Public API: render_frame(...) returns a PIL Image; save_frame(path, ...) writes it.
Rendering never mutates events or detection dictionaries.
"""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont


COLORS = {'detected': (30, 220, 90), 'joint_reject': (255, 145, 30),
          'joint_candidate': (255, 220, 60),
          'budget_deferred': (160, 170, 185)}
HEADER = 56


def render_frame(events, detections, width, height, *, candidates=(),
                 show_candidates=False, window=0, start_s=0, end_s=0):
    """Render full-resolution event pixels below a separate status header.

    Positive pixels are red, negative blue, and mixed polarity magenta.
    Boxes use exclusive x1/y1, matching detections.csv.
    """
    events = np.asarray(events).reshape(-1, 4)
    pixels = np.full((height, width, 3), 15, dtype=np.uint8)
    if len(events):
        x, y = events[:, 0].astype(int), events[:, 1].astype(int)
        positive = np.zeros((height, width), bool)
        negative = np.zeros_like(positive)
        pos, neg = events[:, 3] > 0, events[:, 3] < 0
        positive[y[pos], x[pos]] = True
        negative[y[neg], x[neg]] = True
        pixels[positive] = (245, 75, 75)
        pixels[negative] = (70, 145, 255)
        pixels[positive & negative] = (220, 100, 235)
    image = Image.new('RGB', (width, height+HEADER), (22, 27, 34))
    image.paste(Image.fromarray(pixels), (0, HEADER))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((8, 5), f'Window {window:06d} | {start_s:.4f} - {end_s:.4f} s | Events {len(events)} | Detections {len(detections)}', font=font, fill='white')
    draw.text((8, 22), '+ red / - blue / mixed purple | green: detected', font=font, fill=(190, 200, 210))
    if show_candidates:
        draw.text((8, 38), 'yellow: joint pending | orange: joint rejected | gray: budget', font=font, fill=(190, 200, 210))
    label_rects = []

    def draw_box(candidate, detected):
        raw = candidate['box']
        x0, y0 = max(0, int(raw[0])), max(0, int(raw[1]))
        x1, y1 = min(width-1, int(raw[2])-1), min(height-1, int(raw[3])-1)
        if x1 < x0 or y1 < y0:
            return
        state = candidate.get('state', 'budget_deferred')
        color = COLORS['detected'] if detected else COLORS.get(state, COLORS['budget_deferred'])
        draw.rectangle((x0, y0+HEADER, x1, y1+HEADER), outline=color, width=2 if detected else 1)
        tag = 'JOINT' if detected else {'joint_reject': 'REJECT', 'joint_candidate': 'PENDING',
                                      'budget_deferred': 'BUDGET'}.get(state, state.upper())
        rotation_label = f"{candidate['rotation_q']:.2f}" if candidate.get('observable', False) else 'N/A'
        label = (f"ID {candidate.get('track_id', '?')} {tag} Q={candidate.get('joint_score', 0):.2f} "
                 f"Rot={rotation_label} Om={candidate.get('rotation_omega', 0):.0f} "
                 f"F={candidate.get('rotation_fit', 0):.2f} G={candidate.get('rotation_rigidity', 0):.2f}")
        label_width = int(draw.textlength(label, font=font))+6
        tx = max(0, min(x0, width-label_width))
        ty = max(HEADER, y0+HEADER-14)
        for offset in range(0, height, 15):
            found = False
            for proposed_y in (ty-offset, ty+offset):
                if proposed_y < HEADER or proposed_y+13 >= height+HEADER:
                    continue
                rect = (tx, proposed_y, min(width-1, tx+label_width), proposed_y+13)
                if not any(rect[0] <= r[2] and rect[2] >= r[0] and rect[1] <= r[3] and rect[3] >= r[1] for r in label_rects):
                    ty = proposed_y
                    found = True
                    break
            if found:
                break
        label_rects.append((tx, ty, min(width-1, tx+label_width), ty+13))
        if abs(ty-(y0+HEADER-14)) > 15:
            draw.line((x0, y0+HEADER, tx, ty+7), fill=color, width=1)
        draw.rectangle((tx, ty, min(width-1, tx+label_width), min(height+HEADER-1, ty+13)), fill=(12, 15, 20))
        draw.text((tx+3, ty), label, font=font, fill=color)

    if show_candidates:
        for candidate in candidates:
            if not candidate.get('published', False):
                draw_box(candidate, False)
    # Draw final detections last so their outlines have priority.
    for detection in detections:
        draw_box(detection, True)
    return image


def save_frame(path, events, detections, width, height, **kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    render_frame(events, detections, width, height, **kwargs).save(path)
