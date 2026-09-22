# EvDetMAV reproduction

This folder contains a standalone reproduction of the algorithm flow from
`EvDetMAV: Generalized MAV Detection from Moving Event Cameras`.

Pipeline implemented in `evdetmav_detector.py`:

1. Density-aware saliency map generation
   - Split one event period `Delta T` into `n` slices.
   - For each slice, build positive and negative binary event images.
   - Accumulate the intersection `I_s^t = I_p^t intersection I_n^t`.
   - Scale the accumulated saliency map to 0-255 and threshold with the paper
     default `tau_s = 50`.

2. Spatio-temporal feature extraction
   - Rank saliency clusters by saliency score and keep the paper default
     `K = 4`.
   - Split each local event stream into `m` slices.
   - Compute the paper features:
     - `f_d`: positive-event density per slice.
     - `f_s`: cosine similarity of consecutive normalized slice images.
     - `f_p`: principal-direction similarity from consecutive slice point sets.
   - Apply moving-average filtering and score whether each feature contains
     peaks and valleys. The maximum periodicity score is 6. The paper default
     threshold `tau_p = 3` is used.

3. Clustering-based MAV detection
   - Initialize potential areas from connected saliency components.
   - Merge nearby rectangles using the paper's minimum rectangle-distance idea.
   - Refine accepted areas with connected saliency regions and a fitted 2D
     Gaussian consistency check.
   - Keep refined cyan boxes as the final detections by default. Use
     `--merge-propellers` only when you want one merged MAV box.

The paper does not specify all engineering thresholds, so unspecified values
are command-line parameters. The reported `tau_s`, `tau_p`, and `K` defaults
are preserved.

## Examples

Per-period H5 directory, matching the EventMAV dataset style:

```powershell
python scripts\EventMAV\evdetmav_detector.py `
  --input inputs\170\Event\h5 `
  --out outputs\evdetmav_170 `
  --limit-files 5
```

Continuous NPZ stream with 30 ms sliding windows:

```powershell
python scripts\EventMAV\evdetmav_detector.py `
  --input inputs\170\Event\output_events.npz `
  --out outputs\evdetmav_170_npz `
  --window-ms 30 `
  --step-ms 30 `
  --max-windows 20
```

Use the strict saliency intersection described in the paper:

```powershell
python scripts\EventMAV\evdetmav_detector.py `
  --input inputs\170\Event\h5 `
  --out outputs\evdetmav_strict `
  --intersection-radius 0
```

## Outputs

```text
evdetmav_detections_batch.csv
source_manifest.csv
per_file/*_detections.csv
saliency/*.png
event_boxes_evdetmav/*.png
segmentation/*.png
```

CSV boxes use exclusive `bbox_x1` and `bbox_y1` coordinates.

By default, `evdetmav_detections_batch.csv` stores the refined `propeller_*`
boxes. These are the cyan boxes in the visualization. Passing
`--merge-propellers` restores the earlier single green `mav` box behavior.

## Repository setup

Install the detector dependencies and inspect its options:

```sh
python -m pip install -r requirements.txt
python main.py --help
```

The repository tracks the detector, figure-generation scripts, and experiment
scripts/documentation. Local FRED datasets, generated videos, Python environments,
caches, and locally installed fonts are intentionally excluded from Git; they
remain available in the local project folder.

FRED data: https://huggingface.co/datasets/GabrieleMagrini/FRED

Scripts in `output/FRED_current/source/` document the local six-sequence experiment.
They need additional video/plotting dependencies and local resources as noted in
`output/FRED_current/README.md`; installing the core requirements alone does not
set up the FRED ECF decoder or local fonts. The figure PDFs included here are
illustrations, not an accuracy benchmark.
