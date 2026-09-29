# NeuroSpot-YOLO

YOLO-based object detectors for quantifying neuropathology in whole-slide images (WSI).
Each model lives in its own top-level folder with everything needed to run it on a new
slide or reproduce its training: weights, inference/heatmap scripts, training config, and
a model card documenting validation performance and limitations.

## Models

| Model | Target | Status |
|---|---|---|
| [`amyloid_plaque/`](amyloid_plaque/) | Amyloid-β plaques (4G8 IHC) | Released |

More neuropathology targets (tau, TDP-43, Lewy body) are planned; each will follow the same
layout as `amyloid_plaque/` below.

## Layout (per model)

```
<model>/
  README.md          # usage, validation metrics, limitations -- read this before using the model
  infer_wsi.py        # run the model on one or more whole-slide images
  make_heatmap.py      # detection + density figure from infer_wsi.py's output
  requirements.txt
  weights/
    download_weights.sh   # fetches the released weights (too large to commit to git)
    SHA256SUMS
  training/            # scripts and config to reproduce training from scratch
  examples/             # a sample output on an unseen slide
```

## Quick start

See [`amyloid_plaque/README.md`](amyloid_plaque/README.md) for the currently released model.
In brief:

```bash
cd amyloid_plaque
pip install -r requirements.txt
bash weights/download_weights.sh
python infer_wsi.py --slides /path/to/slides/ --out results/ --dsa
```

## Contributing a new model

Follow the layout above: a self-contained folder with its own `README.md` covering intended
use, training data, validation metrics (on a genuinely held-out set), and known failure modes.
Weights should be distributed as a GitHub release asset (referenced by `weights/download_weights.sh`
+ `weights/SHA256SUMS`), not committed directly, to keep the repository small.
