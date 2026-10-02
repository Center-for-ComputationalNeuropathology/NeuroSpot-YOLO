# NeuroSpot-YOLO

YOLO-based object detectors for quantifying neuropathology in whole-slide images (WSI).
Each model lives in its own top-level folder with everything needed to run it on a new
slide or reproduce its training: weights, inference/heatmap scripts, training config, and
a model card documenting validation performance and limitations.

## Models

| Model | Target | Status |
|---|---|---|
| [`amyloid_plaque/`](amyloid_plaque/) | Amyloid-β plaques (4G8 IHC) | Released |
| [`ptdp43_inclusion/`](ptdp43_inclusion/) | pTDP-43 neuronal cytoplasmic inclusions (pTDP-43 IHC) | Released |
| [`tau_tangle/`](tau_tangle/) | Neurofibrillary tangles (AT8 p-tau IHC) | Released — see limitations |
| [`lewy_body/`](lewy_body/) | Lewy bodies (α-synuclein IHC) | **Experimental** — test release, low accuracy |

The Lewy body detector is an experimental test release: it is included so it can be tried,
but it does not yet perform acceptably on held-out patients (see its README).

## Layout (per model)

```
<model>/
  README.md          # usage, validation metrics, limitations -- read this before using the model
  infer_wsi.py        # run the model on one or more whole-slide images
  make_heatmap.py      # detection + density figure from infer_wsi.py's output
  requirements.txt
  weights/
    <model>.pt          # trained weights, stored via Git LFS (see .gitattributes)
    SHA256SUMS
  training/            # scripts and config to reproduce training from scratch
  examples/             # a sample output on an unseen slide
```

## Quick start

Each model's `README.md` has its own details and limitations
([`amyloid_plaque/`](amyloid_plaque/README.md), [`ptdp43_inclusion/`](ptdp43_inclusion/README.md),
[`tau_tangle/`](tau_tangle/README.md)).
In brief (same interface for every model):

```bash
git lfs install   # once per machine, if not already set up -- see amyloid_plaque/README.md
cd amyloid_plaque
pip install -r requirements.txt
python infer_wsi.py --slides /path/to/slides/ --out results/ --dsa
```

## Contributing a new model

Follow the layout above: a self-contained folder with its own `README.md` covering intended
use, training data, validation metrics (on a genuinely held-out set), and known failure modes.
Weights go in `weights/` tracked via Git LFS (`git lfs track "*.pt"`, already set up for the
whole repo in `.gitattributes`), with a `SHA256SUMS` alongside for integrity checking.
