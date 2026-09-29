#!/usr/bin/env python3
"""
Plot pTDP-43 inclusion detections from infer_wsi.py on the slide thumbnail: detections as dots (left)
and a smoothed density map in inclusions / mm^2 of tissue (right).

Example:
  python make_heatmap.py --slide slide.svs --detections results/slide_detections.csv \
      --out results/slide_heatmap.png
"""
import argparse

import matplotlib
import numpy as np
import openslide
from scipy.ndimage import gaussian_filter

from infer_wsi import TRAIN_MPP_UM, tissue_mask

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide", required=True)
    ap.add_argument("--detections", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.25, help="plot detections with conf >= this")
    ap.add_argument("--sigma-um", type=float, default=400, help="density smoothing (Gaussian sigma, um)")
    ap.add_argument("--bin-um", type=float, default=50)
    a = ap.parse_args()

    slide = openslide.OpenSlide(a.slide)
    W, H = slide.dimensions
    mpp = float(slide.properties.get("openslide.mpp-x", TRAIN_MPP_UM))
    d = np.loadtxt(a.detections, delimiter=",", skiprows=1).reshape(-1, 5)
    d = d[d[:, 4] >= a.conf]
    cx, cy = (d[:, 0] + d[:, 2]) / 2, (d[:, 1] + d[:, 3]) / 2

    try:
        thumb = np.array(slide.get_thumbnail((1600, 1600)).convert("RGB"))
    except AttributeError:   # openslide-python < 1.3 with Pillow >= 10 (Image.ANTIALIAS removed)
        lvl = slide.level_count - 1
        small = slide.read_region((0, 0), lvl, slide.level_dimensions[lvl]).convert("RGB")
        small.thumbnail((1600, 1600))
        thumb = np.array(small)
    k = thumb.shape[1] / W
    mask, ds = tissue_mask(slide)

    b = a.bin_um / mpp
    nx, ny = int(np.ceil(W / b)), int(np.ceil(H / b))
    grid, _, _ = np.histogram2d(cy, cx, bins=[ny, nx], range=[[0, ny * b], [0, nx * b]])
    dens = gaussian_filter(grid, a.sigma_um / a.bin_um) / (a.bin_um / 1000) ** 2
    yy, xx = np.mgrid[0:ny, 0:nx]
    tm = mask[np.clip(((yy + 0.5) * b / ds).astype(int), 0, mask.shape[0] - 1),
              np.clip(((xx + 0.5) * b / ds).astype(int), 0, mask.shape[1] - 1)]
    dens = np.ma.masked_where(~tm, dens)

    fig, ax = plt.subplots(1, 2, figsize=(16, 8 * H / W + 1))
    ext = (0, W * k, H * k, 0)
    ax[0].imshow(thumb, extent=ext)
    ax[0].scatter(cx * k, cy * k, s=2, c="#e0201b", linewidths=0)
    ax[0].set_title(f"pTDP-43 inclusion detections (n = {len(d):,}, conf >= {a.conf})")
    ax[1].imshow(thumb, extent=ext)
    im = ax[1].imshow(dens, extent=(0, nx * b * k, ny * b * k, 0), cmap="Spectral_r", alpha=0.7,
                      interpolation="bilinear")
    ax[1].set_title("pTDP-43 inclusion density")
    fig.colorbar(im, ax=ax[1], fraction=0.035, label="inclusions / mm²")
    for x in ax:
        x.set_xlim(0, W * k)
        x.set_ylim(H * k, 0)
        x.axis("off")
    fig.tight_layout()
    fig.savefig(a.out, dpi=150, facecolor="white")
    print(f"written {a.out}")


if __name__ == "__main__":
    main()
