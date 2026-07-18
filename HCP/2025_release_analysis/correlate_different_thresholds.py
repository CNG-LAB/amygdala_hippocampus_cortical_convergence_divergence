#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Supplementary robustness figure script:
- plots 5% and 15% dominance/sharedness maps
- makes Kendall scatter plots vs 10% using the SAME helper function
  plot_metric_scatter_by_community(...)
- stitches each result into a 4-panel figure:
    [map 5%] [scatter 5 vs 10] [map 15%] [scatter 15 vs 10]

File resolution behavior:
This script requires the exact expected filename for each threshold:
    *_top{threshold}.npy
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from brainspace.datasets import load_conte69
from brainspace.plotting import plot_hemispheres
from brainspace.utils.parcellation import map_to_labels

# ---------------------------------------------------------------------
# Local project imports
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "utils"))
from project_config import ProjectPaths

PATHS = ProjectPaths.from_environment()

from plotting_hippamyg import (
    plot_metric_scatter_by_community,
    YEO7_COLORS,
    gray_rgba,
)

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
BASE_GROUP_DIR = PATHS.output_root / "group_level"
OUT_DIR = BASE_GROUP_DIR / "supp_threshold_maps_vs_10"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GLASSER_LABELING_PATH = str(PATHS.resource_root / "glasser.csv")
GLASSER_TSV_PATH = str(PATHS.resource_root / "atlas-Glasser_dseg.tsv")

TOP_REF = 10
DO_MSR = True
MSR_PARAMS = dict(
    dist_l_path=str(PATHS.resource_root / "Glasser32k_dist_L.npy"),
    dist_r_path=str(PATHS.resource_root / "Glasser32k_dist_R.npy"),
    parc_l_path=str(PATHS.resource_root / "glasser-360_conte69_lh.label.gii"),
    parc_r_path=str(PATHS.resource_root / "glasser-360_conte69_rh.label.gii"),
    n_perm=1000,
    n_proc=8,
    seed=42,
)

MAP_SIZE = (1600, 1000)
MAP_ZOOM = 1.2
DPI = 600

yeo7_dict = {
    "visual": YEO7_COLORS[0],
    "somatomotor": YEO7_COLORS[1],
    "dorsal attention": YEO7_COLORS[2],
    "ventral attention": YEO7_COLORS[3],
    "limbic": YEO7_COLORS[4],
    "frontoparietal": YEO7_COLORS[5],
    "default mode": YEO7_COLORS[6],
}


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    stem: str
    cmap: str
    value_range: tuple[float, float]
    scatter_xlim: tuple[float, float]
    scatter_ylim: tuple[float, float]


METRICS = [
    MetricSpec(
        key="dom_glasso",
        label="Dominance GLASSO",
        stem="hippamyg_dominance_presence_full360_glasso",
        cmap="icefire",
        value_range=(-1.0, 1.0),
        scatter_xlim=(-1.1, 1.1),
        scatter_ylim=(-1.1, 1.1),
    ),
    MetricSpec(
        key="share_glasso",
        label="Sharedness GLASSO",
        stem="hippamyg_shared_presence_full360_glasso",
        cmap="magma",
        value_range=(0.0, 2.0),
        scatter_xlim=(-0.1, 2.1),
        scatter_ylim=(-0.1, 2.1),
    ),
    MetricSpec(
        key="dom_pearson",
        label="Dominance Pearson",
        stem="hippamyg_dominance_presence_full360_pearson",
        cmap="icefire",
        value_range=(-1.0, 1.0),
        scatter_xlim=(-1.1, 1.1),
        scatter_ylim=(-1.1, 1.1),
    ),
    MetricSpec(
        key="share_pearson",
        label="Sharedness Pearson",
        stem="hippamyg_shared_presence_full360_pearson",
        cmap="magma",
        value_range=(0.0, 2.0),
        scatter_xlim=(-0.1, 2.1),
        scatter_ylim=(-0.1, 2.1),
    ),
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def require_exists(path: str | Path) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")


def load_full360_map(base_group_dir: Path, top_percent: int, stem: str) -> np.ndarray:
    """Load the exact full-cortex map for the requested threshold."""
    top_dir = base_group_dir / f"top{top_percent}_hippamyg"
    require_exists(top_dir)

    path = top_dir / f"{stem}_top{top_percent}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing exact file for top{top_percent}: {path}")
    arr = np.load(path)
    arr = np.asarray(arr).squeeze()
    if arr.ndim != 1 or arr.shape[0] != 360:
        raise ValueError(
            f"Expected 1D full360 array of length 360 in {path}, got shape {arr.shape}"
        )
    print(f"[LOAD] top{top_percent}: {path}")
    return arr.astype(float)


def save_surface_map(
    full360: np.ndarray,
    metric: MetricSpec,
    top_percent: int,
    surf_lh,
    surf_rh,
    labeling: np.ndarray,
    mask_labels: np.ndarray,
    out_path: Path,
) -> None:
    vertices = map_to_labels(full360, labeling, mask=mask_labels, fill=np.nan)

    plot_hemispheres(
        surf_lh,
        surf_rh,
        array_name=vertices,
        size=MAP_SIZE,
        cmap=metric.cmap,
        color_bar=True,
        color_range=metric.value_range,
        nan_color=gray_rgba,
        zoom=MAP_ZOOM,
        interactive=False,
        screenshot=True,
        filename=str(out_path),
        suppress_warnings=True,
        transparent_bg=True,
    )


def save_scatter_plot(
    x_360: np.ndarray,
    y_360: np.ndarray,
    metric: MetricSpec,
    top_x: int,
    top_y: int,
    out_path: Path,
) -> None:
    plot_metric_scatter_by_community(
        x_360=x_360,
        y_360=y_360,
        tsv_path=GLASSER_TSV_PATH,
        index_col="index",
        scheme_col="community_yeo7",
        out_path=str(out_path),
        xlabel=f"{metric.label} ({top_x}%)",
        ylabel=f"{metric.label} ({top_y}%)",
        title="",
        xlim=metric.scatter_xlim,
        ylim=metric.scatter_ylim,
        add_identity=False,
        add_reg=True,
        legend=False,
        corr_method="kendall",
        custom_palette=yeo7_dict,
        msr=DO_MSR,
        msr_params=MSR_PARAMS,
        msr_mask_mode="finite",  # pairwise-valid finite parcels only
    )


def stitch_four_panel(
    img_paths: list[Path],
    titles: list[str],
    out_path: Path,
    main_title: str,
) -> None:
    imgs = [mpimg.imread(str(p)) for p in img_paths]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), constrained_layout=True)

    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(title, fontsize=11)

    fig.suptitle(main_title, fontsize=14)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    require_exists(GLASSER_LABELING_PATH)
    require_exists(GLASSER_TSV_PATH)

    labeling = np.genfromtxt(GLASSER_LABELING_PATH, delimiter=",")
    labeling = np.asarray(labeling, dtype=float).ravel()
    mask_labels = labeling != 0

    surf_lh, surf_rh = load_conte69()

    maps_dir = OUT_DIR / "maps"
    scat_dir = OUT_DIR / "scatters"
    comp_dir = OUT_DIR / "composites"
    maps_dir.mkdir(exist_ok=True)
    scat_dir.mkdir(exist_ok=True)
    comp_dir.mkdir(exist_ok=True)

    for metric in METRICS:
        print(f"\n[METRIC] {metric.label}")

        # load threshold maps
        arr_10 = load_full360_map(BASE_GROUP_DIR, TOP_REF, metric.stem)
        arr_5 = load_full360_map(BASE_GROUP_DIR, 5, metric.stem)
        arr_15 = load_full360_map(BASE_GROUP_DIR, 15, metric.stem)

        # surface maps for 5 and 15
        map5_png = maps_dir / f"{metric.key}_map_top5.png"
        map15_png = maps_dir / f"{metric.key}_map_top15.png"

        save_surface_map(
            full360=arr_5,
            metric=metric,
            top_percent=5,
            surf_lh=surf_lh,
            surf_rh=surf_rh,
            labeling=labeling,
            mask_labels=mask_labels,
            out_path=map5_png,
        )
        save_surface_map(
            full360=arr_15,
            metric=metric,
            top_percent=15,
            surf_lh=surf_lh,
            surf_rh=surf_rh,
            labeling=labeling,
            mask_labels=mask_labels,
            out_path=map15_png,
        )

        # scatter plots: 5 vs 10, 15 vs 10
        sc5_png = scat_dir / f"{metric.key}_scatter_top5_vs_top10_kendall.png"
        sc15_png = scat_dir / f"{metric.key}_scatter_top15_vs_top10_kendall.png"

        save_scatter_plot(
            x_360=arr_5,
            y_360=arr_10,
            metric=metric,
            top_x=5,
            top_y=10,
            out_path=sc5_png,
        )
        save_scatter_plot(
            x_360=arr_15,
            y_360=arr_10,
            metric=metric,
            top_x=15,
            top_y=10,
            out_path=sc15_png,
        )

        # stitched 4-panel figure
        panel_png = comp_dir / f"{metric.key}_maps_and_scatter_vs10.png"
        stitch_four_panel(
            img_paths=[map5_png, sc5_png, map15_png, sc15_png],
            titles=[
                "Map 5%",
                "Kendall scatter: 5% vs 10%",
                "Map 15%",
                "Kendall scatter: 15% vs 10%",
            ],
            out_path=panel_png,
            main_title=metric.label,
        )

        print(f"[DONE] Saved composite: {panel_png}")

    print(f"\nAll outputs saved under: {OUT_DIR}")


if __name__ == "__main__":
    main()
