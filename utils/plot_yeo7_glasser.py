#!/usr/bin/env python3
"""Plot the Figure 1 Glasser parcels colored by Yeo-7 network."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from brainspace.datasets import load_conte69
from brainspace.plotting import plot_surf
from brainspace.utils.parcellation import map_to_labels

from plotting_hippamyg import YEO7_COLORS
from project_config import ProjectPaths


NETWORKS = [
    "visual",
    "somatomotor",
    "dorsal attention",
    "ventral attention",
    "limbic",
    "frontoparietal",
    "default mode",
]
ABBREVIATIONS = ["VIS", "SOM", "DAN", "VAN", "LIM", "FPN", "DMN"]


def load_network_indices(path: Path) -> np.ndarray:
    table = pd.read_csv(path, sep="\t").sort_values("index")
    names = table["community_yeo7"].astype(str).str.strip().str.lower()
    mapping = {name: index for index, name in enumerate(NETWORKS, start=1)}
    indices = names.map(mapping)
    if indices.isna().any() or len(indices) != 360:
        invalid = sorted(set(names[indices.isna()]))
        raise ValueError(f"Expected 360 valid Yeo-7 assignments; invalid labels: {invalid}")
    return indices.to_numpy(dtype=int)


def crop_surface_image(image: np.ndarray, padding: int = 10) -> np.ndarray:
    foreground = np.any(image[..., :3] < 0.99, axis=-1)
    midpoint = image.shape[0] // 2
    columns = np.flatnonzero(foreground.any(axis=0))
    top_rows = np.flatnonzero(foreground[:midpoint].any(axis=1))
    bottom_rows = np.flatnonzero(foreground[midpoint:].any(axis=1))
    if not (columns.size and top_rows.size and bottom_rows.size):
        return image

    left = max(0, columns[0] - padding)
    right = min(image.shape[1], columns[-1] + padding + 1)
    top = image[
        max(0, top_rows[0] - padding) : min(midpoint, top_rows[-1] + padding + 1),
        left:right,
    ]
    bottom = image[
        midpoint + max(0, bottom_rows[0] - padding) :
        midpoint + min(image.shape[0] - midpoint, bottom_rows[-1] + padding + 1),
        left:right,
    ]
    return np.vstack((top, bottom))


def render_surface(values: np.ndarray, output_path: Path) -> None:
    surface_left, _ = load_conte69()
    surface_left.append_array(values[: surface_left.n_points], name="yeo7", at="point")
    cmap = ListedColormap(["#e6e6e6", *YEO7_COLORS])
    plot_surf(
        surfs={"lh": surface_left},
        layout=[["lh"], ["lh"]],
        view=[["lateral"], ["medial"]],
        array_name="yeo7",
        cmap=cmap,
        color_range=(0, 7),
        color_bar=False,
        size=(400, 800),
        zoom=1.2,
        screenshot=True,
        filename=str(output_path),
        transparent_bg=False,
    )


def assemble_figure(surface_image: np.ndarray, output_stem: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(3.5, 4), dpi=600)
    grid = fig.add_gridspec(2, 1, height_ratios=[0.15, 1.0], hspace=0.05)
    legend_axis = fig.add_subplot(grid[0])
    legend_axis.axis("off")

    handles = [
        Patch(facecolor=color, edgecolor="none", label=label)
        for color, label in zip(YEO7_COLORS, ABBREVIATIONS)
    ]
    first = legend_axis.legend(
        handles=handles[:4], loc="lower center", bbox_to_anchor=(0.5, 0.5),
        ncol=4, frameon=False, fontsize=10,
    )
    legend_axis.add_artist(first)
    legend_axis.legend(
        handles=handles[4:], loc="upper center", bbox_to_anchor=(0.5, 0.5),
        ncol=3, frameon=False, fontsize=10,
    )

    image_axis = fig.add_subplot(grid[1])
    image_axis.imshow(surface_image)
    image_axis.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    fig.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    paths = ProjectPaths.from_environment()
    network_indices = load_network_indices(paths.resource_root / "atlas-Glasser_dseg.tsv")
    labels = np.ravel(np.loadtxt(paths.resource_root / "glasser.csv", delimiter=",")).astype(int)
    vertex_values = map_to_labels(
        network_indices,
        labels,
        mask=labels != 0,
        fill=0,
    )

    output_dir = paths.output_root / "schematics" / "cortex"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = output_dir / "yeo7_glasser_lh_only_with_legend"
    with TemporaryDirectory() as temporary_dir:
        surface_path = Path(temporary_dir) / "yeo7_surface.png"
        render_surface(vertex_values, surface_path)
        assemble_figure(crop_surface_image(plt.imread(surface_path)), output_stem)


if __name__ == "__main__":
    main()
