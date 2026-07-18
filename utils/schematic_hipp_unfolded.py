#!/usr/bin/env python3
"""Plot the Figure 1 unfolded DeKraker15 hippocampal schematic."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Patch
import nibabel as nib
import numpy as np


N_SUBFIELDS = 5
N_AP_BINS = 3
N_LABELS = N_SUBFIELDS * N_AP_BINS
SUBFIELD_NAMES = ["Subiculum", "CA1", "CA2", "CA3", "CA4"]

HEMISPHERE = os.environ.get("HIPPAMYG_HEMISPHERE", "L").upper()
COLLAPSE_AXIS = 2
FLIP_AP_AXIS = True
FLIP_PD_AXIS = True
AP_TINT_MAX = 0.4


def load_labelmap(path: Path) -> np.ndarray:
    labels = np.rint(nib.load(path).get_fdata()).astype(np.int16)
    observed = set(np.unique(labels)) - {0}
    expected = set(range(1, N_LABELS + 1))
    if observed != expected:
        raise ValueError(f"Expected labels 1-{N_LABELS} in {path}; found {sorted(observed)}")
    return labels


def collapse_to_2d(labels: np.ndarray, axis: int) -> np.ndarray:
    if labels.ndim == 2:
        return labels
    if labels.ndim != 3:
        raise ValueError(f"Expected a 2D or 3D label map, got {labels.shape}")
    moved = np.moveaxis(labels, axis, -1)
    counts = np.stack(
        [(moved == label).sum(axis=-1) for label in range(1, N_LABELS + 1)],
        axis=-1,
    )
    result = np.argmax(counts, axis=-1).astype(np.int16) + 1
    result[counts.sum(axis=-1) == 0] = 0
    return result


def display_orientation(labels: np.ndarray) -> np.ndarray:
    if FLIP_AP_AXIS:
        labels = labels[::-1, :]
    if FLIP_PD_AXIS:
        labels = labels[:, ::-1]
    return labels


def parcel_colors() -> tuple[mcolors.ListedColormap, np.ndarray]:
    base = np.asarray(plt.get_cmap("Set1").colors[:N_SUBFIELDS])
    tints = np.linspace(0.0, AP_TINT_MAX, N_AP_BINS)
    colors = []
    for label in range(N_LABELS):
        rgb = base[label // N_AP_BINS]
        tint = tints[label % N_AP_BINS]
        colors.append((*tuple(rgb + (1.0 - rgb) * tint), 1.0))
    return mcolors.ListedColormap(colors), base


def subfield_boundaries(labels: np.ndarray) -> np.ndarray:
    groups = np.zeros_like(labels)
    mask = labels > 0
    groups[mask] = ((labels[mask] - 1) // N_AP_BINS) + 1
    boundary = np.zeros_like(mask)
    vertical = (groups[1:] != groups[:-1]) & mask[1:] & mask[:-1]
    horizontal = (groups[:, 1:] != groups[:, :-1]) & mask[:, 1:] & mask[:, :-1]
    boundary[1:] |= vertical
    boundary[:, 1:] |= horizontal
    return boundary


def plot_schematic(labels: np.ndarray, output_stem: Path) -> None:
    labels = display_orientation(collapse_to_2d(labels, COLLAPSE_AXIS))
    cmap, base_colors = parcel_colors()
    cmap = copy.copy(cmap)
    cmap.set_bad((0.98, 0.98, 0.98, 1.0))

    display = labels.astype(float)
    display[display == 0] = np.nan
    display -= 1

    fig, ax = plt.subplots(figsize=(3.8, 3.0), constrained_layout=True)
    ax.imshow(
        np.ma.masked_invalid(display),
        origin="lower",
        interpolation="nearest",
        cmap=cmap,
        norm=BoundaryNorm(np.arange(-0.5, N_LABELS, 1.0), cmap.N),
        aspect="auto",
    )

    boundary = subfield_boundaries(labels)
    overlay = np.zeros((*boundary.shape, 4), dtype=float)
    overlay[boundary, 3] = 1.0
    ax.imshow(overlay, origin="lower", interpolation="nearest", aspect="auto")

    ax.set_xlabel("Proximal-distal", fontsize=8)
    ax.set_ylabel("Anterior-posterior", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    handles = [
        Patch(facecolor=base_colors[i], edgecolor="none", label=name)
        for i, name in enumerate(SUBFIELD_NAMES)
    ]
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=8,
        title="Subfield hue",
        title_fontsize=9,
    )

    fig.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if HEMISPHERE not in {"L", "R"}:
        raise ValueError("HIPPAMYG_HEMISPHERE must be L or R")
    output_root = Path(os.environ["HIPPAMYG_OUTPUT_ROOT"])
    input_path = output_root / f"{HEMISPHERE}_hipp_majority_unfold_DeKraker15.nii.gz"
    output_dir = output_root / "schematics" / "hippocampus"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"{HEMISPHERE}_hipp_majority_unfold_DeKraker15_schematic"
    plot_schematic(load_labelmap(input_path), stem)


if __name__ == "__main__":
    main()
