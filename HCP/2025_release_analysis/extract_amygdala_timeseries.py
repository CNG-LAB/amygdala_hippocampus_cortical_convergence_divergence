#!/usr/bin/env python3
"""Extract fractional-mask-weighted amygdala time series and basic QC."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "utils"))
from numerical import weighted_timeseries

LABELS = (7001, 7003, 7005, 7006, 7007, 7008, 7009, 7010, 7015)
NAMES = (
    "Lateral",
    "Basal",
    "Central",
    "Medial",
    "Cortical",
    "Accessory-Basal",
    "Cortico-Amygdaloid-Transition",
    "Anterior-Amygdaloid-Area",
    "Paralaminar",
)


def load_masks(mask_dir: Path, reference: nib.spatialimages.SpatialImage) -> tuple[np.ndarray, np.ndarray]:
    columns = []
    voxel_counts = []
    for hemisphere in ("L", "R"):
        for label in LABELS:
            path = mask_dir / f"soft_amyg_{hemisphere}_{label}_2mm_pool.nii.gz"
            if not path.exists():
                raise FileNotFoundError(path)
            image = nib.load(path)
            if reference.shape[:3] != image.shape[:3] or not np.allclose(
                reference.affine, image.affine, atol=1e-6
            ):
                raise ValueError(f"Mask grid differs from BOLD grid: {path}")
            weights = image.get_fdata(dtype=np.float32).reshape(-1)
            if not np.isfinite(weights).all() or np.any(weights < 0) or np.any(weights > 1):
                raise ValueError(f"Invalid fractional weights: {path}")
            if weights.sum(dtype=np.float64) <= 0:
                raise ValueError(f"Empty mask: {path}")
            columns.append(weights)
            voxel_counts.append(np.count_nonzero(weights))
    return np.column_stack(columns).astype(np.float32), np.asarray(voxel_counts)


def write_tsv(path: Path, header: list[str], rows: np.ndarray) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bold", type=Path)
    parser.add_argument("mask_dir", type=Path)
    parser.add_argument("outstem", type=Path)
    args = parser.parse_args()

    bold_image = nib.load(args.bold)
    if len(bold_image.shape) != 4:
        raise ValueError(f"Expected 4D BOLD, got {bold_image.shape}")
    bold = bold_image.get_fdata(dtype=np.float32).reshape(-1, bold_image.shape[3])
    weights, voxel_counts = load_masks(args.mask_dir, bold_image)
    timeseries, weight_sum, effective_n = weighted_timeseries(
        bold, weights, os.environ.get("AMYG_ACC_DTYPE", "float32")
    )

    args.outstem.parent.mkdir(parents=True, exist_ok=True)
    header = [str(label) for label in LABELS]
    write_tsv(Path(f"{args.outstem}_L_timeseries.tsv"), header, timeseries[:, :9])
    write_tsv(Path(f"{args.outstem}_R_timeseries.tsv"), header, timeseries[:, 9:])

    qc_header = ["id", "name", "voxels", "weight_sum", "signal_std", "effective_n"]
    for hemisphere, offset in (("L", 0), ("R", 9)):
        rows = []
        for index, (label, name) in enumerate(zip(LABELS, NAMES)):
            column = offset + index
            rows.append(
                [label, name, voxel_counts[column], weight_sum[column], np.std(timeseries[:, column], ddof=1), effective_n[column]]
            )
        write_tsv(Path(f"{args.outstem}_{hemisphere}_QC.tsv"), qc_header, np.asarray(rows, dtype=object))


if __name__ == "__main__":
    main()
