#!/usr/bin/env python3
"""Build one subject's 406-node Pearson and GLASSO connectivity matrices.

Node order is fixed as left amygdala (9), right amygdala (9), left
hippocampus (15), right hippocampus (15), and retained Glasser parcels (358).
The numerical definitions and output matrix filenames match the manuscript
analysis.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import zscore

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "utils"))
from numerical import fisher_z_off_diagonal

N_AMYG_PER_HEMISPHERE = 9
N_HIPP_PER_HEMISPHERE = 15
GLASSER_DROP_PARCELS = {120, 300}
N_EXPECTED_TIMEPOINTS = 4 * 1200
MAX_HIPP_LABEL_NAN_FRACTION = 0.30


def load_func_gifti(path: Path) -> np.ndarray:
    """Load a functional GIFTI as vertices × time."""
    image = nib.load(path)
    return np.asarray([array.data for array in image.darrays]).T


def load_label_gifti(path: Path) -> np.ndarray:
    return np.asarray(nib.load(path).darrays[0].data, dtype=int)


def load_amygdala(path: Path) -> np.ndarray:
    frame = pd.read_csv(path, sep="\t")
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.shape[1] != N_AMYG_PER_HEMISPHERE:
        raise ValueError(f"Expected 9 numeric amygdala columns in {path}, got {numeric.shape[1]}")
    values = numeric.to_numpy(dtype=float).T
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite amygdala values in {path}")
    return values


def parcel_means(
    data: np.ndarray,
    labels: np.ndarray,
    expected: int,
    max_nan_fraction: float = MAX_HIPP_LABEL_NAN_FRACTION,
) -> np.ndarray:
    """Average positive hippocampal labels in ascending order."""
    data = np.asarray(data, dtype=float)
    labels = np.asarray(labels)
    label_ids = np.sort(np.unique(labels[labels > 0]))
    if label_ids.size != expected:
        raise ValueError(f"Expected {expected} hippocampal labels, got {label_ids.size}")

    means = np.zeros((label_ids.size, data.shape[1]), dtype=float)
    bad_labels = []
    for row, label in enumerate(label_ids):
        values = data[labels == label]
        nan_fraction = np.isnan(values).sum() / values.size
        if nan_fraction > max_nan_fraction:
            bad_labels.append((int(label), float(nan_fraction)))
            continue
        means[row] = np.nanmean(values, axis=0)

    if bad_labels:
        details = ", ".join(
            f"{label} ({fraction:.3f})" for label, fraction in bad_labels
        )
        raise RuntimeError(
            "Hippocampal labels exceed the maximum NaN fraction "
            f"of {max_nan_fraction:.3f}: {details}"
        )
    if not np.isfinite(means).all():
        raise ValueError("Non-finite hippocampal parcel time series")
    return means


def parcellate_cortex(data: np.ndarray, labels: np.ndarray) -> np.ndarray:
    retained = [parcel for parcel in range(1, 361) if parcel not in GLASSER_DROP_PARCELS]
    missing = [parcel for parcel in retained if not np.any(labels == parcel)]
    if missing:
        raise ValueError(f"Glasser labels missing parcels: {missing}")
    return np.asarray([data[labels == parcel].mean(axis=0) for parcel in retained])


def load_node_timeseries(subject: str) -> tuple[np.ndarray, Path]:
    work_root = Path(os.environ["HIPPAMYG_WORK_ROOT"])
    hipp_root = Path(os.environ["HIPPUNFOLD_ROOT"])
    resource_root = Path(os.environ["HIPPAMYG_RESOURCE_ROOT"])

    subject_root = work_root / subject
    func_dir = subject_root / "func_native_2025"
    timeseries_dir = subject_root / "timeseries"
    hipp_dir = hipp_root / subject / "hippunfold" / f"sub-{subject}"

    amyg_left = load_amygdala(timeseries_dir / f"{subject}_amyg_softROIts_mass_L_timeseries.tsv")
    amyg_right = load_amygdala(timeseries_dir / f"{subject}_amyg_softROIts_mass_R_timeseries.tsv")

    hipp_left = load_func_gifti(
        hipp_dir / "func" / f"sub-{subject}_hemi-L_space-T1w_den-2mm_label-hipp_desc-REST_tclean.func.gii"
    )
    hipp_right = load_func_gifti(
        hipp_dir / "func" / f"sub-{subject}_hemi-R_space-T1w_den-2mm_label-hipp_desc-REST_tclean.func.gii"
    )
    labels_left = load_label_gifti(
        hipp_dir / "surf" / f"sub-{subject}_hemi-L_space-unfold_den-2mm_label-hipp_DeKraker15.label.gii"
    )
    labels_right = load_label_gifti(
        hipp_dir / "surf" / f"sub-{subject}_hemi-R_space-unfold_den-2mm_label-hipp_DeKraker15.label.gii"
    )
    if hipp_left.shape[0] != labels_left.size or hipp_right.shape[0] != labels_right.size:
        raise ValueError("Hippocampal surface and label vertex counts differ")

    hipp_left_parcels = parcel_means(hipp_left, labels_left, N_HIPP_PER_HEMISPHERE)
    hipp_right_parcels = parcel_means(hipp_right, labels_right, N_HIPP_PER_HEMISPHERE)

    cortex_left = load_func_gifti(func_dir / "rfMRI_Cortex_Left_tclean.func.gii")
    cortex_right = load_func_gifti(func_dir / "rfMRI_Cortex_Right_tclean.func.gii")
    cortex = np.concatenate([cortex_left, cortex_right], axis=0)
    glasser_labels = np.genfromtxt(resource_root / "glasser.csv", delimiter=",").astype(int).ravel()
    if glasser_labels.size != cortex.shape[0]:
        raise ValueError("Glasser label count does not match cortical vertex count")
    cortex_parcels = parcellate_cortex(cortex, glasser_labels)

    nodes = np.vstack(
        [amyg_left, amyg_right, hipp_left_parcels, hipp_right_parcels, cortex_parcels]
    )
    if nodes.shape != (406, N_EXPECTED_TIMEPOINTS):
        raise ValueError(
            f"Expected 406 nodes × 4800 time points, got {nodes.shape}. "
            "Four complete HCP runs are required."
        )
    return nodes, func_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject")
    args = parser.parse_args()

    nodes, output_dir = load_node_timeseries(args.subject)
    standardized = zscore(nodes, axis=1, ddof=1, nan_policy="raise")
    if not np.isfinite(standardized).all():
        raise ValueError("Non-finite values after node-wise standardization")

    pearson = np.corrcoef(standardized)
    np.fill_diagonal(pearson, 1.0)
    pearson_z = fisher_z_off_diagonal(pearson)
    np.save(output_dir / "connectivity_matrix_REST_406x406_r.npy", pearson)
    np.save(output_dir / "connectivity_matrix_REST_406x406_z.npy", pearson_z)

    toolbox_root = os.environ.get("ACTFLOW_TOOLBOX_ROOT", "")
    if toolbox_root:
        sys.path.insert(0, toolbox_root)
    from ActflowToolbox.connectivity_estimation import graphicalLassoCV

    partial, cv_results = graphicalLassoCV(
        standardized,
        kFolds=4,
        optMethod="loglikelihood",
        saveFiles=0,
        outDir="",
        foldsScheme="blocked",
    )
    np.fill_diagonal(partial, 0.0)
    partial_z = fisher_z_off_diagonal(partial)
    np.save(output_dir / "connectivity_matrix_REST_406x406_glasso_partial.npy", partial)
    np.save(output_dir / "connectivity_matrix_REST_406x406_glasso_z.npy", partial_z)
    (output_dir / "glasso_lambda1_REST.txt").write_text(str(cv_results["bestParam"]))


if __name__ == "__main__":
    main()
