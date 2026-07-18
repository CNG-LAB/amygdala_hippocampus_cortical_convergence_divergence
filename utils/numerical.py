"""Small numerical kernels shared by the subject-level pipeline."""

from __future__ import annotations

import numpy as np


def fisher_z_off_diagonal(matrix: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    result = np.zeros_like(matrix, dtype=float)
    off_diagonal = ~np.eye(matrix.shape[0], dtype=bool)
    result[off_diagonal] = np.arctanh(
        np.clip(matrix[off_diagonal], -1 + eps, 1 - eps)
    )
    return result


def pool6(array: np.ndarray) -> np.ndarray:
    """Exact 6 × 6 × 6 block-mean pooling."""
    if array.ndim != 3 or any(size % 6 for size in array.shape):
        raise ValueError(f"Expected a divisible 3D fine grid, got {array.shape}")
    nx, ny, nz = (size // 6 for size in array.shape)
    return array.reshape(nx, 6, ny, 6, nz, 6).mean(axis=(1, 3, 5)).astype(float)


def weighted_timeseries(
    bold_2d: np.ndarray,
    weights: np.ndarray,
    accumulation_dtype: str = "float32",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mask-weighted signals, effective weight sums, and sample sizes."""
    finite_voxels = np.isfinite(bold_2d).all(axis=1)
    bold = np.where(finite_voxels[:, None], bold_2d, 0.0)
    effective_weights = np.where(finite_voxels[:, None], weights, 0.0)
    weight_sum = effective_weights.sum(axis=0, dtype=np.float64)
    if np.any(weight_sum <= 0):
        raise ValueError("At least one mask has no finite BOLD voxels")

    dtype = np.float64 if accumulation_dtype == "float64" else np.float32
    numerator = bold.astype(dtype, copy=False).T @ effective_weights.astype(dtype, copy=False)
    timeseries = numerator / weight_sum[None, :]
    squared_sum = np.square(effective_weights).sum(axis=0, dtype=np.float64)
    effective_n = np.square(weight_sum) / squared_sum
    return timeseries, weight_sum, effective_n


def subdivide_hippocampal_labels(
    labelmap: np.ndarray,
    n_subfields: int = 5,
    n_sections: int = 3,
) -> np.ndarray:
    """Apply the manuscript's subfield-major longitudinal binning rule."""
    labels = np.asarray(labelmap, dtype=np.int32).copy()
    boundaries = np.linspace(0, labels.shape[0], n_sections + 1).astype(int)
    for section, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        slab = labels[start:stop]
        slab[slab > 0] += n_subfields * section

    result = np.zeros_like(labels)
    for section in range(n_sections):
        for subfield in range(n_subfields):
            old = subfield + 1 + n_subfields * section
            new = subfield * n_sections + section + 1
            result[labels == old] = new
    return result
