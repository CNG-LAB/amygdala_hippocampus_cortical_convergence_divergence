#!/usr/bin/env python3
"""Compute the sample-derived Pearson cortical reference gradients.

This prerequisite loads the configured cohort's Pearson connectivity matrices,
averages them in Fisher-z space, and embeds the retained 358-parcel cortical
block using the settings reported in the manuscript. It must be run before the
group dominance/sharedness and joint hippocampus-amygdala gradient scripts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "utils"))

from helper import kept_sorted_from_removed, run_gradients, zavg_ignore
from project_config import ProjectPaths


N_AMYGDALA_PER_HEMISPHERE = 9
N_HIPPOCAMPUS_PER_HEMISPHERE = 15
REMOVED_GLASSER_PARCELS = [120, 300]
N_COMPONENTS = 10


def main() -> None:
    paths = ProjectPaths.from_environment()
    subjects = [
        line.strip()
        for line in paths.subject_list.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not subjects:
        raise RuntimeError("Subject list is empty")
    if len(set(subjects)) != len(subjects):
        raise RuntimeError("Subject list contains duplicate IDs")

    kept_sorted = kept_sorted_from_removed(REMOVED_GLASSER_PARCELS)
    n_subcortical = 2 * (
        N_AMYGDALA_PER_HEMISPHERE + N_HIPPOCAMPUS_PER_HEMISPHERE
    )
    n_total = n_subcortical + len(kept_sorted)
    filename = f"connectivity_matrix_REST_{n_total}x{n_total}_z.npy"

    matrices = []
    for subject in subjects:
        matrix_path = paths.work_root / subject / "func_native_2025" / filename
        if not matrix_path.exists():
            raise FileNotFoundError(matrix_path)
        matrix = np.load(matrix_path)
        if matrix.shape != (n_total, n_total):
            raise ValueError(
                f"{subject}: got Pearson matrix shape {matrix.shape}, "
                f"expected ({n_total}, {n_total})"
            )
        matrices.append(matrix.astype(float, copy=False))

    group_connectivity = zavg_ignore(
        np.stack(matrices, axis=0),
        drop_zeros=False,
        axis=0,
    )
    cortex_indices = np.arange(n_subcortical, n_total, dtype=int)
    cortex_connectivity = group_connectivity[
        np.ix_(cortex_indices, cortex_indices)
    ]
    gradients, eigenvalues, _ = run_gradients(
        cortex_connectivity,
        n_components=N_COMPONENTS,
        kernel="normalized_angle",
        random_state=0,
    )

    output_dir = paths.output_root / "group_level_gradients" / "top10_union_pearson"
    output_dir.mkdir(parents=True, exist_ok=True)
    gradient_path = output_dir / "cortex_intrinsic_gradients.npy"
    eigenvalue_path = output_dir / "cortex_intrinsic_eigenvalues.npy"
    kept_path = output_dir / "kept_sorted.npy"
    np.save(gradient_path, gradients)
    np.save(eigenvalue_path, eigenvalues)
    np.save(kept_path, kept_sorted)

    print(f"Loaded {len(subjects)} subjects; matrix size {n_total}x{n_total}.")
    print(f"Pearson cortex-cortex base matrix shape: {cortex_connectivity.shape}")
    print(f"Saved cortical gradients: {gradient_path}")
    print(f"Saved cortical eigenvalues: {eigenvalue_path}")
    print(f"Saved retained-cortex indices: {kept_path}")


if __name__ == "__main__":
    main()
