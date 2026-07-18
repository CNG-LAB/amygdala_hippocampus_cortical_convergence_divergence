#!/usr/bin/env python3
"""Create group-majority DeKraker15 volume and surface label maps.

For each hemisphere, this script majority-votes the configured cohort's
unfolded NIfTI label maps and canonical 2-mm surface GIFTI labels. Ties are
resolved toward the smallest label, matching NumPy's ``argmax`` rule.
"""

from __future__ import annotations

import os
from pathlib import Path

import nibabel as nib
import numpy as np

N_LABELS = 15


def read_subjects(path: Path) -> list[str]:
    subjects = [
        line.strip().removeprefix("sub-")
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not subjects:
        raise ValueError("Subject list is empty")
    if len(set(subjects)) != len(subjects):
        raise ValueError("Subject list contains duplicate IDs")
    return subjects


def subject_root(root: Path, subject: str) -> Path:
    return root / subject / "hippunfold" / f"sub-{subject}"


def volume_label_path(root: Path, subject: str, hemisphere: str) -> Path:
    sub = f"sub-{subject}"
    return (
        subject_root(root, subject)
        / "anat"
        / f"{sub}_hemi-{hemisphere}_space-unfold_label-hipp_atlas-multihist7_subfields-DeKraker15.nii.gz"
    )


def surface_label_path(root: Path, subject: str, hemisphere: str) -> Path:
    sub = f"sub-{subject}"
    return (
        subject_root(root, subject)
        / "surf"
        / f"{sub}_hemi-{hemisphere}_space-unfold_den-2mm_label-hipp_DeKraker15.label.gii"
    )


def validate_labels(labels: np.ndarray, path: Path) -> np.ndarray:
    values = np.asarray(labels)
    if not np.all(np.isfinite(values)) or not np.all(values == np.rint(values)):
        raise ValueError(f"Non-integer or non-finite labels in {path}")
    values = values.astype(np.int16)
    unexpected = set(np.unique(values)) - set(range(N_LABELS + 1))
    if unexpected:
        raise ValueError(f"Unexpected labels {sorted(unexpected)} in {path}")
    return values


def vote_volume(paths: list[Path]) -> tuple[nib.Nifti1Image, np.ndarray]:
    reference = nib.load(paths[0])
    counts = np.zeros((*reference.shape, N_LABELS + 1), dtype=np.uint16)
    for index, path in enumerate(paths):
        image = reference if index == 0 else nib.load(path)
        if image.shape != reference.shape or not np.allclose(
            image.affine, reference.affine, atol=1e-6
        ):
            raise ValueError(f"Volume grid mismatch: {path}")
        labels = validate_labels(image.get_fdata(), path)
        for label in range(N_LABELS + 1):
            counts[..., label] += labels == label
    return reference, np.argmax(counts, axis=-1).astype(np.int16)


def vote_surface(paths: list[Path]) -> np.ndarray:
    counts = None
    shape = None
    for path in paths:
        image = nib.load(path)
        if len(image.darrays) != 1:
            raise ValueError(f"Expected one data array in {path}, got {len(image.darrays)}")
        labels = validate_labels(image.darrays[0].data, path)
        if labels.ndim != 1:
            raise ValueError(f"Expected one-dimensional surface labels in {path}")
        if counts is None:
            shape = labels.shape
            counts = np.zeros((*shape, N_LABELS + 1), dtype=np.uint16)
        elif labels.shape != shape:
            raise ValueError(f"Surface vertex-count mismatch: {path}")
        for label in range(N_LABELS + 1):
            counts[..., label] += labels == label
    if counts is None:
        raise ValueError("No surface label arrays supplied")
    # np.argmax returns the first maximum, preserving smallest-label tie breaks.
    return np.argmax(counts, axis=-1).astype(np.int16)


def require_paths(paths: list[Path], description: str) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} {description}; first missing file: {missing[0]}"
        )


def main() -> None:
    subject_list = Path(os.environ["HIPPAMYG_SUBJECT_LIST"])
    hippunfold_root = Path(os.environ["HIPPUNFOLD_ROOT"])
    output_root = Path(os.environ["HIPPAMYG_OUTPUT_ROOT"])
    output_root.mkdir(parents=True, exist_ok=True)
    subjects = read_subjects(subject_list)

    for hemisphere in ("L", "R"):
        volume_paths = [
            volume_label_path(hippunfold_root, subject, hemisphere)
            for subject in subjects
        ]
        surface_paths = [
            surface_label_path(hippunfold_root, subject, hemisphere)
            for subject in subjects
        ]
        require_paths(volume_paths, f"{hemisphere}-hemisphere volume label maps")
        require_paths(surface_paths, f"{hemisphere}-hemisphere surface label maps")

        reference, volume_majority = vote_volume(volume_paths)
        header = reference.header.copy()
        header.set_data_dtype(np.int16)
        volume_output = (
            output_root / f"{hemisphere}_hipp_majority_unfold_DeKraker15.nii.gz"
        )
        nib.save(
            nib.Nifti1Image(volume_majority, reference.affine, header),
            volume_output,
        )

        surface_output = output_root / f"{hemisphere}_hipp_majority_labels.npy"
        np.save(surface_output, vote_surface(surface_paths), allow_pickle=False)

        print(f"[{hemisphere}] Saved volume labels:  {volume_output}")
        print(f"[{hemisphere}] Saved surface labels: {surface_output}")


if __name__ == "__main__":
    main()
