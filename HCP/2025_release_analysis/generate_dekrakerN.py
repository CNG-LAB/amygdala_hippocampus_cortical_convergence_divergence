#!/usr/bin/env python3
"""Divide five HippUnfold subfields into three longitudinal bins."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

N_SUBFIELDS = 5
N_AP_SECTIONS = 3

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "utils"))
from numerical import subdivide_hippocampal_labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject")
    args = parser.parse_args()

    root = os.environ.get("HIPPUNFOLD_ROOT")
    if not root:
        raise RuntimeError("HIPPUNFOLD_ROOT is not set; source config.sh first")
    subject = f"sub-{args.subject}"
    base = Path(root) / args.subject / "hippunfold" / subject / "anat"

    for hemisphere in ("L", "R"):
        source = base / f"{subject}_hemi-{hemisphere}_space-unfold_label-hipp_atlas-multihist7_subfields.nii.gz"
        if not source.exists():
            raise FileNotFoundError(source)
        image = nib.load(source)
        result = subdivide_hippocampal_labels(
            image.get_fdata(),
            n_subfields=N_SUBFIELDS,
            n_sections=N_AP_SECTIONS,
        )
        n_labels = N_SUBFIELDS * N_AP_SECTIONS
        output = base / f"{subject}_hemi-{hemisphere}_space-unfold_label-hipp_atlas-multihist7_subfields-DeKraker{n_labels}.nii.gz"
        header = image.header.copy()
        header.set_data_dtype(np.int32)
        nib.save(nib.Nifti1Image(result, image.affine, header), output)


if __name__ == "__main__":
    main()
