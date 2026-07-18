#!/usr/bin/env python3
"""Pool aligned 0.333 mm binary masks to 2 mm fractional masks."""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "utils"))
from numerical import pool6


def check_fine_grid(image: nib.Nifti1Image, reference: nib.Nifti1Image, atol: float) -> None:
    expected_shape = tuple(size * 6 for size in reference.shape[:3])
    if image.shape[:3] != expected_shape:
        raise ValueError(f"Fine-grid shape {image.shape[:3]} != {expected_shape}")
    expected_affine = reference.affine.copy()
    expected_affine[:3, :3] /= 6
    if not np.allclose(image.affine, expected_affine, atol=atol):
        raise ValueError("Fine-grid affine is not the aligned one-sixth reference affine")


def audit_mass(paths: list[Path], label: str, eps: float) -> None:
    total = sum((nib.load(path).get_fdata(dtype=np.float64) for path in paths))
    maximum = float(np.max(total))
    if maximum > 1 + 5 * eps:
        raise ValueError(f"{label} mask mass exceeds one: {maximum:.8f}")
    print(f"{label} maximum pooled mass: {maximum:.8f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("fine_pattern")
    parser.add_argument("output_pattern")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--eps", type=float, default=1e-6)
    args = parser.parse_args()

    reference = nib.load(args.reference)
    fine_paths = [Path(path) for path in sorted(glob.glob(args.fine_pattern))]
    if len(fine_paths) != 18:
        raise ValueError(f"Expected 18 fine-grid masks, found {len(fine_paths)}")

    first = nib.load(fine_paths[0])
    check_fine_grid(first, reference, args.atol)
    for path in fine_paths[1:]:
        image = nib.load(path)
        if image.shape[:3] != first.shape[:3] or not np.allclose(
            image.affine, first.affine, atol=args.atol
        ):
            raise ValueError(f"Fine-grid mismatch: {path}")

    output_paths = []
    for source in fine_paths:
        fields = source.name.split("_")
        if len(fields) < 4 or fields[0] != "bin" or fields[1] not in {"L", "R"}:
            raise ValueError(f"Unexpected fine-mask filename: {source.name}")
        hemisphere, label = fields[1:3]
        output = Path(args.output_pattern.format(hemi=hemisphere, lab=label))
        output_paths.append(output)

        if output.exists() and not args.force:
            existing = nib.load(output)
            if existing.shape[:3] != reference.shape[:3] or not np.allclose(
                existing.affine, reference.affine, atol=args.atol
            ):
                raise ValueError(f"Existing output grid differs from reference: {output}")
            continue

        binary = nib.load(source).get_fdata(dtype=np.float64) > 0.5
        fractional = pool6(binary.astype(np.float64))
        header = reference.header.copy()
        header.set_data_dtype(np.float32)
        image = nib.Nifti1Image(fractional, reference.affine, header)
        image.set_sform(reference.affine, code=int(reference.header["sform_code"]))
        image.set_qform(reference.affine, code=int(reference.header["qform_code"]))
        nib.save(image, output)

    left = [path for path in output_paths if "_L_" in path.name]
    right = [path for path in output_paths if "_R_" in path.name]
    audit_mass(left, "Left amygdala", args.eps)
    audit_mass(right, "Right amygdala", args.eps)
    audit_mass(output_paths, "Bilateral amygdala", args.eps)


if __name__ == "__main__":
    main()
