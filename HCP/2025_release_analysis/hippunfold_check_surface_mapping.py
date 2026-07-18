#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run single-subject QC for HippUnfold fMRI volume-to-surface mapping.

What it does
------------
For each hemisphere:
1) loads hippocampal functional GIFTI
2) loads DeKraker label GIFTI
3) loads unfolded hippocampal surface
4) checks vertex-count consistency
5) computes per-vertex QC:
   - temporal mean
   - temporal std
   - finite fraction across time
   - zero-variance mask
6) computes per-label QC:
   - vertex count
   - number of connected components on the mesh
   - NaN fraction
   - mean/std of parcel mean time series
7) saves:
   - per-hemisphere TSV
   - per-hemisphere QC PNG
   - text summary

Usage
-----
python HCP/2025_release_analysis/hippunfold_check_surface_mapping.py 100206
python HCP/2025_release_analysis/hippunfold_check_surface_mapping.py sub-100206

Notes
-----
- This is a fast QC script. It is good for catching broken mapping, NaN-heavy data,
  label fragmentation, and obvious bad hemispheres.
- It does NOT replace anatomical overlay QC on the original T1w / fMRI volume.
"""

from __future__ import annotations

import os
import argparse
from typing import Dict, List, Tuple

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from scipy import sparse
from scipy.sparse.csgraph import connected_components


# ----------------------------
# Pipeline defaults.
# ----------------------------
DEFAULT_HIPPUNFOLD_ROOT = os.environ.get("HIPPUNFOLD_ROOT", "")
DEFAULT_N_LABELS = 15
DEFAULT_MAX_LABEL_NAN_FRAC = 0.30


# ----------------------------
# I/O helpers
# ----------------------------
def require_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")


def load_func_gii(path: str) -> np.ndarray:
    """
    Returns
    -------
    X : ndarray, shape (V, T)
        Vertex x time matrix.
    """
    g = nib.load(path)
    X = np.array([da.data for da in g.darrays], dtype=np.float64)  # (T, V)
    if X.ndim != 2:
        raise RuntimeError(f"Unexpected func.gii array shape at {path}: {X.shape}")
    return X.T  # (V, T)


def load_label_gii(path: str) -> np.ndarray:
    g = nib.load(path)
    if len(g.darrays) != 1:
        raise RuntimeError(f"Expected exactly 1 darray in label.gii: {path}")
    return np.asarray(g.darrays[0].data, dtype=np.int32)


def load_surf_gii(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    coords : (V, 3)
    faces  : (F, 3)
    """
    g = nib.load(path)
    if len(g.darrays) < 2:
        raise RuntimeError(f"Surface file has too few darrays: {path}")
    coords = np.asarray(g.darrays[0].data, dtype=np.float64)
    faces = np.asarray(g.darrays[1].data, dtype=np.int32)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise RuntimeError(f"Unexpected surface coords shape {coords.shape} in {path}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise RuntimeError(f"Unexpected surface faces shape {faces.shape} in {path}")
    return coords, faces


# ----------------------------
# Graph / mesh helpers
# ----------------------------
def build_vertex_adjacency(n_vertices: int, faces: np.ndarray) -> sparse.csr_matrix:
    """
    Undirected adjacency from triangular faces.
    """
    i = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    j = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])

    data = np.ones(i.shape[0], dtype=np.uint8)
    A = sparse.coo_matrix((data, (i, j)), shape=(n_vertices, n_vertices))
    A = A + A.T
    A.data[:] = 1
    A = A.tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def n_connected_components_for_label(
    label_vertices: np.ndarray,
    adjacency: sparse.csr_matrix,
) -> int:
    """
    Count connected components within one label's induced subgraph.
    """
    idx = np.flatnonzero(label_vertices)
    if idx.size == 0:
        return 0
    subA = adjacency[idx][:, idx]
    n_comp, _ = connected_components(subA, directed=False, return_labels=True)
    return int(n_comp)


# ----------------------------
# QC computations
# ----------------------------
def compute_vertex_metrics(X_vt: np.ndarray) -> Dict[str, np.ndarray]:
    """
    X_vt : (V, T)
    """
    finite_mask = np.isfinite(X_vt)
    finite_frac = finite_mask.mean(axis=1)

    X_clean = np.where(finite_mask, X_vt, np.nan)
    mean_v = np.nanmean(X_clean, axis=1)
    std_v = np.nanstd(X_clean, axis=1, ddof=1)

    zero_var = np.isfinite(std_v) & (std_v == 0)
    nan_any = ~np.all(finite_mask, axis=1)

    return {
        "mean": mean_v,
        "std": std_v,
        "finite_frac": finite_frac,
        "zero_var": zero_var.astype(np.int32),
        "nan_any": nan_any.astype(np.int32),
    }


def compute_label_metrics(
    X_vt: np.ndarray,
    labels_v: np.ndarray,
    adjacency: sparse.csr_matrix,
    max_label_nan_frac: float,
) -> List[Dict[str, float]]:
    """
    Per-label QC table.
    """
    rows: List[Dict[str, float]] = []
    label_ids = np.sort(np.unique(labels_v))
    label_ids = label_ids[label_ids > 0]

    for lab in label_ids:
        idx = np.flatnonzero(labels_v == lab)
        Xi = X_vt[idx, :]

        finite_mask = np.isfinite(Xi)
        Xi_clean = np.where(finite_mask, Xi, np.nan)

        nan_frac = float(np.isnan(Xi_clean).sum() / Xi_clean.size) if Xi_clean.size > 0 else np.nan

        # parcel mean TS using nan-robust averaging across vertices
        ts = np.nanmean(Xi_clean, axis=0)
        ts_mean = float(np.nanmean(ts))
        ts_std = float(np.nanstd(ts, ddof=1)) if np.isfinite(ts).sum() > 1 else np.nan

        # zero-var vertices inside parcel
        v_std = np.nanstd(Xi_clean, axis=1, ddof=1)
        zero_var_frac = float(np.mean(np.isfinite(v_std) & (v_std == 0))) if v_std.size > 0 else np.nan

        # connected components on surface
        label_mask = (labels_v == lab)
        n_comp = n_connected_components_for_label(label_mask, adjacency)

        rows.append({
            "label": int(lab),
            "n_vertices": int(idx.size),
            "n_components": int(n_comp),
            "nan_frac": nan_frac,
            "zero_var_vertex_frac": zero_var_frac,
            "parcel_ts_mean": ts_mean,
            "parcel_ts_std": ts_std,
            "flag_nan_frac_gt_thresh": int(np.isfinite(nan_frac) and (nan_frac > max_label_nan_frac)),
            "flag_many_components": int(n_comp > 1),
            "flag_low_parcel_std": int(np.isfinite(ts_std) and (ts_std < 1e-6)),
        })

    return rows


# ----------------------------
# Plotting
# ----------------------------
def choose_2d_coords(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Use the two axes with highest variance.
    Works for unfolded surface; still okay if slightly 3D.
    """
    var = np.var(coords, axis=0)
    order = np.argsort(var)[::-1]
    x = coords[:, order[0]]
    y = coords[:, order[1]]
    return x, y


def scatter_panel(ax, x, y, values, title, cmap="viridis", vmin=None, vmax=None, s=6):
    m = np.isfinite(values)
    if not np.any(m):
        ax.text(0.5, 0.5, "no finite data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    sc = ax.scatter(x[m], y[m], c=values[m], s=s, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)


def save_qc_figure(
    out_png: str,
    coords: np.ndarray,
    labels_v: np.ndarray,
    vertex_metrics: Dict[str, np.ndarray],
    hemi: str,
    subject: str,
) -> None:
    x, y = choose_2d_coords(coords)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))

    scatter_panel(axes[0, 0], x, y, vertex_metrics["mean"], f"{hemi}: temporal mean", cmap="viridis")
    scatter_panel(axes[0, 1], x, y, vertex_metrics["std"], f"{hemi}: temporal std", cmap="magma")
    scatter_panel(axes[0, 2], x, y, vertex_metrics["finite_frac"], f"{hemi}: finite fraction", cmap="cividis", vmin=0, vmax=1)

    scatter_panel(axes[1, 0], x, y, labels_v.astype(float), f"{hemi}: labels", cmap="tab20")
    scatter_panel(axes[1, 1], x, y, vertex_metrics["zero_var"].astype(float), f"{hemi}: zero-var vertices", cmap="Reds", vmin=0, vmax=1)
    scatter_panel(axes[1, 2], x, y, vertex_metrics["nan_any"].astype(float), f"{hemi}: any-NaN vertices", cmap="Reds", vmin=0, vmax=1)

    fig.suptitle(f"HippUnfold mapping QC — {subject} — {hemi}", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("subject", help="Subject ID, with or without 'sub-' prefix.")
    ap.add_argument("--hippunfold-root", default=DEFAULT_HIPPUNFOLD_ROOT)
    ap.add_argument("--n-labels", type=int, default=DEFAULT_N_LABELS)
    ap.add_argument("--max-label-nan-frac", type=float, default=DEFAULT_MAX_LABEL_NAN_FRAC)
    args = ap.parse_args()

    subject = args.subject.strip()
    if subject.startswith("sub-"):
        subject = subject[4:]
    subdir = os.path.join(args.hippunfold_root, subject, "hippunfold", f"sub-{subject}")
    func_dir = os.path.join(subdir, "func")
    surf_dir = os.path.join(subdir, "surf")
    qc_dir = os.path.join(subdir, "qc_mapping")
    os.makedirs(qc_dir, exist_ok=True)

    summary_lines: List[str] = []
    summary_lines.append(f"subject\t{subject}")
    summary_lines.append(f"subdir\t{subdir}")

    for hemi in ("L", "R"):
        func_path = os.path.join(
            func_dir,
            f"sub-{subject}_hemi-{hemi}_space-T1w_den-2mm_label-hipp_desc-REST_tclean.func.gii",
        )
        label_path = os.path.join(
            surf_dir,
            f"sub-{subject}_hemi-{hemi}_space-unfold_den-2mm_label-hipp_DeKraker{args.n_labels}.label.gii",
        )
        surf_path = os.path.join(
            surf_dir,
            f"sub-{subject}_hemi-{hemi}_space-unfold_den-2mm_label-hipp_midthickness.surf.gii",
        )

        require_exists(func_path)
        require_exists(label_path)
        require_exists(surf_path)

        X = load_func_gii(func_path)          # (V, T)
        labels = load_label_gii(label_path)   # (V,)
        coords, faces = load_surf_gii(surf_path)

        if X.shape[0] != labels.shape[0]:
            raise RuntimeError(
                f"{hemi}: func/label vertex mismatch: func={X.shape[0]} label={labels.shape[0]}"
            )
        if X.shape[0] != coords.shape[0]:
            raise RuntimeError(
                f"{hemi}: func/surface vertex mismatch: func={X.shape[0]} surf={coords.shape[0]}"
            )

        adjacency = build_vertex_adjacency(X.shape[0], faces)
        vertex_metrics = compute_vertex_metrics(X)
        label_rows = compute_label_metrics(
            X_vt=X,
            labels_v=labels,
            adjacency=adjacency,
            max_label_nan_frac=args.max_label_nan_frac,
        )

        # Save per-label TSV
        out_tsv = os.path.join(qc_dir, f"sub-{subject}_hemi-{hemi}_hipp_mapping_qc.tsv")
        header = [
            "label",
            "n_vertices",
            "n_components",
            "nan_frac",
            "zero_var_vertex_frac",
            "parcel_ts_mean",
            "parcel_ts_std",
            "flag_nan_frac_gt_thresh",
            "flag_many_components",
            "flag_low_parcel_std",
        ]
        with open(out_tsv, "w", encoding="utf-8") as f:
            f.write("\t".join(header) + "\n")
            for row in label_rows:
                f.write("\t".join(str(row[h]) for h in header) + "\n")

        # Save figure
        out_png = os.path.join(qc_dir, f"sub-{subject}_hemi-{hemi}_hipp_mapping_qc.png")
        save_qc_figure(
            out_png=out_png,
            coords=coords,
            labels_v=labels,
            vertex_metrics=vertex_metrics,
            hemi=hemi,
            subject=f"sub-{subject}",
        )

        # Summary
        finite_frac_global = float(np.mean(vertex_metrics["finite_frac"]))
        zero_var_frac_global = float(np.mean(vertex_metrics["zero_var"]))
        nan_any_frac_global = float(np.mean(vertex_metrics["nan_any"]))

        n_labels_found = int(np.sum(np.unique(labels) > 0))
        bad_nan_labels = int(sum(r["flag_nan_frac_gt_thresh"] for r in label_rows))
        fragmented_labels = int(sum(r["flag_many_components"] for r in label_rows))
        low_std_labels = int(sum(r["flag_low_parcel_std"] for r in label_rows))

        summary_lines.extend([
            f"{hemi}_func_path\t{func_path}",
            f"{hemi}_label_path\t{label_path}",
            f"{hemi}_surf_path\t{surf_path}",
            f"{hemi}_shape_VxT\t{X.shape[0]}x{X.shape[1]}",
            f"{hemi}_n_labels_found\t{n_labels_found}",
            f"{hemi}_mean_finite_fraction\t{finite_frac_global:.6f}",
            f"{hemi}_zero_var_vertex_fraction\t{zero_var_frac_global:.6f}",
            f"{hemi}_any_nan_vertex_fraction\t{nan_any_frac_global:.6f}",
            f"{hemi}_n_labels_nanflag\t{bad_nan_labels}",
            f"{hemi}_n_labels_fragmented\t{fragmented_labels}",
            f"{hemi}_n_labels_lowstd\t{low_std_labels}",
            f"{hemi}_tsv\t{out_tsv}",
            f"{hemi}_png\t{out_png}",
        ])

        print(f"\n[{hemi}]")
        print(f"  shape (V,T): {X.shape}")
        print(f"  labels found: {n_labels_found}")
        print(f"  mean finite fraction: {finite_frac_global:.6f}")
        print(f"  zero-var vertex fraction: {zero_var_frac_global:.6f}")
        print(f"  any-NaN vertex fraction: {nan_any_frac_global:.6f}")
        print(f"  labels with nan_frac > {args.max_label_nan_frac:.2f}: {bad_nan_labels}")
        print(f"  fragmented labels (components > 1): {fragmented_labels}")
        print(f"  low-std labels: {low_std_labels}")
        print(f"  wrote: {out_tsv}")
        print(f"  wrote: {out_png}")

    summary_path = os.path.join(qc_dir, f"sub-{subject}_hipp_mapping_qc_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    print(f"\nSummary written to: {summary_path}")


if __name__ == "__main__":
    main()
