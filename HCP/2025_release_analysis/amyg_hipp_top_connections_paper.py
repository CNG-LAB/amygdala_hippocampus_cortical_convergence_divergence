#!/usr/bin/env python3
"""Run the manuscript's group-level dominance and sharedness analyses.

The script averages the configured cohort's Pearson and GLASSO matrices, builds
the method-specific positive top-k masks, computes cortical dominance and
sharedness, and produces the Figure 1-3 analyses, tables, and reliability
summaries. Set paths through ``config.sh`` and select 5%, 10%, or 15% with
``HIPPAMYG_TOP_PERCENT``. Only threshold maps are run for non-10% values.
"""

from __future__ import annotations

import os
import shutil
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from scipy.stats import kendalltau, pearsonr, rankdata, spearmanr
from statsmodels.stats.multitest import fdrcorrection

from brainspace.datasets import load_conte69
from brainspace.utils.parcellation import map_to_labels
from brainspace.plotting import plot_hemispheres
from surfplot import Plot  # used for LH-only composite seed maps

# ---------------------------------------------------------------------
# Local project imports
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "utils"))

from project_config import ProjectPaths

PATHS = ProjectPaths.from_environment()

from helper import (
    _p_to_stars,
    calc_dominance_normalized,
    calc_sharedness_normalized,
    compute_msr_statistics,
    compute_seedwise_topk_mask,
    load_glasso_support_prevalence,
    map_top_to_full360,
    seedwise_topk_masks_from_prevalence,
    zavg_ignore,
)

from hippamyg_labels import (
    AMYGDALA_SHORT,
    _AMYGDALA_LUT_RGBA,
    region_indices,
    build_hippocampus_labels,
    get_amygdala_labels,
    amygdala_broad_group_indices,
)

from figure3_heatmaps import make_row_c_heatmaps
from group_analysis import (
    compute_seed_scores_loso,
    corr_r_n,
    overlap_only_scatter_arrays,
    robust_range_pos,
    robust_range_sym,
    strength_stats_from_mask,
)
from split_half_reliability import run_split_half_reliability

from plotting_hippamyg import (
    YEO7_COLORS,
    create_matplotlib_grid,
    get_hipp_label_ids,
    gray_rgba,
    load_hipp_surfaces_and_labels,
    map_gradients_to_hipp_vertices,
    plot_amygdala_volumes_sharedscale,
    plot_and_save_hipp_gradients,
    plot_and_save_hipp_unfolded_gradients,
    plot_community_bars_with_strips,
    plot_metric_scatter_by_community,
    plot_seed_maps,
)


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
N_HIPP_LABELS = 15                          # hippocampal labels per hemisphere
GLASSER_DROP_PARCELS = [120, 300]           # dropped Glasser 1-based IDs
TOP_PERCENT = int(os.environ.get("HIPPAMYG_TOP_PERCENT", "10"))

SUBJECT_LIST_FILE = str(PATHS.subject_list)
BASE_DIR = str(PATHS.work_root)

# Per-vertex Glasser parcellation labels on fsLR/Conte69
GLASSER_LABELING_PATH = str(PATHS.resource_root / "glasser.csv")

# Glasser communities TSV (index + Yeo + Mesulam columns)
GLASSER_TSV_PATH = str(PATHS.resource_root / "atlas-Glasser_dseg.tsv")

OUTPUT_ROOT = str(PATHS.output_root / "group_level" / f"top{TOP_PERCENT}_hippamyg")
os.makedirs(OUTPUT_ROOT, exist_ok=True)

COMMUNITY_SPECS = [
    dict(
        name="Yeo7",
        tsv_path=GLASSER_TSV_PATH,
        sep="\t",
        index_col="index",
        scheme_col="community_yeo7",
    ),
]

yeo7_dict = {
    "visual": YEO7_COLORS[0],
    "somatomotor": YEO7_COLORS[1],
    "dorsal attention": YEO7_COLORS[2],
    "ventral attention": YEO7_COLORS[3],
    "limbic": YEO7_COLORS[4],
    "frontoparietal": YEO7_COLORS[5],
    "default mode": YEO7_COLORS[6]
}

# Figure toggles
PLOT_SEED_MAPS = False          # if False: skip individual seed maps
MAKE_COMPOSITE_GRIDS = False     # if False: skip LH-only composite grids

# Which community-level plots to generate
COMMUNITY_PLOT_TYPES = {"bar_strip", "glasso_vs_pearson_scatter"}

DO_MSR = os.environ.get("HIPPAMYG_RUN_MSR", "1") == "1"

MSR_PARAMS = dict(
    dist_l_path=str(PATHS.resource_root / "Glasser32k_dist_L.npy"),
    dist_r_path=str(PATHS.resource_root / "Glasser32k_dist_R.npy"),
    parc_l_path=str(PATHS.resource_root / "glasser-360_conte69_lh.label.gii"),
    parc_r_path=str(PATHS.resource_root / "glasser-360_conte69_rh.label.gii"),
    n_perm=1000,
    n_proc=8,
    seed=42,
)
PREFERRED_FONT = "Lato"

# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------
def require_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")


def load_subject_list(path: str) -> List[str]:
    require_exists(path)
    with open(path, "r") as f:
        subs = [s.strip() for s in f if s.strip() and not s.strip().startswith("#")]
    if not subs:
        raise RuntimeError(f"No subjects in {path}")
    if len(set(subs)) != len(subs):
        raise RuntimeError(f"Duplicate subject IDs in {path}")
    return subs


def load_stack(subjects: List[str], fname_pattern: str, n_total: int) -> np.ndarray:
    """Load one matrix for every listed subject."""
    mats = []
    expected_name = fname_pattern.format(N=n_total)

    for sid in subjects:
        fpath = os.path.join(BASE_DIR, sid, "func_native_2025", expected_name)
        if not os.path.exists(fpath):
            raise FileNotFoundError(fpath)
        arr = np.load(fpath)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] != n_total:
            raise ValueError(f"{sid}: got {arr.shape}, expected ({n_total}, {n_total})")
        mats.append(arr)

    mats = np.asarray(mats, dtype=float)
    print(f"[INFO] Loaded {len(subjects)} subjects for {expected_name}")
    return mats


def map_seedblock_to_vertices(
    seed_block: np.ndarray,
    kept_glp_1based: np.ndarray,
    labeling: np.ndarray,
    mask_labels: np.ndarray,
) -> np.ndarray:
    """Map (n_seeds_block × n_ctx) matrix to (n_seeds_block × n_vertices)."""
    # Convert Glasser labels to zero-based indices.
    kept_0based = kept_glp_1based.astype(int) - 1

    full360_block = map_top_to_full360(seed_block, kept_0based, full_len=360, fill=np.nan)

    # Map parcel values to surface vertices.
    mapped = []
    for row in full360_block:
        v = map_to_labels(row, labeling, mask=mask_labels, fill=np.nan)
        mapped.append(v)
    return np.asarray(mapped)


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
# -------------------------
# 1) Subjects and indices
# -------------------------
subjects = load_subject_list(SUBJECT_LIST_FILE)
print(f"[INFO] Using {len(subjects)} subjects from {SUBJECT_LIST_FILE}")

# Positive GLASSO support prevalence.
prevalence, kept_glp = load_glasso_support_prevalence(
    subject_list_file=SUBJECT_LIST_FILE,
    base_dir=BASE_DIR,
    n_amyg_per_hemi=9,
    n_hipp_per_hemi=N_HIPP_LABELS,
    removed_glasser=GLASSER_DROP_PARCELS,
    positive_only=True,
    verbose=False,
)
prevalence = np.asarray(prevalence, dtype=float)
kept_glp = np.asarray(kept_glp, dtype=int)

n_seeds, n_ctx = prevalence.shape
expected_seeds = 2 * 9 + 2 * N_HIPP_LABELS
if n_seeds != expected_seeds:
    raise RuntimeError(
        f"Prevalence n_seeds={n_seeds}, expected {expected_seeds} "
        f"(2*9 amyg + 2*{N_HIPP_LABELS} hipp)"
    )
N_TOTAL = n_seeds + n_ctx
print(f"[INFO] Prevalence shape: seeds={n_seeds}, cortex={n_ctx}, N_TOTAL={N_TOTAL}")

# Per-seed Top-10% GLASSO prevalence mask
mask_glasso_seedwise = seedwise_topk_masks_from_prevalence(
    prevalence, top_percent=TOP_PERCENT
)  # shape: (n_seeds, n_ctx)

# Seed index layout
idx_dict = region_indices(N_HIPP_LABELS)
amyg_L_idx = idx_dict["Left"]["amyg_indices"]
amyg_R_idx = idx_dict["Right"]["amyg_indices"]
hipp_L_idx = idx_dict["Left"]["hipp_indices"]
hipp_R_idx = idx_dict["Right"]["hipp_indices"]

amyg_rows_all = np.r_[amyg_L_idx, amyg_R_idx]
hipp_rows_all = np.r_[hipp_L_idx, hipp_R_idx]

# Structure totals used for normalized counts.
n_total_amyg = len(amyg_rows_all)  # 18
n_total_hipp = len(hipp_rows_all)  # 30
print(f"[INFO] Normalization Totals: Amyg={n_total_amyg}, Hipp={n_total_hipp}")

# -------------------------
# 2) Group-level GLASSO and Pearson
# -------------------------
# GLASSO z
glasso_z_stack = load_stack(
    subjects, fname_pattern="connectivity_matrix_REST_{N}x{N}_glasso_z.npy", n_total=N_TOTAL
)
group_glasso_r = zavg_ignore(glasso_z_stack, drop_zeros=True, axis=0)

# Pearson z
pearson_z_stack = load_stack(
    subjects, fname_pattern="connectivity_matrix_REST_{N}x{N}_z.npy", n_total=N_TOTAL
)
group_pearson_r = zavg_ignore(pearson_z_stack, drop_zeros=False, axis=0)

ctx_start = N_TOTAL - n_ctx
ctx_indices = np.arange(ctx_start, N_TOTAL, dtype=int)
seed_rows = np.arange(0, n_seeds, dtype=int)

glasso_seed2ctx = group_glasso_r[seed_rows][:, ctx_indices]   # (n_seeds, n_ctx)
pearson_seed2ctx = group_pearson_r[seed_rows][:, ctx_indices]

# Top-10% Pearson mask (by raw r, positive-only)
mask_pearson_seedwise = compute_seedwise_topk_mask(
    pearson_seed2ctx,
    top_percent=TOP_PERCENT,
    positive_only=True,
)

# Apply per-method masks (for per-seed maps)
# GLASSO: Top-10% prevalence
glasso_seed2ctx_masked = np.where(mask_glasso_seedwise, glasso_seed2ctx, np.nan)

# Pearson: Top-10% positive Pearson r (no GLASSO gating)
pearson_seed2ctx_masked = np.where(mask_pearson_seedwise, pearson_seed2ctx, np.nan)

# Categorical overlap codes for GLASSO vs Pearson:
# 1 = GLASSO-only, 2 = Pearson-only, 3 = Both
conj_code = np.full(glasso_seed2ctx.shape, np.nan, dtype=float)

g = mask_glasso_seedwise
p = mask_pearson_seedwise

conj_code[g & ~p] = 1.0   # GLASSO-only
conj_code[~g & p] = 2.0   # Pearson-only
conj_code[g & p] = 3.0    # Both

# -------------------------
# 3) Surfaces and labels
# -------------------------
labeling = np.genfromtxt(GLASSER_LABELING_PATH, delimiter=",")
labeling = np.asarray(labeling, dtype=float).ravel()
mask_labels = labeling != 0

surf_lh, surf_rh = load_conte69()

# Amygdala labels
amyg_labels_L = get_amygdala_labels("Left", style="long")
amyg_labels_R = get_amygdala_labels("Right", style="long")

# Hippocampus labels + AP bin indices (subfield-major ordering)
hipp_labels_L, hipp_ap_bins_L = build_hippocampus_labels(
    "Left", n_labels=N_HIPP_LABELS, order="subfield-major"
)
hipp_labels_R, hipp_ap_bins_R = build_hippocampus_labels(
    "Right", n_labels=N_HIPP_LABELS, order="subfield-major"
)

hipp_ap_bins_L = np.asarray(hipp_ap_bins_L, dtype=int)  # values in {1..K}, here K=3
hipp_ap_bins_R = np.asarray(hipp_ap_bins_R, dtype=int)

out_glasso_amyg = os.path.join(OUTPUT_ROOT, "glasso_amygdala_connectivity")
out_glasso_hipp = os.path.join(OUTPUT_ROOT, "glasso_hippocampus_connectivity")
out_pearson_amyg = os.path.join(OUTPUT_ROOT, "pearson_amygdala_connectivity")
out_pearson_hipp = os.path.join(OUTPUT_ROOT, "pearson_hippocampus_connectivity")
for d in [out_glasso_amyg, out_glasso_hipp, out_pearson_amyg, out_pearson_hipp]:
    os.makedirs(d, exist_ok=True)

# -------------------------
# 4) Map per-seed GLASSO & Pearson to vertices
# -------------------------
print("[INFO] Mapping seed→cortex maps to vertices...")

mapped_glasso: Dict[str, Dict[str, np.ndarray]] = {"Amygdala": {}, "Hippocampus": {}}
mapped_pearson: Dict[str, Dict[str, np.ndarray]] = {"Amygdala": {}, "Hippocampus": {}}
mapped_conj: Dict[str, Dict[str, np.ndarray]] = {"Amygdala": {}, "Hippocampus": {}}

mapped_sources = (
    (mapped_glasso, glasso_seed2ctx_masked),
    (mapped_pearson, pearson_seed2ctx_masked),
    (mapped_conj, conj_code),
)
for hemi, amyg_indices, hipp_indices in (
    ("Left", amyg_L_idx, hipp_L_idx),
    ("Right", amyg_R_idx, hipp_R_idx),
):
    for mapped, seed_matrix in mapped_sources:
        mapped["Amygdala"][hemi] = map_seedblock_to_vertices(
            seed_matrix[amyg_indices, :],
            kept_glp,
            labeling,
            mask_labels,
        )
        mapped["Hippocampus"][hemi] = map_seedblock_to_vertices(
            seed_matrix[hipp_indices, :],
            kept_glp,
            labeling,
            mask_labels,
        )

# -------------------------
# 5) Per-seed maps (full brain) + LH-only composite grids
# -------------------------
print("[PLOT] Per-seed maps and LH-only composite grids...")

# --- 5A. Full per-seed maps (LH+RH) ---
if PLOT_SEED_MAPS:
    print("[PLOT] Per-seed GLASSO & Pearson maps (Top-10% GLASSO prevalence)...")

    seed_plot_specs = (
        (
            "Amygdala",
            (amyg_labels_L, amyg_labels_R),
            (
                (mapped_glasso, out_glasso_amyg, "glasso"),
                (mapped_pearson, out_pearson_amyg, "pearson"),
            ),
        ),
        (
            "Hippocampus",
            (hipp_labels_L, hipp_labels_R),
            (
                (mapped_glasso, out_glasso_hipp, "glasso"),
                (mapped_pearson, out_pearson_hipp, "pearson"),
            ),
        ),
    )
    for structure, labels_by_hemi, method_specs in seed_plot_specs:
        for mapped, out_dir, tag in method_specs:
            for hemi, labels in zip(("Left", "Right"), labels_by_hemi):
                plot_seed_maps(
                    surf_lh,
                    surf_rh,
                    mapped=mapped,
                    structure=structure,
                    hemi=hemi,
                    labels=labels,
                    out_dir=out_dir,
                    tag=tag,
                    top_percent=TOP_PERCENT,
                )
else:
    print("[PLOT] Skipping per-seed maps (PLOT_SEED_MAPS=False).")

# 5B. LH-only composite grids for Figures 1 and 2
if MAKE_COMPOSITE_GRIDS:
    print("[GRID] Building left-hemisphere-only composite grids using surfplot...")

    # Left-hemisphere seeds used in the composite panels
    # struct, side, labels, rows, cols, fheight, fwidth
    structures_for_grid = [
        ("Amygdala", "Left", amyg_labels_L, 3, 3, 1.5, 2.5),
        ("Hippocampus", "Left", hipp_labels_L, 5, 3, 1.5, 2.5),
    ]

    methods = ["conjunction"] #["glasso", "pearson", "conjunction"]

    # Temporary dir for per-seed LH-only PNGs that feed the grids
    grid_tmp_dir = os.path.join(OUTPUT_ROOT, "tmp_grid_lh_only")

    # Clear temporary panels before rebuilding the grids.
    if os.path.isdir(grid_tmp_dir):
        shutil.rmtree(grid_tmp_dir)
    os.makedirs(grid_tmp_dir, exist_ok=True)

    # Determine number of LH vertices
    if hasattr(surf_lh, "n_points"):
        n_lh_vertices_global = surf_lh.n_points
    else:
        n_lh_vertices_global = None  # will fall back to half in first loop

    for method in methods:
        if method == "glasso":
            mapped_dict = mapped_glasso
        elif method == "pearson":
            mapped_dict = mapped_pearson
        elif method == "conjunction":
            mapped_dict = mapped_conj
        else:
            raise ValueError(f"Unknown method '{method}'")

        for (struct, side, labels, rows, cols, fheight, fwidth) in structures_for_grid:
            block = mapped_dict[struct][side]  # (n_seeds_block, n_vertices_total)
            n_seeds_block = block.shape[0]
            if n_seeds_block == 0:
                continue

            # Determine the left-hemisphere vertex count.
            if n_lh_vertices_global is None:
                n_lh_vertices = block.shape[1] // 2
            else:
                n_lh_vertices = n_lh_vertices_global

            grid_pngs: List[str] = []
            grid_titles: List[str] = []

            # Seed plotting order.
            if struct == "Hippocampus":
                # block rows are already in subfield-major order (hipp_labels_L)
                # The 5×3 grid uses:
                #   rows  = subfields
                #   cols  = AP bins (A to P)
                labels_for_grid = hipp_labels_L          # subfield-major
                row_indices = list(range(n_seeds_block)) # identity mapping
            else:
                # Amygdala (and any other case): keep original order
                labels_for_grid = labels
                row_indices = list(range(n_seeds_block))


            for grid_idx, row_idx in enumerate(row_indices):
                vals_full = block[row_idx]
                vals_lh = vals_full[:n_lh_vertices]

                label_name = labels_for_grid[grid_idx] if grid_idx < len(labels_for_grid) else f"{struct}_{row_idx+1}"

                # Strip redundant hemisphere prefix from the label shown above each brain
                if side == "Left" and label_name.startswith("Left "):
                    label_name = label_name[5:]
                elif side == "Right" and label_name.startswith("Right "):
                    label_name = label_name[6:]

                safe_label = label_name.replace(" ", "_")
                grid_fname = os.path.join(
                    grid_tmp_dir,
                    f"{method}_LH_{struct}_{side}_seed{row_idx+1}_{safe_label}.png",
                )

                if not os.path.exists(grid_fname):
                    p = Plot(
                        surf_lh,
                        views=["lateral", "medial"],
                        zoom=1.3,
                        size=(500, 300),
                        brightness=0.7,
                    )


                    if method == "conjunction":
                        cmap = ListedColormap(["#1f77b4", "#ff7f0e", "#8c564b"])
                        p.add_layer(vals_lh, cmap=cmap, cbar=False)
                    else:
                        p.add_layer(vals_lh, cmap="viridis", cbar=True)

                    fig = p.build()
                    for ax in fig.axes:
                        tick_labels = ax.get_xticklabels() or ax.get_yticklabels()
                        plt.setp(tick_labels, fontsize=10)
                    fig.savefig(grid_fname, dpi=300, bbox_inches="tight", pad_inches=0.02)
                    plt.close(fig)

                grid_pngs.append(grid_fname)
                grid_titles.append(label_name)

            out_grid = os.path.join(
                OUTPUT_ROOT,
                f"FIG_GRID_{struct}_{side}_LHonly_{method}_top{TOP_PERCENT}.png",
            )
            create_matplotlib_grid(
                grid_pngs,
                grid_titles,
                rows,
                cols,
                fheight,
                fwidth,
                out_grid,
            )

else:
    print("[GRID] Skipping composite grids (MAKE_COMPOSITE_GRIDS=False).")

# -------------------------
# 6) Hipp-Amyg + Broad-Group Metrics (Normalized)
# -------------------------
print("[METRICS] Calculating Normalized Dominance & Sharedness...")

# --- Amygdala vs Hippocampus ---

# GLASSO
amyg_count_ctx_gl = mask_glasso_seedwise[amyg_rows_all, :].sum(axis=0).astype(float)
hipp_count_ctx_gl = mask_glasso_seedwise[hipp_rows_all, :].sum(axis=0).astype(float)

D_pres_ctx_gl = calc_dominance_normalized(
    amyg_count_ctx_gl,
    hipp_count_ctx_gl,
    n_total_amyg,
    n_total_hipp,
)
S_pres_ctx_gl = calc_sharedness_normalized(
    amyg_count_ctx_gl,
    hipp_count_ctx_gl,
    n_total_amyg,
    n_total_hipp,
)

# Pearson
amyg_count_ctx_pe = mask_pearson_seedwise[amyg_rows_all, :].sum(axis=0).astype(float)
hipp_count_ctx_pe = mask_pearson_seedwise[hipp_rows_all, :].sum(axis=0).astype(float)

D_pres_ctx_pe = calc_dominance_normalized(
    amyg_count_ctx_pe,
    hipp_count_ctx_pe,
    n_total_amyg,
    n_total_hipp,
)
S_pres_ctx_pe = calc_sharedness_normalized(
    amyg_count_ctx_pe,
    hipp_count_ctx_pe,
    n_total_amyg,
    n_total_hipp,
)

# -------------------------
# 7) Map metrics to full-360 and vertices
# -------------------------
def expand_vec(vec: np.ndarray, kept_1based: np.ndarray) -> np.ndarray:
    kept_0based = kept_1based.astype(int) - 1
    expanded_2d = map_top_to_full360(vec[None, :], kept_0based, full_len=360, fill=np.nan)
    return expanded_2d[0]

# --- GLASSO: Amyg-Hipp ---
D_pres_full360_gl = expand_vec(D_pres_ctx_gl, kept_glp)
S_pres_full360_gl = expand_vec(S_pres_ctx_gl, kept_glp)

D_pres_vertices_gl = map_to_labels(D_pres_full360_gl, labeling, mask=mask_labels, fill=np.nan)
S_pres_vertices_gl = map_to_labels(S_pres_full360_gl, labeling, mask=mask_labels, fill=np.nan)

np.save(os.path.join(OUTPUT_ROOT, f"hippamyg_dominance_presence_ctx_glasso_top{TOP_PERCENT}.npy"), D_pres_ctx_gl)
np.save(os.path.join(OUTPUT_ROOT, f"hippamyg_shared_presence_ctx_glasso_top{TOP_PERCENT}.npy"), S_pres_ctx_gl)
np.save(os.path.join(OUTPUT_ROOT, f"hippamyg_dominance_presence_full360_glasso_top{TOP_PERCENT}.npy"), D_pres_full360_gl)
np.save(os.path.join(OUTPUT_ROOT, f"hippamyg_shared_presence_full360_glasso_top{TOP_PERCENT}.npy"), S_pres_full360_gl)
np.save(os.path.join(OUTPUT_ROOT, f"hippamyg_dominance_presence_vertices_glasso_top{TOP_PERCENT}.npy"), D_pres_vertices_gl)
np.save(os.path.join(OUTPUT_ROOT, f"hippamyg_shared_presence_vertices_glasso_top{TOP_PERCENT}.npy"), S_pres_vertices_gl)

# --- Pearson: Amyg-Hipp ---
D_pres_full360_pe = expand_vec(D_pres_ctx_pe, kept_glp)
S_pres_full360_pe = expand_vec(S_pres_ctx_pe, kept_glp)

D_pres_vertices_pe = map_to_labels(D_pres_full360_pe, labeling, mask=mask_labels, fill=np.nan)
S_pres_vertices_pe = map_to_labels(S_pres_full360_pe, labeling, mask=mask_labels, fill=np.nan)

np.save(os.path.join(OUTPUT_ROOT, f"hippamyg_dominance_presence_ctx_pearson_top{TOP_PERCENT}.npy"), D_pres_ctx_pe)
np.save(os.path.join(OUTPUT_ROOT, f"hippamyg_shared_presence_ctx_pearson_top{TOP_PERCENT}.npy"), S_pres_ctx_pe)
np.save(os.path.join(OUTPUT_ROOT, f"hippamyg_dominance_presence_full360_pearson_top{TOP_PERCENT}.npy"), D_pres_full360_pe)
np.save(os.path.join(OUTPUT_ROOT, f"hippamyg_shared_presence_full360_pearson_top{TOP_PERCENT}.npy"), S_pres_full360_pe)
np.save(os.path.join(OUTPUT_ROOT, f"hippamyg_dominance_presence_vertices_pearson_top{TOP_PERCENT}.npy"), D_pres_vertices_pe)
np.save(os.path.join(OUTPUT_ROOT, f"hippamyg_shared_presence_vertices_pearson_top{TOP_PERCENT}.npy"), S_pres_vertices_pe)

# -------------------------
# 8) Plot GLASSO & Pearson presence metrics on surface
# -------------------------
print("[PLOT] Metric Surface maps...")

DOM_RANGE = (-1, 1)
SHARE_RANGE = (0, 2)  # Sharedness max is 2.0

for name, arr, rng, cmap in [
    # Amyg vs Hipp
    ("dom_glasso", D_pres_vertices_gl, DOM_RANGE, "icefire"),
    ("share_glasso", S_pres_vertices_gl, SHARE_RANGE, "magma"),
    ("dom_pearson", D_pres_vertices_pe, DOM_RANGE, "icefire"),
    ("share_pearson", S_pres_vertices_pe, SHARE_RANGE, "magma"),
]:
    plot_hemispheres(
        surf_lh, surf_rh, array_name=arr, size=(1600, 1000), cmap=cmap, color_bar=True, color_range=rng,
        nan_color= gray_rgba, zoom=1.2, interactive=False, screenshot=True,
        filename=os.path.join(OUTPUT_ROOT, f"MAP_{name}_top{TOP_PERCENT}.png"),
        suppress_warnings=True, transparent_bg=True,

    )


# -------------------------
# 9B) Community Bar Plots (Bar + Strip Overlay)
# -------------------------
metrics_to_plot = (
    (D_pres_full360_gl, "Dominance_GLASSO", (-1.1, 1.1), None),
    (S_pres_full360_gl, "Sharedness_GLASSO", (0, 2.0), (0.0, 2.0)),
    (D_pres_full360_pe, "Dominance_Pearson", (-1.1, 1.1), None),
    (S_pres_full360_pe, "Sharedness_Pearson", (0, 2.0), (0.0, 2.0)),
)


if "bar_strip" in COMMUNITY_PLOT_TYPES:
    print("[COMMUNITY] Generating Bar Summaries with Strip overlays...")

    comm_bar_out_dir = os.path.join(OUTPUT_ROOT, "communities_bar_plots")
    os.makedirs(comm_bar_out_dir, exist_ok=True)

    for data, metric_name, ylim, inset_range in metrics_to_plot:
        for spec in COMMUNITY_SPECS:
            require_exists(spec["tsv_path"])

            out_name = f"bar_strip_{metric_name}_{spec['name']}.png"
            out_path = os.path.join(comm_bar_out_dir, out_name)

            plot_community_bars_with_strips(
                data_360=data,
                tsv_path=spec["tsv_path"],
                index_col=spec["index_col"],
                scheme_col=spec["scheme_col"],
                out_path=out_path,
                title="",
                ylabel="",
                ylim=ylim,
                yscale="linear",
                orientation="horizontal",
                inset=False,
                inset_range=inset_range,
                inset_loc="lower right",
                custom_palette=yeo7_dict
            )


else:
    print("[COMMUNITY] Skipping bar+strip plots (not in COMMUNITY_PLOT_TYPES).")

# -------------------------
# 9D) Cross-method consistency: GLASSO vs Pearson scatters
# -------------------------
if "glasso_vs_pearson_scatter" in COMMUNITY_PLOT_TYPES:

    print("[COMMUNITY] GLASSO vs Pearson scatters for Dominance / Sharedness (Yeo7)...")

    # Use Yeo7 for coloring
    yeo7_spec = next((s for s in COMMUNITY_SPECS if s["name"] == "Yeo7"), None)

    if yeo7_spec is not None and os.path.exists(yeo7_spec["tsv_path"]):
        scatter_gp_out_dir = os.path.join(OUTPUT_ROOT, "scatter_glasso_vs_pearson")
        os.makedirs(scatter_gp_out_dir, exist_ok=True)

        scatter_specs = (
            (
                "Dominance",
                D_pres_full360_gl,
                D_pres_full360_pe,
                (-1.1, 1.1),
            ),
            (
                "Sharedness",
                S_pres_full360_gl,
                S_pres_full360_pe,
                (-0.1, 2.1),
            ),
        )
        for metric_name, glasso_values, pearson_values, limits in scatter_specs:
            out_path = os.path.join(
                scatter_gp_out_dir,
                f"scatter_{metric_name}_GLASSO_vs_Pearson_{yeo7_spec['name']}.png",
            )
            plot_metric_scatter_by_community(
                x_360=glasso_values,
                y_360=pearson_values,
                tsv_path=yeo7_spec["tsv_path"],
                index_col=yeo7_spec["index_col"],
                scheme_col=yeo7_spec["scheme_col"],
                out_path=out_path,
                xlabel=f"{metric_name} (GLASSO)",
                ylabel=f"{metric_name} (Pearson)",
                title="",
                xlim=limits,
                ylim=limits,
                add_identity=False,
                add_reg=True,
                legend=False,
                corr_method="kendall",
                custom_palette=yeo7_dict,
                msr=DO_MSR,
                msr_params=MSR_PARAMS,
                msr_mask_mode="finite",
            )

    else:
        print("[COMMUNITY] No Yeo17 spec found or TSV missing; skipping GLASSO vs Pearson scatters.")
else:
    print("[COMMUNITY] Skipping glasso vs pearson scatter plots (not in COMMUNITY_PLOT_TYPES).")

print(f"[DONE] top{TOP_PERCENT} maps finished.")
if TOP_PERCENT != 10:
    raise SystemExit("Threshold-sensitivity run complete; later sections are top-10 only.")
# =============================================================================
# FIG2 EXTRA (strength corroboration)
#
# Maps (with colorbars):
#   Net strength difference (Amyg − Hipp) computed as nanmean POSITIVE strength
#   within each method’s seedwise mask. Outside method union-mask -> NaN (gray).
#
# Scatters (Yeo-7 colors, overlap-only, no p-values):
#   x = count-based dominance (D_pres_full360_*)
#   y = strength-based dominance (sdom_360_*)
#   Only overlap parcels (both Hipp & Amyg present) are shown.
#   Title contains r and n only (no p).
#
# Uses existing: plot_hemispheres + plot_metric_scatter_by_community
# =============================================================================

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
POSITIVE_ONLY = True

# Output
OUT_DIR = os.path.join(OUTPUT_ROOT, "fig2_strength_block")
os.makedirs(OUT_DIR, exist_ok=True)

# Map dimensions.
MAP_SIZE = (1100, 700)
MAP_ZOOM = 1.25

# Scatter styling
SCAT_ALPHA = 0.75
SCAT_XLIM = (-1.1, 1.1)
SCAT_YLIM = (-1.1, 1.1)

# ---------------------------------------------------------------------
# 1) Compute netdiff + strength-dominance in ctx space
# ---------------------------------------------------------------------
net_gl_ctx, sdom_gl_ctx, overlap_gl_ctx, union_gl_ctx = strength_stats_from_mask(
    glasso_seed2ctx, mask_glasso_seedwise, amyg_rows_all, hipp_rows_all, positive_only=POSITIVE_ONLY
)
net_pe_ctx, sdom_pe_ctx, overlap_pe_ctx, union_pe_ctx = strength_stats_from_mask(
    pearson_seed2ctx, mask_pearson_seedwise, amyg_rows_all, hipp_rows_all, positive_only=POSITIVE_ONLY
)

# Expand overlap/union to 360
overlap_gl_360 = (expand_vec(overlap_gl_ctx.astype(float), kept_glp) > 0)
union_gl_360   = (expand_vec(union_gl_ctx.astype(float), kept_glp) > 0)

overlap_pe_360 = (expand_vec(overlap_pe_ctx.astype(float), kept_glp) > 0)
union_pe_360   = (expand_vec(union_pe_ctx.astype(float), kept_glp) > 0)

# Expand maps to 360 and set outside-mask to NaN (gray)
net_gl_360  = expand_vec(net_gl_ctx, kept_glp)
net_pe_360  = expand_vec(net_pe_ctx, kept_glp)
sdom_gl_360 = expand_vec(sdom_gl_ctx, kept_glp)
sdom_pe_360 = expand_vec(sdom_pe_ctx, kept_glp)

net_gl_360_plot  = np.full(360, np.nan); net_gl_360_plot[union_gl_360]  = net_gl_360[union_gl_360]
net_pe_360_plot  = np.full(360, np.nan); net_pe_360_plot[union_pe_360]  = net_pe_360[union_pe_360]
sdom_gl_360_plot = np.full(360, np.nan); sdom_gl_360_plot[union_gl_360] = sdom_gl_360[union_gl_360]
sdom_pe_360_plot = np.full(360, np.nan); sdom_pe_360_plot[union_pe_360] = sdom_pe_360[union_pe_360]

# ---------------------------------------------------------------------
# 2) MAPS with colorbars (net-diff only)
# ---------------------------------------------------------------------

map_paths = {}
for method, map_data in (
    ("GLASSO", net_gl_360_plot),
    ("Pearson", net_pe_360_plot),
):
    map_path = os.path.join(
        OUT_DIR,
        f"MAP_netdiff_strength_{method}_top{TOP_PERCENT}.png",
    )
    plot_hemispheres(
        surf_lh,
        surf_rh,
        array_name=map_to_labels(
            map_data,
            labeling,
            mask=mask_labels,
            fill=np.nan,
        ),
        size=MAP_SIZE,
        cmap="icefire",
        color_bar="bottom",
        color_range="sym",
        nan_color=gray_rgba,
        share="b",
        zoom=MAP_ZOOM,
        interactive=False,
        screenshot=True,
        filename=map_path,
        suppress_warnings=True,
        transparent_bg=True,
    )
    map_paths[method] = map_path

print("[OK] Saved maps:")
print("  ", map_paths["GLASSO"])
print("  ", map_paths["Pearson"])

# ---------------------------------------------------------------------
# 3) Scatters: count vs strength dominance in overlap parcels
# ---------------------------------------------------------------------
scatter_paths = {}
for method, count_dominance, strength_dominance, overlap in (
    ("GLASSO", D_pres_full360_gl, sdom_gl_360_plot, overlap_gl_360),
    ("Pearson", D_pres_full360_pe, sdom_pe_360_plot, overlap_pe_360),
):
    x_values, y_values = overlap_only_scatter_arrays(
        count_dominance,
        strength_dominance,
        overlap,
    )
    correlation, n_parcels = corr_r_n(x_values, y_values)
    scatter_path = os.path.join(
        OUT_DIR,
        f"SCAT_countdom_vs_strengthdom_{method}_overlapOnly_top{TOP_PERCENT}.png",
    )
    plot_metric_scatter_by_community(
        x_360=x_values,
        y_360=y_values,
        tsv_path=GLASSER_TSV_PATH,
        index_col="index",
        scheme_col="community_yeo7",
        out_path=scatter_path,
        xlabel="Count-based dominance",
        ylabel="Strength-based dominance",
        title=f"{method} (overlap only): r={correlation:.2f}, n={n_parcels}",
        xlim=SCAT_XLIM,
        ylim=SCAT_YLIM,
        add_identity=False,
        add_reg=True,
        legend=False,
        alpha=SCAT_ALPHA,
        custom_palette=yeo7_dict,
        msr=DO_MSR,
        msr_params=MSR_PARAMS,
        msr_mask_mode="finite_and_in_mask360",
        msr_mask_360=overlap,
    )
    scatter_paths[method] = scatter_path

print("[OK] Saved scatters:")
print("  ", scatter_paths["GLASSO"])
print("  ", scatter_paths["Pearson"])

# ---------------------------------------------------------------------
# 4) Combine maps and scatters into the manuscript 2 × 2 panel
# ---------------------------------------------------------------------
panel_specs = (
    (map_paths["GLASSO"], "GLASSO: Net strength diff"),
    (map_paths["Pearson"], "Pearson: Net strength diff"),
    (scatter_paths["GLASSO"], "GLASSO: Count vs Strength dominance"),
    (scatter_paths["Pearson"], "Pearson: Count vs Strength dominance"),
)
fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
for axis, (image_path, title) in zip(axes.flat, panel_specs):
    axis.imshow(mpimg.imread(image_path))
    axis.axis("off")
    axis.set_title(title, fontsize=11)

png_combo = os.path.join(
    OUT_DIR,
    f"FIG2_strength_block_2x2_top{TOP_PERCENT}.png",
)
fig.savefig(png_combo, dpi=600, bbox_inches="tight")
plt.close(fig)
print("[OK] Combined panel:", png_combo)

# =============================================================================
# FIGURE 3 (SUBCORTEX FOCUS): Pearson vs GLASSO side-by-side
# - LOSO seed-scoring version only
#
# Row A: Preference-to-sharedness (seed-wise) -> Hipp surfaces + Amyg volumes
# Row B: Preference-to-dominance (seed-wise) -> Hipp surfaces + Amyg volumes
# Row C: Amygdala x Hippocampus FC heatmaps (18×30): Pearson vs GLASSO
#
# Outputs:
#   FIG3_DIR/
#     seed_scores_*.npy
#     subcortex_overlays/*.png
#     composites/*.png
#     heatmaps/*.png
# =============================================================================

# -------------------------
# Output folders
# -------------------------
FIG3_DIR = os.path.join(OUTPUT_ROOT, "fig3_subcortex_sidebyside")
OUT_OVERLAYS = os.path.join(FIG3_DIR, "subcortex_overlays")
OUT_COMPOSITES = os.path.join(FIG3_DIR, "composites")
OUT_HEAT = os.path.join(FIG3_DIR, "heatmaps")
os.makedirs(FIG3_DIR, exist_ok=True)
os.makedirs(OUT_OVERLAYS, exist_ok=True)
os.makedirs(OUT_COMPOSITES, exist_ok=True)
os.makedirs(OUT_HEAT, exist_ok=True)

# -------------------------
# Subcortical visualization resources
# -------------------------
# Hipp canonical labels + surfaces
HIPP_LABELS_L_NPY = str(PATHS.output_root / "L_hipp_majority_labels.npy")
HIPP_LABELS_R_NPY = str(PATHS.output_root / "R_hipp_majority_labels.npy")
HIPP_RESOURCE_ROOT = Path(os.environ["HIPPUNFOLD_RESOURCE_ROOT"])
HIPP_SURF_L_GII = str(HIPP_RESOURCE_ROOT / "canonical_surfs/tpl-avg_space-canonical_den-2mm_label-hipp_midthickness.surf.gii")
HIPP_SURF_R_GII = str(HIPP_RESOURCE_ROOT / "canonical_surfs/tpl-avg_space-canonical_den-2mm_label-hipp_midthickness.surf.gii")

HIPP_UNFOLD_LABELMAP_L_NII = str(PATHS.output_root / "L_hipp_majority_unfold_DeKraker15.nii.gz")

HIPP_UNFOLD_LABELMAP_R_NII = str(PATHS.output_root / "R_hipp_majority_unfold_DeKraker15.nii.gz")

# Amygdala label images (for nucleus overlays)
AMYG_LABEL_L = str(PATHS.resource_root / "lh.AmygLabels.mgz")
AMYG_LABEL_R = str(PATHS.resource_root / "rh.AmygLabels.mgz")
AMYG_LABEL_ORDER = [7001, 7003, 7005, 7006, 7007, 7008, 7009, 7010, 7015]

# Colormaps
CMAP_HUB = plt.get_cmap("magma")     # hubness (0..)
CMAP_PREF = plt.get_cmap("icefire")  # preference (-..+)

# -------------------------
# Compute union masks in pruned cortex space for each method
# (restricting to parcels with at least one seed from hipp or amyg present)
# -------------------------
# Family-wise counts in pruned cortical space.
amyg_count_ctx_gl = mask_glasso_seedwise[amyg_rows_all, :].sum(axis=0).astype(int)
hipp_count_ctx_gl = mask_glasso_seedwise[hipp_rows_all, :].sum(axis=0).astype(int)
union_ctx_gl = (amyg_count_ctx_gl + hipp_count_ctx_gl) > 0

amyg_count_ctx_pe = mask_pearson_seedwise[amyg_rows_all, :].sum(axis=0).astype(int)
hipp_count_ctx_pe = mask_pearson_seedwise[hipp_rows_all, :].sum(axis=0).astype(int)
union_ctx_pe = (amyg_count_ctx_pe + hipp_count_ctx_pe) > 0

# -------------------------
# Compute LOSO scores (Pearson/GLASSO)
# -------------------------

hub_pe_loso, pref_pe_loso = compute_seed_scores_loso(
    pearson_seed2ctx, mask_pearson_seedwise, amyg_rows_all, hipp_rows_all, n_total_amyg, n_total_hipp, union_ctx_pe, positive_only=True
)
hub_gl_loso, pref_gl_loso = compute_seed_scores_loso(
    glasso_seed2ctx, mask_glasso_seedwise, amyg_rows_all, hipp_rows_all, n_total_amyg, n_total_hipp, union_ctx_gl, positive_only=True
)

# Save arrays

np.save(os.path.join(FIG3_DIR, f"hub_seed_pearson_LOSO_top{TOP_PERCENT}.npy"), hub_pe_loso)
np.save(os.path.join(FIG3_DIR, f"hub_seed_glasso_LOSO_top{TOP_PERCENT}.npy"), hub_gl_loso)
np.save(os.path.join(FIG3_DIR, f"pref_seed_pearson_LOSO_top{TOP_PERCENT}.npy"), pref_pe_loso)
np.save(os.path.join(FIG3_DIR, f"pref_seed_glasso_LOSO_top{TOP_PERCENT}.npy"), pref_gl_loso)

print("[FIG3] Saved seed-score arrays to:", FIG3_DIR)

# Shared display ranges (keep stable across Pearson/GLASSO)

# Shared color range.
hub_vmin, hub_vmax = robust_range_pos(
    hub_pe_loso,
    hub_gl_loso,
    q=100.0
)
pref_vmin, pref_vmax = robust_range_sym(
    pref_pe_loso,
    pref_gl_loso,
    q=100.0
)

# # fixed range
# hub_vmin, hub_vmax = 0.0, 2.0
# pref_vmin, pref_vmax = -1.0, 1.0


def _stitch_horiz(img_paths: list[str], out_path: str, title: str | None = None):
    imgs = [mpimg.imread(p) for p in img_paths if p and os.path.exists(p)]
    if not imgs:
        return
    fig, axes = plt.subplots(1, len(imgs), figsize=(len(imgs)*4.0, 4.0), constrained_layout=True)
    if len(imgs) == 1:
        axes = [axes]
    for ax, im in zip(axes, imgs):
        ax.imshow(im)
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=12)
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

def _stitch_vert(img_paths: list[str], out_path: str, title: str | None = None, height_ratios: list[float] | None = None):
    imgs = [mpimg.imread(p) for p in img_paths if p and os.path.exists(p)]
    if not imgs:
        return

    n = len(imgs)
    if height_ratios is None:
        height_ratios = [1.0] * n
    if len(height_ratios) != n:
        raise ValueError("height_ratios length must match number of images")

    # Make figure height proportional to ratios (keeps resolution reasonable)
    fig_h = 3.2 * float(np.sum(height_ratios))
    fig, axes = plt.subplots(
        n, 1,
        figsize=(6.0, fig_h),
        constrained_layout=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    if n == 1:
        axes = [axes]

    for ax, im in zip(axes, imgs):
        ax.imshow(im)
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=12)

    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


# Load subcortical resources once
label_L, label_R, verts_dict, faces_dict = load_hipp_surfaces_and_labels(
    labeling_path_L=HIPP_LABELS_L_NPY,
    labeling_path_R=HIPP_LABELS_R_NPY,
    surf_path_L=HIPP_SURF_L_GII,
    surf_path_R=HIPP_SURF_R_GII,
)
uniq_L = get_hipp_label_ids(label_L, expected_n=N_HIPP_LABELS)
uniq_R = get_hipp_label_ids(label_R, expected_n=N_HIPP_LABELS)


img_L_amyg = nb.load(AMYG_LABEL_L)
img_R_amyg = nb.load(AMYG_LABEL_R)

def plot_seedscore_overlays(seed_scores_48: np.ndarray, method: str, variant: str, metric: str,
                           vmin: float, vmax: float, cmap):
    """
    metric: "hubness" or "preference"
    variant: currently "LOSO" in this cleaned paper script
    Saves:
      - Hipp Left + Right pngs
      - combined amygdala PNG
    Returns filepaths for stitching.
    """
    seed_scores_48 = np.asarray(seed_scores_48, float)
    if seed_scores_48.shape != (glasso_seed2ctx.shape[0],):
        raise ValueError(f"Expected seed_scores shape ({glasso_seed2ctx.shape[0]},), got {seed_scores_48.shape}")

    # emulate gradients_top shape (n_seeds, 1)
    G = seed_scores_48[:, None]

    # --- Hipp surface mapping (uses indices into seed order)
    gv = map_gradients_to_hipp_vertices(
        gradients_top=G,
        left_hipp_indices=hipp_L_idx,
        right_hipp_indices=hipp_R_idx,
        labeling_L=label_L,
        labeling_R=label_R,
        uniq_L=uniq_L,
        uniq_R=uniq_R,
        n_components=1,
    )

    per_vmin = np.array([vmin], dtype=float)
    per_vmax = np.array([vmax], dtype=float)

    hipp_tpl = os.path.join(
        OUT_OVERLAYS,
        f"FIG3_{metric}_{variant}_{method}" + "_{side}_hipp_G{gradient}.png"
    )
    plot_and_save_hipp_gradients(
        vertices_dict=verts_dict,
        faces_dict=faces_dict,
        grad_vertex_dicts=gv,
        per_grad_vmin=per_vmin,
        per_grad_vmax=per_vmax,
        cmap=cmap,
        title_template="",
        output_file_template=hipp_tpl,
        n_to_plot=1
    )
    hipp_L_png = hipp_tpl.format(side="left", gradient=1)
    hipp_R_png = hipp_tpl.format(side="right", gradient=1)

    # --- True 2D unfolded hippocampus map
    # Uses the exact same cmap/vmin/vmax as the folded Fig. 3 hippocampus plot.
    hipp_unfolded_out = os.path.join(
        OUT_OVERLAYS,
        f"FIG3_{metric}_{variant}_{method}_hipp_unfolded_LR.png"
    )

    unfolded_cbar_label = (
        "Preference to sharedness"
        if ("sharedness" in metric or "hubness" in metric)
        else "Preference to dominance"
    )

    plot_and_save_hipp_unfolded_gradients(
        values=seed_scores_48,
        left_hipp_indices=hipp_L_idx,
        right_hipp_indices=hipp_R_idx,
        labelmap_l_path=HIPP_UNFOLD_LABELMAP_L_NII,
        labelmap_r_path=HIPP_UNFOLD_LABELMAP_R_NII,
        out_path=hipp_unfolded_out,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        n_hipp=N_HIPP_LABELS,
        collapse_axis=2,
        flip_ap=True,
        flip_pd=True,
        boundary_mode="subfield",
        title="",
        cbar_label=unfolded_cbar_label,
        save_pdf=True,
    )

    # --- Amyg volumes (both hemis in one figure, naming varies -> locate by prefix)
    out_prefix = f"FIG3_{metric}_{variant}_{method}_amyg"
    plot_amygdala_volumes_sharedscale(
        gradients_top=G,
        label_img_L=img_L_amyg,
        label_img_R=img_R_amyg,
        label_order=AMYG_LABEL_ORDER,
        left_indices=amyg_L_idx,
        right_indices=amyg_R_idx,
        per_grad_vmin=per_vmin,
        per_grad_vmax=per_vmax,
        out_dir=OUT_OVERLAYS,
        title_prefix=f"{method.upper()} {metric} ({variant}) — Amygdala",
        output_prefix=out_prefix,
        n_to_plot=1,
        cmap=cmap,
        ortho_layout="vertical",
        mirror_right_from_left=True,
        draw_contours=True,
    )
    amyg_L_png = os.path.join(OUT_OVERLAYS, f"{out_prefix}_L_gradient_{1}.png")
    amyg_R_png = os.path.join(OUT_OVERLAYS, f"{out_prefix}_R_gradient_{1}.png")

    # Stitch the left and right amygdala panels.
    amyg_LR_png = os.path.join(OUT_OVERLAYS, f"{out_prefix}_LR_gradient_{1}.png")
    _stitch_horiz([amyg_L_png, amyg_R_png], amyg_LR_png, title=None)

    return hipp_L_png, hipp_R_png, amyg_LR_png


def make_side_by_side_panel(method_left: str, method_right: str,
                            left_files: tuple[str, str, str],
                            right_files: tuple[str, str, str],
                            out_path: str, main_title: str):
    """
    left_files/right_files: (hipp_L, hipp_R, amyg_png)
    Creates one composite: left column (Pearson) | right column (GLASSO)
    Each column: [Hipp_L + Hipp_R stitched horizontally] stacked over [Amyg]
    """
    # stitch hippocampus halves per method
    tmp_left_hipp = os.path.join(OUT_COMPOSITES, "_tmp_left_hipp.png")
    tmp_right_hipp = os.path.join(OUT_COMPOSITES, "_tmp_right_hipp.png")
    _stitch_horiz([left_files[0], left_files[1]], tmp_left_hipp, title=None)
    _stitch_horiz([right_files[0], right_files[1]], tmp_right_hipp, title=None)

    # stitch columns vertically (hipp over amyg)
    tmp_left_col = os.path.join(OUT_COMPOSITES, "_tmp_left_col.png")
    tmp_right_col = os.path.join(OUT_COMPOSITES, "_tmp_right_col.png")
    _stitch_vert([tmp_left_hipp, left_files[2]], tmp_left_col, title=method_left, height_ratios=[1.0, 1.35])
    _stitch_vert([tmp_right_hipp, right_files[2]], tmp_right_col, title=method_right, height_ratios=[1.0, 1.35])


    # stitch columns side-by-side
    _stitch_horiz([tmp_left_col, tmp_right_col], out_path, title=main_title)

    # Remove temporary panels.
    for p in [tmp_left_hipp, tmp_right_hipp, tmp_left_col, tmp_right_col]:
        if os.path.exists(p):
            os.remove(p)

# Produce the two LOSO overlay panels in manuscript order.
for pearson_scores, glasso_scores, metric, limits, cmap, filename, title in (
    (
        hub_pe_loso,
        hub_gl_loso,
        "hubness_to_sharedness",
        (hub_vmin, hub_vmax),
        CMAP_HUB,
        "FIG3A_hubness_LOSO_Pearson_vs_GLASSO.png",
        "FIG3A: Hubness-to-sharedness (LOSO)",
    ),
    (
        pref_pe_loso,
        pref_gl_loso,
        "preference_to_dom_map",
        (pref_vmin, pref_vmax),
        CMAP_PREF,
        "FIG3B_pref_LOSO_Pearson_vs_GLASSO.png",
        "FIG3B: Preference-to-dominance (LOSO)",
    ),
):
    pearson_files = plot_seedscore_overlays(
        pearson_scores,
        "pearson",
        "LOSO",
        metric,
        limits[0],
        limits[1],
        cmap,
    )
    glasso_files = plot_seedscore_overlays(
        glasso_scores,
        "glasso",
        "LOSO",
        metric,
        limits[0],
        limits[1],
        cmap,
    )
    make_side_by_side_panel(
        "Pearson",
        "GLASSO",
        pearson_files,
        glasso_files,
        os.path.join(OUT_COMPOSITES, filename),
        main_title=title,
    )

print("[FIG3] Saved composite overlay panels to:", OUT_COMPOSITES)

# -------------------------
# Row C: Amygdala × hippocampus FC heatmaps
# -------------------------
# The Figure 3C module defines the aggregation, support mask, sign,
# orientation, scaling, styling, and output filenames.
(
    AH_pe_full,
    AH_gl_full,
    AH_any_prev,
    AH_pos_prev,
    INDEXED_AMYG_PALETTE,
) = make_row_c_heatmaps(
    pearson_z_stack=pearson_z_stack,
    glasso_z_stack=glasso_z_stack,
    amyg_rows_all=amyg_rows_all,
    hipp_rows_all=hipp_rows_all,
    amyg_labels_L=amyg_labels_L,
    amyg_labels_R=amyg_labels_R,
    hipp_labels_L=hipp_labels_L,
    hipp_labels_R=hipp_labels_R,
    out_heat=OUT_HEAT,
    preferred_font=PREFERRED_FONT,
)


# =============================================================================
# Figure 3D correlations and scatter plots
# Partial Spearman controls and 2x2 hippocampus/amygdala scatter plots.
# =============================================================================

FIG3D_SCORE_MODE = "LOSO"

def _safe_fdrcorr(pvals, alpha=0.05):
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan, dtype=float)
    m = np.isfinite(p)
    if not m.any(): return out
    _, q = fdrcorrection(p[m], alpha=alpha, method="indep")
    out[m] = q
    return out

def _intrinsic_bars_from_AH(AH: np.ndarray, sign_policy: str = "pos"):
    """Intrinsic AH strength per seed (mean across opposite structure)."""
    X = np.asarray(AH, float).copy()
    sp = str(sign_policy).lower().strip()

    if sp == "pos":
        X[~np.isfinite(X)] = np.nan
        X[X <= 0] = np.nan
    elif sp == "pos0":
        X[~np.isfinite(X)] = np.nan
        X[X < 0] = 0.0
    elif sp == "abs":
        X = np.abs(X)
    else:
        X[~np.isfinite(X)] = np.nan
        X = np.abs(X)

    amyg_x = np.nanmean(X, axis=1)
    hipp_x = np.nanmean(X, axis=0)
    return amyg_x, hipp_x

def _mean_pos_seed2ctx_strength(seed2ctx: np.ndarray, mask_seedwise: np.ndarray | None, positive_only=True):
    """Per-seed mean positive seed->cortex strength within that method's own mask."""
    X = np.asarray(seed2ctx, float)
    if mask_seedwise is None:
        M = np.isfinite(X)
    else:
        M = np.asarray(mask_seedwise, bool) & np.isfinite(X)

    if positive_only:
        Xp = np.where(M, np.clip(X, 0.0, None), np.nan)
    else:
        Xp = np.where(M, X, np.nan)
    return np.nanmean(Xp, axis=1)

def _partial_spearman(x, y, controls_2d):
    """Partial Spearman via rank-transform then residualize with linear regression."""
    x, y, C = np.asarray(x, float), np.asarray(y, float), np.asarray(controls_2d, float)
    if C.ndim == 1: C = C[:, None]

    m = np.isfinite(x) & np.isfinite(y)
    for j in range(C.shape[1]): m &= np.isfinite(C[:, j])

    n = int(m.sum())
    if n < 4: return np.nan, np.nan, n

    xr, yr = rankdata(x[m]), rankdata(y[m])
    Cr = np.column_stack([rankdata(C[m, j]) for j in range(C.shape[1])])
    A = np.column_stack([np.ones((n, 1)), Cr])

    bx, *_ = np.linalg.lstsq(A, xr, rcond=None)
    by, *_ = np.linalg.lstsq(A, yr, rcond=None)

    rx, ry = xr - A @ bx, yr - A @ by
    r, p = pearsonr(rx, ry)
    return float(r), float(p), n

def _build_hipp_ap_colors_LR(hipp_ap_bins_L, hipp_ap_bins_R, palette="Oranges"):
    """Builds L+R hippocampus colors based on AP bins."""
    bL, bR = np.asarray(hipp_ap_bins_L, int).ravel(), np.asarray(hipp_ap_bins_R, int).ravel()
    bins = np.concatenate([bL, bR])
    uniq = sorted(np.unique(bins).tolist())
    pal = sns.color_palette(palette, len(uniq))
    b2i = {b:i for i,b in enumerate(uniq)}
    return [pal[b2i[int(b)]] for b in bL] + [pal[b2i[int(b)]] for b in bR]

# Statistical printouts with cortical-strength controls.

hub_pe_fig3d = hub_pe_loso
hub_gl_fig3d = hub_gl_loso
pref_pe_fig3d = pref_pe_loso
pref_gl_fig3d = pref_gl_loso

print("\n" + "="*80)
print("EXTRINSIC CORTICAL ROUTING VS INTRINSIC AH CONNECTIVITY (GLASSO)")
print("="*80)

n_amyg = len(amyg_rows_all)

# Extract selected extrinsic scores
amyg_hub_gl, hipp_hub_gl = hub_gl_fig3d[:n_amyg], hub_gl_fig3d[n_amyg:]
amyg_pref_gl, hipp_pref_gl = pref_gl_fig3d[:n_amyg], pref_gl_fig3d[n_amyg:]

# Metric 1: unconditional mean positive amygdala–hippocampus coupling per seed
amyg_bars_gl, hipp_bars_gl = _intrinsic_bars_from_AH(AH_gl_full, sign_policy="pos")

# Controls
ctrl_gl_all = _mean_pos_seed2ctx_strength(glasso_seed2ctx, mask_glasso_seedwise)
amyg_ctrl_gl, hipp_ctrl_gl = ctrl_gl_all[:n_amyg], ctrl_gl_all[n_amyg:]

def print_stats_row(name, x, y, ctrl):
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(ctrl)
    if m.sum() < 4: return
    r0, p0 = spearmanr(x[m], y[m])
    rp, pp, _ = _partial_spearman(x[m], y[m], ctrl[m])
    print(f"{name:<40} | r0={r0:5.2f} (p={p0:.3f}) | r_part={rp:5.2f} (p={pp:.3f})")

print("\n--- PARTIAL SPEARMAN CONTROLS (Controlling for Cortical Strength) ---")
print_stats_row("Hippocampus (Metric 1) -> Sharedness", hipp_bars_gl, hipp_hub_gl, hipp_ctrl_gl)
print_stats_row("Hippocampus (Metric 1) -> Preference", hipp_bars_gl, hipp_pref_gl, hipp_ctrl_gl)
print_stats_row("Amygdala (Metric 1) -> Sharedness", amyg_bars_gl, amyg_hub_gl, amyg_ctrl_gl)
print_stats_row("Amygdala (Metric 1) -> Preference", amyg_bars_gl, amyg_pref_gl, amyg_ctrl_gl)
print("="*80 + "\n")

# =============================================================================
# Figure 3D 2×2 scatter plots
# =============================================================================
def plot_fig3d_intrinsic_vs_extrinsic_2x2(pearson_dict, glasso_dict, metric_name, out_path):
    """Creates a 2x2 scatter grid: Hipp/Amyg vs Pearson/GLASSO for a given metric."""

    amyg_palette = [
        (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)
        for color in _AMYGDALA_LUT_RGBA.values()
    ]
    amyg_colors = amyg_palette + amyg_palette
    hipp_colors = _build_hipp_ap_colors_LR(hipp_ap_bins_L, hipp_ap_bins_R)

    def _extract(d):
        am_x, hi_x = _intrinsic_bars_from_AH(np.asarray(d["AH"], float), sign_policy="pos")
        score = np.asarray(d["score"], float).ravel()
        am_y, hi_y = score[:n_amyg], score[n_amyg:]

        ctrl = _mean_pos_seed2ctx_strength(d["seed2ctx"], d["mask_seedwise"], positive_only=True)
        return am_x, hi_x, am_y, hi_y, ctrl[:n_amyg], ctrl[n_amyg:]

    am_x_pe, hi_x_pe, am_y_pe, hi_y_pe, am_c_pe, hi_c_pe = _extract(pearson_dict)
    am_x_gl, hi_x_gl, am_y_gl, hi_y_gl, am_c_gl, hi_c_gl = _extract(glasso_dict)

    panels = [
        ("Hippocampus", "Pearson", hi_x_pe, hi_y_pe, hipp_colors, hi_c_pe),
        ("Hippocampus", "GLASSO",  hi_x_gl, hi_y_gl, hipp_colors, hi_c_gl),
        ("Amygdala",    "Pearson", am_x_pe, am_y_pe, amyg_colors, am_c_pe),
        ("Amygdala",    "GLASSO",  am_x_gl, am_y_gl, amyg_colors, am_c_gl),
    ]

    rho0, p0, rhop, pp, nn = [], [], [], [], []
    for (struct, method, x, y, colors, ctrl) in panels:
        m = np.isfinite(x) & np.isfinite(y)
        n = int(m.sum())
        r0, p = spearmanr(np.asarray(x)[m], np.asarray(y)[m]) if n >= 4 else (np.nan, np.nan)
        rp_, pp_, _ = _partial_spearman(x, y, ctrl) if n >= 4 else (np.nan, np.nan, n)

        rho0.append(r0); p0.append(p); rhop.append(rp_); pp.append(pp_); nn.append(n)

    q0 = _safe_fdrcorr(p0)
    qp = _safe_fdrcorr(pp)

    sns.set_context("talk", font_scale=1.4)
    sns.set_style("ticks")

    fig, axes = plt.subplots(2, 2, figsize=(10, 15), constrained_layout=True, facecolor="white")

    # Common y limits across all panels.
    all_y_vals = np.concatenate([p[3][np.isfinite(p[3])] for p in panels])
    global_ymin = np.min(all_y_vals) - 0.05
    global_ymax = np.max(all_y_vals) + 0.05
    # Add headroom for the statistics annotation.
    plot_ymax = global_ymax + ((global_ymax - global_ymin) * 0.15)

    # Method-specific x limits shared within each column.
    pe_x_vals = np.concatenate([p[2][np.isfinite(p[2])] for p in panels if p[1] == "Pearson"])
    gl_x_vals = np.concatenate([p[2][np.isfinite(p[2])] for p in panels if p[1] == "GLASSO"])

    # Add 5% horizontal padding.
    pe_padding = (np.max(pe_x_vals) - np.min(pe_x_vals)) * 0.05
    pe_xlim = (np.min(pe_x_vals) - pe_padding, np.max(pe_x_vals) + pe_padding)

    gl_padding = (np.max(gl_x_vals) - np.min(gl_x_vals)) * 0.05
    gl_xlim = (np.min(gl_x_vals) - gl_padding, np.max(gl_x_vals) + gl_padding)

    for i, (struct, method, x, y, colors, ctrl) in enumerate(panels):
        ax = axes[i // 2, i % 2]
        m = np.isfinite(x) & np.isfinite(y)
        xv, yv, cv = x[m], y[m], [colors[j] for j, ok in enumerate(m) if ok]

        # Scatter plot.
        ax.scatter(xv, yv, c=cv, s=140, edgecolor="white", linewidth=1.0, alpha=0.95, zorder=5)

        ax.set_title(f"{struct} — {method}", fontweight="bold", pad=12)
        ax.set_xlabel("Intrinsic AH FC (expected +r)")
        ax.set_ylabel(f"Extrinsic Cortical {metric_name}")
        sns.despine(ax=ax, offset=10, trim=False)

        ax.set_ylim(global_ymin, plot_ymax)

        if method == "Pearson":
            ax.set_xlim(pe_xlim)
        else: # GLASSO
            ax.set_xlim(gl_xlim)

        # Statistics annotation.
        star0 = _p_to_stars(q0[i])
        starp = _p_to_stars(qp[i])

        txt = (
            f"ρ = {rho0[i]:.2f}{star0}\n"
            f"partial ρ = {rhop[i]:.2f}{starp}"        )

        ax.text(0.05, 0.95, txt, transform=ax.transAxes, va="top", ha="left",
                fontsize=18, bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7", alpha=0.9))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIG 3D] Saved {metric_name} grid to: {out_path}")

# =============================================================================
# Figure 3D panels
# =============================================================================
# Main Text Figure: Sharedness
plot_fig3d_intrinsic_vs_extrinsic_2x2(
    pearson_dict={"AH": AH_pe_full, "score": hub_pe_fig3d, "seed2ctx": pearson_seed2ctx, "mask_seedwise": mask_pearson_seedwise},
    glasso_dict={"AH": AH_gl_full, "score": hub_gl_fig3d, "seed2ctx": glasso_seed2ctx, "mask_seedwise": mask_glasso_seedwise},
    metric_name="Sharedness",
    out_path=os.path.join(FIG3_DIR, f"FIG3D_Sharedness_Intrinsic_vs_Extrinsic_{FIG3D_SCORE_MODE}_top{TOP_PERCENT}.png")
)


# Supplementary Figure: Dominance
plot_fig3d_intrinsic_vs_extrinsic_2x2(
    pearson_dict={"AH": AH_pe_full, "score": pref_pe_fig3d, "seed2ctx": pearson_seed2ctx, "mask_seedwise": mask_pearson_seedwise},
    glasso_dict={"AH": AH_gl_full, "score": pref_gl_fig3d, "seed2ctx": glasso_seed2ctx, "mask_seedwise": mask_glasso_seedwise},
    metric_name="Preference",
    out_path=os.path.join(FIG3_DIR, f"SUPP_FIG_Preference_Intrinsic_vs_Extrinsic_{FIG3D_SCORE_MODE}_top{TOP_PERCENT}.png")
)

#------------------------------------------------------------
# Legend for this part
#------------------------------------------------------------

def create_horizontal_scatter_legend(out_path):
    """
    Create the three-row legend used in Figure 3.
    Hippocampus (1 col x 3 rows) | Amygdala (3 cols x 3 rows)
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.5), gridspec_kw={'width_ratios': [1, 2.5]})

    axes[0].axis('off')
    axes[1].axis('off')

    # Hippocampal anterior–posterior bins.
    hipp_colors = sns.color_palette("Oranges", 3)
    hipp_labels = ["Anterior", "Middle", "Posterior"]

    hipp_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
               markersize=20, markeredgecolor='white', markeredgewidth=1)
        for c in hipp_colors
    ]

    leg_hipp = axes[0].legend(hipp_handles, hipp_labels, title="Hippocampus (A-P Axis)",
                              loc='center', ncol=1, frameon=False,
                              fontsize=20, title_fontproperties={'weight': 'bold', 'size': 20})
    leg_hipp._legend_box.align = "center"
    leg_hipp.get_title().set_fontweight('bold')

    # Amygdala nuclei.
    amyg_colors = INDEXED_AMYG_PALETTE
    amyg_labels = AMYGDALA_SHORT

    amyg_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
               markersize=20, markeredgecolor='white', markeredgewidth=1)
        for c in amyg_colors
    ]

    leg_amyg = axes[1].legend(amyg_handles, amyg_labels, title="Amygdala Nuclei",
                              loc='center', ncol=3, frameon=False,
                              fontsize=20, title_fontproperties={'weight': 'bold', 'size': 15},
                              columnspacing=1.5, handletextpad=0.4)
    leg_amyg._legend_box.align = "center"
    leg_amyg.get_title().set_fontweight('bold')

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=600, bbox_inches='tight', transparent=True)

    base, _ = os.path.splitext(out_path)
    plt.savefig(base + ".pdf", bbox_inches='tight', transparent=True)
    plt.close(fig)
    print(f"[DONE] Saved 3-row standalone legend to: {out_path}")

# =============================================================================
# Figure 3D legend
# =============================================================================
legend_output_file = os.path.join(FIG3_DIR, "FIG3D_Scatter_Legend.png")
create_horizontal_scatter_legend(legend_output_file)
#
#------------------------------------------------------------
# Surface map: Neither / GLASSO-only / Pearson-only / Overlap (union across seeds) -- FIG 1 BOTTOM PANEL
# ------------------------------------------------------------
# 1) Union across seeds -> parcel-wise membership (n_ctx,)
g_any = np.any(mask_glasso_seedwise, axis=0)   # (n_ctx,)
p_any = np.any(mask_pearson_seedwise, axis=0)  # (n_ctx,)

# category codes (n_ctx,)
# 0 = Neither, 1 = GLASSO-only, 2 = Pearson-only, 3 = Overlap
cat_ctx = np.zeros_like(g_any, dtype=float)
cat_ctx[g_any & ~p_any] = 1.0
cat_ctx[~g_any & p_any] = 2.0
cat_ctx[g_any & p_any]  = 3.0

# 2) Expand n_ctx -> full 360 parcel vector (NaN for dropped parcels)
cat_full360 = expand_vec(cat_ctx, kept_glp)

# 3) Map parcels -> vertices
cat_vertices = map_to_labels(cat_full360, labeling, mask=mask_labels, fill=np.nan)

# 4) Discrete colormap
#    Order must match codes: 0,1,2,3
cmap4 = ListedColormap([
    "#bdbdbd",  # 0 Neither (light gray)
    "#1f77b4",  # 1 GLASSO-only (blue)
    "#ff7f0e",  # 2 Pearson-only (orange)
    "#8c564b",  # 3 Overlap (brown)
])

out_path = os.path.join(
    OUTPUT_ROOT,
    f"MAP_union_categories_neither_glasso_pearson_overlap_top{TOP_PERCENT}.png"
)

# 5) Plot on surface
# color_range matches the discrete codes.
plot_hemispheres(
    surf_lh, surf_rh,
    array_name=cat_vertices,
    size=(1600, 1000),
    cmap=cmap4,
    color_bar=True,
    color_range=(0, 3),
    nan_color= gray_rgba,  # outside cortex / unlabeled vertices
    zoom=1.2,
    interactive=False,
    screenshot=True,
    filename=out_path,
    suppress_warnings=True,
    transparent_bg=True,
)

print("[DONE] Saved:", out_path)

# Save the category legend separately for figure assembly.
legend_path = os.path.join(
    OUTPUT_ROOT,
    f"LEGEND_union_categories_top{TOP_PERCENT}.png"
)
fig, ax = plt.subplots(figsize=(6, 1.2))
ax.axis("off")
labels = ["Neither", "GLASSO only", "Pearson only", "Overlap"]
handles = [plt.Line2D([0], [0], marker="s", linestyle="", markersize=12, markerfacecolor=cmap4(i))
           for i in range(4)]
ax.legend(handles, labels, loc="center", ncol=4, frameon=False, handletextpad=0.6, columnspacing=1.2)
plt.tight_layout()
plt.savefig(legend_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print("[DONE] Saved legend:", legend_path)

# Community-stacked proportion distribution (e.g., Yeo7) - SAME COLORS AS SURFACE
# ------------------------------------------------------------
# Mosaic plot styling.
sns.set_context("poster", font_scale=0.9)
sns.set_style("white")

# --- FONT CONFIGURATION ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = PREFERRED_FONT

COLORS = {
    "Neither": "#e0e0e0",      # Light Gray
    "GLASSO only": "#1f77b4",  # Blue
    "Pearson only": "#ff7f0e", # Orange
    "Overlap": "#8c564b",      # Brown
}

# Community assignments.
df = pd.read_csv(GLASSER_TSV_PATH, sep="\t")
df["index"] = pd.to_numeric(df["index"], errors="coerce").astype(int)
df = df.set_index("index")

parcel_ids = kept_glp.astype(int)
scheme_col = "community_yeo7"
comm_labels = df.loc[parcel_ids, scheme_col].astype(str).values

g_any = np.any(mask_glasso_seedwise, axis=0)
p_any = np.any(mask_pearson_seedwise, axis=0)

cat_code = np.full(len(parcel_ids), "Neither", dtype=object)
cat_code[g_any & ~p_any] = "GLASSO only"
cat_code[~g_any & p_any] = "Pearson only"
cat_code[g_any & p_any]  = "Overlap"

tmp = pd.DataFrame({"community": comm_labels, "category": cat_code})
ct = pd.crosstab(tmp["community"], tmp["category"])

col_order = ["Neither", "GLASSO only", "Pearson only", "Overlap"]
for c in col_order:
    if c not in ct.columns: ct[c] = 0
ct = ct[col_order]

# Normalize within each cortical community.
ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
n_per_comm = ct.sum(axis=1)

# ------------------------------------------------------------
# Order communities by the GLASSO-only proportion.
# ------------------------------------------------------------
glasso_sort_metric = ct_pct["GLASSO only"]

sort_table = pd.DataFrame(
    {
        "glasso_sort": glasso_sort_metric,
        "overlap": ct_pct["Overlap"],
        "pearson_only": ct_pct["Pearson only"],
        "community_label": ct_pct.index.astype(str),
    },
    index=ct_pct.index,
)

community_order = (
    sort_table
    .sort_values(
        by=["glasso_sort", "overlap", "pearson_only", "community_label"],
        ascending=[False, False, True, True],
        kind="mergesort",  # deterministic/stable
    )
    .index
)

ct = ct.loc[community_order]
ct_pct = ct_pct.loc[community_order]
n_per_comm = n_per_comm.loc[community_order]

fig, ax = plt.subplots(figsize=(16, 10))

ct_pct.plot(
    kind="bar",
    stacked=True,
    ax=ax,
    color=[COLORS[c] for c in col_order],
    width=1.0,
    edgecolor="white",
    linewidth=1.5,
    legend=False         # NO LEGEND
)

# Percent labels within segments.
for c in ax.containers:
    labels = [f'{v.get_height():.0f}%' if v.get_height() > 5 else '' for v in c]

    ax.bar_label(
        c,
        labels=labels,
        label_type='center',
        color='black',
        fontsize=28,
        fontweight='bold'
    )

# Parcel counts above bars.
for i, comm_name in enumerate(ct_pct.index):
    total_n = int(n_per_comm[comm_name])
    ax.text(
        i,
        102,
        f"N={total_n}",
        ha='center',
        va='bottom',
        fontsize=20,
        fontweight='bold',
        color='#333333'
    )

sns.despine(left=True, bottom=True, right=True, top=True)

ax.set_yticks([])
ax.set_ylabel("")
ax.set_xlabel("")

plt.xticks(
    rotation=20,
    ha='right',
    fontweight='bold',
    rotation_mode="anchor",
    fontsize=27
)

plt.subplots_adjust(bottom=0.25)

out_path = os.path.join(
    OUTPUT_ROOT,
    f"STACKED_MOSAIC_LATO_by_{scheme_col}_top{TOP_PERCENT}.png"
)
plt.savefig(out_path, dpi=600, bbox_inches="tight")
plt.close(fig)

print(f"[DONE] Saved Lato mosaic plot: {out_path}")

# ============================================================
# Pairwise-domain correlations with cortical functional gradients
# ============================================================
# ============================================================
# Analysis paths.
PEARSON_CTX_DIR = str(PATHS.output_root / "group_level_gradients/top10_union_pearson")
GRADS_CTX_PATH = os.path.join(PEARSON_CTX_DIR, "cortex_intrinsic_gradients.npy")

KEPT_SORTED_PATH = os.path.join(PEARSON_CTX_DIR, "kept_sorted.npy")

MAP_DIR = str(PATHS.output_root / "group_level/top10_hippamyg")

DOM_PEARSON_PATH  = os.path.join(MAP_DIR, "hippamyg_dominance_presence_full360_pearson_top10.npy")
SHR_PEARSON_PATH  = os.path.join(MAP_DIR, "hippamyg_shared_presence_full360_pearson_top10.npy")
DOM_GLASSO_PATH   = os.path.join(MAP_DIR, "hippamyg_dominance_presence_full360_glasso_top10.npy")
SHR_GLASSO_PATH   = os.path.join(MAP_DIR, "hippamyg_shared_presence_full360_glasso_top10.npy")

OUT_DIR = os.path.join(MAP_DIR, "method_compare_ctxgrad_corrmats")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_PNG = os.path.join(OUT_DIR, f"corr_dom_shared_vs_ctxG1-3_top{TOP_PERCENT}_MSR_FDR_pairwiseMask.png")
OUT_PDF = os.path.join(OUT_DIR, f"corr_dom_shared_vs_ctxG1-3_top{TOP_PERCENT}_MSR_FDR_pairwiseMask.pdf")

# Correlation type for MSR + empirical r
CORR_KIND = "kendall"   # "pearson", "spearman", or "kendall"

# MSR parameters
MSR_N_PERM = 1000
MSR_N_PROC = 8
MSR_SEED = 42

# Plot style
DPI = 600
LINEWIDTHS = 3
ANNOT_SIZE = 28

# ============================================================
def load_1d(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file:\n  {path}")
    arr = np.load(path)
    arr = np.asarray(arr).squeeze()
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D array in {path}, got shape {arr.shape}")
    return arr.astype(float)

def load_int_1d(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing retained-cortex index file:\n  {path}\n"
            "Run compute_cortical_gradients.py before the group analyses."
        )
    arr = np.load(path)
    arr = np.asarray(arr).squeeze()
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D int array in {path}, got shape {arr.shape}")
    return arr.astype(int)

def expand_ctx_grad_to_full360(grads_ctx: np.ndarray, kept_sorted: np.ndarray, grad_index: int) -> np.ndarray:
    if grads_ctx.ndim != 2:
        raise ValueError(f"Expected grads_ctx to be 2D (N_kept, n_components), got {grads_ctx.shape}")
    if grad_index < 0 or grad_index >= grads_ctx.shape[1]:
        raise ValueError(f"grad_index={grad_index} out of range for grads_ctx with {grads_ctx.shape[1]} comps")
    if kept_sorted.ndim != 1:
        raise ValueError("kept_sorted must be 1D")
    if len(kept_sorted) != grads_ctx.shape[0]:
        raise ValueError(
            f"Length mismatch: len(kept_sorted)={len(kept_sorted)} but grads_ctx.shape[0]={grads_ctx.shape[0]}"
        )
    full = np.full(360, np.nan, float)
    full[kept_sorted] = grads_ctx[:, grad_index]
    return full

def _corr(x: np.ndarray, y: np.ndarray, kind: str):
    """Empirical correlation on already-masked vectors."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan, np.nan, int(m.sum())

    if kind == "pearson":
        r, p = pearsonr(x[m], y[m])
    elif kind == "spearman":
        r, p = spearmanr(x[m], y[m])
    elif kind == "kendall":
        r, p = kendalltau(x[m], y[m])
    else:
        raise ValueError("kind must be one of: 'pearson', 'spearman', 'kendall'")
    return float(r), float(p), int(m.sum())

# ============================================================
# Input arrays.
if not os.path.exists(GRADS_CTX_PATH):
    raise FileNotFoundError(
        f"Missing cortex gradients:\n  {GRADS_CTX_PATH}\n"
        "Run compute_cortical_gradients.py before this analysis."
    )

grads_ctx = np.load(GRADS_CTX_PATH)  # (N_kept, n_components)
kept_sorted = load_int_1d(KEPT_SORTED_PATH)

ctx_g1 = expand_ctx_grad_to_full360(grads_ctx, kept_sorted, grad_index=0)
ctx_g2 = expand_ctx_grad_to_full360(grads_ctx, kept_sorted, grad_index=1)
ctx_g3 = expand_ctx_grad_to_full360(grads_ctx, kept_sorted, grad_index=2)

dom_p = load_1d(DOM_PEARSON_PATH)
shr_p = load_1d(SHR_PEARSON_PATH)
dom_g = load_1d(DOM_GLASSO_PATH)
shr_g = load_1d(SHR_GLASSO_PATH)

for name, arr in [
    ("dom_p", dom_p), ("shr_p", shr_p), ("dom_g", dom_g), ("shr_g", shr_g),
    ("ctx_g1", ctx_g1), ("ctx_g2", ctx_g2), ("ctx_g3", ctx_g3)
]:
    if arr.shape[0] != 360:
        raise ValueError(f"{name} expected shape (360,), got {arr.shape}")

# ============================================================
# Target and reference matrices.
target_data = np.column_stack([dom_p, shr_p, dom_g, shr_g])   # (360, 4)
ref_data    = np.column_stack([ctx_g1, ctx_g2, ctx_g3])       # (360, 3)

row_labels = ["Dom\nPe", "Shared\nPe", "Dom\nGL", "Shared\nGL"]
col_labels = ["CrtFC\nG1", "CrtFC\nG2", "CrtFC\nG3"]

user_mask = np.ones(360, dtype=bool)

# ============================================================
# Pairwise empirical correlations and MSR p-values.
R = np.full((target_data.shape[1], ref_data.shape[1]), np.nan, dtype=float)
P_msr = np.full_like(R, np.nan, dtype=float)
N_used = np.zeros_like(R, dtype=int)

for i in range(target_data.shape[1]):
    for j in range(ref_data.shape[1]):
        x = target_data[:, i].astype(float, copy=False)
        y = ref_data[:, j].astype(float, copy=False)

        # Pairwise-valid cortical domain.
        pair_mask = user_mask & np.isfinite(x) & np.isfinite(y)
        n_pair = int(pair_mask.sum())
        N_used[i, j] = n_pair

        # Empirical correlation on the pairwise-valid domain
        r_emp, _, _ = _corr(x[pair_mask], y[pair_mask], CORR_KIND)
        R[i, j] = r_emp

        if n_pair < 3:
            P_msr[i, j] = np.nan
            print(f"[PAIR {i+1},{j+1}] n={n_pair} < 3 -> r=NA, p_msr=NA")
            continue

        # Generate MSR nulls on the same pairwise-valid domain.
        r_mat, p_mat = compute_msr_statistics(
            target_data=x[:, None],          # (360, 1)
            ref_data=y[:, None],             # (360, 1)
            mask_idx=pair_mask,              # pairwise-valid mask only
            target_idxs=[0],
            ref_idxs=[0],
            dist_l_path=MSR_PARAMS["dist_l_path"],
            dist_r_path=MSR_PARAMS["dist_r_path"],
            parc_l_path=MSR_PARAMS["parc_l_path"],
            parc_r_path=MSR_PARAMS["parc_r_path"],
            n_proc=MSR_N_PROC,
            n_perm=MSR_N_PERM,
            seed=MSR_SEED + 1000 * i + 10 * j,
            metric=CORR_KIND,
        )

        # Keep the empirical r from the explicit pairwise calculation above.
        # p-value comes from MSR on the same parcel domain.
        P_msr[i, j] = p_mat[0, 0]

# ============================================================
# FDR across the 4 × 3 test family.
Q_fdr = np.full_like(P_msr, np.nan, dtype=float)
valid_p = np.isfinite(P_msr)

if valid_p.any():
    _, qvals = fdrcorrection(P_msr[valid_p].ravel(), alpha=0.05, method="indep")
    Q_fdr[valid_p] = qvals
else:
    print("[WARN] No finite MSR p-values found; FDR skipped.")

# Annotation = empirical r + FDR stars
annot = np.empty_like(R, dtype=object)
for i in range(R.shape[0]):
    for j in range(R.shape[1]):
        if not np.isfinite(R[i, j]):
            annot[i, j] = ""
        else:
            q = Q_fdr[i, j]
            if not np.isfinite(q):
                stars = ""
            elif q < 1e-3:
                stars = "***"
            elif q < 1e-2:
                stars = "**"
            elif q < 5e-2:
                stars = "*"
            else:
                stars = ""
            annot[i, j] = f"{R[i, j]:.2f}{stars}"

# Console summary
print("\n[PAIRWISE MASKS] Number of parcels used per cell:")
print(N_used)

print("\n[PAIRWISE MASKS] Empirical correlation matrix:")
print(np.array2string(R, precision=3, suppress_small=False))

print("\n[PAIRWISE MASKS] Raw MSR p-values:")
print(np.array2string(P_msr, precision=4, suppress_small=False))

print("\n[PAIRWISE MASKS] FDR-corrected q-values:")
print(np.array2string(Q_fdr, precision=4, suppress_small=False))

# ============================================================
# Correlation heatmap.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": PREFERRED_FONT,
    "axes.edgecolor": "black",
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

sns.set_context("talk", font_scale=2)
plt.style.use("seaborn-v0_8-white")

fig_w = max(3 * 2.2 + 2, 7.0)
fig_h = max(4 * 2.75 + 2, 6.0)

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
sns.heatmap(
    R,
    annot=annot,
    fmt="",
    cmap="RdBu_r",
    center=0,
    vmin=-1,
    vmax=1,
    square=False,
    linewidths=LINEWIDTHS,
    linecolor="white",
    xticklabels=col_labels,
    yticklabels=row_labels,
    ax=ax,
    annot_kws={"weight": "bold", "size": ANNOT_SIZE},
    cbar_kws={"label": f"{CORR_KIND.capitalize()}", "shrink": 0.9, "ticks": [-1, -0.5, 0, 0.5, 1]},
)

ax.xaxis.tick_top()
ax.xaxis.set_label_position("top")
plt.xticks(rotation=0)
plt.yticks(rotation=90, va="center")

plt.tight_layout()
fig.savefig(OUT_PNG, dpi=DPI, bbox_inches="tight")
fig.savefig(OUT_PDF, dpi=DPI, bbox_inches="tight")
plt.close(fig)

print("\nSaved:")
print("  ", OUT_PNG)
print("  ", OUT_PDF)
print("\nAnnotation convention: value = empirical correlation on pairwise-valid parcels; stars = FDR-corrected MSR significance.")

# ---------------------------------------------------------------------
# TSV exports
# ---------------------------------------------------------------------
EXPORT_DISPLAY_ONLY_MATRICES = False  # row-scaled / display-masked matrices are not exported

print("[EXPORT] Running final comprehensive TSV exports...")

# -------------------------
# Helpers
# -------------------------
def _base_glasser_table(glasser_tsv_path: str) -> pd.DataFrame:
    idx360 = pd.Index(np.arange(1, 361), name="index")
    if os.path.exists(glasser_tsv_path):
        df = pd.read_csv(glasser_tsv_path, sep="\t")
        if "index" not in df.columns:
            raise ValueError(f"'index' column not found in {glasser_tsv_path}")
        df["index"] = pd.to_numeric(df["index"], errors="coerce")
        df = df.dropna(subset=["index"]).copy()
        df["index"] = df["index"].astype(int)
        df = df.set_index("index", drop=False).sort_index()
        # Ensure all 360 Glasser parcels are represented.
        if not np.array_equal(df.index.to_numpy(), np.arange(1, 361)):
            df = df.reindex(idx360)
            if "index" not in df.columns:
                df["index"] = df.index
            else:
                df["index"] = df.index
        return df
    else:
        return pd.DataFrame({"index": np.arange(1, 361)}, index=idx360)

def _full360_numeric(arr, kept_1based=None, full_len=360):
    if arr is None:
        return None
    x = np.asarray(arr).squeeze()
    if x.ndim != 1:
        warnings.warn(f"[EXPORT] Expected 1D array, got shape {x.shape}; skipping.")
        return None

    idx = pd.Index(np.arange(1, full_len + 1), name="index")

    if x.shape[0] == full_len:
        return pd.Series(x.astype(float), index=idx)

    if kept_1based is not None and x.shape[0] == len(kept_1based):
        out = np.full(full_len, np.nan, dtype=float)
        out[np.asarray(kept_1based, int) - 1] = x.astype(float)
        return pd.Series(out, index=idx)

    warnings.warn(
        f"[EXPORT] Could not expand numeric vector of length {x.shape[0]} to full {full_len}; skipping."
    )
    return None

def _full360_object(vals, kept_1based=None, full_len=360):
    if vals is None:
        return None
    vals = list(vals)
    idx = pd.Index(np.arange(1, full_len + 1), name="index")

    if len(vals) == full_len:
        return pd.Series(vals, index=idx, dtype="object")

    if kept_1based is not None and len(vals) == len(kept_1based):
        out = np.empty(full_len, dtype=object)
        out[:] = pd.NA
        out[np.asarray(kept_1based, int) - 1] = vals
        return pd.Series(out, index=idx, dtype="object")

    warnings.warn(
        f"[EXPORT] Could not expand object vector of length {len(vals)} to full {full_len}; skipping."
    )
    return None

def _add_full360_numeric(df: pd.DataFrame, colname: str, arr, kept_1based=None):
    s = _full360_numeric(arr, kept_1based=kept_1based, full_len=360)
    if s is not None:
        df[colname] = s.reindex(df.index)

def _add_full360_object(df: pd.DataFrame, colname: str, vals, kept_1based=None):
    s = _full360_object(vals, kept_1based=kept_1based, full_len=360)
    if s is not None:
        df[colname] = s.reindex(df.index)

def _connected_seed_strings(mask_seedwise: np.ndarray, seed_labels: list[str]) -> list[str]:
    out = []
    for k in range(mask_seedwise.shape[1]):
        idx = np.where(mask_seedwise[:, k])[0]
        names = [seed_labels[i] for i in idx]
        out.append("; ".join(names))
    return out

def add_seed_column(df, colname, arr, expected_len):
    x = np.asarray(arr, dtype=float).squeeze()
    if x.shape != (expected_len,):
        raise ValueError(f"{colname}: expected ({expected_len},), got {x.shape}")
    df[colname] = x

# -------------------------
# Seed labels / metadata
# -------------------------
all_seed_labels = amyg_labels_L + amyg_labels_R + hipp_labels_L + hipp_labels_R
if len(all_seed_labels) != n_seeds:
    raise RuntimeError(
        f"[EXPORT] len(all_seed_labels)={len(all_seed_labels)} != n_seeds={n_seeds}"
    )

# -------------------------
# Connected-seed labels for the export table
# -------------------------
seeds_gl_str_list = _connected_seed_strings(mask_glasso_seedwise, all_seed_labels)
seeds_pe_str_list = _connected_seed_strings(mask_pearson_seedwise, all_seed_labels)

# -------------------------
# 1) FINAL PARCEL-WISE TABLE
# -------------------------
parcel_df = _base_glasser_table(GLASSER_TSV_PATH)

# Primary parcel-wise metrics
_add_full360_numeric(parcel_df, "Dominance_GLASSO", D_pres_ctx_gl, kept_1based=kept_glp)
_add_full360_numeric(parcel_df, "Sharedness_GLASSO", S_pres_ctx_gl, kept_1based=kept_glp)
_add_full360_numeric(parcel_df, "Dominance_Pearson", D_pres_ctx_pe, kept_1based=kept_glp)
_add_full360_numeric(parcel_df, "Sharedness_Pearson", S_pres_ctx_pe, kept_1based=kept_glp)

# Family-wise counts per parcel
_add_full360_numeric(parcel_df, "Amyg_Count_GLASSO", amyg_count_ctx_gl, kept_1based=kept_glp)
_add_full360_numeric(parcel_df, "Hipp_Count_GLASSO", hipp_count_ctx_gl, kept_1based=kept_glp)
_add_full360_numeric(parcel_df, "Amyg_Count_Pearson", amyg_count_ctx_pe, kept_1based=kept_glp)
_add_full360_numeric(parcel_df, "Hipp_Count_Pearson", hipp_count_ctx_pe, kept_1based=kept_glp)


# Connected seed strings and simple union booleans
_add_full360_object(parcel_df, "Connected_Seeds_GLASSO", seeds_gl_str_list, kept_1based=kept_glp)
_add_full360_object(parcel_df, "Connected_Seeds_Pearson", seeds_pe_str_list, kept_1based=kept_glp)

_add_full360_numeric(parcel_df, "In_Union_Mask_GLASSO", np.any(mask_glasso_seedwise, axis=0).astype(float), kept_1based=kept_glp)
_add_full360_numeric(parcel_df, "In_Union_Mask_Pearson", np.any(mask_pearson_seedwise, axis=0).astype(float), kept_1based=kept_glp)

# FIG2 strength corroboration parcel-wise results
_add_full360_numeric(parcel_df, "StrengthDiff_AmygMinusHipp_GLASSO", net_gl_ctx, kept_1based=kept_glp)
_add_full360_numeric(parcel_df, "StrengthDiff_AmygMinusHipp_Pearson", net_pe_ctx, kept_1based=kept_glp)
_add_full360_numeric(parcel_df, "StrengthDominance_GLASSO", sdom_gl_ctx, kept_1based=kept_glp)
_add_full360_numeric(parcel_df, "StrengthDominance_Pearson", sdom_pe_ctx, kept_1based=kept_glp)
_add_full360_numeric(parcel_df, "StrengthOverlap_GLASSO", overlap_gl_ctx.astype(float), kept_1based=kept_glp)
_add_full360_numeric(parcel_df, "StrengthOverlap_Pearson", overlap_pe_ctx.astype(float), kept_1based=kept_glp)
_add_full360_numeric(parcel_df, "StrengthUnion_GLASSO", union_gl_ctx.astype(float), kept_1based=kept_glp)
_add_full360_numeric(parcel_df, "StrengthUnion_Pearson", union_pe_ctx.astype(float), kept_1based=kept_glp)

# Bottom-panel categorical map: neither / GLASSO only / Pearson only / overlap
_add_full360_numeric(parcel_df, "MethodOverlap_Code", cat_ctx, kept_1based=kept_glp)

cat_label_ctx = np.empty(len(cat_ctx), dtype=object)
cat_label_ctx[:] = pd.NA
code_to_label = {
    0.0: "Neither",
    1.0: "GLASSO only",
    2.0: "Pearson only",
    3.0: "Overlap",
}
for code, label in code_to_label.items():
    cat_label_ctx[np.asarray(cat_ctx) == code] = label
_add_full360_object(parcel_df, "MethodOverlap_Label", cat_label_ctx, kept_1based=kept_glp)

# Cortical intrinsic gradients.
_add_full360_numeric(parcel_df, "CtxIntrinsic_Gradient1", ctx_g1, kept_1based=None)
_add_full360_numeric(parcel_df, "CtxIntrinsic_Gradient2", ctx_g2, kept_1based=None)
_add_full360_numeric(parcel_df, "CtxIntrinsic_Gradient3", ctx_g3, kept_1based=None)

# Useful bookkeeping
parcel_df["Dropped_Glasser_Parcel"] = parcel_df["index"].isin(GLASSER_DROP_PARCELS).astype(int)
parcel_df["Kept_In_Primary_Cortex_Set"] = (~parcel_df["Dropped_Glasser_Parcel"].astype(bool)).astype(int)

parcel_out_tsv = os.path.join(
    OUTPUT_ROOT,
    f"glasser360_hippamyg_metrics_top{TOP_PERCENT}.tsv"
)
parcel_df.to_csv(parcel_out_tsv, sep="\t", index=False, float_format="%.6f")
print(f"[EXPORT] Saved parcel-wise TSV: {parcel_out_tsv}")

# -------------------------
# 2) FINAL SEED-WISE TABLE
# -------------------------
seed_df = pd.DataFrame({
    "seed_row": np.arange(n_seeds, dtype=int),
    "seed_label": all_seed_labels,
    "structure": (["Amygdala"] * (len(amyg_labels_L) + len(amyg_labels_R))) + (["Hippocampus"] * (len(hipp_labels_L) + len(hipp_labels_R))),
    "hemisphere": (["Left"] * len(amyg_labels_L)) + (["Right"] * len(amyg_labels_R)) + (["Left"] * len(hipp_labels_L)) + (["Right"] * len(hipp_labels_R)),
})

# Amygdala broad group metadata
amyg_group_L = [""] * len(amyg_labels_L)
for grp_name, grp_idx in amygdala_broad_group_indices().items():
    for ii in grp_idx:
        if 0 <= ii < len(amyg_group_L):
            amyg_group_L[ii] = grp_name
amyg_group_all = amyg_group_L + amyg_group_L + ([""] * (len(hipp_labels_L) + len(hipp_labels_R)))
seed_df["amyg_broad_group"] = amyg_group_all

# Hipp AP bin metadata
hipp_ap_all = ([np.nan] * (len(amyg_labels_L) + len(amyg_labels_R))) + list(hipp_ap_bins_L.astype(float)) + list(hipp_ap_bins_R.astype(float))
seed_df["hipp_ap_bin"] = hipp_ap_all

# Seed mask sizes
add_seed_column(seed_df, "TopK_Count_GLASSO", mask_glasso_seedwise.sum(axis=1).astype(float), n_seeds)
add_seed_column(seed_df, "TopK_Count_Pearson", mask_pearson_seedwise.sum(axis=1).astype(float), n_seeds)

# Mean positive seed->cortex strength within each method's own mask
add_seed_column(
    seed_df,
    "CtxMeanPosStrength_GLASSO",
    _mean_pos_seed2ctx_strength(glasso_seed2ctx, mask_glasso_seedwise, positive_only=True),
    n_seeds,
)
add_seed_column(
    seed_df,
    "CtxMeanPosStrength_Pearson",
    _mean_pos_seed2ctx_strength(pearson_seed2ctx, mask_pearson_seedwise, positive_only=True),
    n_seeds,
)

# FIG3 seed scores
add_seed_column(seed_df, "HubnessToSharedness_Pearson_LOSO", hub_pe_loso, n_seeds)
add_seed_column(seed_df, "HubnessToSharedness_GLASSO_LOSO", hub_gl_loso, n_seeds)
add_seed_column(seed_df, "PreferenceToDominance_Pearson_LOSO", pref_pe_loso, n_seeds)
add_seed_column(seed_df, "PreferenceToDominance_GLASSO_LOSO", pref_gl_loso, n_seeds)

# Intrinsic AH expected positive FC per seed
am_tmp, hi_tmp = _intrinsic_bars_from_AH(AH_pe_full, sign_policy="pos")
add_seed_column(seed_df, "Intrinsic_AH_ExpectedPosFC_Pearson", np.r_[am_tmp, hi_tmp], n_seeds)
am_tmp, hi_tmp = _intrinsic_bars_from_AH(AH_gl_full, sign_policy="pos")
add_seed_column(seed_df, "Intrinsic_AH_ExpectedPosFC_GLASSO", np.r_[am_tmp, hi_tmp], n_seeds)

# GLASSO AH prevalence summaries per seed
add_seed_column(seed_df, "Intrinsic_AH_AnyPrev_GLASSO", np.r_[AH_any_prev.mean(axis=1), AH_any_prev.mean(axis=0)], n_seeds)
add_seed_column(seed_df, "Intrinsic_AH_PosPrev_GLASSO", np.r_[AH_pos_prev.mean(axis=1), AH_pos_prev.mean(axis=0)], n_seeds)

seed_out_tsv = os.path.join(
    OUTPUT_ROOT,
    f"seedwise_hippamyg_metrics_top{TOP_PERCENT}.tsv"
)
seed_df.to_csv(seed_out_tsv, sep="\t", index=False, float_format="%.6f")
print(f"[EXPORT] Saved seed-wise TSV: {seed_out_tsv}")

# -------------------------
# 3) AMYGDALA x HIPPOCAMPUS PAIRWISE TABLE
# -------------------------
amyg_pair_labels = amyg_labels_L + amyg_labels_R
hipp_pair_labels = hipp_labels_L + hipp_labels_R
pair_rows = []
for i, a_lbl in enumerate(amyg_pair_labels):
    for j, h_lbl in enumerate(hipp_pair_labels):
        row = {
            "amyg_row": i,
            "hipp_col": j,
            "amyg_seed": a_lbl,
            "hipp_seed": h_lbl,
            "Pearson_FC": float(AH_pe_full[i, j]),
            "GLASSO_FC": float(AH_gl_full[i, j]),
        }
        if "AH_any_prev" in globals():
            row["GLASSO_AnyPrev"] = float(AH_any_prev[i, j])
        if "AH_pos_prev" in globals():
            row["GLASSO_PosPrev"] = float(AH_pos_prev[i, j])

        if EXPORT_DISPLAY_ONLY_MATRICES and "AH_gl_full_plot" in globals():
            row["GLASSO_FC_DisplayMasked"] = float(AH_gl_full_plot[i, j]) if np.isfinite(AH_gl_full_plot[i, j]) else np.nan

        pair_rows.append(row)

pair_df = pd.DataFrame(pair_rows)
pair_out_tsv = os.path.join(
    OUTPUT_ROOT,
    f"amygdala_x_hippocampus_pairwise_metrics_top{TOP_PERCENT}.tsv"
)
pair_df.to_csv(pair_out_tsv, sep="\t", index=False, float_format="%.6f")
print(f"[EXPORT] Saved pairwise AH TSV: {pair_out_tsv}")

# -------------------------
# 4) MAP-vs-GRADIENT CORRELATION SUMMARY TABLE
# -------------------------
corr_rows = []
for i, rlab in enumerate(row_labels):
    for j, clab in enumerate(col_labels):
        corr_rows.append({
            "target_map": str(rlab).replace("\n", "_"),
            "reference_gradient": str(clab).replace("\n", "_"),
            "corr_kind": CORR_KIND,
            "r_empirical": float(R[i, j]) if np.isfinite(R[i, j]) else np.nan,
            "p_msr": float(P_msr[i, j]) if np.isfinite(P_msr[i, j]) else np.nan,
            "q_fdr": float(Q_fdr[i, j]) if np.isfinite(Q_fdr[i, j]) else np.nan,
            "n_used": int(N_used[i, j]),
        })
corr_df = pd.DataFrame(corr_rows)
corr_out_tsv = os.path.join(
    OUTPUT_ROOT,
    f"map_vs_ctxgradient_correlations_top{TOP_PERCENT}.tsv"
)
corr_df.to_csv(corr_out_tsv, sep="\t", index=False, float_format="%.6f")
print(f"[EXPORT] Saved map-vs-gradient TSV: {corr_out_tsv}")

# -------------------------
# 5) FIG3D SCATTER STATS TABLES
# -------------------------
def _build_fig3d_stats(metric_name: str, pearson_score: np.ndarray, glasso_score: np.ndarray) -> pd.DataFrame:
    rows = []
    n_am = len(amyg_rows_all)
    n_hi = len(hipp_rows_all)

    for method_name, AH, score, seed2ctx, mask_seedwise in [
        ("Pearson", AH_pe_full, pearson_score, pearson_seed2ctx, mask_pearson_seedwise),
        ("GLASSO",  AH_gl_full, glasso_score, glasso_seed2ctx, mask_glasso_seedwise),
    ]:
        am_x, hi_x = _intrinsic_bars_from_AH(AH, sign_policy="pos")
        ctrl = _mean_pos_seed2ctx_strength(seed2ctx, mask_seedwise, positive_only=True)

        specs = [
            ("Amygdala", am_x, np.asarray(score[:n_am], float), np.asarray(ctrl[:n_am], float)),
            ("Hippocampus", hi_x, np.asarray(score[n_am:n_am+n_hi], float), np.asarray(ctrl[n_am:n_am+n_hi], float)),
        ]

        for structure, x, y, c in specs:
            m = np.isfinite(x) & np.isfinite(y)
            n0 = int(m.sum())
            if n0 >= 4:
                r0, p0 = spearmanr(x[m], y[m])
            else:
                r0, p0 = (np.nan, np.nan)

            rp, pp, npart = _partial_spearman(x, y, c)

            rows.append({
                "metric": metric_name,
                "score_mode": FIG3D_SCORE_MODE,
                "method": method_name,
                "structure": structure,
                "n_zero_order": n0,
                "spearman_r": float(r0) if np.isfinite(r0) else np.nan,
                "spearman_p": float(p0) if np.isfinite(p0) else np.nan,
                "n_partial": int(npart),
                "partial_spearman_r": float(rp) if np.isfinite(rp) else np.nan,
                "partial_spearman_p": float(pp) if np.isfinite(pp) else np.nan,
            })

    out = pd.DataFrame(rows)

    # FDR over zero-order p's and partial p's separately across all rows in this table
    out["spearman_q_fdr"] = _safe_fdrcorr(out["spearman_p"].to_numpy(float))
    out["partial_spearman_q_fdr"] = _safe_fdrcorr(out["partial_spearman_p"].to_numpy(float))
    return out


fig3d_stats = pd.concat([
    _build_fig3d_stats("Sharedness", hub_pe_fig3d, hub_gl_fig3d),
    _build_fig3d_stats("Preference", pref_pe_fig3d, pref_gl_fig3d),
], axis=0, ignore_index=True)

fig3d_out_tsv = os.path.join(
    OUTPUT_ROOT,
    f"fig3d_intrinsic_vs_extrinsic_stats_{FIG3D_SCORE_MODE}_top{TOP_PERCENT}.tsv"
)
fig3d_stats.to_csv(fig3d_out_tsv, sep="\t", index=False, float_format="%.6f")
print(f"[EXPORT] Saved FIG3D stats TSV: {fig3d_out_tsv}")

# -------------------------
# 6) Community category counts / percentages from the mosaic plot block
# -------------------------
ct_out = ct.reset_index()
ct_counts_tsv = os.path.join(
    OUTPUT_ROOT,
    f"union_category_counts_by_community_top{TOP_PERCENT}.tsv"
)
ct_out.to_csv(ct_counts_tsv, sep="\t", index=False)
print(f"[EXPORT] Saved category-count TSV: {ct_counts_tsv}")

ct_pct_out = ct_pct.reset_index()
ct_pct_tsv = os.path.join(
    OUTPUT_ROOT,
    f"union_category_percent_by_community_top{TOP_PERCENT}.tsv"
)
ct_pct_out.to_csv(ct_pct_tsv, sep="\t", index=False, float_format="%.6f")
print(f"[EXPORT] Saved category-percent TSV: {ct_pct_tsv}")

print("[EXPORT] Final comprehensive TSV exports finished.")

# ============================================================
# Split-half reliability
# ============================================================
run_split_half_reliability(
    top_percent=TOP_PERCENT,
    output_root=OUTPUT_ROOT,
    subjects=subjects,
    base_dir=BASE_DIR,
    n_total=N_TOTAL,
    seed_rows=seed_rows,
    ctx_indices=ctx_indices,
    pearson_z_stack=pearson_z_stack,
    glasso_z_stack=glasso_z_stack,
    amyg_rows_all=amyg_rows_all,
    hipp_rows_all=hipp_rows_all,
    kept_glp=kept_glp,
    n_ctx=n_ctx,
    amyg_L_idx=amyg_L_idx,
    amyg_R_idx=amyg_R_idx,
    hipp_L_idx=hipp_L_idx,
    hipp_R_idx=hipp_R_idx,
    n_total_amyg=n_total_amyg,
    n_total_hipp=n_total_hipp,
)
