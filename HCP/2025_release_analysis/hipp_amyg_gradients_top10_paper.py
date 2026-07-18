#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute joint hippocampus–amygdala gradients for Pearson or GLASSO.

The script loads group seed-to-cortex connectivity, applies the GLASSO
consensus mask, aligns right-hemisphere diffusion gradients to the left, and
projects the gradients to Glasser cortex and subcortical surfaces.

Configuration is loaded from the environment after sourcing ``config.sh``.

Important assumptions: The script assumes the same fixed node order as the subject-level connectivity script. For GLASSO, zeros are treated as absent edges during group averaging, and positive support prevalence is used for top-10% union selection. For Pearson, positive group-average correlations are used for top-10% union selection. The top-10% union is used for cortical projection/mapping, while gradients are computed from the full hippocampus–amygdala to kept-cortex matrix.

Output: Group-level gradient arrays, eigenvalue summaries, top-10% cortical union indices, cortical gradient projection maps, hippocampal surface gradient plots, amygdala volumetric gradient plots, Yeo-7 summaries, MSR correspondence heatmaps, and TSV summaries.
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import nibabel as nb
import pandas as pd
import seaborn as sns
from statsmodels.stats.multitest import fdrcorrection
from brainspace.datasets import load_conte69
from brainspace.utils.parcellation import map_to_labels
from surfplot import Plot

# ===== Utils on path =====
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "utils"))
from project_config import ProjectPaths

PATHS = ProjectPaths.from_environment()
from hippamyg_labels import build_hippocampus_labels, get_amygdala_labels
from helper import (
    union_topk_cols,
    sanitize_removed_list,
    kept_sorted_from_removed,
    correlate_gradients_to_columns,
    map_top_to_full360,
    zavg_ignore,
    plot_eigen_spectrum,
    load_glasso_support_prevalence,
    compute_seedwise_topk_mask,
    consensus_mask_from_prevalence,
    run_gradients_hemi_align_right_to_left,
    compute_msr_statistics,
)


# =========================
# Analysis constants
# =========================
N_HIPP_PARCELS_PER_HEMI = 15          # hippocampus parcels per hemisphere
REMOVED_GLASSER_PARCELS = [120, 300]  # 1-based Glasser IDs to exclude
N_COMPONENTS = 10                     # gradient components
TOP_PERCENT = 10
# Group-level consensus: keep edges present in at least this % of subjects (GLASSO only).
# Set to 0 or None to disable consensus masking.
CONSENSUS_PERCENT = 10

# Select which connectivity to analyze: 'glasso' or 'pearson'
METHOD = os.environ.get("HIPPAMYG_METHOD", "pearson").lower()

# =========================
# 1) Subjects
# =========================
subject_list_file = str(PATHS.subject_list)
if not os.path.exists(subject_list_file):
    raise FileNotFoundError(f"Subject list not found: {subject_list_file}")

with open(subject_list_file, 'r') as f:
    subjects = [line.strip() for line in f if line.strip()]
if not subjects:
    print("No subjects found. Exiting.")
    sys.exit(1)
if len(set(subjects)) != len(subjects):
    raise RuntimeError(f"Duplicate subject IDs in {subject_list_file}")

# =========================
# 2) Paths and STRICT filename
# =========================
base_dir = str(PATHS.work_root)

removed0     = sanitize_removed_list(REMOVED_GLASSER_PARCELS)        # 0-based removed
kept_sorted  = kept_sorted_from_removed(REMOVED_GLASSER_PARCELS)     # 0-based kept

n_amyg_per_hemi = 9
n_hipp_per_hemi = int(N_HIPP_PARCELS_PER_HEMI)
n_subregions    = 2 * (n_amyg_per_hemi + n_hipp_per_hemi)
cortex_len_expected = len(kept_sorted)
T_expected      = n_subregions + cortex_len_expected

method_lower = METHOD.lower()
if method_lower == "glasso":
    mat_suffix = "glasso_z"
elif method_lower == "pearson":
    mat_suffix = "z"
else:
    raise ValueError("METHOD must be 'glasso' or 'pearson'.")

expected_filename = f"connectivity_matrix_REST_{T_expected}x{T_expected}_{mat_suffix}.npy"

print(f"METHOD: {METHOD}")
print(f"Expected subject matrix shape: {T_expected}×{T_expected} "
      f"(amyg+hipp={n_subregions}, cortex kept={cortex_len_expected})")
print(f"Strict filename: {expected_filename}")
print(f"Removed Glasser parcels (0-based): {removed0.tolist()}")

output_root = str(PATHS.output_root / "group_level_gradients")
output_dir = os.path.join(output_root, f"top{TOP_PERCENT}_union_{method_lower}")
os.makedirs(output_dir, exist_ok=True)

# =========================
# 3) Load subject matrices (z-space)
# =========================
all_mats = []

for subj in subjects:
    func_dir = os.path.join(base_dir, subj, "func_native_2025")
    mat_path = os.path.join(func_dir, expected_filename)
    if not os.path.exists(mat_path):
        raise FileNotFoundError(mat_path)
    z_mat = np.load(mat_path)

    if z_mat.ndim != 2 or z_mat.shape[0] != z_mat.shape[1] or z_mat.shape[0] != T_expected:
        raise ValueError(f"{subj}: got {z_mat.shape}, expected ({T_expected}, {T_expected})")

    all_mats.append(z_mat.astype(float, copy=False))

all_mats = np.stack(all_mats, axis=0)  # (n_subj_valid, T_expected, T_expected)
n_subj_valid, T, _ = all_mats.shape
print(f"Loaded {n_subj_valid} subjects; matrix size {T}×{T}.")

# Average across subjects in z-space, then back to r
# - GLASSO: drop_zeros=True (zeros are treated as "no edge")
# - PEARSON: drop_zeros=False (zeros are valid correlations)
drop_zeros_flag = (method_lower == "glasso")
conn = zavg_ignore(all_mats, drop_zeros=drop_zeros_flag, axis=0)
print(f"Group-average connectivity computed (drop_zeros={drop_zeros_flag}).")

# =========================
# 4) Indexing (order fixed) and hippamyg→cortex matrix
# =========================
# Order assumption:
# [L-Amyg(9), R-Amyg(9), L-Hipp(n_hipp), R-Hipp(n_hipp), Cortex(pruned in ascending original index)]
hipp_amyg_indices = np.arange(0, n_subregions, dtype=int)

cortex_start = n_subregions
if T - cortex_start != cortex_len_expected:
    print(f"Unexpected cortex block length {T - cortex_start} vs expected {cortex_len_expected}. Exiting.")
    sys.exit(1)

cortex_cols_abs = cortex_start + np.arange(cortex_len_expected, dtype=int)
hippamyg2cortex = conn[np.ix_(hipp_amyg_indices, cortex_cols_abs)]  # (2*(9+n_hipp) × cortex_len_expected)

print(f"hippamyg2cortex shape: {hippamyg2cortex.shape}")

# =========================
# 5) Top-10% union (relative to kept cortex columns)
#     - GLASSO: union of per-seed Top-10% GLASSO support prevalence (positive-only)
#     - PEARSON: union of per-seed Top-10% positive Pearson r
# =========================
if method_lower == "glasso":
    print("Computing GLASSO support prevalence (positive edges only) for Top-10% union...")
    prevalence_gl, kept_glp_1based = load_glasso_support_prevalence(
        subject_list_file=subject_list_file,
        base_dir=base_dir,
        n_amyg_per_hemi=n_amyg_per_hemi,
        n_hipp_per_hemi=n_hipp_per_hemi,
        removed_glasser=REMOVED_GLASSER_PARCELS,
        positive_only=True,
        verbose=False,
    )
    prevalence_gl = np.asarray(prevalence_gl, dtype=float)
    kept_glp_1based = np.asarray(kept_glp_1based, dtype=int)

    if prevalence_gl.shape != (n_subregions, cortex_len_expected):
        raise RuntimeError(
            f"Prevalence shape {prevalence_gl.shape} incompatible with "
            f"expected (n_seeds={n_subregions}, n_ctx={cortex_len_expected})."
        )

    kept_from_prev_0based = kept_glp_1based - 1
    if not np.array_equal(kept_sorted, kept_from_prev_0based):
        print("[WARN] kept_sorted from REMOVED_GLASSER_PARCELS does not exactly match "
              "kept_glp returned by load_glasso_support_prevalence.")

    # Consensus mask + diagnostics (GLASSO only)
    if CONSENSUS_PERCENT is not None and CONSENSUS_PERCENT > 0:
        consensus_fraction = CONSENSUS_PERCENT / 100.0
        consensus_mask = consensus_mask_from_prevalence(
            prevalence_gl,
            min_fraction=consensus_fraction,
        )

        if consensus_mask.shape != hippamyg2cortex.shape:
            raise RuntimeError(
                f"Consensus mask shape {consensus_mask.shape} does not match "
                f"hippamyg2cortex shape {hippamyg2cortex.shape}."
            )

        total_edges = int(hippamyg2cortex.size)
        surviving_edges = int(consensus_mask.sum())
        pct_surviving = 100.0 * surviving_edges / float(total_edges)

        print(f"Total edges in hippamyg2cortex:      {total_edges}")
        print(
            f"Surviving edges after ≥{CONSENSUS_PERCENT}% consensus: "
            f"{surviving_edges} ({pct_surviving:.2f}% of edges)"
        )

        # Apply mask: edges failing consensus are zeroed out
        hippamyg2cortex = np.where(consensus_mask, hippamyg2cortex, 0.0)

        # Diagnostics: per-seed surviving edge counts
        per_seed_counts = consensus_mask.sum(axis=1)
        print(
            "Per-seed surviving edges (after consensus): "
            f"min={per_seed_counts.min()}, "
            f"median={np.median(per_seed_counts):.1f}, "
            f"max={per_seed_counts.max()}"
        )

        # Diagnostics: per-seed std of connectivity after masking
        per_seed_std = hippamyg2cortex.std(axis=1)
        print(
            "Per-seed connectivity std (after consensus masking): "
            f"min={per_seed_std.min():.4f}, "
            f"median={np.median(per_seed_std):.4f}, "
            f"max={per_seed_std.max():.4f}"
        )

        if per_seed_counts.min() < 20:
            print(
                "[WARN] Some seeds have fewer than 20 surviving edges after consensus masking; "
                "consider lowering CONSENSUS_PERCENT or inspecting those seeds."
            )
    else:
        print("CONSENSUS_PERCENT is 0/None; skipping consensus masking for GLASSO.")

    # Union of per-row Top-k (by prevalence)
    top10_rel_in_matrix = union_topk_cols(
        prevalence_gl,
        top_percent=TOP_PERCENT,
        use_abs=False,
    )
    if top10_rel_in_matrix.size == 0:
        print("Top-10% GLASSO-prevalence union is empty; check data distribution. Exiting.")
        sys.exit(1)

elif method_lower == "pearson":
    print("Computing Top-10% union from group-average Pearson r (positive-only, per seed)...")
    # Per-seed top-10% mask restricted to positive correlations.
    mask_pearson_seedwise = compute_seedwise_topk_mask(
        hippamyg2cortex,
        top_percent=TOP_PERCENT,
        positive_only=True,
    )
    top10_rel_in_matrix = np.where(mask_pearson_seedwise.any(axis=0))[0]
    if top10_rel_in_matrix.size == 0:
        print("Top-10% union (Pearson, positive-only) is empty; check data distribution. Exiting.")
        sys.exit(1)
else:
    raise ValueError(f"Unknown METHOD '{METHOD}'. Use 'glasso' or 'pearson'.")

hippamyg2cortex_top = hippamyg2cortex[:, top10_rel_in_matrix]
# Map to original 0..359 Glasser indices
top10_indices = kept_sorted[top10_rel_in_matrix]
np.save(os.path.join(output_dir, "top10_indices.npy"), top10_indices)

print(f"Top-{TOP_PERCENT}% union size ({METHOD}): {hippamyg2cortex_top.shape[1]} "
      f"(of {hippamyg2cortex.shape[1]} cortex columns present)")

# =========================
# 6) Gradient analysis on FULL hippamyg→kept-cortex (NOT reduced)
#     (For GLASSO, this uses the consensus-masked matrix if enabled.)
# =========================
print(f"Performing group-level gradient analysis on full hippamyg→kept-cortex matrix ({METHOD})...")

# --- Hemisphere row indices (must match fixed row order) ---
# [L-Amyg(9), R-Amyg(9), L-Hipp(n_hipp), R-Hipp(n_hipp)]
n_amyg = n_amyg_per_hemi
n_hipp = n_hipp_per_hemi

left_rows = np.r_[np.arange(0, n_amyg),
                  np.arange(2*n_amyg, 2*n_amyg + n_hipp)].astype(int)

right_rows = np.r_[np.arange(n_amyg, 2*n_amyg),
                   np.arange(2*n_amyg + n_hipp, 2*n_amyg + 2*n_hipp)].astype(int)

print(f"Hemi alignment (Option A): left_rows={left_rows.size}, right_rows={right_rows.size}")


gradients_top, lambdas_top, normalized_eigenvalues_top, hemi_meta = run_gradients_hemi_align_right_to_left(
    hippamyg2cortex,
    left_rows=left_rows,
    right_rows=right_rows,
    n_components=N_COMPONENTS,
    kernel="normalized_angle",
    random_state=0,
    procrustes_center=True,
    procrustes_scale=True,
)

# Optional QC saves
np.save(os.path.join(output_dir, "hemi_gradients_left.npy"), hemi_meta["gradients_left"])
np.save(os.path.join(output_dir, "hemi_gradients_right_aligned.npy"), hemi_meta["gradients_right_aligned"])
np.save(os.path.join(output_dir, "hemi_lambdas_left.npy"), hemi_meta["lambdas_left"])
np.save(os.path.join(output_dir, "hemi_lambdas_right.npy"), hemi_meta["lambdas_right"])

# =========================
# 7) Plot eigenvalue spectrum
# =========================
fig, ax = plt.subplots(1, figsize=(4.4, 6))
plot_eigen_spectrum(
    normalized_eigenvalues_top,
    ax=ax,
    title=f'Group-level Eigenvalue Spectrum ({METHOD}, Top-10% mask used only for mapping)',
)
plt.tight_layout()
eigenvalues_plot_path = os.path.join(output_dir, "group_normalized_eigenvalues_top10_union.png")
plt.savefig(eigenvalues_plot_path)
plt.close(fig)

# Save gradients & eigenvalues
np.save(os.path.join(output_dir, "group_gradients_top10_union.npy"), gradients_top)
np.save(os.path.join(output_dir, "group_eigenvalues_top10_union.npy"), lambdas_top)

print("Gradient analysis complete.")
print(f"Gradients/eigenvalues saved to {output_dir}")
print(f"Eigenvalue plot saved as {eigenvalues_plot_path}")

# =========================================================
# 7b) Plot eigenvalue spectrum (Left & Right Split)
# =========================================================

# Hemisphere-specific eigenvalues.
lambdas_L = hemi_meta["lambdas_left"]
lambdas_R = hemi_meta["lambdas_right"]

norm_evals_L = lambdas_L / np.sum(lambdas_L)
norm_evals_R = lambdas_R / np.sum(lambdas_R)

fig, ax = plt.subplots(1, figsize=(5, 5))

ax.plot(range(1, len(norm_evals_L) + 1), norm_evals_L, 
        marker='o', color='#1f77b4', label='Left Hemi (Reference)')

ax.plot(range(1, len(norm_evals_R) + 1), norm_evals_R, 
        marker='s', color='#ff7f0e', label='Right Hemi (Aligned)')

ax.set_title(f'Variance Explained by Hemisphere\n({METHOD.capitalize()})', fontsize=14, fontweight='bold')
ax.set_xlabel('Gradient Component', fontsize=12)
ax.set_ylabel('Normalized Eigenvalue (Variance Explained)', fontsize=12)
ax.set_xticks(range(1, max(len(norm_evals_L), len(norm_evals_R)) + 1))
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(frameon=False, fontsize=11)

sns.despine(ax=ax)
plt.tight_layout()

eigenvalues_plot_path = os.path.join(output_dir, "group_normalized_eigenvalues_split_hemi.png")
fig.savefig(eigenvalues_plot_path, dpi=300, bbox_inches='tight')
plt.close(fig)

# Save arrays used by downstream projection and reporting steps.
np.save(os.path.join(output_dir, "group_norm_evals_left.npy"), norm_evals_L)
np.save(os.path.join(output_dir, "group_norm_evals_right.npy"), norm_evals_R)

print(f"Split eigenvalue plot saved as {eigenvalues_plot_path}")

# =========================
# 8) Correlate gradients with cortex
# and
# 9) Map correlations back to full 360 array
# =========================
# - gradients_top was computed from the full hippamyg2cortex matrix above.
# - For GLASSO, hippamyg2cortex is already consensus-masked if CONSENSUS_PERCENT > 0.
# - The top-10% projection is retained as a diagnostic/supplementary output.
# - Figure 4 uses the full retained-cortex projection.

# --- Top-10% diagnostic projection ---
gradientr_fc_top10 = correlate_gradients_to_columns(
    gradients_top,
    hippamyg2cortex_top,
)

corr_top10_full360 = map_top_to_full360(
    gradientr_fc_top10,
    top10_indices,
    full_len=360,
    fill=np.nan,
)

np.save(
    os.path.join(output_dir, "group_gradients_fc_corr_top10union_full360.npy"),
    corr_top10_full360,
)

np.save(
    os.path.join(output_dir, "group_gradients_fc_corr_top10only.npy"),
    gradientr_fc_top10,
)


# --- Main full-cortex projection for Figure 4 and MSR/cortical-gradient comparison ---
gradientr_fc_full = correlate_gradients_to_columns(
    gradients_top,
    hippamyg2cortex,
)

corr_full_length = map_top_to_full360(
    gradientr_fc_full,
    kept_sorted,
    full_len=360,
    fill=np.nan,
)

np.save(
    os.path.join(output_dir, "group_gradients_fc_corr_full360.npy"),
    corr_full_length,
)

np.save(
    os.path.join(output_dir, "group_gradients_fc_corr_kept358_fullprojection.npy"),
    gradientr_fc_full,
)

# =========================
# 10) Surface mapping & plot (first 2 components)
# =========================
glasser_labeling_path = str(PATHS.resource_root / "glasser.csv")
labeling = np.genfromtxt(glasser_labeling_path, delimiter=',')
surf_lh, surf_rh = load_conte69()
mask = labeling != 0

n_comp = gradients_top.shape[1]
n_grad_to_plot = min(2, n_comp)  # Figure 4 uses G1 and G2

# Number of vertices in the left fsLR-32k surface.
n_pts_lh = surf_lh.n_points

grad_pngs = []

for i in range(n_grad_to_plot):
    # Map the cortical projection from parcels to surface vertices.
    mapped_vals = map_to_labels(
        corr_full_length[i, :],
        labeling,
        mask=mask,
        fill=np.nan
    )

    data_lr = {
        "left":  mapped_vals[:n_pts_lh],
        "right": mapped_vals[n_pts_lh:],
    }

    # Surface panels without labels or a color bar.
    p = Plot(
        surf_lh,
        surf_rh,
        layout="grid",            # rows = views, cols = hemis
        views=["lateral", "medial"],
        mirror_views=True,
        zoom=1.5,
        size=(500, 400),
    )

    p.add_layer(
        data_lr,
        cmap="viridis",
        cbar=False,
    )
    fig = p.build(
        figsize=(6, 4),
        colorbar=False,
    )

    out_png = os.path.join(
        output_dir,
        f"{method_lower}_gradientmap_hippamyg2cortex_fc_fullcortex_grad{i+1}_brains_only.png",
    )
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    grad_pngs.append(out_png)
    print(f"Saved brains-only surfplot for Grad {i+1}: {out_png}")

    




grad_files = list(grad_pngs)

imgs = [mpimg.imread(f) for f in grad_files]

fig, axes = plt.subplots(
    1, len(imgs),
    figsize=(len(imgs) * 4, 4),
    constrained_layout=True,
)

if len(imgs) == 1:
    axes = [axes]

for ax, img in zip(axes, imgs):
    ax.imshow(img)
    ax.axis("off")

combined_out = os.path.join(
    output_dir,
    f"{method_lower}_gradientmap_hippamyg2cortex_fc_fullcortex_gradients_columns.png",
)
fig.savefig(combined_out, dpi=600, bbox_inches="tight")
plt.close(fig)

print(f"Saved combined column layout: {combined_out}")

# ------------------------------------------------------------------
#    Plotting gradients on hippocampus surface and amygdala volume
# ------------------------------------------------------------------
from plotting_hippamyg import (
    resolve_indices_and_labels,
    load_hipp_surfaces_and_labels,
    get_hipp_label_ids,
    map_gradients_to_hipp_vertices,
    compute_shared_limits,
    plot_and_save_hipp_gradients,
    plot_and_save_hipp_unfolded_gradients,
    plot_amygdala_volumes_sharedscale,
    plot_community_bars_with_strips,
    YEO7_COLORS
)

n_hipp   = int(N_HIPP_PARCELS_PER_HEMI)
n_grads  = min(2, gradients_top.shape[1])   # Figure 4 uses G1 and G2
cmap_cm  = plt.get_cmap('viridis')

# --- Indices and labels from utils.hippamyg_labels ---
idxlab = resolve_indices_and_labels(n_hipp)
left_amyg_indices,  right_amyg_indices  = idxlab['Left']['amyg_indices'],  idxlab['Right']['amyg_indices']
left_hipp_indices,  right_hipp_indices  = idxlab['Left']['hipp_indices'],  idxlab['Right']['hipp_indices']

# --- Paths for hipp labels and surfaces ---
HIPP_LABELS_L_NPY = str(PATHS.output_root / "L_hipp_majority_labels.npy")
HIPP_LABELS_R_NPY = str(PATHS.output_root / "R_hipp_majority_labels.npy")

HIPP_RESOURCE_ROOT = Path(os.environ["HIPPUNFOLD_RESOURCE_ROOT"])
HIPP_SURF_L_GII = str(HIPP_RESOURCE_ROOT / "canonical_surfs/tpl-avg_space-canonical_den-2mm_label-hipp_midthickness.surf.gii")
HIPP_SURF_R_GII = str(HIPP_RESOURCE_ROOT / "canonical_surfs/tpl-avg_space-canonical_den-2mm_label-hipp_midthickness.surf.gii")

HIPP_UNFOLD_LABELMAP_L_NII = str(PATHS.output_root / "L_hipp_majority_unfold_DeKraker15.nii.gz")

HIPP_UNFOLD_LABELMAP_R_NII = str(PATHS.output_root / "R_hipp_majority_unfold_DeKraker15.nii.gz")

# --- Load hippocampal surfaces & per-vertex labels ---
labeling_L, labeling_R, vertices_dict, faces_dict = load_hipp_surfaces_and_labels(
    labeling_path_L=HIPP_LABELS_L_NPY,
    labeling_path_R=HIPP_LABELS_R_NPY,
    surf_path_L=HIPP_SURF_L_GII,
    surf_path_R=HIPP_SURF_R_GII,
)

# check expected #labels (e.g., 15 per hemi)
uniq_L = get_hipp_label_ids(labeling_L, expected_n=n_hipp)
uniq_R = get_hipp_label_ids(labeling_R, expected_n=n_hipp)

# --- Map gradient components -> hippocampal vertex arrays (L/R) ---
grad_vertex_dicts = map_gradients_to_hipp_vertices(
    gradients_top,
    left_hipp_indices, right_hipp_indices,
    labeling_L, labeling_R,
    uniq_L, uniq_R,
    n_components=n_grads
)

# --- Shared per-gradient color limits across Hipp (L/R) + Amyg (L/R) ---
per_grad_vmin, per_grad_vmax = compute_shared_limits(
    gradients_top,
    left_hipp_indices, right_hipp_indices,
    left_amyg_indices, right_amyg_indices,
    n_to_plot=n_grads
)

# --- Plot hippocampus (surfaces) ---
hipp_out_tpl = os.path.join(output_dir, '{side}_hippocampus_gradient{gradient}.png')
plot_and_save_hipp_gradients(
    vertices_dict, faces_dict, grad_vertex_dicts,
    per_grad_vmin, per_grad_vmax,
    cmap_cm,
    title_template='Hippocampus {side} Gradient {gradient}',
    output_file_template=hipp_out_tpl,
    n_to_plot=n_grads
)

# --- True 2D unfolded hippocampus maps
# Uses exact same cmap and per-gradient vmin/vmax as folded hippocampus and amygdala.
for g in range(n_grads):
    plot_and_save_hipp_unfolded_gradients(
        values=gradients_top[:, g],
        left_hipp_indices=left_hipp_indices,
        right_hipp_indices=right_hipp_indices,
        labelmap_l_path=HIPP_UNFOLD_LABELMAP_L_NII,
        labelmap_r_path=HIPP_UNFOLD_LABELMAP_R_NII,
        out_path=os.path.join(
            output_dir,
            f"LR_hippocampus_unfolded_labelmap_gradient{g + 1}.png"
        ),
        cmap=cmap_cm,
        vmin=per_grad_vmin[g],
        vmax=per_grad_vmax[g],
        n_hipp=n_hipp,
        collapse_axis=2,
        flip_ap=True,
        flip_pd=True,
        boundary_mode="subfield",
        title=f"Hippocampus unfolded Gradient {g + 1}",
        cbar_label="Gradient value",
        save_pdf=True,
    )

# --- Plot amygdala (volumes) with the same limits ---
label_order = [7001, 7003, 7005, 7006, 7007, 7008, 7009, 7010, 7015]
label_img_L_amyg = nb.load(PATHS.resource_root / "lh.AmygLabels.mgz")
label_img_R_amyg = nb.load(PATHS.resource_root / "rh.AmygLabels.mgz")

plot_amygdala_volumes_sharedscale(
    gradients_top,
    label_img_L=label_img_L_amyg,
    label_img_R=label_img_R_amyg,  # kept for signature; ignored if mirror_right_from_left=True
    label_order=label_order,
    left_indices=left_amyg_indices,
    right_indices=right_amyg_indices,
    per_grad_vmin=per_grad_vmin,
    per_grad_vmax=per_grad_vmax,
    out_dir=output_dir,
    title_prefix="Amygdala",
    output_prefix="volumetric_amygdala_gradients",
    n_to_plot=n_grads,
    cmap=cmap_cm,
    ortho_layout="vertical",
    mirror_right_from_left=True,
    draw_contours=True,
)


print("Hippocampus and Amygdala gradient plotting complete (shared color scale per gradient).")

# ===================================================================
# Paper Figure 4B: Yeo-7 summaries for cortical gradient projections
# + compact subcortical metadata used by TSV export
# ===================================================================
NETWORK_MAPPING_TSV_PATH = str(PATHS.resource_root / "atlas-Glasser_dseg.tsv")
TSV_INDEX_COLUMN = "index"
GRADIENTS_TO_PLOT = [0, 1]  # Figure 4 uses G1 and G2.

yeo7_dict = {
    "visual": YEO7_COLORS[0],
    "somatomotor": YEO7_COLORS[1],
    "dorsal attention": YEO7_COLORS[2],
    "ventral attention": YEO7_COLORS[3],
    "limbic": YEO7_COLORS[4],
    "frontoparietal": YEO7_COLORS[5],
    "default mode": YEO7_COLORS[6],
}

# Use the cortical projection matrix computed above; shape = (n_components, 360).
corr_full360 = corr_full_length
if corr_full360.ndim != 2 or corr_full360.shape[1] != 360:
    raise RuntimeError(f"Expected corr_full360 shape (n_components, 360), got {corr_full360.shape}.")

for grad_idx in GRADIENTS_TO_PLOT:
    if grad_idx >= corr_full360.shape[0]:
        print(f"[Yeo7] Skipping G{grad_idx + 1}: only {corr_full360.shape[0]} gradients available.")
        continue

    barstrip_out = os.path.join(
        output_dir,
        f"barstrip_yeo7_gradient{grad_idx + 1}_fc_fullcortex.png",
    )
    plot_community_bars_with_strips(
        data_360=corr_full360[grad_idx, :],
        tsv_path=NETWORK_MAPPING_TSV_PATH,
        index_col=TSV_INDEX_COLUMN,
        scheme_col="community_yeo7",
        out_path=barstrip_out,
        title=f"Bar+Strip — Gradient {grad_idx + 1} FC Corr (full retained cortex) — Yeo-7 networks",
        ylabel=f"FC corr with Subcortical Gradient {grad_idx + 1}",
        ylim=(-1.0, 1.0),
        yscale="linear",
        orientation="horizontal",
        inset=False,
        custom_palette=yeo7_dict,
    )
    print(f"[Yeo7] Saved bar+strip plot: {barstrip_out}")

# -------------------------------------------------------------------
# Subcortical label metadata for TSV export
# -------------------------------------------------------------------
amyg_base = get_amygdala_labels(side=None, style="long")
left_labels_amygdala = [f"Left Amygdala {label}" for label in amyg_base]
right_labels_amygdala = [f"Right Amygdala {label}" for label in amyg_base]

n_hipp_from_idx = len(left_hipp_indices)
if n_hipp_from_idx % 5 != 0:
    raise RuntimeError("N_HIPP_PARCELS_PER_HEMI must be a multiple of 5 for hippocampal AP labels.")

left_names_full, left_ap_bins = build_hippocampus_labels("Left", n_hipp_from_idx, order="subfield-major")
right_names_full, right_ap_bins = build_hippocampus_labels("Right", n_hipp_from_idx, order="subfield-major")

def _prefix_hipp(label: str) -> str:
    side, rest = label.split(" ", 1)
    return f"{side} Hippocampus {rest}"

left_labels_hipp_names = [_prefix_hipp(label) for label in left_names_full]
right_labels_hipp_names = [_prefix_hipp(label) for label in right_names_full]
left_labels_hipp_ap = [f"AP {ap}" for ap in left_ap_bins]
right_labels_hipp_ap = [f"AP {ap}" for ap in right_ap_bins]

combined_labels_ordered = (
    left_labels_amygdala
    + right_labels_amygdala
    + list(left_labels_hipp_names)
    + list(right_labels_hipp_names)
)
combined_region_types_ordered = (
    ["Amygdala"] * len(left_labels_amygdala)
    + ["Amygdala"] * len(right_labels_amygdala)
    + ["Hippocampus"] * n_hipp_from_idx
    + ["Hippocampus"] * n_hipp_from_idx
)
combined_ap_positions_ordered = (
    ["Amygdala"] * len(left_labels_amygdala)
    + ["Amygdala"] * len(right_labels_amygdala)
    + list(left_labels_hipp_ap)
    + list(right_labels_hipp_ap)
)

print("[TSV metadata] Subcortical label arrays prepared.")
# ===================================================================
# Export TSV summaries (cortex & subcortex, GLASSO + Pearson)
# ===================================================================
print("\n=== TSV EXPORT: Cortical & Subcortical Gradients (GLASSO + Pearson) ===")

# Directories for each method (must match earlier output_dir convention)
method_dirs = {
    "glasso": os.path.join(output_root, "top10_union_glasso"),
    "pearson": os.path.join(output_root, "top10_union_pearson"),
}

grad_methods = {}
for meth, mdir in method_dirs.items():
    corr_path = os.path.join(mdir, "group_gradients_fc_corr_full360.npy")
    grads_path = os.path.join(mdir, "group_gradients_top10_union.npy")
    top10_path = os.path.join(mdir, "top10_indices.npy")

    if all(os.path.exists(p) for p in (corr_path, grads_path, top10_path)):
        corr_full = np.load(corr_path)       # (n_comp, 360)
        grads = np.load(grads_path)          # (n_seeds, n_comp)
        top10_idx = np.load(top10_path)      # (n_top_union,) 0-based Glasser IDs
        grad_methods[meth] = {
            "corr_full360": corr_full,
            "gradients": grads,
            "top10_indices": top10_idx,
        }
        print(f"[TSV] Found results for method '{meth}' in {mdir}.")
    else:
        missing = [p for p in (corr_path, grads_path, top10_path) if not os.path.exists(p)]
        print(f"[TSV] Missing files for method '{meth}' ({', '.join(missing)}); skipping this method.")

if not grad_methods:
    print("[TSV] No methods with complete outputs found; skipping TSV export.")
else:
    # -------------------------------------------------------------
    # A) Cortex TSV (360 parcels: GLASSER + gradient–FC correlations)
    # -------------------------------------------------------------
    df_glasser = pd.read_csv(NETWORK_MAPPING_TSV_PATH, sep="\t")

    if "index" not in df_glasser.columns:
        raise RuntimeError(f"Column 'index' not found in Glasser TSV: {NETWORK_MAPPING_TSV_PATH}")

    # Core value table: one row per parcel 1..360
    parcel_ids = np.arange(1, 361, dtype=int)
    df_ctx = pd.DataFrame({"index": parcel_ids})
    df_ctx.set_index("index", inplace=True)

    # Number of gradient components to export (common across methods)
    n_comp_common = min(m["corr_full360"].shape[0] for m in grad_methods.values())

    for meth, mats in grad_methods.items():
        corr_full = mats["corr_full360"]           # (n_comp, 360)
        top10_idx_0b = np.asarray(mats["top10_indices"], dtype=int)
        top10_idx_1b = top10_idx_0b + 1            # convert to 1-based Glasser IDs

        if corr_full.shape[1] != 360:
            print(f"[TSV] WARNING: corr_full360 for '{meth}' has width {corr_full.shape[1]} != 360; skipping.")
            continue

        # Add one column per gradient: Grad1_FCCorr_GLASSO / PEARSON, etc.
        for g in range(n_comp_common):
            if g >= corr_full.shape[0]:
                print(f"[TSV] WARNING: gradient index {g} out of range for method '{meth}'.")
                continue
            col = f"Grad{g+1}_FCCorr_{meth.upper()}"
            df_ctx[col] = corr_full[g, :]

        # Boolean flag: whether parcel is in Top-{TOP_PERCENT}% union for this method
        df_ctx[f"In_Top10Union_{meth.upper()}"] = df_ctx.index.isin(top10_idx_1b)

    df_glasser["index"] = pd.to_numeric(df_glasser["index"], errors="coerce")
    df_glasser.dropna(subset=["index"], inplace=True)
    df_glasser["index"] = df_glasser["index"].astype(int)
    df_glasser.set_index("index", inplace=True, drop=False)

    # Join atlas metadata + gradient metrics
    df_cortex_final = df_glasser.join(df_ctx, how="left", rsuffix="_grad")
    out_tsv_ctx = os.path.join(
        output_root,
        f"glasser360_hippamyg_gradients_fc_top{TOP_PERCENT}.tsv",
    )
    df_cortex_final.to_csv(out_tsv_ctx, sep="\t", index=False, float_format="%.6f")
    print(f"[TSV] Saved cortical gradients TSV: {out_tsv_ctx}")

    # ---------------------------------------------------------
    # B) Subcortical TSV (Hippocampus + Amygdala gradient coords)
    # ---------------------------------------------------------
    try:
        n_subregions = len(combined_labels_ordered)
    except NameError:
        print("[TSV] Subcortical label arrays not found; skipping subcortical TSV.")
    else:
        df_sub = pd.DataFrame(
            {
                "Seed_Index": np.arange(n_subregions, dtype=int),
                "Label": combined_labels_ordered,
                "Region": combined_region_types_ordered,
                "AP_Position": combined_ap_positions_ordered,
            }
        )

        # Hemisphere from label prefix
        sides = []
        for lbl in combined_labels_ordered:
            side = "Unknown"
            if isinstance(lbl, str):
                if lbl.startswith("Left "):
                    side = "Left"
                elif lbl.startswith("Right "):
                    side = "Right"
            sides.append(side)
        df_sub["Side"] = sides

        # Attach gradient coordinates per method:
        #   Grad1_GLASSO, Grad2_GLASSO, ... and Grad1_PEARSON, Grad2_PEARSON, ...
        for meth, mats in grad_methods.items():
            G = mats["gradients"]  # (n_subregions, n_components)
            if G.shape[0] != n_subregions:
                print(
                    f"[TSV] WARNING: gradients for method '{meth}' have "
                    f"{G.shape[0]} rows != expected {n_subregions}; skipping this method in subcortical TSV."
                )
                continue
            n_comp_m = min(n_comp_common, G.shape[1])
            for g in range(n_comp_m):
                col = f"Grad{g+1}_{meth.upper()}"
                df_sub[col] = G[:, g]

        out_tsv_sub = os.path.join(
            output_root,
            f"hippamyg_subcortical_gradients_top{TOP_PERCENT}.tsv",
        )
        df_sub.to_csv(out_tsv_sub, sep="\t", index=False, float_format="%.6f")
        print(f"[TSV] Saved subcortical gradients TSV: {out_tsv_sub}")

# ===================================================================
# Load sample-derived Pearson cortical gradients generated upstream
# ===================================================================
# Figure 4C tests joint subcortical gradient projections against Pearson-derived
# cortical functional gradients (CrtFC G1-G3) plus dominance/sharedness maps.

ctx_out_dir = os.path.join(output_root, "top10_union_pearson")
cortex_intrinsic_path = os.path.join(ctx_out_dir, "cortex_intrinsic_gradients.npy")
kept_sorted_path = os.path.join(ctx_out_dir, "kept_sorted.npy")
if not os.path.exists(cortex_intrinsic_path) or not os.path.exists(kept_sorted_path):
    raise FileNotFoundError(
        "Missing sample-derived cortical-gradient prerequisites. Run "
        "compute_cortical_gradients.py before the joint-gradient analyses."
    )

grads_ctx = np.load(cortex_intrinsic_path)
saved_kept_sorted = np.load(kept_sorted_path)
if grads_ctx.ndim != 2 or grads_ctx.shape[0] != len(kept_sorted):
    raise ValueError(
        f"Cortical gradients have shape {grads_ctx.shape}; expected "
        f"({len(kept_sorted)}, n_components)"
    )
if not np.array_equal(saved_kept_sorted, kept_sorted):
    raise ValueError(
        "Saved kept_sorted.npy does not match REMOVED_GLASSER_PARCELS"
    )
ctx_gradients = {"pearson": grads_ctx}
# ===================================================================
# MSR ANALYSIS: Modular Reference Map Loader
# ===================================================================
print("\n=== START: Moran Spectral Randomization (Modular) ===")

# --- Configuration ---
n_perm = 1000
MSR_SUBCORT_GRADS = [0, 1]  # Analyze G1 and G2 of Subcortical Projections

# Standard Paths
GLASSER_L_GII = str(PATHS.resource_root / "glasser-360_conte69_lh.label.gii")
GLASSER_R_GII = str(PATHS.resource_root / "glasser-360_conte69_rh.label.gii")
DIST_L_NPY = str(PATHS.resource_root / "Glasser32k_dist_L.npy")
DIST_R_NPY = str(PATHS.resource_root / "Glasser32k_dist_R.npy")

extra_maps_dir = str(PATHS.output_root / "group_level/top10_hippamyg")

# -------------------------------------------------------------------
# Reference maps tested against the joint-gradient projections.
# -------------------------------------------------------------------
reference_maps_config = [
    # --- Intrinsic Cortex Gradients (Pearson) ---
    {'type': 'gradient', 'data': ctx_gradients.get('pearson'), 'index': 0, 'label': 'Pear CrtFC G1'},
    {'type': 'gradient', 'data': ctx_gradients.get('pearson'), 'index': 1, 'label': 'Pear CrtFC G2'},
    {'type': 'gradient', 'data': ctx_gradients.get('pearson'), 'index': 2, 'label': 'Pear CrtFC G3'},

    # --- Extra Maps (From Disk) ---
    {'type': 'map', 'path': os.path.join(extra_maps_dir, f"hippamyg_shared_presence_full360_{method_lower}_top10.npy"), 
     'label': 'Shared.'},
    {'type': 'map', 'path': os.path.join(extra_maps_dir, f"hippamyg_dominance_presence_full360_{method_lower}_top10.npy"), 
     'label': 'Dom.'},
]

# ===================================================================
# Load and stack the configured reference maps.
# ===================================================================

# Joint-gradient cortical projections.
proj_dir = os.path.join(output_root, f"top10_union_{method_lower}")
proj_fc_corr_path = os.path.join(proj_dir, "group_gradients_fc_corr_full360.npy")
if not os.path.exists(proj_fc_corr_path):
    raise FileNotFoundError(f"Missing target file: {proj_fc_corr_path}")
subcort_data_transposed = np.load(proj_fc_corr_path).T  # (360, N)

# Reference maps.
ref_arrays = []
ref_labels = []

print("Loading Reference Maps...")
for item in reference_maps_config:
    # Map retained-cortex gradients to the full Glasser atlas.
    if item['type'] == 'gradient':
        source_data = item['data']  # expected shape: (n_kept_cortex, n_components)
        if source_data is None:
            raise RuntimeError(f"Reference gradient '{item['label']}' is None.")

        col_idx = item['index']
        if source_data.ndim != 2 or source_data.shape[0] != len(kept_sorted):
            raise RuntimeError(
                f"Reference gradient '{item['label']}' has shape {source_data.shape}; "
                f"expected ({len(kept_sorted)}, n_components)."
            )
        if col_idx >= source_data.shape[1]:
            raise RuntimeError(
                f"Reference gradient '{item['label']}' requested component {col_idx}, "
                f"but only {source_data.shape[1]} components are available."
            )

        full_map = np.full(360, np.nan)
        full_map[kept_sorted] = source_data[:, col_idx]
        ref_arrays.append(full_map)
        ref_labels.append(item['label'])
        print(f"  + Added: {item['label']}")

    # B. Handle 'map' type (expects file path to shape (360,))
    elif item['type'] == 'map':
        path = item.get('path')
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Missing reference map for {item['label']}: {path}")

        loaded_map = np.load(path).squeeze()
        if loaded_map.ndim != 1 or loaded_map.shape[0] != 360:
            raise RuntimeError(
                f"Reference map '{item['label']}' has shape {loaded_map.shape}; expected (360,)."
            )

        ref_arrays.append(loaded_map)
        ref_labels.append(item['label'])
        print(f"  + Added: {item['label']} (from disk)")

    else:
        raise ValueError(f"Unknown reference map type: {item['type']}")

# Stack into (360, N_refs)
reference_data_combined = np.column_stack(ref_arrays)
ref_idxs_to_test = list(range(len(ref_arrays)))

# --- 3. Run MSR Analysis ---
r_vals = np.full((len(MSR_SUBCORT_GRADS), len(ref_idxs_to_test)), np.nan, dtype=float)
p_vals = np.full_like(r_vals, np.nan, dtype=float)

for ii, target_grad_idx in enumerate(MSR_SUBCORT_GRADS):
    for jj, ref_idx in enumerate(ref_idxs_to_test):

        x = subcort_data_transposed[:, target_grad_idx]
        y = reference_data_combined[:, ref_idx]
        ref_label = ref_labels[ref_idx]

        # Pair-specific domain:
        # - Cortical gradients: all retained cortex
        # - Dominance/sharedness: finite overlap, because those maps are top-10-defined
        pair_mask = np.zeros(360, dtype=bool)
        pair_mask[kept_sorted] = True
        pair_mask &= np.isfinite(x) & np.isfinite(y)

        if pair_mask.sum() < 3:
            print(
                f"[WARN] Skipping AmygHipp G{target_grad_idx + 1} vs {ref_label}: "
                f"only {pair_mask.sum()} finite parcels."
            )
            continue

        r_tmp, p_tmp = compute_msr_statistics(
            target_data=x[:, None],
            ref_data=y[:, None],
            mask_idx=pair_mask,
            target_idxs=[0],
            ref_idxs=[0],
            dist_l_path=DIST_L_NPY,
            dist_r_path=DIST_R_NPY,
            parc_l_path=GLASSER_L_GII,
            parc_r_path=GLASSER_R_GII,
            n_proc=8,
            n_perm=n_perm,
            metric="kendall",
        )

        r_vals[ii, jj] = r_tmp[0, 0]
        p_vals[ii, jj] = p_tmp[0, 0]

        print(
            f"AmygHipp G{target_grad_idx + 1} vs {ref_label}: "
            f"n={pair_mask.sum()}, tau={r_vals[ii, jj]:.3f}, p={p_vals[ii, jj]:.4f}"
        )

np.save(os.path.join(output_dir, f"msr_rvals_top{TOP_PERCENT}.npy"), r_vals)

# --- FDR correction across the full tested family ---
q_vals = np.full_like(p_vals, np.nan, dtype=float)
valid_p = np.isfinite(p_vals)

if valid_p.any():
    _, q_flat = fdrcorrection(
        p_vals[valid_p].ravel(),
        alpha=0.05,
        method="indep"
    )
    q_vals[valid_p] = q_flat
else:
    print("[WARN] No finite MSR p-values found; skipping FDR correction.")

# Optional: save raw and FDR-corrected p-values
np.save(os.path.join(output_dir, f"msr_raw_pvals_top{TOP_PERCENT}.npy"), p_vals)
np.save(os.path.join(output_dir, f"msr_fdr_qvals_top{TOP_PERCENT}.npy"), q_vals)

print("\nRaw MSR p-values:")
print(np.array2string(p_vals, precision=4, suppress_small=False))

print("\nFDR-corrected q-values:")
print(np.array2string(q_vals, precision=4, suppress_small=False))
# --- 4. Plot Heatmap ---
print("\nGenerating heatmap...")
sns.set_context("talk", font_scale=1.5)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": "Lato",
    "axes.edgecolor": "black",
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    })


plt.style.use('seaborn-v0_8-white')

y_labels = [f"AmygHipp G{g+1}" for g in MSR_SUBCORT_GRADS]
x_labels = ref_labels 

annot_labels = np.empty_like(r_vals, dtype=object)
for i in range(r_vals.shape[0]):
    for j in range(r_vals.shape[1]):
        q = q_vals[i, j]
        star = "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else ""
        annot_labels[i, j] = f"{r_vals[i, j]:.2f}{star}"

fig_hm, ax_hm = plt.subplots(figsize=(len(x_labels)*2 + 2, 5)) # Dynamic width
sns.heatmap(
    r_vals, annot=annot_labels, fmt="", cmap="RdBu_r",
    center=0, vmin=-1, vmax=1, square=False, linewidths=3, linecolor='white',
    xticklabels=x_labels, yticklabels=y_labels, ax=ax_hm,
    annot_kws={"weight": "bold", "size": 22}, 
    cbar_kws={"label": "Kendall", "shrink": 0.9}
)

ax_hm.xaxis.tick_top()
ax_hm.xaxis.set_label_position('top')
plt.yticks(rotation=0)
heatmap_path = os.path.join(output_dir, f"heatmap_modular_top{TOP_PERCENT}_FDR.png")
plt.savefig(heatmap_path, dpi=600, bbox_inches='tight')
plt.close(fig_hm)
print(f"Saved modular heatmap to: {heatmap_path}")
