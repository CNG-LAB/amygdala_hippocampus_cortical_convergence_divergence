# -*- coding: utf-8 -*-
"""
General helpers for hipp/amyg -> cortex analyses.
"""

from __future__ import annotations
import os
from typing import Iterable, Tuple, Optional, Dict, Sequence
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from brainspace.gradient import GradientMaps
from neuromaps import nulls, datasets
from neuromaps.points import get_surface_distance

# ---------------------------------------------------------------------
# Basic transforms
# ---------------------------------------------------------------------
def inverse_fisher_z_transform(z: np.ndarray) -> np.ndarray:
    """Inverse Fisher z."""
    z = np.asarray(z, dtype=float)
    return np.tanh(z)


# ---------------------------------------------------------------------
# Averaging helpers
# ---------------------------------------------------------------------
def zavg_ignore(arr_z: np.ndarray, drop_zeros: bool = True, axis: int = 0) -> np.ndarray:
    """
    Average z-matrices across subjects, ignoring NaNs and (optionally) zeros.

    Parameters
    ----------
    arr_z : array, shape (N, T, T) or (N, ...)
        Stacked subject z-conn data.
    drop_zeros : bool
        If True, treat exact zeros as missing (→ NaN) before averaging.
    axis : int
        Axis to average over (usually subjects axis).

    Returns
    -------
    avg_r : array
        Averaged matrix back in r-space.
    """
    A = np.array(arr_z, dtype=float, copy=True)
    if drop_zeros:
        A[np.isclose(A, 0.0)] = np.nan
    valid = np.isfinite(A)
    count = valid.sum(axis=axis)
    zmean = np.divide(
        np.where(valid, A, 0.0).sum(axis=axis),
        count,
        out=np.zeros_like(count, dtype=float),
        where=count > 0,
    )
    return inverse_fisher_z_transform(zmean)


# ---------------------------------------------------------------------
# Robust correlation
# ---------------------------------------------------------------------
def safe_pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    """
    Pearson r guarded against zero-variance vectors and SciPy return changes.
    Returns 0.0 if either vector is (near) constant.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if (np.std(x) < 1e-12) or (np.std(y) < 1e-12):
        return 0.0
    out = pearsonr(x, y)
    # SciPy ≥1.11 returns PearsonRResult(statistic, pvalue)
    if hasattr(out, "statistic"):
        return float(out.statistic)
    if hasattr(out, "correlation"):
        return float(out.correlation)
    return float(out[0])


# ---------------------------------------------------------------------
# Column selection: union of per-row Top-K
# ---------------------------------------------------------------------

def load_glasso_support_prevalence(
    subject_list_file: str,
    base_dir: str,
    *,
    n_amyg_per_hemi: int = 9,
    n_hipp_per_hemi: int = 15,
    removed_glasser: Sequence[int] = (120, 300),
    positive_only: bool = False,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deterministically compute GLASSO support prevalence across subjects.

    Uses the exact filename:
      connectivity_matrix_REST_{N_TOTAL}x{N_TOTAL}_glasso_partial.npy

    Returns:
      prevalence: (2*9 + 2*n_hipp_per_hemi, n_ctx) in [0..1]
      kept_glp:   1-based Glasser IDs (length n_ctx)

    If positive_only is True, prevalence counts only strictly positive
    partial correlations (M > 0). Otherwise, any non-zero finite edge
    (M != 0) counts as present.
    """
    kept_glp = _discover_kept_glasser(removed_glasser)  # 1..360 \ removed
    n_ctx    = kept_glp.size
    n_seeds  = 2 * n_amyg_per_hemi + 2 * n_hipp_per_hemi
    N_TOTAL  = n_seeds + n_ctx

    seed_rows = np.arange(0, n_seeds)
    ctx_start = N_TOTAL - n_ctx

    with open(subject_list_file, "r") as f:
        subjects = [
            s.strip() for s in f
            if s.strip() and not s.strip().startswith("#")
        ]

    edge_sum = np.zeros((n_seeds, n_ctx), dtype=float)
    n_valid = 0

    for sid in subjects:
        fpath = os.path.join(
            base_dir, sid, "func_native_2025",
            f"connectivity_matrix_REST_{N_TOTAL}x{N_TOTAL}_glasso_partial.npy"
        )
        if not os.path.exists(fpath):
            if verbose:
                print(f"[prevalence] {sid}: missing file for N_TOTAL={N_TOTAL}: {os.path.basename(fpath)}")
            continue

        M = np.load(fpath)
        if M.ndim != 2 or M.shape[0] != M.shape[1] or M.shape[0] != N_TOTAL:
            if verbose:
                print(f"[prevalence] {sid}: bad shape {M.shape}, expected ({N_TOTAL},{N_TOTAL})")
            continue

        if positive_only:
            supp = np.isfinite(M) & (M > 0)
        else:
            supp = np.isfinite(M) & (M != 0)

        edge_sum += supp[seed_rows][:, ctx_start:].astype(float)
        n_valid += 1

    if n_valid != len(subjects):
        raise RuntimeError(
            f"Expected {len(subjects)} valid GLASSO partial matrices, found {n_valid} "
            f"(N_TOTAL={N_TOTAL})."
        )

    prevalence = edge_sum / n_valid
    return prevalence, kept_glp

def consensus_mask_from_prevalence(
    prevalence: np.ndarray,
    min_fraction: float = 0.10,
) -> np.ndarray:
    """
    Build a boolean consensus mask (n_seeds × n_ctx) from GLASSO-support prevalence.

    Parameters
    ----------
    prevalence : array, shape (n_seeds, n_ctx)
        Prevalence values in [0, 1].
    min_fraction : float
        Minimum fraction of subjects required for an edge to be kept.
        For 10% consensus, use 0.10.

    Returns
    -------
    mask : bool array, shape (n_seeds, n_ctx)
        True where prevalence >= min_fraction and finite.
    """
    p = np.asarray(prevalence, dtype=float)
    if p.ndim != 2:
        raise ValueError("`prevalence` must be 2D [n_seeds, n_ctx].")
    if min_fraction is None or min_fraction <= 0:
        # No additional filtering: keep all finite prevalence entries
        return np.isfinite(p)
    return np.isfinite(p) & (p >= float(min_fraction))
    

def seedwise_topk_masks_from_prevalence(
    prevalence: np.ndarray,
    top_percent: int = 10,
) -> np.ndarray:
    """
    Boolean masks of Top-k cortical parcels per seed, based on prevalence.

    Parameters
    ----------
    prevalence : array, shape (n_seeds, n_ctx)
        Prevalence values in [0, 1] (typically). Each row is one seed.
    top_percent : int, optional
        Percentage of cortical parcels to select per seed (by prevalence),
        e.g. 10 -> top 10% of parcels. Must be > 0 to select anything.

    Returns
    -------
    mask : bool array, shape (n_seeds, n_ctx)
        For each seed (row), True entries indicate the selected parcels.
        - If a seed has no positive prevalence (all <= 0 or no finite values),
          its entire row in `mask` is False (empty mask).
        - If a seed has m>0 positive parcels and k>m, it selects only those m
          parcels (no zero-prevalence parcels are ever selected).
    """
    prevalence = np.asarray(prevalence, dtype=float)
    if prevalence.ndim != 2:
        raise ValueError("`prevalence` must be 2D [n_seeds, n_ctx].")

    n_seeds, n_ctx = prevalence.shape

    # Use the manuscript definition of k.
    k = max(1, int(np.ceil(top_percent / 100.0 * n_ctx)))

    mask = np.zeros((n_seeds, n_ctx), dtype=bool)

    for r in range(n_seeds):
        row = prevalence[r]

        finite = np.isfinite(row)
        if not finite.any():
            # No finite values: keep mask[r] all False
            continue

        # Indices with strictly positive prevalence
        pos_idx = np.where((row > 0) & finite)[0]
        n_pos = pos_idx.size

        if n_pos == 0:
            # All finite entries are <= 0: empty mask for this seed
            continue

        # Limit k to the number of positive parcels.
        k_eff = min(k, n_pos)

        # Take top k_eff among the positive ones only
        pos_vals = row[pos_idx]
        local_idx = np.argpartition(pos_vals, -k_eff)[-k_eff:]
        local_idx = local_idx[np.argsort(pos_vals[local_idx])[::-1]]

        selected = pos_idx[local_idx]
        mask[r, selected] = True

    return mask
    
def compute_seedwise_topk_mask(
    matrix: np.ndarray,
    top_percent: int = 10,
    positive_only: bool = False,
) -> np.ndarray:
    """
    Build a boolean mask (n_rows × n_cols) indicating the Top-K columns
    per row based on the raw values (not absolute value).

    Parameters
    ----------
    matrix : array (n_rows, n_cols)
        Input connectivity matrix (e.g., Pearson r or Fisher z).
    top_percent : int
        Percentage of columns to select (1..100).
    positive_only : bool
        If True, only strictly positive entries (value > 0) are considered
        as candidates for Top-K. Negative and zero entries are ignored.

    Returns
    -------
    mask : bool array (n_rows, n_cols)
        For each row, True = selected parcel.
        - If a row has no finite candidates (or no positive values when
          positive_only=True), the entire row is False.
        - If a row has m candidates and k > m, only those m are selected.
    """
    mat = np.asarray(matrix, dtype=float)
    if mat.ndim != 2:
        raise ValueError("`matrix` must be 2D [n_rows, n_cols].")

    n_rows, n_cols = mat.shape
    k = max(1, int(np.ceil(top_percent / 100.0 * n_cols)))

    mask = np.zeros((n_rows, n_cols), dtype=bool)

    for r in range(n_rows):
        row = mat[r]
        finite = np.isfinite(row)

        if positive_only:
            cand_idx = np.where((row > 0) & finite)[0]
        else:
            cand_idx = np.where(finite)[0]

        n_cand = cand_idx.size
        if n_cand == 0:
            # No usable values in this row
            continue

        k_eff = min(k, n_cand)
        vals = row[cand_idx]

        # Top-k by raw value (no abs)
        if k_eff < n_cand:
            local_idx = np.argpartition(vals, -k_eff)[-k_eff:]
        else:
            local_idx = np.arange(n_cand)

        selected = cand_idx[local_idx]
        mask[r, selected] = True

    return mask

    
def _discover_kept_glasser(removed_glasser: Sequence[int]) -> np.ndarray:
    """Return 1-based Glasser IDs that are kept (e.g., 1..360 minus removed)."""
    removed = set(removed_glasser or [])
    kept = np.array([p for p in range(1, 361) if p not in removed], dtype=int)
    if kept.size == 0:
        raise ValueError("No Glasser parcels left after removal.")
    return kept

def _topk_argpartition(values: np.ndarray, k: int) -> np.ndarray:
    """Indices of the top-k elements of `values` (ties broken arbitrarily)."""
    if k <= 0:
        return np.empty(0, dtype=int)
    if k >= values.size:
        return np.arange(values.size, dtype=int)
    idx = np.argpartition(values, -k)[-k:]
    # Optional: sort descending within the top-k block (not required for union)
    return idx[np.argsort(values[idx])][::-1]


def union_topk_cols(
    mat: np.ndarray,
    top_percent: int,
    use_abs: bool = False,
) -> np.ndarray:
    """
    Union of per-row top-K columns in a (rows × cols) matrix.

    Parameters
    ----------
    mat : array (n_rows × n_cols)
        Values used to rank columns; NaNs/±inf are ignored.
    top_percent : int
        Percent (1..100). K = ceil(p/100 * n_cols), min 1.
    use_abs : bool
        If True, rank by abs(values).

    Returns
    -------
    idx_union : 1D int array
        Sorted unique column indices (0..n_cols-1).
    """
    mat = np.asarray(mat, dtype=float)
    nrows, ncols = mat.shape
    topK = max(1, int(np.ceil(float(top_percent) / 100.0 * ncols)))
    union: set[int] = set()

    for i in range(nrows):
        row = mat[i]
        if use_abs:
            row = np.abs(row)
        # ignore non-finite by mapping to -inf (so they never appear in top-k)
        safe = np.where(np.isfinite(row), row, -np.inf)
        idx = _topk_argpartition(safe, topK)
        union.update(idx.tolist())

    return np.array(sorted(union), dtype=int)


# ---------------------------------------------------------------------
# Glasser indexing utilities
# ---------------------------------------------------------------------
def sanitize_removed_list(lst: Iterable[int], low: int = 1, high: int = 360) -> np.ndarray:
    """
    From a 1-based list of parcel IDs, keep those within [low, high], return 0-based sorted unique.
    """
    a = np.array([int(x) for x in lst], dtype=int)
    a = a[(a >= low) & (a <= high)]
    a = np.unique(a) - low
    return a


def kept_sorted_from_removed(removed_1based: Iterable[int], n_total: int = 360) -> np.ndarray:
    """
    Given removed 1-based IDs, return sorted kept 0-based IDs in [0..n_total-1].
    """
    removed0 = sanitize_removed_list(removed_1based, low=1, high=n_total)  # now 0-based
    kept = np.setdiff1d(np.arange(n_total, dtype=int), removed0, assume_unique=False)
    return kept


def map_top_to_full360(corr_top: np.ndarray, top_abs_idx: np.ndarray, full_len: int = 360, fill: float = np.nan) -> np.ndarray:
    """
    Place (n_comp × n_top) values into a (n_comp × full_len) array at absolute indices; fill elsewhere.
    """
    corr_top = np.asarray(corr_top, dtype=float)
    top_abs_idx = np.asarray(top_abs_idx, dtype=int)
    out = np.full((corr_top.shape[0], full_len), fill, dtype=float)
    out[:, top_abs_idx] = corr_top
    return out


# ---------------------------------------------------------------------
# Gradient + correlation helpers
# ---------------------------------------------------------------------
def run_gradients(
    SxC: np.ndarray,
    n_components: int = 10,
    kernel: str = "normalized_angle",
    random_state: Optional[int] = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit BrainSpace gradients on a seeds×cortex matrix.

    Returns
    -------
    gradients : (n_seeds × n_components)
    lambdas   : (n_components,)
    lambdas_norm : (n_components,) normalized eigenvalues
    """
    if GradientMaps is None:
        raise ImportError("brainspace is not available. Install/activate it to use run_gradients().")

    SxC = np.asarray(SxC, dtype=float)
    gm = GradientMaps(n_components=n_components, kernel=kernel, random_state=random_state)
    gm.fit(SxC)
    grads = gm.gradients_
    lam = gm.lambdas_
    lam_norm = lam / np.sum(lam) if np.sum(lam) > 0 else lam
    return grads, lam, lam_norm


def correlate_gradients_to_columns(
    gradients: np.ndarray,
    SxC_subset: np.ndarray,
    corr_fn=safe_pearsonr,
) -> np.ndarray:
    """
    Correlate each gradient component (length = n_seeds) to each column of SxC_subset.

    Returns
    -------
    corr : array (n_components × n_cols_subset)
    """
    G = np.asarray(gradients, dtype=float)
    X = np.asarray(SxC_subset, dtype=float)

    n_comp = G.shape[1]
    n_cols = X.shape[1]
    out = np.zeros((n_comp, n_cols), dtype=float)

    for ic in range(n_comp):
        g = G[:, ic]
        for k in range(n_cols):
            out[ic, k] = corr_fn(g, X[:, k])
    return out


def run_gradients_hemi_align_right_to_left(
    SxC: np.ndarray,
    left_rows: np.ndarray,
    right_rows: np.ndarray,
    n_components: int = 10,
    kernel: str = "normalized_angle",
    random_state: Optional[int] = 0,
    procrustes_center: bool = False,
    procrustes_scale: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
      1) Fit gradients on LEFT rows only (defines the reference space)
      2) Fit gradients on RIGHT rows only
      3) Align RIGHT gradients -> LEFT gradients using BrainSpace procrustes
      4) Concatenate into a single gradients array.

    Parameters
    ----------
    SxC : (n_seeds, n_ctx) array
        Seed-by-cortex connectivity matrix.
    left_rows, right_rows : arrays of equal length
        Row indices for homologous seeds in left vs right hemisphere.
        ``left_rows[i]`` and ``right_rows[i]`` must describe the same seed type.
    procrustes_center, procrustes_scale : bool
        If False/False: rotation/reflection only (closest to “keep magnitudes”).
        If True: allows removing mean/scale before alignment (changes interpretation slightly).

    Returns
    -------
    gradients_all : (n_seeds, n_components)
        Left gradients (unaltered) + Right gradients (aligned to left space).
    lambdas_proxy : (n_components,)
        A proxy eigen-spectrum for continuity (mean of left/right lambdas).
        This is not the spectrum of a single joint embedding.
    lambdas_norm_proxy : (n_components,)
        Normalized proxy eigenvalues.
    meta : dict
        Contains left/right gradients and lambdas for QC.
    """
    # Import here because these dependencies are only needed for plotting.
    from brainspace.gradient import GradientMaps
    from brainspace.gradient.alignment import procrustes

    SxC = np.asarray(SxC, dtype=float)
    left_rows = np.asarray(left_rows, dtype=int)
    right_rows = np.asarray(right_rows, dtype=int)

    if SxC.ndim != 2:
        raise ValueError(f"SxC must be 2D (seeds×cortex). Got {SxC.shape}.")
    if left_rows.size != right_rows.size:
        raise ValueError(
            "left_rows and right_rows must have equal length (homologous mapping). "
            f"Got {left_rows.size} vs {right_rows.size}."
        )
    if np.intersect1d(left_rows, right_rows).size > 0:
        raise ValueError("left_rows and right_rows overlap; they must be disjoint.")
    if left_rows.min() < 0 or right_rows.min() < 0:
        raise ValueError("left_rows/right_rows must be non-negative.")
    if left_rows.max() >= SxC.shape[0] or right_rows.max() >= SxC.shape[0]:
        raise ValueError("left_rows/right_rows contain indices outside SxC row range.")

    X_L = SxC[left_rows, :]
    X_R = SxC[right_rows, :]

    # 1) Fit LEFT gradients (reference space)
    gm_L = GradientMaps(n_components=n_components, kernel=kernel, random_state=random_state)
    gm_L.fit(X_L)
    G_L = np.asarray(gm_L.gradients_, dtype=float)
    lam_L = np.asarray(gm_L.lambdas_, dtype=float)

    # 2) Fit RIGHT gradients (independent)
    gm_R = GradientMaps(n_components=n_components, kernel=kernel, random_state=random_state)
    gm_R.fit(X_R)
    G_R = np.asarray(gm_R.gradients_, dtype=float)
    lam_R = np.asarray(gm_R.lambdas_, dtype=float)

    # 3) Align RIGHT -> LEFT in gradient space (keeps LEFT fixed)
    G_R_aligned = procrustes(G_R, G_L, center=procrustes_center, scale=procrustes_scale)

    # 4) Stitch back into a full gradients array
    G_all = np.zeros((SxC.shape[0], n_components), dtype=float)
    G_all[left_rows, :] = G_L
    G_all[right_rows, :] = G_R_aligned

    # Proxy eigenspectrum used by the manuscript plot.
    lam_proxy = 0.5 * (lam_L + lam_R)
    lam_norm_proxy = lam_proxy / lam_proxy.sum() if lam_proxy.sum() > 0 else lam_proxy

    meta = dict(
        left_rows=left_rows.copy(),
        right_rows=right_rows.copy(),
        gradients_left=G_L,
        gradients_right_raw=G_R,
        gradients_right_aligned=G_R_aligned,
        lambdas_left=lam_L,
        lambdas_right=lam_R,
        procrustes_center=np.array([procrustes_center]),
        procrustes_scale=np.array([procrustes_scale]),
    )

    return G_all, lam_proxy, lam_norm_proxy, meta

# ---------------------------------------------------------------------
# Eigenvalue-spectrum plotting
# ---------------------------------------------------------------------

def plot_eigen_spectrum(
    normalized_eigs: np.ndarray,
    ax=None,
    title: str = "Eigenvalue Spectrum",
    *,
    show_cumulative: bool = True,
    annotate_top: int = 0,
    percent: bool = True,
    marker_size: int = 45,
    line_width: float = 2.0,
    bar_alpha: float = 0.25,
):
    """Plot normalized eigenvalues and their cumulative sum.

    Parameters
    ----------
    normalized_eigs : array-like
        1D eigenvalues, typically normalized to sum to 1.
    ax : matplotlib axis or None
        If None, a new figure/axis is created.
    show_cumulative : bool
        If True, adds a cumulative explained-variance curve on a secondary y-axis.
        Return value becomes (ax, ax_cum) in that case.
    annotate_top : int
        If >0, annotate the top-k components with their value.
    percent : bool
        If True, y-axis is shown as %.
    """

    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mtick
        import seaborn as sns
    except ImportError as e:
        raise ImportError(
            "This function requires seaborn, pandas, and matplotlib. "
            "Install with: pip install seaborn pandas matplotlib"
        ) from e

    ne = np.asarray(normalized_eigs, dtype=float).ravel()
    if ne.size == 0:
        raise ValueError("normalized_eigs is empty.")
    if not np.all(np.isfinite(ne)):
        raise ValueError("normalized_eigs contains non-finite values (nan/inf).")

    cum = np.cumsum(ne)

    created_ax = ax is None
    if created_ax:
        _, ax = plt.subplots(1, figsize=(5, 4.4))

    df = pd.DataFrame(
        {
            "Component": np.arange(1, ne.size + 1, dtype=int),
            "Eigen": ne,
        }
    )

    # Restrict style changes to this plotting context.
    with sns.axes_style("whitegrid"), sns.plotting_context("talk"):
        sns.barplot(data=df, x="Component", y="Eigen", ax=ax, alpha=bar_alpha, errorbar=None, native_scale=True)
        sns.lineplot(data=df, x="Component", y="Eigen", ax=ax, linewidth=line_width)
        sns.scatterplot(data=df, x="Component", y="Eigen", ax=ax, s=marker_size, legend=False)

        ax.set_title(title, pad=10)
        ax.set_xlabel("Gradient")
        ax.set_ylabel("Normalized Eigenvalue" + (" (%)" if percent else ""))

        if percent:
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))

        if ne.size > 25:
            step = max(1, ne.size // 10)
            ax.set_xticks(np.arange(1, ne.size + 1, step))

        sns.despine(ax=ax)

        if annotate_top and annotate_top > 0:
            k = int(min(annotate_top, ne.size))
            top_idx = np.argsort(ne)[::-1][:k]
            for i in top_idx:
                x = int(i + 1)
                y = float(ne[i])
                label = f"{y*100:.1f}%" if percent else f"{y:.3g}"
                ax.annotate(
                    label,
                    xy=(x, y),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

        ax_cum = None
        if show_cumulative:
            ax_cum = ax.twinx()
            df2 = pd.DataFrame({"Component": df["Component"], "Cumulative": cum})
            sns.lineplot(
                data=df2,
                x="Component",
                y="Cumulative",
                ax=ax_cum,
                linewidth=1.8,
                linestyle="--",
            )
            ax_cum.set_ylabel("Cumulative" + (" (%)" if percent else ""))
            if percent:
                ax_cum.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
            ax_cum.set_ylim(0, max(1.0, float(np.nanmax(cum)) * 1.02))
            sns.despine(ax=ax_cum, left=True)

        return (ax, ax_cum) if show_cumulative else ax


# ---------------------------------------------------------------------
# Metric Calculation (Normalized)
# ---------------------------------------------------------------------

def calc_dominance_normalized(count_a: np.ndarray, count_b: np.ndarray, total_a: int, total_b: int) -> np.ndarray:
    """
    Calculates Dominance Ratio based on Normalized Proportions.
    D = (Norm_A - Norm_B) / (Norm_A + Norm_B)
    Range: -1 (B only) to +1 (A only).
    """
    a = np.asarray(count_a, dtype=float)
    b = np.asarray(count_b, dtype=float)
    
    # Normalize by the structural capacity (total seeds)
    norm_a = a / float(total_a)
    norm_b = b / float(total_b)
    
    # Calculate Ratio
    with np.errstate(divide="ignore", invalid="ignore"):
        num = norm_a - norm_b
        denom = norm_a + norm_b
        D = num / denom
        
    # Handle 0/0 -> NaN
    return D

def calc_sharedness_normalized(count_a: np.ndarray, count_b: np.ndarray, total_a: int, total_b: int) -> np.ndarray:
    """
    Calculates Sharedness based on Balance and Magnitude of Normalized Proportions.
    S = Balance_Ratio * Total_Magnitude
      = (min_norm / max_norm) * (norm_a + norm_b)
      
    Range: 0 to 2.0 (since max norm_a=1 and max norm_b=1)
    """
    a = np.asarray(count_a, dtype=float)
    b = np.asarray(count_b, dtype=float)
    
    # Normalize
    norm_a = a / float(total_a)
    norm_b = b / float(total_b)
    
    # Calculate Balance and Magnitude
    with np.errstate(divide="ignore", invalid="ignore"):
        _min = np.minimum(norm_a, norm_b)
        _max = np.maximum(norm_a, norm_b)
        
        balance = _min / _max
        magnitude = norm_a + norm_b
        
        S = balance * magnitude
        
    return S


def get_or_compute_geodesic_distmat(dist_l_path, dist_r_path, parc_l_path, parc_r_path, n_proc=8):
    """
    Load or compute parcel-wise geodesic distance matrices (LH, RH) for an fsLR 32k atlas surface.

    Returns
    -------
    dist_L, dist_R : np.ndarray
        Square (N_L, N_L) and (N_R, N_R) distance matrices.
    """
    if os.path.exists(dist_l_path) and os.path.exists(dist_r_path):
        print(f"Loading existing distance matrices:\n  L: {dist_l_path}\n  R: {dist_r_path}")
        # mmap_mode makes this much lighter on RAM if matrices are large
        dist_L = np.load(dist_l_path, mmap_mode="r")
        dist_R = np.load(dist_r_path, mmap_mode="r")
    else:
        print("Distance matrices not found. Computing geodesic distances (this is slow)...")
        fslr = datasets.fetch_atlas("fsLR", "32k")
        surf_L, surf_R = fslr["midthickness"]

        print(f"  Computing Left Hemisphere (n_proc={n_proc})...")
        dist_L = get_surface_distance(surface=surf_L, parcellation=parc_l_path, n_proc=n_proc)
        print(f"  Computing Right Hemisphere (n_proc={n_proc})...")
        dist_R = get_surface_distance(surface=surf_R, parcellation=parc_r_path, n_proc=n_proc)

        np.save(dist_l_path, np.asarray(dist_L))
        np.save(dist_r_path, np.asarray(dist_R))
        print("Saved new distance matrices to disk.")

        # Reload as memmaps (optional, but keeps behavior consistent)
        dist_L = np.load(dist_l_path, mmap_mode="r")
        dist_R = np.load(dist_r_path, mmap_mode="r")

    # Validate matrix shapes.
    if dist_L.ndim != 2 or dist_L.shape[0] != dist_L.shape[1]:
        raise ValueError(f"Left distmat must be square 2D; got {dist_L.shape}")
    if dist_R.ndim != 2 or dist_R.shape[0] != dist_R.shape[1]:
        raise ValueError(f"Right distmat must be square 2D; got {dist_R.shape}")

    return dist_L, dist_R

def _p_to_stars(p):
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""
    
def _mask_to_bool(mask_idx, n_parcels: int) -> np.ndarray:
    """
    Convert mask_idx (bool mask or integer indices) into a boolean mask of length n_parcels.
    """
    mask_idx = np.asarray(mask_idx)
    if mask_idx.dtype == bool:
        if mask_idx.shape[0] != n_parcels:
            raise ValueError(f"Boolean mask length {mask_idx.shape[0]} != n_parcels {n_parcels}")
        return mask_idx
    # integer indices
    mask_bool = np.zeros(n_parcels, dtype=bool)
    mask_bool[mask_idx.astype(int)] = True
    return mask_bool


def compute_msr_statistics(
    target_data, ref_data, mask_idx, target_idxs, ref_idxs,
    dist_l_path, dist_r_path, parc_l_path, parc_r_path,
    n_proc=8, n_perm=1000, seed=42, metric='pearson'
):
    """
    MSR-corrected correlations between target maps and reference maps.

    Scientific details:
    - Nulls are generated on the tested parcel set by assigning NaN outside
      ``mask_idx`` before calling ``nulls.moran``.
    - No mean-imputation: NaNs are handled by neuromaps' internal masking.
    """
    # 0) Select correlation function
    if metric == 'pearson':
        corr_func = pearsonr
    elif metric == 'spearman':
        corr_func = spearmanr
    elif metric == 'kendall':
        corr_func = kendalltau
    else:
        raise ValueError(f"Unknown metric '{metric}'. Use 'pearson', 'spearman', or 'kendall'.")

    target_data = np.asarray(target_data)
    ref_data = np.asarray(ref_data)

    if target_data.ndim != 2 or ref_data.ndim != 2:
        raise ValueError(f"target_data and ref_data must be 2D (parcels x maps). Got {target_data.shape}, {ref_data.shape}")
    if target_data.shape[0] != ref_data.shape[0]:
        raise ValueError(f"target_data and ref_data must have same #parcels. Got {target_data.shape[0]} vs {ref_data.shape[0]}")

    n_parcels = target_data.shape[0]
    mask_bool = _mask_to_bool(mask_idx, n_parcels)

    # 1) Geometry
    dist_L, dist_R = get_or_compute_geodesic_distmat(dist_l_path, dist_r_path, parc_l_path, parc_r_path, n_proc=n_proc)

    # Data must be ordered as [left parcels, right parcels] to match the distance matrices.
    nL, nR = dist_L.shape[0], dist_R.shape[0]
    if nL + nR != n_parcels:
        raise ValueError(
            f"dist_L + dist_R sizes ({nL}+{nR}={nL+nR}) do not match data length ({n_parcels}). "
            "The data and distance matrices use different parcel ordering or inclusion."
        )

    distmat = (dist_L, dist_R)

    n_t, n_r = len(target_idxs), len(ref_idxs)
    p_values = np.ones((n_t, n_r), dtype=float)
    r_values = np.zeros((n_t, n_r), dtype=float)

    print(f"Computing MSR ({metric}, n_perm={n_perm}) for {n_t} target(s) vs {n_r} reference(s)...")

    for i, t_idx in enumerate(target_idxs):
        # Generate nulls on the analysis domain only.
        x = target_data[:, t_idx].astype(float, copy=True)
        x[~mask_bool] = np.nan

        # Generate nulls for this target
        rotated_nulls = nulls.moran(
            data=x,
            distmat=distmat,
            n_perm=n_perm,
            seed=seed + int(t_idx),
            n_proc=1
        )

        for j, r_idx in enumerate(ref_idxs):
            y = ref_data[:, r_idx].astype(float, copy=False)

            # Valid samples = in mask and finite in both vectors
            valid = mask_bool & np.isfinite(x) & np.isfinite(y)

            # Insufficient data to define a correlation.
            if valid.sum() < 3:
                r_emp = 0.0
                p_val = 1.0
                r_values[i, j] = r_emp
                p_values[i, j] = p_val
                print(f"  Target G{t_idx+1} vs Ref G{r_idx+1}: {metric}=NA (n<3), p_moran=1.0000")
                continue

            r_emp = corr_func(x[valid], y[valid])[0]
            if not np.isfinite(r_emp):
                # Undefined correlation, for example from zero variance.
                r_emp = 0.0

            r_values[i, j] = r_emp

            # Null correlations (same valid mask for y; null map already has NaNs outside analysis set)
            null_r = np.zeros(n_perm, dtype=float)
            for p in range(n_perm):
                x_null = rotated_nulls[:, p]
                v = valid & np.isfinite(x_null)
                if v.sum() < 3:
                    null_r[p] = 0.0
                else:
                    rr = corr_func(x_null[v], y[v])[0]
                    null_r[p] = 0.0 if not np.isfinite(rr) else rr

            n_extreme = np.sum(np.abs(null_r) >= np.abs(r_emp))
            p_val = (1 + n_extreme) / (1 + n_perm)
            p_values[i, j] = p_val

            print(f"  Target G{t_idx+1} vs Ref G{r_idx+1}: {metric}={r_emp:.2f}, p_moran={p_val:.4f}")

    return r_values, p_values

__all__ = [
    "inverse_fisher_z_transform",
    "zavg_ignore",
    "safe_pearsonr",
    "union_topk_cols",
    "load_glasso_support_prevalence",
    "consensus_mask_from_prevalence",
    "compute_seedwise_topk_mask",
    "seedwise_topk_masks_from_prevalence",
    "sanitize_removed_list",
    "kept_sorted_from_removed",
    "map_top_to_full360",
    "run_gradients",
    "correlate_gradients_to_columns",
    "run_gradients_hemi_align_right_to_left",
    "plot_eigen_spectrum",
    "calc_dominance_normalized",
    "calc_sharedness_normalized",
    "get_or_compute_geodesic_distmat",
    "compute_msr_statistics",
]
