"""
Numerical summaries shared by the manuscript group analyses. 
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr

from helper import calc_dominance_normalized, calc_sharedness_normalized


def strength_stats_from_mask(
    seed2ctx: np.ndarray,
    mask_seedwise: np.ndarray,
    amyg_rows: np.ndarray,
    hipp_rows: np.ndarray,
    positive_only: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Summarize masked amygdala and hippocampus connection strengths."""
    strengths = seed2ctx.astype(float, copy=False)
    if positive_only:
        strengths = np.clip(strengths, 0.0, None)
    strengths = np.where(mask_seedwise, strengths, np.nan)

    amyg_count = mask_seedwise[amyg_rows, :].sum(axis=0).astype(int)
    hipp_count = mask_seedwise[hipp_rows, :].sum(axis=0).astype(int)
    overlap = (amyg_count > 0) & (hipp_count > 0)
    union = (amyg_count > 0) | (hipp_count > 0)

    amyg_mean = np.nanmean(strengths[amyg_rows, :], axis=0)
    hipp_mean = np.nanmean(strengths[hipp_rows, :], axis=0)

    # Treat an absent seed family as zero for the difference and ratio.
    amyg_zero_filled = np.nan_to_num(amyg_mean, nan=0.0)
    hipp_zero_filled = np.nan_to_num(hipp_mean, nan=0.0)
    net_difference = amyg_zero_filled - hipp_zero_filled

    denominator = amyg_zero_filled + hipp_zero_filled
    strength_dominance = np.full_like(denominator, np.nan, dtype=float)
    valid = denominator > 0
    strength_dominance[valid] = (
        amyg_zero_filled[valid] - hipp_zero_filled[valid]
    ) / denominator[valid]
    return net_difference, strength_dominance, overlap, union


def overlap_only_scatter_arrays(
    x_360: np.ndarray,
    y_360: np.ndarray,
    overlap_360: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return copies of two maps with non-overlap parcels set to NaN."""
    x_values = np.asarray(x_360, float).copy()
    y_values = np.asarray(y_360, float).copy()
    x_values[~overlap_360] = np.nan
    y_values[~overlap_360] = np.nan
    return x_values, y_values


def corr_r_n(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """Return Pearson r and the finite paired sample size."""
    finite = np.isfinite(x) & np.isfinite(y)
    n_pairs = int(finite.sum())
    if n_pairs < 3:
        return np.nan, n_pairs
    correlation, _ = pearsonr(x[finite], y[finite])
    return float(correlation), n_pairs


def robust_range_pos(
    a: np.ndarray,
    b: np.ndarray,
    q: float = 99.0,
) -> tuple[float, float]:
    values = np.concatenate([np.ravel(a), np.ravel(b)]).astype(float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    vmax = max(float(np.percentile(values, q)), 1e-6)
    return 0.0, vmax


def robust_range_sym(
    a: np.ndarray,
    b: np.ndarray,
    q: float = 99.0,
) -> tuple[float, float]:
    values = np.concatenate([np.ravel(a), np.ravel(b)]).astype(float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return -1.0, 1.0
    vmax = max(float(np.percentile(np.abs(values), q)), 1e-6)
    return -vmax, vmax


def seed_weighted_mean(
    seed2ctx_row: np.ndarray,
    mask_row: np.ndarray,
    map_ctx: np.ndarray,
    positive_only: bool = True,
    extra_ctx_mask: np.ndarray | None = None,
) -> float:
    """Average a cortical map using one seed's masked FC values as weights."""
    weights = np.asarray(seed2ctx_row, float).copy()
    mask = np.asarray(mask_row, bool)
    cortical_map = np.asarray(map_ctx, float)

    if weights.shape[0] != cortical_map.shape[0]:
        raise ValueError("seed2ctx_row length != map_ctx length")
    if mask.shape[0] != cortical_map.shape[0]:
        raise ValueError("mask_row length != map_ctx length")

    weights[~mask] = np.nan
    if positive_only:
        weights = np.where(
            np.isfinite(weights),
            np.clip(weights, 0.0, None),
            np.nan,
        )

    if extra_ctx_mask is not None:
        extra_mask = np.asarray(extra_ctx_mask, bool)
        if extra_mask.shape[0] != cortical_map.shape[0]:
            raise ValueError("extra_ctx_mask length mismatch")
        weights[~extra_mask] = np.nan

    # Undefined map parcels must not contribute to the denominator.
    weights[~np.isfinite(cortical_map)] = np.nan
    denominator = np.nansum(weights)
    if denominator <= 0:
        return np.nan
    return float(np.nansum(weights * cortical_map) / denominator)


def compute_seed_scores_loso(
    seed2ctx: np.ndarray,
    mask_seedwise: np.ndarray,
    amyg_rows: np.ndarray,
    hipp_rows: np.ndarray,
    total_amyg: int,
    total_hipp: int,
    union_ctx: np.ndarray,
    positive_only: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute leave-one-seed-out hubness and preference scores."""
    n_seeds = seed2ctx.shape[0]
    hubness = np.full(n_seeds, np.nan, dtype=float)
    preference = np.full(n_seeds, np.nan, dtype=float)

    amyg_set = set(amyg_rows.tolist())
    hipp_set = set(hipp_rows.tolist())
    amyg_count_all = mask_seedwise[amyg_rows, :].sum(axis=0).astype(float)
    hipp_count_all = mask_seedwise[hipp_rows, :].sum(axis=0).astype(float)

    for seed_index in range(n_seeds):
        remaining_amyg = total_amyg - (1 if seed_index in amyg_set else 0)
        remaining_hipp = total_hipp - (1 if seed_index in hipp_set else 0)
        if remaining_amyg <= 0 or remaining_hipp <= 0:
            continue

        amyg_count = amyg_count_all.copy()
        hipp_count = hipp_count_all.copy()
        if seed_index in amyg_set:
            amyg_count -= mask_seedwise[seed_index, :].astype(float)
        if seed_index in hipp_set:
            hipp_count -= mask_seedwise[seed_index, :].astype(float)

        dominance_loso = calc_dominance_normalized(
            amyg_count,
            hipp_count,
            remaining_amyg,
            remaining_hipp,
        )
        sharedness_loso = calc_sharedness_normalized(
            amyg_count,
            hipp_count,
            remaining_amyg,
            remaining_hipp,
        )
        preference[seed_index] = seed_weighted_mean(
            seed2ctx[seed_index, :],
            mask_seedwise[seed_index, :],
            dominance_loso,
            positive_only=positive_only,
            extra_ctx_mask=union_ctx,
        )
        hubness[seed_index] = seed_weighted_mean(
            seed2ctx[seed_index, :],
            mask_seedwise[seed_index, :],
            sharedness_loso,
            positive_only=positive_only,
            extra_ctx_mask=union_ctx,
        )

    return hubness, preference


def mean_positive_r_from_z_stack(z_stack: np.ndarray) -> np.ndarray:
    """Return the subject mean of positive r, including zero-valued edges."""
    r_stack = np.tanh(np.asarray(z_stack, dtype=float))
    return np.nanmean(np.clip(r_stack, 0.0, None), axis=0)
