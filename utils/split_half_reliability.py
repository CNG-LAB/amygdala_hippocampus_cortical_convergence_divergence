"""Split-half reliability analysis for the cortical maps and seed scores."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from group_analysis import (
    compute_seed_scores_loso,
    mean_positive_r_from_z_stack,
    strength_stats_from_mask,
)
from helper import (
    calc_dominance_normalized,
    calc_sharedness_normalized,
    compute_seedwise_topk_mask,
    map_top_to_full360,
    seedwise_topk_masks_from_prevalence,
    zavg_ignore,
)


def run_split_half_reliability(
    *,
    top_percent: int,
    output_root: str,
    subjects: list[str],
    base_dir: str,
    n_total: int,
    seed_rows: np.ndarray,
    ctx_indices: np.ndarray,
    pearson_z_stack: np.ndarray,
    glasso_z_stack: np.ndarray,
    amyg_rows_all: np.ndarray,
    hipp_rows_all: np.ndarray,
    kept_glp: np.ndarray,
    n_ctx: int,
    amyg_L_idx: np.ndarray,
    amyg_R_idx: np.ndarray,
    hipp_L_idx: np.ndarray,
    hipp_R_idx: np.ndarray,
    n_total_amyg: int,
    n_total_hipp: int,
) -> None:
    """Run 200 deterministic splits and write reliability summaries."""
    TOP_PERCENT = top_percent
    OUTPUT_ROOT = output_root
    BASE_DIR = base_dir
    N_TOTAL = n_total

    SH_TOP_PERCENT = TOP_PERCENT
    SH_N_SPLITS = 200
    SH_SEED = 20260428

    SH_INCLUDE_SEED_SCORES = True
    SH_INCLUDE_AH_MATRICES = True

    SH_OUT_DIR = os.path.join(OUTPUT_ROOT, f"split_half_reliability_top{SH_TOP_PERCENT}")
    os.makedirs(SH_OUT_DIR, exist_ok=True)

    rng = np.random.default_rng(SH_SEED)

    def _load_glasso_positive_support_stack(subjects_to_scan):
        """Load positive GLASSO seed-to-cortex support for every subject."""
        fname = f"connectivity_matrix_REST_{N_TOTAL}x{N_TOTAL}_glasso_partial.npy"

        support_list = []

        for sid in subjects_to_scan:
            fpath = os.path.join(BASE_DIR, sid, "func_native_2025", fname)
            if not os.path.exists(fpath):
                raise FileNotFoundError(fpath)
            M = np.load(fpath, mmap_mode="r")

            if M.ndim != 2 or M.shape != (N_TOTAL, N_TOTAL):
                raise ValueError(f"{sid}: got GLASSO partial shape {M.shape}")

            block = np.asarray(M[seed_rows[:, None], ctx_indices], dtype=float)
            support_list.append(np.isfinite(block) & (block > 0.0))

        return np.stack(support_list, axis=0).astype(bool)


    def _align_subject_stacks():
        """
        Align Pearson z, GLASSO z, and GLASSO partial-support stacks to the same subject order.
        Uses already-loaded pearson_z_stack and glasso_z_stack from the main script.
        """
        print("[SH] Loading GLASSO positive support stack...")
        support_stack_all = _load_glasso_positive_support_stack(subjects)
        common_sids = list(subjects)

        # Extract only seed->cortex z blocks to keep memory and runtime reasonable.
        pearson_seed2ctx_z = pearson_z_stack[:, seed_rows[:, None], ctx_indices].astype(float, copy=False)
        glasso_seed2ctx_z = glasso_z_stack[:, seed_rows[:, None], ctx_indices].astype(float, copy=False)
        glasso_support_pos = support_stack_all.astype(bool, copy=False)

        if SH_INCLUDE_AH_MATRICES:
            pearson_ah_z = pearson_z_stack[:, amyg_rows_all[:, None], hipp_rows_all].astype(float, copy=False)
            glasso_ah_z = glasso_z_stack[:, amyg_rows_all[:, None], hipp_rows_all].astype(float, copy=False)
        else:
            pearson_ah_z = None
            glasso_ah_z = None

        with open(os.path.join(SH_OUT_DIR, "common_subjects_used.txt"), "w") as f:
            f.write("\n".join(common_sids) + "\n")

        print(f"[SH] Common subjects used: {len(common_sids)}")
        print(f"[SH] seed->cortex stack shape: {pearson_seed2ctx_z.shape}")
        return common_sids, pearson_seed2ctx_z, glasso_seed2ctx_z, glasso_support_pos, pearson_ah_z, glasso_ah_z


    def _expand_ctx_to_full360(vec_ctx):
        vec_ctx = np.asarray(vec_ctx).squeeze()
        if vec_ctx.ndim != 1 or vec_ctx.shape[0] != n_ctx:
            raise ValueError(f"Expected ctx vector of length {n_ctx}, got {vec_ctx.shape}")

        kept_0based = kept_glp.astype(int) - 1
        out = map_top_to_full360(vec_ctx[None, :], kept_0based, full_len=360, fill=np.nan)
        return out[0]


    def _corr_pair(x, y, method="pearson"):
        x = np.asarray(x, float).ravel()
        y = np.asarray(y, float).ravel()
        m = np.isfinite(x) & np.isfinite(y)

        n = int(m.sum())
        if n < 3:
            return np.nan, np.nan, n

        if np.nanstd(x[m]) < 1e-12 or np.nanstd(y[m]) < 1e-12:
            return np.nan, np.nan, n

        if method == "pearson":
            r, p = pearsonr(x[m], y[m])
        elif method == "spearman":
            r, p = spearmanr(x[m], y[m])
        else:
            raise ValueError("method must be 'pearson' or 'spearman'")

        return float(r), float(p), n


    def _dice_bool(a, b):
        a = np.asarray(a, bool)
        b = np.asarray(b, bool)
        denom = int(a.sum() + b.sum())
        if denom == 0:
            return np.nan
        return float(2 * np.logical_and(a, b).sum() / denom)


    def _jaccard_bool(a, b):
        a = np.asarray(a, bool)
        b = np.asarray(b, bool)
        denom = int(np.logical_or(a, b).sum())
        if denom == 0:
            return np.nan
        return float(np.logical_and(a, b).sum() / denom)


    def _cohen_kappa_categorical(a, b):
        """
        Cohen kappa for finite categorical arrays. Used for method-overlap category maps.
        """
        a = np.asarray(a).ravel()
        b = np.asarray(b).ravel()
        m = np.isfinite(a.astype(float)) & np.isfinite(b.astype(float))
        if int(m.sum()) < 2:
            return np.nan, int(m.sum())

        aa = a[m].astype(int)
        bb = b[m].astype(int)

        cats = np.union1d(np.unique(aa), np.unique(bb))
        if cats.size < 2:
            return np.nan, int(m.sum())

        n = aa.size
        po = np.mean(aa == bb)

        pe = 0.0
        for c in cats:
            pe += np.mean(aa == c) * np.mean(bb == c)

        if np.isclose(1.0 - pe, 0.0):
            return np.nan, int(n)

        return float((po - pe) / (1.0 - pe)), int(n)

    def _make_ipsi_mask_ah():
        """
        AH matrix shape: amygdala rows = L9,R9; hippocampus cols = L15,R15.
        """
        n_aL = len(amyg_L_idx)
        n_aR = len(amyg_R_idx)
        n_hL = len(hipp_L_idx)
        n_hR = len(hipp_R_idx)

        mask = np.zeros((n_aL + n_aR, n_hL + n_hR), dtype=bool)
        mask[:n_aL, :n_hL] = True
        mask[n_aL:n_aL + n_aR, n_hL:n_hL + n_hR] = True
        return mask


    ah_ipsi_mask = _make_ipsi_mask_ah()

    def _compute_half_outputs(pearson_seed2ctx_z_half, glasso_seed2ctx_z_half, glasso_support_pos_half,
                              pearson_ah_z_half=None, glasso_ah_z_half=None):
        """
        Compute all split-half outputs from subject-level half stacks.
        """
        # Group seed->cortex matrices using the same z-averaging logic as the main script.
        seed2ctx_pe = zavg_ignore(pearson_seed2ctx_z_half, drop_zeros=False, axis=0)
        seed2ctx_gl = zavg_ignore(glasso_seed2ctx_z_half, drop_zeros=True, axis=0)

        # GLASSO top-k by positive support prevalence.
        prevalence_gl = glasso_support_pos_half.mean(axis=0)
        mask_gl = seedwise_topk_masks_from_prevalence(prevalence_gl, top_percent=SH_TOP_PERCENT)

        # Pearson top-k by group positive r.
        mask_pe = compute_seedwise_topk_mask(
            seed2ctx_pe,
            top_percent=SH_TOP_PERCENT,
            positive_only=True,
        )

        # Counts: Amyg vs Hipp.
        amyg_count_gl = mask_gl[amyg_rows_all, :].sum(axis=0).astype(float)
        hipp_count_gl = mask_gl[hipp_rows_all, :].sum(axis=0).astype(float)

        amyg_count_pe = mask_pe[amyg_rows_all, :].sum(axis=0).astype(float)
        hipp_count_pe = mask_pe[hipp_rows_all, :].sum(axis=0).astype(float)

        D_gl = calc_dominance_normalized(amyg_count_gl, hipp_count_gl, n_total_amyg, n_total_hipp)
        S_gl = calc_sharedness_normalized(amyg_count_gl, hipp_count_gl, n_total_amyg, n_total_hipp)

        D_pe = calc_dominance_normalized(amyg_count_pe, hipp_count_pe, n_total_amyg, n_total_hipp)
        S_pe = calc_sharedness_normalized(amyg_count_pe, hipp_count_pe, n_total_amyg, n_total_hipp)

        # Strength corroboration maps.
        net_gl, sdom_gl, overlap_gl, union_gl = strength_stats_from_mask(
            seed2ctx_gl, mask_gl, amyg_rows_all, hipp_rows_all, positive_only=True
        )
        net_pe, sdom_pe, overlap_pe, union_pe = strength_stats_from_mask(
            seed2ctx_pe, mask_pe, amyg_rows_all, hipp_rows_all, positive_only=True
        )

        maps_ctx = {
            "Dominance_GLASSO": D_gl,
            "Sharedness_GLASSO": S_gl,
            "Dominance_Pearson": D_pe,
            "Sharedness_Pearson": S_pe,
            "StrengthDiff_AmygMinusHipp_GLASSO": net_gl,
            "StrengthDiff_AmygMinusHipp_Pearson": net_pe,
            "StrengthDominance_GLASSO": sdom_gl,
            "StrengthDominance_Pearson": sdom_pe,
        }

        maps360 = {name: _expand_ctx_to_full360(val) for name, val in maps_ctx.items()}

        # Method-overlap category map.
        g_any = np.any(mask_gl, axis=0)
        p_any = np.any(mask_pe, axis=0)

        cat_ctx = np.zeros(n_ctx, dtype=float)
        cat_ctx[g_any & ~p_any] = 1.0
        cat_ctx[~g_any & p_any] = 2.0
        cat_ctx[g_any & p_any] = 3.0
        cat360 = _expand_ctx_to_full360(cat_ctx)

        out = {
            "seed2ctx_gl": seed2ctx_gl,
            "seed2ctx_pe": seed2ctx_pe,
            "prevalence_gl": prevalence_gl,
            "mask_gl": mask_gl,
            "mask_pe": mask_pe,
            "maps_ctx": maps_ctx,
            "maps360": maps360,
            "cat360": cat360,
            "union_ctx_gl": union_gl,
            "union_ctx_pe": union_pe,
            "overlap_ctx_gl": overlap_gl,
            "overlap_ctx_pe": overlap_pe,
        }

        if SH_INCLUDE_SEED_SCORES:
            seed_scores = {}

            hub_pe_loso, pref_pe_loso = compute_seed_scores_loso(
                seed2ctx_pe, mask_pe,
                amyg_rows_all, hipp_rows_all,
                n_total_amyg, n_total_hipp,
                union_pe, positive_only=True
            )
            hub_gl_loso, pref_gl_loso = compute_seed_scores_loso(
                seed2ctx_gl, mask_gl,
                amyg_rows_all, hipp_rows_all,
                n_total_amyg, n_total_hipp,
                union_gl, positive_only=True
            )


            seed_scores["HubnessToSharedness_Pearson_LOSO"] = hub_pe_loso
            seed_scores["HubnessToSharedness_GLASSO_LOSO"] = hub_gl_loso
            seed_scores["PreferenceToDominance_Pearson_LOSO"] = pref_pe_loso
            seed_scores["PreferenceToDominance_GLASSO_LOSO"] = pref_gl_loso

            out["seed_scores"] = seed_scores

        if SH_INCLUDE_AH_MATRICES:
            if pearson_ah_z_half is None or glasso_ah_z_half is None:
                raise ValueError("AH z-stacks were requested but not provided.")

            AH_pe = mean_positive_r_from_z_stack(pearson_ah_z_half)
            AH_gl = mean_positive_r_from_z_stack(glasso_ah_z_half)

            # Positive support prevalence for GLASSO AH.
            AH_gl_pos_prev = np.mean(np.isfinite(glasso_ah_z_half) & (glasso_ah_z_half > 0.0), axis=0)

            out["ah_mats"] = {
                "AH_ExpectedPosFC_Pearson": AH_pe,
                "AH_ExpectedPosFC_GLASSO": AH_gl,
                "AH_PosPrevalence_GLASSO": AH_gl_pos_prev,
            }

        return out

    def _compare_halves(res_a, res_b):
        """
        Return one wide row of reliability metrics for a split.
        """
        row = {}

        # Continuous cortical maps.
        for name in res_a["maps360"].keys():
            r, p, n = _corr_pair(res_a["maps360"][name], res_b["maps360"][name], method="pearson")
            row[f"{name}__pearson_r"] = r
            row[f"{name}__pearson_p_naive"] = p
            row[f"{name}__n_parcels"] = n

            rho, pp, _ = _corr_pair(res_a["maps360"][name], res_b["maps360"][name], method="spearman")
            row[f"{name}__spearman_rho"] = rho
            row[f"{name}__spearman_p_naive"] = pp

        # Top-k mask reliability.
        for method_label, key in [("GLASSO", "mask_gl"), ("Pearson", "mask_pe")]:
            ma = res_a[key]
            mb = res_b[key]

            seedwise_dice = [_dice_bool(ma[i], mb[i]) for i in range(ma.shape[0])]
            seedwise_jaccard = [_jaccard_bool(ma[i], mb[i]) for i in range(ma.shape[0])]
            row[f"SeedwiseMaskDice_{method_label}__mean"] = float(np.nanmean(seedwise_dice))
            row[f"SeedwiseMaskJaccard_{method_label}__mean"] = float(np.nanmean(seedwise_jaccard))

            row[f"UnionMaskDice_{method_label}"] = _dice_bool(np.any(ma, axis=0), np.any(mb, axis=0))
            row[f"UnionMaskJaccard_{method_label}"] = _jaccard_bool(np.any(ma, axis=0), np.any(mb, axis=0))

        # Method-overlap categorical map.
        kappa, n_k = _cohen_kappa_categorical(res_a["cat360"], res_b["cat360"])
        row["MethodOverlapCategory__cohen_kappa"] = kappa
        row["MethodOverlapCategory__n_parcels"] = n_k

        # Seed-score reliability across 48 seeds.
        if SH_INCLUDE_SEED_SCORES:
            for name in res_a["seed_scores"].keys():
                r, p, n = _corr_pair(res_a["seed_scores"][name], res_b["seed_scores"][name], method="pearson")
                row[f"{name}__seed_pearson_r"] = r
                row[f"{name}__seed_pearson_p_naive"] = p
                row[f"{name}__n_seeds"] = n

                rho, pp, _ = _corr_pair(res_a["seed_scores"][name], res_b["seed_scores"][name], method="spearman")
                row[f"{name}__seed_spearman_rho"] = rho
                row[f"{name}__seed_spearman_p_naive"] = pp

        # Amygdala x hippocampus matrix reliability.
        if SH_INCLUDE_AH_MATRICES:
            for name in res_a["ah_mats"].keys():
                A = res_a["ah_mats"][name]
                B = res_b["ah_mats"][name]

                r, p, n = _corr_pair(A, B, method="pearson")
                row[f"{name}__matrix_pearson_r_all"] = r
                row[f"{name}__matrix_pearson_p_naive_all"] = p
                row[f"{name}__n_cells_all"] = n

                rho, pp, _ = _corr_pair(A, B, method="spearman")
                row[f"{name}__matrix_spearman_rho_all"] = rho
                row[f"{name}__matrix_spearman_p_naive_all"] = pp

                r, p, n = _corr_pair(A[ah_ipsi_mask], B[ah_ipsi_mask], method="pearson")
                row[f"{name}__matrix_pearson_r_ipsi"] = r
                row[f"{name}__matrix_pearson_p_naive_ipsi"] = p
                row[f"{name}__n_cells_ipsi"] = n

                rho, pp, _ = _corr_pair(A[ah_ipsi_mask], B[ah_ipsi_mask], method="spearman")
                row[f"{name}__matrix_spearman_rho_ipsi"] = rho
                row[f"{name}__matrix_spearman_p_naive_ipsi"] = pp

        return row


    def _summarize_wide(df_wide):
        id_cols = {"split", "n_half_A", "n_half_B"}
        metric_cols = [
            c for c in df_wide.columns
            if c not in id_cols
            and not c.endswith("__pearson_p_naive")
            and not c.endswith("__spearman_p_naive")
            and not c.endswith("__seed_pearson_p_naive")
            and not c.endswith("__seed_spearman_p_naive")
            and not c.endswith("__matrix_pearson_p_naive_all")
            and not c.endswith("__matrix_spearman_p_naive_all")
            and not c.endswith("__matrix_pearson_p_naive_ipsi")
            and not c.endswith("__matrix_spearman_p_naive_ipsi")
            and not c.startswith("MethodOverlapCategory__n")
            and not c.endswith("__n_parcels")
            and not c.endswith("__n_seeds")
            and not c.endswith("__n_cells_all")
            and not c.endswith("__n_cells_ipsi")
        ]

        rows = []
        for c in metric_cols:
            vals = pd.to_numeric(df_wide[c], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]

            if vals.size == 0:
                rows.append({
                    "metric": c,
                    "n_splits_finite": 0,
                    "mean": np.nan,
                    "sd": np.nan,
                    "median": np.nan,
                    "q025": np.nan,
                    "q975": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                })
                continue

            rows.append({
                "metric": c,
                "n_splits_finite": int(vals.size),
                "mean": float(np.mean(vals)),
                "sd": float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan,
                "median": float(np.median(vals)),
                "q025": float(np.quantile(vals, 0.025)),
                "q975": float(np.quantile(vals, 0.975)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            })

        out = pd.DataFrame(rows).sort_values("median", ascending=False)
        return out


    def _plot_selected_summary(summary_df, out_png):
        selected_metrics = [
            "Dominance_GLASSO__pearson_r",
            "Sharedness_GLASSO__pearson_r",
            "Dominance_Pearson__pearson_r",
            "Sharedness_Pearson__pearson_r",
            "StrengthDominance_GLASSO__pearson_r",
            "StrengthDominance_Pearson__pearson_r",
            "SeedwiseMaskDice_GLASSO__mean",
            "SeedwiseMaskDice_Pearson__mean",
            "UnionMaskDice_GLASSO",
            "UnionMaskDice_Pearson",
            "MethodOverlapCategory__cohen_kappa",
        ]

        sub = summary_df[summary_df["metric"].isin(selected_metrics)].copy()
        if sub.empty:
            print("[SH][WARN] No selected summary metrics found for plotting.")
            return

        sub["metric"] = pd.Categorical(sub["metric"], categories=selected_metrics, ordered=True)
        sub = sub.sort_values("metric")

        y = np.arange(len(sub))
        med = sub["median"].to_numpy(float)
        lo = sub["q025"].to_numpy(float)
        hi = sub["q975"].to_numpy(float)

        fig, ax = plt.subplots(figsize=(9, max(5, 0.38 * len(sub))))
        ax.errorbar(
            med, y,
            xerr=np.vstack([med - lo, hi - med]),
            fmt="o",
            capsize=3,
            lw=1.5,
        )
        ax.axvline(0.0, color="0.6", lw=1, ls="--")
        ax.axvline(0.8, color="0.8", lw=1, ls=":")
        ax.set_yticks(y)
        ax.set_yticklabels(sub["metric"].astype(str))
        ax.set_xlabel("Split-half reliability: median with 95% split interval")
        ax.set_title(f"Split-half reliability, top {SH_TOP_PERCENT}%, n_splits={SH_N_SPLITS}")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[SH] Saved selected-metric plot: {out_png}")


    (
        common_sids,
        pearson_seed2ctx_z_common,
        glasso_seed2ctx_z_common,
        glasso_support_pos_common,
        pearson_ah_z_common,
        glasso_ah_z_common,
    ) = _align_subject_stacks()

    n_common = len(common_sids)
    if n_common % 2 != 0:
        print(f"[SH][INFO] Odd number of common subjects ({n_common}); halves will be {n_common // 2} and {n_common - n_common // 2}.")

    records = []

    print(f"[SH] Running {SH_N_SPLITS} deterministic random split-halves...")

    for split_i in range(SH_N_SPLITS):
        perm = rng.permutation(n_common)
        n_a = n_common // 2
        idx_a = np.sort(perm[:n_a])
        idx_b = np.sort(perm[n_a:])

        res_a = _compute_half_outputs(
            pearson_seed2ctx_z_common[idx_a],
            glasso_seed2ctx_z_common[idx_a],
            glasso_support_pos_common[idx_a],
            pearson_ah_z_common[idx_a] if SH_INCLUDE_AH_MATRICES else None,
            glasso_ah_z_common[idx_a] if SH_INCLUDE_AH_MATRICES else None,
        )

        res_b = _compute_half_outputs(
            pearson_seed2ctx_z_common[idx_b],
            glasso_seed2ctx_z_common[idx_b],
            glasso_support_pos_common[idx_b],
            pearson_ah_z_common[idx_b] if SH_INCLUDE_AH_MATRICES else None,
            glasso_ah_z_common[idx_b] if SH_INCLUDE_AH_MATRICES else None,
        )

        row = {
            "split": split_i,
            "n_half_A": int(len(idx_a)),
            "n_half_B": int(len(idx_b)),
        }
        row.update(_compare_halves(res_a, res_b))
        records.append(row)

        if (split_i + 1) % 10 == 0 or (split_i + 1) == SH_N_SPLITS:
            print(f"[SH] Finished split {split_i + 1}/{SH_N_SPLITS}")

    df_wide = pd.DataFrame.from_records(records)

    wide_tsv = os.path.join(SH_OUT_DIR, "split_half_reliability_wide.tsv")
    df_wide.to_csv(wide_tsv, sep="\t", index=False, float_format="%.6f")
    print(f"[SH] Saved wide split-half table: {wide_tsv}")

    summary_df = _summarize_wide(df_wide)
    summary_tsv = os.path.join(SH_OUT_DIR, "split_half_reliability_summary.tsv")
    summary_df.to_csv(summary_tsv, sep="\t", index=False, float_format="%.6f")
    print(f"[SH] Saved summary table: {summary_tsv}")
    plot_png = os.path.join(SH_OUT_DIR, "split_half_reliability_selected_metrics.png")
    _plot_selected_summary(summary_df, plot_png)

    print("\n[SH] Key split-half reliability summaries:")
    key_show = [
        "Dominance_GLASSO__pearson_r",
        "Sharedness_GLASSO__pearson_r",
        "Dominance_Pearson__pearson_r",
        "Sharedness_Pearson__pearson_r",
        "StrengthDominance_GLASSO__pearson_r",
        "StrengthDominance_Pearson__pearson_r",
        "SeedwiseMaskDice_GLASSO__mean",
        "SeedwiseMaskDice_Pearson__mean",
        "UnionMaskDice_GLASSO",
        "UnionMaskDice_Pearson",
        "MethodOverlapCategory__cohen_kappa",
        "HubnessToSharedness_GLASSO_LOSO__seed_pearson_r",
        "HubnessToSharedness_Pearson_LOSO__seed_pearson_r",
        "AH_ExpectedPosFC_GLASSO__matrix_pearson_r_ipsi",
        "AH_ExpectedPosFC_Pearson__matrix_pearson_r_ipsi",
    ]

    print(
        summary_df[summary_df["metric"].isin(key_show)]
        .loc[:, ["metric", "median", "q025", "q975", "mean", "sd", "n_splits_finite"]]
        .sort_values("metric")
        .to_string(index=False)
    )

    print("\n[SH DONE]")
