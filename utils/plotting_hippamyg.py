"""Plotting helpers used by the analysis scripts."""

from __future__ import annotations

import os
import hashlib
from typing import Dict, List, Tuple, Optional, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
import nibabel as nb
from nilearn import plotting as niplotting
from brainspace.plotting import plot_hemispheres


from hippamyg_labels import (
    region_indices,
    build_hippocampus_labels,
    get_amygdala_labels,
)

from helper import compute_msr_statistics, _p_to_stars

# Canonical Yeo-7 colors in network order: visual, somatomotor, dorsal
# attention, ventral attention, limbic, frontoparietal, and default mode.
YEO7_COLORS = [
    (0.470588, 0.0705882, 0.52549),
    (0.27451, 0.509804, 0.705882),
    (0.0, 0.462745, 0.054902),
    (0.768627, 0.227451, 0.980392),
    (0.862745, 0.972549, 0.643137),
    (0.901961, 0.580392, 0.133333),
    (0.803922, 0.243137, 0.305882),
]

__all__ = [
    "gray_rgba",
    "YEO7_COLORS",
    "plot_seed_maps",
    "create_matplotlib_grid",
    # community plots
    "plot_community_bars_with_strips",
    "plot_metric_scatter_by_community",
    # hipp/amyg plotting & helpers
    "load_hipp_surfaces_and_labels",
    "get_hipp_label_ids",
    "map_gradients_to_hipp_vertices",
    "compute_shared_limits",
    "plot_and_save_hipp_gradients",
    "plot_and_save_hipp_unfolded_gradients",
    "plot_amygdala_scalar_volumes",
    "plot_amygdala_volumes_sharedscale",
    "resolve_indices_and_labels",
]

# Converts to (R, G, B, A)
gray_rgba = mcolors.to_rgba("#bdbdbd")


def _tab20_palette_for_categories(categories, seed: int = 0) -> dict:
    cats = sorted(set(map(str, categories)))
    base = sns.color_palette("tab20", 20)
    base_hex = [mcolors.to_hex(color) for color in base]
    used, palette = set(), {}
    for cat in cats:
        h = int(hashlib.sha1(cat.lower().encode("utf-8")).hexdigest()[:8], 16)
        idx = (h + seed) % 20
        start = idx
        while idx in used:
            idx = (idx + 1) % 20
            if idx == start:
                break
        used.add(idx)
        palette[cat] = base_hex[idx]
    return palette


# =========================
# Composite figure
# =========================

def plot_seed_maps(
    surf_lh,
    surf_rh,
    mapped: Dict[str, Dict[str, np.ndarray]],
    structure: str,
    hemi: str,
    labels: List[str],
    out_dir: str,
    tag: str,
    top_percent: int,
) -> None:
    """
    Plot per-seed maps for a given structure ('Amygdala' or 'Hippocampus') and hemisphere.
    Write one full-brain PNG per seed.
    """
    os.makedirs(out_dir, exist_ok=True)

    block = mapped[structure][hemi]  # shape: (n_seeds_block, n_vertices)
    if not isinstance(block, np.ndarray) or block.ndim != 2:
        raise ValueError(
            f"mapped[{structure}][{hemi}] must be 2D (n_seeds × n_vertices); "
            f"got {None if block is None else getattr(block, 'shape', 'unknown')}"
        )

    n_seeds_block = block.shape[0]
    if n_seeds_block == 0:
        print(f"[PLOT] No seeds for {structure} {hemi} ({tag}); skipping.")
        return

    for i in range(n_seeds_block):
        vertex_vals = block[i]
        label = labels[i] if i < len(labels) else f"{structure}_{i+1}"
        fname = os.path.join(
            out_dir,
            f"{tag}_top{top_percent}_{hemi}_{structure.lower()}2cortex_{label.replace(' ', '_')}.png",
        )

        print(f"[PLOT] {structure} {hemi} ({tag}) — seed {i+1}/{n_seeds_block} → {fname}")

        plot_hemispheres(
            surf_lh,
            surf_rh,
            array_name=vertex_vals,
            size=(1400, 900),
            cmap="viridis",
            nan_color=gray_rgba,
            color_bar=True,
            zoom=1.2,
            interactive=False,
            screenshot=True,
            filename=fname,
            suppress_warnings=True,
            transparent_bg=True,
        )

def crop_image_margin(img_data):
    """Crop transparent or white margins from an RGB or RGBA image."""
    if img_data.shape[2] == 4:
        non_empty = img_data[..., 3] > 0
    else:
        non_empty = np.any(img_data < 1.0, axis=2)

    if not np.any(non_empty):
        return img_data

    coords = np.argwhere(non_empty)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    return img_data[y0:y1, x0:x1]


def create_matplotlib_grid(
    image_paths: List[str], 
    titles: List[str], 
    rows: int, 
    cols: int, 
    fheight: int,
    fwidth: int,
    out_file: str, 
    main_title: str = ""
) -> None:
    """
    Stitches brain PNGs into a tight grid.
    """
    if not image_paths: 
        return
    
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Lato'

    figsize_height = rows * fheight 
    figsize_width = cols * fwidth  

    
    fig, axes = plt.subplots(rows, cols, figsize=(figsize_width, figsize_height))
    
    if rows * cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i < len(image_paths) and os.path.exists(image_paths[i]):
            img = mpimg.imread(image_paths[i])
            img_cropped = crop_image_margin(img)
            ax.imshow(img_cropped)
            display_title = titles[i].replace('_', ' ')
            ax.set_title(display_title, fontsize=10, fontweight='bold', y=0.85)
            ax.axis('off')
        else:
            ax.axis('off')
            
    if main_title:
        plt.suptitle(main_title, fontsize=16, y=1.0)

    plt.subplots_adjust(wspace=0.01, hspace=0.01)

    plt.savefig(out_file, dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"[GRID] Saved composite figure: {out_file}")

# =========================
# Community plots
# =========================
def plot_community_bars_with_strips(
    data_360: np.ndarray,
    tsv_path: str,
    index_col: str,
    scheme_col: str,
    out_path: str,
    title: str,
    ylabel: str,
    ylim: Tuple[float, float] = None,
    yscale: str = "linear",          # "linear" or "symlog" (value axis)
    orientation: str = "vertical",   # "vertical" or "horizontal"
    symlog_linthresh: float = 0.05,  # linear window around 0 for symlog
    inset: bool = False,
    inset_range: Tuple[float, float] = None,  # range on value axis (e.g. (0, 0.3))
    inset_loc: str = "lower left",
    inset_size: Tuple[float, float] = (0.45, 0.45),  # fraction of parent axes (w, h)
    custom_palette: list | dict | None = None,
) -> None:
    """
    Bar plot (mean + 95% CI) with overlaid strip plot (individual parcels),
    annotated with n per community.

    - orientation = "vertical": communities on x, values on y
    - orientation = "horizontal": communities on y, values on x
    - yscale = "symlog": useful for skewed metrics (e.g., sharedness)
    - inset: if True, draw a zoomed-in inset on the value axis (using inset_range)
    """
    orientation = orientation.lower()
    yscale = yscale.lower()

    if orientation not in ("vertical", "horizontal"):
        raise ValueError("orientation must be 'vertical' or 'horizontal'")
    if yscale not in ("linear", "symlog"):
        raise ValueError("yscale must be 'linear' or 'symlog'")

    # Atlas and community metadata.
    df_atlas = pd.read_csv(tsv_path, sep="\t")
    if index_col not in df_atlas.columns:
        print(f"[WARN] Index col '{index_col}' not in TSV. Skipping {title}.")
        return

    df_atlas[index_col] = (
        pd.to_numeric(df_atlas[index_col], errors="coerce")
        .fillna(-1)
        .astype(int)
    )

    # Parcel values.
    if data_360.shape[0] != 360:
        raise ValueError(f"data_360 must have shape (360,), got {data_360.shape}")

    parcel_ids = np.arange(1, 361, dtype=int)
    df_vals = pd.DataFrame({index_col: parcel_ids, "Value": data_360})

    df_merged = pd.merge(df_atlas, df_vals, on=index_col, how="inner")
    df_clean = df_merged.replace({np.inf: np.nan, -np.inf: np.nan})
    df_clean = df_clean.dropna(subset=["Value", scheme_col])

    if df_clean.empty:
        print(f"[WARN] No valid data for {title}")
        return

    df_clean[scheme_col] = df_clean[scheme_col].astype(str)

    grp = df_clean.groupby(scheme_col)["Value"]
    stats = grp.agg(["mean", "count", "max", "min"])
    communities = list(stats.index)  # preserve TSV order

    # Plot styling.
    sns.set_context("paper", font_scale=1.75) 
    sns.set_style("ticks")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": "Lato",
        "axes.edgecolor": "black",
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
    })

    if orientation == "vertical":
        fig, ax = plt.subplots(figsize=(4, 5)) 
    else:
        fig, ax = plt.subplots(figsize=(6, 4))

    ax.set_facecolor("white")
    ax.grid(axis="y" if orientation == "vertical" else "x", which="major")
    ax.grid(axis="x" if orientation == "vertical" else "y", which="major", visible=False)
    ax.set_axisbelow(True)

    if custom_palette is not None:
        if isinstance(custom_palette, dict):
            comm_colors = [custom_palette.get(c, "#333333") for c in communities]
        else:
            # The color list follows the community order in the atlas table.
            if len(custom_palette) < len(communities):
                print(f"[WARN] custom_palette has {len(custom_palette)} colors but {len(communities)} communities found.")
            comm_colors = [custom_palette[i % len(custom_palette)] for i in range(len(communities))]
    else:
        # Deterministic categorical palette
        palette_map = _tab20_palette_for_categories(communities, seed=0)
        comm_colors = [palette_map[c] for c in communities]


    # Bar and parcel-level strip plots
    if orientation == "vertical":
        x_key, y_key = scheme_col, "Value"
    else:
        x_key, y_key = "Value", scheme_col
        
        

    sns.barplot(
        data=df_clean, x=x_key, y=y_key, order=communities,
        palette=comm_colors,
        alpha=1.0,
        errorbar=("ci", 95),
        capsize=0.15,
        err_kws={'linewidth': 1.5, 'color': 'black'},
        edgecolor="none",
        width=0.75,
        ax=ax,
        zorder=2
    )

    sns.stripplot(
        data=df_clean, x=x_key, y=y_key, order=communities,
        color="#333333",
        alpha=0.6,
        jitter=0.25,
        size=4.0,
        edgecolor="white",
        linewidth=0.5,
        ax=ax,
        zorder=3
    )

    # Value-axis limits.
    sns.despine(ax=ax, offset=10, trim=False)

    data_min = float(df_clean["Value"].min())
    data_max = float(df_clean["Value"].max())
    data_range = data_max - data_min if data_max != data_min else 1.0

    if ylim is None:
        axis_min = data_min - 0.1 * data_range
        axis_max = data_max + 0.15 * data_range
    else:
        axis_min, axis_max = ylim
        
    if yscale == "symlog":
        axis_min = max(0, axis_min)

    if orientation == "vertical":
        ax.set_ylim(axis_min, axis_max)
        if yscale == "symlog": ax.set_yscale("symlog", linthresh=symlog_linthresh)
        ax.tick_params(axis='x', rotation=45)
    else:
        ax.set_xlim(axis_min, axis_max)
        if yscale == "symlog": ax.set_xscale("symlog", linthresh=symlog_linthresh)

    # Add sample sizes to the category labels.
    if orientation == "vertical":
        new_labels = [f"{c}\n(n={int(stats.loc[c, 'count'])})" for c in communities]
        ax.set_xticklabels(new_labels, ha="right", rotation=40)
    else:
        new_labels = [f"{c} (n={int(stats.loc[c, 'count'])})" for c in communities]
        ax.set_yticklabels(new_labels)

    ax.set_title(title, fontsize=14, pad=15, weight="bold")
    ax.set_xlabel(ylabel if orientation == "horizontal" else "", fontsize=12, weight="bold")
    ax.set_ylabel(ylabel if orientation == "vertical" else "", fontsize=12, weight="bold")

    # Add a zero line if crossing zero
    if axis_min < 0 < axis_max:
        if orientation == "vertical":
            ax.axhline(0, color="black", linestyle="-", linewidth=0.8, zorder=1)
        else:
            ax.axvline(0, color="black", linestyle="-", linewidth=0.8, zorder=1)

    # Optional inset.
    if inset and inset_range is not None:
        axins = inset_axes(ax, width="40%", height="40%", loc=inset_loc, borderpad=1)
        axins.set_facecolor("white")
        
        sns.barplot(data=df_clean, x=x_key, y=y_key, order=communities,
                    palette=comm_colors, alpha=1.0, errorbar=("ci", 95),
                    err_kws={'linewidth': 1.0, 'color': 'black'}, edgecolor="none", ax=axins)
        
        sns.stripplot(data=df_clean, x=x_key, y=y_key, order=communities,
                      color="#333333", alpha=0.6, jitter=0.25, size=2.5,
                      edgecolor="none", ax=axins, zorder=10)
        
        if orientation == "vertical":
            axins.set_ylim(inset_range)
            axins.set_xticks([])
        else:
            axins.set_xlim(inset_range)
            axins.set_yticks([])

        axins.set_xlabel("")
        axins.set_ylabel("")
        sns.despine(ax=axins)

    if inset and inset_range is not None:
        w_pct = f"{inset_size[0]*100:.0f}%"
        h_pct = f"{inset_size[1]*100:.0f}%"
        axins = inset_axes(ax, width=w_pct, height=h_pct, loc=inset_loc, borderpad=1.0)

        axins.set_facecolor("#f9f9fb")
        axins.grid(axis="y" if orientation == "vertical" else "x", which="major")
        axins.grid(axis="x" if orientation == "vertical" else "y", which="major", visible=False)
        axins.set_axisbelow(True)


        if orientation == "vertical":
            axins.set_ylim(inset_range[0], inset_range[1])
            axins.tick_params(axis="x", labelrotation=35, labelsize=7)
            axins.tick_params(axis="y", labelsize=7)
        else:
            axins.set_xlim(inset_range[0], inset_range[1])
            axins.tick_params(axis="y", labelsize=7)
            axins.tick_params(axis="x", labelsize=7)

        axins.set_xlabel("")
        axins.set_ylabel("")

    plt.tight_layout()
    plt.savefig(out_path, dpi=600)
    plt.close(fig)
    print(f"[PLOT] Saved bar summary: {out_path}")


def plot_metric_scatter_by_community(
    x_360: np.ndarray,
    y_360: np.ndarray,
    tsv_path: str,
    index_col: str,
    scheme_col: str,
    out_path: str,
    xlabel: str,
    ylabel: str,
    title: str,
    xlim: Tuple[float, float] = None,
    ylim: Tuple[float, float] = None,
    add_identity: bool = False,
    add_reg: bool = True,
    legend: bool = False,
    alpha: float = 0.8,
    size_360: np.ndarray | None = None,
    size_scale: Tuple[float, float] = (80.0, 140.0),
    size_use_abs: bool = True,
    corr_method: str = "pearson",
    custom_palette: list | dict | None = None,
    msr: bool = False,
    msr_params: dict | None = None,
    msr_mask_mode: str = "finite",          # "finite" or "finite_and_in_mask360"
    msr_mask_360: np.ndarray | None = None,
    msr_annotate: bool = True,              # whether to print p_MSR on the plot
):
    """
    Scatter plot of two parcel-wise metrics (x_360 vs y_360), colored by community.

    Parameters
    ----------
    x_360, y_360 : array, shape (360,)
        Metric values per Glasser parcel (NaNs allowed).
    tsv_path : str
        Path to atlas TSV with community info.
    index_col : str
        Column in TSV that encodes parcel index (1..360).
    scheme_col : str
        Column in TSV that encodes community labels (e.g., 'community_yeo17').
    out_path : str
        Output PNG path.
    xlabel, ylabel : str
        Axis labels.
    title : str
        Figure title.
    xlim, ylim : (float, float), optional
        Axis limits. If None, inferred from data.
    add_identity : bool
        If True, draw y = x line over overlapping axis range.
    add_reg : bool
        If True, draw a global least-squares regression line (no grouping).
    legend : bool
        If True, show community legend; otherwise hide (keeps plot clean).
    alpha : float
        Point alpha.
    size_360 : array-like, optional
        Optional 360-length vector used to modulate point size. If provided,
        entries are aligned by parcel index (1..360). NaNs are allowed.
    size_scale : (float, float)
        Minimum and maximum marker sizes used for scaling size_360.
    size_use_abs : bool
        If True (default), use abs(size_360) before scaling so that strong
        positive or negative values both yield larger markers.
    corr_method: str
        'pearson' or 'spearman' or 'kendall' choose which correlation to use.
    """
    if x_360.shape[0] != 360 or y_360.shape[0] != 360:
        raise ValueError(
            f"x_360 and y_360 must both have shape (360,), "
            f"got {x_360.shape}, {y_360.shape}"
        )

    # Load atlas with community labels
    df_atlas = pd.read_csv(tsv_path, sep="\t")
    if index_col not in df_atlas.columns:
        print(f"[WARN] Index col '{index_col}' not in TSV. Skipping {title}.")
        return
    if scheme_col not in df_atlas.columns:
        print(f"[WARN] Scheme col '{scheme_col}' not in TSV. Skipping {title}.")
        return

    df_atlas[index_col] = (
        pd.to_numeric(df_atlas[index_col], errors="coerce")
        .fillna(-1)
        .astype(int)
    )

    # Build metrics dataframe
    parcel_ids = np.arange(1, 361, dtype=int)
    df_vals = pd.DataFrame({
        index_col: parcel_ids,
        "x": x_360,
        "y": y_360,
    })

    # Merge and clean
    df = pd.merge(df_atlas, df_vals, on=index_col, how="inner")
    df = df.replace({np.inf: np.nan, -np.inf: np.nan})
    df = df.dropna(subset=["x", "y", scheme_col])

    if df.empty:
        print(f"[WARN] No valid data for scatter: {title}")
        return

    df[scheme_col] = df[scheme_col].astype(str)
    
        # Optional per-parcel point sizes from a 360-length vector
    if size_360 is not None:
        size_360 = np.asarray(size_360, dtype=float)

        # Build lookup: index_col (1..360) -> size_source
        size_df = pd.DataFrame({
            index_col: np.arange(1, 361, dtype=int),
            "size_source": size_360,
        })

        df = pd.merge(df, size_df, on=index_col, how="left")

        vals = df["size_source"].to_numpy()
        if size_use_abs:
            vals = np.abs(vals)

        # Replace NaN / inf with 0 before scaling
        vals = np.where(np.isfinite(vals), vals, 0.0)

        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
        s_min, s_max = size_scale

        if vmax > vmin:
            scaled = (vals - vmin) / (vmax - vmin)
            scaled = s_min + scaled * (s_max - s_min)
        else:
            scaled = np.full_like(vals, (s_min + s_max) / 2.0, dtype=float)

        df["pt_size"] = scaled
    else:
        df["pt_size"] = 120.0


    # Correlation estimate.
    if df.shape[0] >= 2:
        r = df["x"].corr(df["y"], method=corr_method)
    else:
        r = np.nan
        
    # MSR-corrected p-value over the same cortical domain.
    p_msr = np.nan
    n_used = int(df.shape[0])
    
    if msr:
        if msr_params is None:
            raise ValueError(
                "msr=True but msr_params=None. Provide:\n"
                "  dist_l_path, dist_r_path, parc_l_path, parc_r_path\n"
                "Optional: n_perm, n_proc, seed, metric"
            )
    
        # Build a 360-length boolean mask matching the correlation domain
        x_all = np.asarray(x_360, float)
        y_all = np.asarray(y_360, float)
    
        finite = np.isfinite(x_all) & np.isfinite(y_all)
    
        if msr_mask_mode == "finite":
            mask_bool_360 = finite
        elif msr_mask_mode == "finite_and_in_mask360":
            if msr_mask_360 is None:
                raise ValueError("msr_mask_mode='finite_and_in_mask360' requires msr_mask_360 (shape (360,), bool).")
            mm = np.asarray(msr_mask_360, bool)
            if mm.shape != (360,):
                raise ValueError(f"msr_mask_360 must have shape (360,), got {mm.shape}")
            mask_bool_360 = finite & mm
        else:
            raise ValueError("msr_mask_mode must be 'finite' or 'finite_and_in_mask360'")
    
        if int(mask_bool_360.sum()) >= 3 and np.isfinite(r):
            # compute_msr_statistics expects 2D arrays (parcels x maps)
            target_data = x_all[:, None]   # (360,1)
            ref_data    = y_all[:, None]   # (360,1)
    
            # Use corr_method unless an MSR metric is specified.
            metric_for_msr = str(msr_params.get("metric", corr_method))
    
            r_vals, p_vals = compute_msr_statistics(
                target_data=target_data,
                ref_data=ref_data,
                mask_idx=mask_bool_360,
                target_idxs=[0],
                ref_idxs=[0],
                dist_l_path=msr_params["dist_l_path"],
                dist_r_path=msr_params["dist_r_path"],
                parc_l_path=msr_params["parc_l_path"],
                parc_r_path=msr_params["parc_r_path"],
                n_proc=int(msr_params.get("n_proc", 8)),
                n_perm=int(msr_params.get("n_perm", 1000)),
                seed=int(msr_params.get("seed", 42)),
                metric=metric_for_msr,
            )
            p_msr = float(p_vals[0, 0])


    # Dynamic Labeling (LaTeX formatting for Greek letters)
    if corr_method == "spearman":
        stats_label = r"\rho"  # Greek rho
    elif corr_method == "kendall":
        stats_label = r"\tau"  # Greek tau
    else:
        stats_label = "r"      # Pearson r
    
    # Styling
    sns.set_style("ticks")
    sns.set_context("poster", font_scale=1.1)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": "Lato",
        "axes.edgecolor": "#444444",
        "axes.linewidth": 1.1,
        "xtick.major.width": 1.1,
        "text.color": "#333333",
        "grid.color": "#e0e0e0",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
    })

    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    ax.set_facecolor("white")
    ax.grid(False)

    # Palette per community (stable, hash-based)
    communities = sorted(df[scheme_col].unique())

    if custom_palette is not None:
        if isinstance(custom_palette, dict):
            palette_map = custom_palette
        else:
            # Create a dict mapping sorted communities to the provided list
            palette_map = {
                comm: custom_palette[i % len(custom_palette)] 
                for i, comm in enumerate(communities)
            }
    else:
        palette_map = _tab20_palette_for_categories(communities, seed=0)


    # The community legend is constructed explicitly below.
    sns.scatterplot(
        data=df,
        x="x",
        y="y",
        hue=scheme_col,
        palette=palette_map,
        s=df["pt_size"].to_numpy(dtype=float),
        edgecolor="white",
        linewidth=0.8,
        alpha=alpha,
        ax=ax,
        legend=False,                           
    )
    
    # Community color legend
    if legend:
        handles = [
            mlines.Line2D(
                [], [],
                linestyle="",
                marker="o",
                markersize=6,  # fixed legend marker size in points
                markerfacecolor=palette_map[c],
                markeredgecolor="white",
                markeredgewidth=0.3,
                label=str(c),
            )
            for c in communities
        ]
        ax.legend(
            handles=handles,
            title=scheme_col,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.0,
            frameon=False,
        )




    # Global regression line (across all parcels)
    if add_reg and df.shape[0] >= 2:
        sns.regplot(
            data=df,
            x="x",
            y="y",
            scatter=False,
            ax=ax,
            color="black",
            line_kws={"linewidth": 2, "alpha": 0.7},
        )

    # Axis limits
    xdata_min, xdata_max = float(df["x"].min()), float(df["x"].max())
    ydata_min, ydata_max = float(df["y"].min()), float(df["y"].max())

    if xlim is None:
        xr = xdata_max - xdata_min if xdata_max > xdata_min else 1.0
        xlim = (xdata_min - 0.05 * xr, xdata_max + 0.05 * xr)
    if ylim is None:
        yr = ydata_max - ydata_min if ydata_max > ydata_min else 1.0
        ylim = (ydata_min - 0.05 * yr, ydata_max + 0.05 * yr)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    # Optional identity line.
    if add_identity:
        line_min = max(xlim[0], ylim[0])
        line_max = min(xlim[1], ylim[1])
        if line_max > line_min:
            ax.plot(
                [line_min, line_max],
                [line_min, line_max],
                linestyle="--",
                linewidth=1.0,
                color="#555555",
                alpha=0.7,
                zorder=1,
            )

    # Labels, title, annotation
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Annotation in top-left
    if np.isfinite(r):
        # Convert MSR p-value to stars (if available)
        stars = ""
        if msr and msr_annotate:
            stars = _p_to_stars(p_msr)
    
        # Build compact annotation: correlation + stars only
        ann = f"${stats_label}$ = {r:.2f}{stars}"
    
        ax.text(
            0.02, 1.02,
            ann,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            color="#555555",
            alpha=1 ,           
        )



    # Axis styling
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
        
    sns.despine(ax=ax, offset=10, trim=True)

    plt.tight_layout()
    fig.savefig(out_path, dpi=600)
    plt.close(fig)
    print(f"[SCATTER] Saved scatter: {out_path}")
    
    return {"r": float(r) if np.isfinite(r) else np.nan, "n": n_used, "p_msr": p_msr}

    


# =========================
# Hippocampus helpers
# =========================
def load_hipp_surfaces_and_labels(
    labeling_path_L: str,
    labeling_path_R: str,
    surf_path_L: str,
    surf_path_R: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    labeling_L = np.asarray(np.load(labeling_path_L))
    labeling_R = np.asarray(np.load(labeling_path_R))
    surf_L = nb.load(surf_path_L); surf_R = nb.load(surf_path_R)
    V_L = surf_L.get_arrays_from_intent('NIFTI_INTENT_POINTSET')[0].data
    F_L = surf_L.get_arrays_from_intent('NIFTI_INTENT_TRIANGLE')[0].data
    V_R = surf_R.get_arrays_from_intent('NIFTI_INTENT_POINTSET')[0].data
    F_R = surf_R.get_arrays_from_intent('NIFTI_INTENT_TRIANGLE')[0].data
    return labeling_L, labeling_R, {'left': V_L, 'right': V_R}, {'left': F_L, 'right': F_R}

def get_hipp_label_ids(labeling: np.ndarray, expected_n: Optional[int] = None) -> np.ndarray:
    mask = labeling != 0
    uniq = np.array(sorted(np.unique(labeling[mask]).astype(int)))
    if expected_n is not None and len(uniq) != int(expected_n):
        raise RuntimeError(f"Unexpected # hippocampal labels on surface: {len(uniq)} (expected {expected_n}).")
    return uniq

def _vertex_values_from_labels(
    labeling: np.ndarray,
    uniq_labels: np.ndarray,
    hemi_row_indices: np.ndarray,
    component_values: np.ndarray,
) -> np.ndarray:
    out = np.full(labeling.shape, np.nan, dtype=float)
    for k, lab in enumerate(uniq_labels):
        out[labeling == lab] = float(component_values[int(hemi_row_indices[k])])
    return out

def map_gradients_to_hipp_vertices(
    gradients_top: np.ndarray,
    left_hipp_indices: np.ndarray,
    right_hipp_indices: np.ndarray,
    labeling_L: np.ndarray,
    labeling_R: np.ndarray,
    uniq_L: np.ndarray,
    uniq_R: np.ndarray,
    n_components: int,
) -> List[Dict[str, np.ndarray]]:
    grad_hipp = []
    for i in range(n_components):
        vals = gradients_top[:, i]
        gl = _vertex_values_from_labels(labeling_L, uniq_L, left_hipp_indices, vals)
        gr = _vertex_values_from_labels(labeling_R, uniq_R, right_hipp_indices, vals)
        grad_hipp.append({'left': gl, 'right': gr})
    return grad_hipp

def compute_shared_limits(
    gradients_top: np.ndarray,
    left_hipp_indices: np.ndarray,
    right_hipp_indices: np.ndarray,
    left_amyg_indices: np.ndarray,
    right_amyg_indices: np.ndarray,
    n_to_plot: int,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    vmins, vmaxs = [], []
    for g in range(n_to_plot):
        vals = np.concatenate([
            gradients_top[left_hipp_indices,  g].ravel(),
            gradients_top[right_hipp_indices, g].ravel(),
            gradients_top[left_amyg_indices,  g].ravel(),
            gradients_top[right_amyg_indices, g].ravel(),
        ])
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            vmin, vmax = -1.0, 1.0
        else:
            vmin, vmax = float(np.min(vals)), float(np.max(vals))
            if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or abs(vmax - vmin) < 1e-12:
                vmin, vmax = vmin - eps, vmax + eps
        vmins.append(vmin); vmaxs.append(vmax)
    return np.array(vmins), np.array(vmaxs)

def _plot_hipp_trisurf_single(
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_values: np.ndarray,
    vmin: float,
    vmax: float,
    cmap,
    title: str,
    output_file: str,
) -> None:
    face_vals = np.nanmean(
        np.vstack([
            vertex_values[faces[:, 0].astype(int)],
            vertex_values[faces[:, 1].astype(int)],
            vertex_values[faces[:, 2].astype(int)],
        ]),
        axis=0
    )
    x, y, z = vertices[:, 0], vertices[:, 1], -vertices[:, 2]
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.view_init(azim=90, elev=-60)
    surf = ax.plot_trisurf(x, y, z, triangles=faces.astype(int),
                           cmap=cmap, linewidth=0, antialiased=False,
                           edgecolor=None, alpha=0.9)
    surf.set_array(face_vals); surf.set_clim(vmin, vmax)
    cbar = fig.colorbar(surf, shrink=0.5, aspect=10, ticks=[vmin, vmax])
    cbar.ax.set_yticklabels([f"{vmin:0.2f}", f"{vmax:0.2f}"])
    cbar.ax.tick_params(labelsize=20); cbar.set_label('Gradient Value', rotation=270, fontsize=20)
    ax.grid(False); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.set_axis_off()
    plt.title(title); os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=600, bbox_inches='tight'); plt.close(fig)

def plot_and_save_hipp_gradients(
    vertices_dict: Dict[str, np.ndarray],
    faces_dict: Dict[str, np.ndarray],
    grad_vertex_dicts: List[Dict[str, np.ndarray]],
    per_grad_vmin: np.ndarray,
    per_grad_vmax: np.ndarray,
    cmap,
    output_file_template: str,
    n_to_plot: int,
    title_template: str = None,
) -> None:
    for i in range(n_to_plot):
        vmin_i, vmax_i = float(per_grad_vmin[i]), float(per_grad_vmax[i])
        _plot_hipp_trisurf_single(
            vertices_dict['left'], faces_dict['left'], grad_vertex_dicts[i]['left'],
            vmin_i, vmax_i, cmap,
            title_template.format(side='Left', gradient=i+1),
            output_file_template.format(side='left', gradient=i+1)
        )
        _plot_hipp_trisurf_single(
            vertices_dict['right'], faces_dict['right'], grad_vertex_dicts[i]['right'],
            vmin_i, vmax_i, cmap,
            title_template.format(side='Right', gradient=i+1),
            output_file_template.format(side='right', gradient=i+1)
        )

def plot_and_save_hipp_unfolded_gradients(  
    values,
    left_hipp_indices,
    right_hipp_indices,
    labelmap_l_path,
    labelmap_r_path,
    out_path,
    cmap,
    vmin,
    vmax,
    *,
    n_hipp=15,
    collapse_axis=2,
    flip_ap=True,
    flip_pd=True,
    boundary_mode="subfield",
    title="",
    cbar_label="",
    figsize=(6.8, 3.0),
    dpi=600,
    save_pdf=True,
):
    """
    Plot hippocampal parcel values on true 2D unfolded hippocampal labelmaps.

    This function does not compute colors.
    Pass the same cmap, vmin, and vmax used by the parent Fig. 3/Fig. 4 script.

    values:
        Full 48-seed vector in standard order:
        [L-Amyg(9), R-Amyg(9), L-Hipp(15), R-Hipp(15)]
    """

    values = np.asarray(values, dtype=float).ravel()
    left_hipp_indices = np.asarray(left_hipp_indices, dtype=int)
    right_hipp_indices = np.asarray(right_hipp_indices, dtype=int)

    if left_hipp_indices.size != n_hipp:
        raise ValueError(f"left_hipp_indices has {left_hipp_indices.size}, expected {n_hipp}")
    if right_hipp_indices.size != n_hipp:
        raise ValueError(f"right_hipp_indices has {right_hipp_indices.size}, expected {n_hipp}")

    vals_l = values[left_hipp_indices]
    vals_r = values[right_hipp_indices]

    labels_l_3d = nb.load(labelmap_l_path).get_fdata().astype(np.int16)
    labels_r_3d = nb.load(labelmap_r_path).get_fdata().astype(np.int16)

    def collapse_mode(labels_3d):
        moved = np.moveaxis(labels_3d, collapse_axis, -1)
        flat = moved.reshape(-1, moved.shape[-1])
        out = np.zeros(flat.shape[0], dtype=np.int16)

        for i, row in enumerate(flat):
            row = row[row > 0]
            if row.size == 0:
                continue
            labs, counts = np.unique(row, return_counts=True)
            out[i] = labs[np.argmax(counts)]

        return out.reshape(moved.shape[:-1])

    def paint(label2d, vals15):
        img = np.full(label2d.shape, np.nan, dtype=float)
        for lab in range(1, n_hipp + 1):
            img[label2d == lab] = vals15[lab - 1]
        return img

    def orient(x):
        y = np.asarray(x).copy()
        if flip_ap:
            y = y[::-1, :]
        if flip_pd:
            y = y[:, ::-1]
        return y

    def make_boundary_mask(label2d):
        mode = str(boundary_mode).lower()

        if mode == "none":
            return np.zeros(label2d.shape, dtype=bool)

        if mode == "parcel":
            b_lab = label2d.astype(np.int16)

        elif mode == "subfield":
            n_subfields = 5
            if n_hipp % n_subfields != 0:
                raise ValueError("n_hipp must be divisible by 5 for subfield boundaries")

            n_ap = n_hipp // n_subfields
            b_lab = np.zeros(label2d.shape, dtype=np.int16)

            mask = label2d > 0
            b_lab[mask] = ((label2d[mask] - 1) // n_ap) + 1

        else:
            raise ValueError("boundary_mode must be 'subfield', 'parcel', or 'none'")

        valid = b_lab > 0
        b = np.zeros(b_lab.shape, dtype=bool)

        row_change = (b_lab[1:, :] != b_lab[:-1, :]) & valid[1:, :] & valid[:-1, :]
        b[1:, :] |= row_change

        col_change = (b_lab[:, 1:] != b_lab[:, :-1]) & valid[:, 1:] & valid[:, :-1]
        b[:, 1:] |= col_change

        return b

    label_l_2d = collapse_mode(labels_l_3d)
    label_r_2d = collapse_mode(labels_r_3d)

    label_l_disp = orient(label_l_2d)
    label_r_disp = orient(label_r_2d)

    img_l_disp = orient(paint(label_l_2d, vals_l))
    img_r_disp = orient(paint(label_r_2d, vals_r))

    if isinstance(cmap, str):
        try:
            cmap_obj = plt.get_cmap(cmap)
        except ValueError:
            if cmap == "icefire":
                cmap_obj = sns.color_palette("icefire", as_cmap=True)
            else:
                raise
    else:
        cmap_obj = cmap

    cmap_obj = cmap_obj.copy()
    cmap_obj.set_bad((0.82, 0.82, 0.82, 1.0))

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    last_im = None
    for ax, img, labels, hemi in zip(
        axes,
        [img_l_disp, img_r_disp],
        [label_l_disp, label_r_disp],
        ["L", "R"],
    ):
        last_im = ax.imshow(
            np.ma.masked_invalid(img),
            origin="lower",
            interpolation="nearest",
            cmap=cmap_obj,
            vmin=float(vmin),
            vmax=float(vmax),
            aspect="auto",
            zorder=1,
        )

        b = make_boundary_mask(labels)
        rgba = np.zeros((*b.shape, 4), dtype=float)
        rgba[b, :3] = 0.0
        rgba[b, 3] = 1.0

        ax.imshow(
            rgba,
            origin="lower",
            interpolation="nearest",
            aspect="auto",
            zorder=10,
        )

        ax.set_title(f"{title} {hemi}".strip(), fontsize=9)
        ax.set_xlabel("Proximal ↔ Distal", fontsize=16)
        ax.set_ylabel("Posterior ↔ Anterior", fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)

    cbar = fig.colorbar(
        last_im,
        ax=list(axes),
        fraction=0.035,
        pad=0.015,
    )
    cbar.ax.tick_params(labelsize=7)

    if cbar_label:
        cbar.set_label(cbar_label, fontsize=8)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")

    if save_pdf:
        fig.savefig(os.path.splitext(out_path)[0] + ".pdf", bbox_inches="tight")

    plt.close(fig)
    print(f"[UNFOLDED] Saved: {out_path}")
    
# =========================
# Amygdala plotting
# =========================


CUT_COORDS_L = [-14.82, 10.64, -21.51]
CUT_COORDS_R = [25.55, 11.94, -20.74]



def _fov_x_center_mm(img: nb.spatialimages.SpatialImage) -> float:
    """World-space x center of the image field-of-view (robust via 8 corners)."""
    shape = img.shape[:3]
    corners_ijk = np.array(
        [
            [0, 0, 0],
            [shape[0] - 1, 0, 0],
            [0, shape[1] - 1, 0],
            [0, 0, shape[2] - 1],
            [shape[0] - 1, shape[1] - 1, 0],
            [shape[0] - 1, 0, shape[2] - 1],
            [0, shape[1] - 1, shape[2] - 1],
            [shape[0] - 1, shape[1] - 1, shape[2] - 1],
        ],
        dtype=float,
    )
    corners_xyz = nb.affines.apply_affine(img.affine, corners_ijk)
    x_min = float(np.min(corners_xyz[:, 0]))
    x_max = float(np.max(corners_xyz[:, 0]))
    return 0.5 * (x_min + x_max)


def mirror_img_world_x(img: nb.Nifti1Image, x0_mm: float | None = None) -> nb.Nifti1Image:
    """
    Mirror an image in world-x about plane x = x0_mm.
    If x0_mm is None, mirrors about the FOV x-center.
    """
    if x0_mm is None:
        x0_mm = _fov_x_center_mm(img)

    # World-space reflection: x' = 2*x0 - x
    S = np.eye(4, dtype=float)
    S[0, 0] = -1.0
    S[0, 3] = 2.0 * float(x0_mm)

    new_affine = S @ img.affine
    data = img.get_fdata(dtype=np.float32)

    hdr = img.header.copy()
    # Keep the header consistent with the affine.
    if hasattr(hdr, "set_qform"):
        hdr.set_qform(new_affine, code=1)
        hdr.set_sform(new_affine, code=1)

    return nb.Nifti1Image(data, new_affine, header=hdr)


def mirror_cut_coords_x(
    cut_coords: list[float] | tuple[float, float, float],
    x0_mm: float,
) -> list[float]:
    """Mirror (x,y,z) cut coords about x=x0_mm."""
    arr = np.asarray(cut_coords, dtype=float).ravel()
    if arr.size != 3:
        raise ValueError(f"cut_coords must be 3 values, got: {cut_coords!r}")
    x, y, z = map(float, arr)
    return [2.0 * float(x0_mm) - x, y, z]


def make_amygdala_scalar_volumes(
    label_img_L: nb.Nifti1Image,
    label_img_R: nb.Nifti1Image,
    label_order: List[int],
    values_L: np.ndarray,
    values_R: np.ndarray,
    fill: float | None = np.nan,
) -> Tuple[nb.Nifti1Image, nb.Nifti1Image]:
    """
    Map per-nucleus scalar values to dense 3D volumes for left & right amygdala.

    Parameters
    ----------
    label_img_L, label_img_R : Nifti1Image
        Label images with integer amygdala label codes.
    label_order : list[int]
        The integer label codes (FreeSurfer LUT) in the order that matches
        `values_L` / `values_R`.
    values_L, values_R : array-like, shape (len(label_order),)
        Scalar values to assign to each label for left / right amygdala.
    fill : float or None, optional
        Background value for voxels that are not part of any label in
        `label_order`. If None, use 0.0. If np.nan (default), voxels are
        omitted from plotting by nilearn.

    Returns
    -------
    img_L, img_R : Nifti1Image
        Scalar volumes with the same shape/affine as the input labels.
    """
    data_L = label_img_L.get_fdata()
    data_R = label_img_R.get_fdata()

    values_L = np.asarray(values_L).ravel()
    values_R = np.asarray(values_R).ravel()

    if values_L.shape[0] != len(label_order) or values_R.shape[0] != len(label_order):
        raise ValueError(
            f"values_L/values_R must have length {len(label_order)}, "
            f"got {values_L.shape[0]} and {values_R.shape[0]}"
        )

    if fill is None:
        vol_L = np.zeros_like(data_L, dtype=np.float32)
        vol_R = np.zeros_like(data_R, dtype=np.float32)
    else:
        vol_L = np.full_like(data_L, fill, dtype=np.float32)
        vol_R = np.full_like(data_R, fill, dtype=np.float32)

    for i, lab in enumerate(label_order):
        lab = int(lab)
        mask_L = data_L == lab
        mask_R = data_R == lab
        if mask_L.any():
            vol_L[mask_L] = float(values_L[i])
        if mask_R.any():
            vol_R[mask_R] = float(values_R[i])

    img_L = nb.Nifti1Image(vol_L, label_img_L.affine, label_img_L.header)
    img_R = nb.Nifti1Image(vol_R, label_img_R.affine, label_img_R.header)
    return img_L, img_R


def _parse_ortho_cut_coords(
    cut_coords: list[float] | tuple[float, float, float] | None,
    name: str = "cut_coords",
) -> tuple[float, float, float]:
    if cut_coords is None:
        raise ValueError(f"{name} must be provided for display_mode='ortho'.")
    arr = np.asarray(cut_coords, dtype=float).ravel()
    if arr.size != 3:
        raise ValueError(
            f"{name} must have exactly 3 values (x, y, z) for display_mode='ortho'. "
            f"Got {arr.size}: {cut_coords!r}"
        )
    return float(arr[0]), float(arr[1]), float(arr[2])

def _add_contours_to_display(display, label_img, color='black', linewidth=0.5):
    """Draw contours for every nonzero label in a NIfTI image."""
    if label_img is None:
        return
    
    # Get all unique label IDs (excluding background 0)
    data = label_img.get_fdata()
    labels = np.unique(data)
    labels = labels[labels != 0]

    # Draw a contour for each label individually
    for lab in labels:
        # Binary image for the current label.
        mask_data = (data == lab).astype(np.float32)
        mask_img = nb.Nifti1Image(mask_data, label_img.affine, label_img.header)
        
        display.add_contours(
            mask_img, 
            levels=[0.5], 
            colors=color, 
            linewidths=linewidth
        )


def plot_amygdala_scalar_volumes(
    img_L: nb.Nifti1Image,
    img_R: nb.Nifti1Image,
    out_dir: str,
    title_L: str | None = None,
    title_R: str | None = None,
    output_prefix: str = "amygdala_scalar",
    index: int | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap=plt.get_cmap("viridis"),
    bg_img=None,
    display_mode: str = "ortho",
    draw_cross: bool = False,
    cut_coords_L: list[float] | tuple[float, float, float] | None = None,
    cut_coords_R: list[float] | tuple[float, float, float] | None = None,
    colorbar: bool = False,
    ortho_layout: Literal["horizontal", "vertical"] = "horizontal",
    contours_L: nb.Nifti1Image | None = None,
    contours_R: nb.Nifti1Image | None = None,
    contour_linewidth: float = 0.5,
) -> Tuple[str, str]:
    """
    Generic plotting helper for a left/right pair of scalar amygdala volumes.

    ``ortho_layout`` selects the standard horizontal display or a vertical
    stack of x, y, and z views when ``display_mode="ortho"``.

    Returns
    -------
    out_L, out_R : str
        Paths of the left and right PNGs.
    """
    os.makedirs(out_dir, exist_ok=True)

    suffix = "" if index is None else f"_gradient_{index}"
    out_L = os.path.join(out_dir, f"{output_prefix}_L{suffix}.png")
    out_R = os.path.join(out_dir, f"{output_prefix}_R{suffix}.png")

    if cut_coords_L is None:
        cut_coords_L = CUT_COORDS_L
    if cut_coords_R is None:
        cut_coords_R = CUT_COORDS_R

    def _plot_one(img, out_path, title, cut_coords, contour_img):
        if display_mode == "ortho" and ortho_layout == "vertical":
            x, y, z = _parse_ortho_cut_coords(cut_coords, name="cut_coords")
            fig, axes = plt.subplots(3, 1, figsize=(6, 10))
            fig.subplots_adjust(hspace=0.05)

            # Display the title on the first panel only.
            titles = [title, None, None]

            for ax, mode, cc, t in zip(
                axes,
                ("x", "y", "z"),
                ([x], [y], [z]),
                titles,
            ):
                display = niplotting.plot_stat_map(
                    img,
                    bg_img=bg_img,
                    display_mode=mode,
                    cut_coords=cc,
                    draw_cross=draw_cross,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    title=t,
                    axes=ax,
                    annotate=False,
                    # Display one color bar on the final panel.
                    colorbar=(colorbar and (mode == "z")),
                )
                _add_contours_to_display(display, contour_img, linewidth=contour_linewidth)

            fig.savefig(out_path, dpi=600, bbox_inches="tight")
            plt.close(fig)
        else:
            # Standard nilearn layout
            fig, ax = plt.subplots(1, figsize=(8, 6))
            display = niplotting.plot_stat_map(
                img,
                bg_img=bg_img,
                display_mode=display_mode,
                draw_cross=draw_cross,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                title=title,
                axes=ax,
                cut_coords=cut_coords,
                annotate=False,
                colorbar=colorbar,
            )
            _add_contours_to_display(display, contour_img, linewidth=contour_linewidth)
            
            fig.savefig(out_path, dpi=600, bbox_inches="tight")
            plt.close(fig)

    _plot_one(img_L, out_L, title_L, cut_coords_L, contours_L)
    _plot_one(img_R, out_R, title_R, cut_coords_R, contours_R)

    return out_L, out_R


def plot_amygdala_volumes_sharedscale(
    gradients_top: np.ndarray,
    label_img_L: nb.spatialimages.SpatialImage,
    label_img_R: nb.spatialimages.SpatialImage,
    label_order: List[int],
    left_indices: np.ndarray,
    right_indices: np.ndarray,
    per_grad_vmin: np.ndarray,
    per_grad_vmax: np.ndarray,
    out_dir: str,
    title_prefix: str = "Amygdala",
    output_prefix: str = "volumetric_amygdala_gradients",
    n_to_plot: int = 3,
    cmap=plt.get_cmap("viridis"),
    ortho_layout: Literal["horizontal", "vertical"] = "horizontal",
    mirror_right_from_left: bool = True,
    cut_coords_L: list[float] | tuple[float, float, float] | None = None,
    cut_coords_R: list[float] | tuple[float, float, float] | None = None,
    draw_contours: bool = False,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    n_pairs = min(len(label_order), len(left_indices), len(right_indices))
    if n_pairs == 0:
        raise ValueError(
            "label_order / left_indices / right_indices have zero overlap (n_pairs == 0)."
        )

    use_labels = list(label_order[:n_pairs])
    n_gradients = gradients_top.shape[1]
    n_to_plot = min(n_to_plot, n_gradients)

    if cut_coords_L is None:
        cut_coords_L = CUT_COORDS_L

    if mirror_right_from_left:
        # Mirror about the field-of-view center of the left label image.
        x0_mm = _fov_x_center_mm(label_img_L if isinstance(label_img_L, nb.Nifti1Image) else nb.Nifti1Image(label_img_L.get_fdata(), label_img_L.affine))
        if cut_coords_R is None:
            cut_coords_R = mirror_cut_coords_x(cut_coords_L, x0_mm)
    else:
        x0_mm = None
        if cut_coords_R is None:
            cut_coords_R = CUT_COORDS_R
            
    # Prepare contour images
    c_img_L = label_img_L
    c_img_R = label_img_R

    # Ensure c_img_R is mirrored if the data is mirrored
    if mirror_right_from_left and draw_contours:
        c_img_R = mirror_img_world_x(label_img_L, x0_mm=x0_mm)

    if not draw_contours:
        c_img_L = None
        c_img_R = None

    for g in range(n_to_plot):
        vmin_g = float(per_grad_vmin[g])
        vmax_g = float(per_grad_vmax[g])

        values_L = np.asarray(gradients_top[left_indices[:n_pairs], g], dtype=float)
        values_R = np.asarray(gradients_top[right_indices[:n_pairs], g], dtype=float)

        if mirror_right_from_left:
            # Build BOTH volumes on left labels, then mirror the "right" in world-x
            img_L, img_R_tmp = make_amygdala_scalar_volumes(
                label_img_L=label_img_L,
                label_img_R=label_img_L,
                label_order=use_labels,
                values_L=values_L,
                values_R=values_R,
                fill=np.nan,
            )
            img_R = mirror_img_world_x(img_R_tmp, x0_mm=x0_mm)
        else:
            # Use the right-hemisphere label image.
            img_L, img_R = make_amygdala_scalar_volumes(
                label_img_L=label_img_L,
                label_img_R=label_img_R,
                label_order=use_labels,
                values_L=values_L,
                values_R=values_R,
                fill=np.nan,
            )

        plot_amygdala_scalar_volumes(
            img_L,
            img_R,
            out_dir=out_dir,
            output_prefix=output_prefix,
            index=g + 1,
            vmin=vmin_g,
            vmax=vmax_g,
            cmap=cmap,
            cut_coords_L=cut_coords_L,
            cut_coords_R=cut_coords_R,
            bg_img=None,
            ortho_layout=ortho_layout,
            contours_L=c_img_L,
            contours_R=c_img_R,
            contour_linewidth=0.5
        )




# =========================
# Index and label lookup
# =========================
def resolve_indices_and_labels(
    n_hipp: int,
    amyg_style: str = "long",
    hipp_order: str = "subfield-major",
) -> Dict[str, Dict[str, np.ndarray | List[str]]]:
    idx = region_indices(n_hipp)
    out: Dict[str, Dict[str, np.ndarray | List[str]]] = {'Left': {}, 'Right': {}}
    for side in ("Left", "Right"):
        out[side]['amyg_indices'] = idx[side]['amyg_indices']
        out[side]['hipp_indices'] = idx[side]['hipp_indices']
        out[side]['amyg_labels'] = get_amygdala_labels(side=side, style=amyg_style)
        hipp_names, _ = build_hippocampus_labels(side=side, n_labels=n_hipp, order=hipp_order)
        out[side]['hipp_labels'] = hipp_names
    return out
