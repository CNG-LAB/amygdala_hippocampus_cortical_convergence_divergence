"""Render the manuscript Figure 3C amygdala–hippocampus heatmaps.

Only the settings used for the manuscript are implemented here: matched
unconditional positive-mean FC, ipsilateral hemispheres, positive-only values,
flipped heatmaps, and both raw and amygdala-row-scaled panels.
"""

from __future__ import annotations

import copy
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from group_analysis import mean_positive_r_from_z_stack
from hippamyg_labels import AMYGDALA_SHORT, _AMYGDALA_LUT_RGBA


FIGSIZE = (8, 12)
DPI = 600
TOP_BAR_TICKS = [0.0, 0.01, 0.02]
TOP_BAR_TICKLABELS = ["0", ".01", ".02"]
RIGHT_BAR_TICKS = [0.0, 0.02, 0.04]
RIGHT_BAR_TICKLABELS = ["0", ".02", ".04"]
MODE_TAG = "agg-matched_unconditional_posmean_gmask-off"


def _set_style(preferred_font: str) -> None:
    sns.set_theme(style="white")
    sns.set_context(
        "paper",
        font_scale=2.6,
        rc={
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
        },
    )
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": [preferred_font],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 26,
        "xtick.labelsize": 26,
        "ytick.labelsize": 26,
    })


def _strip_hemi_prefix(label: str) -> str:
    text = str(label).strip()
    text = re.sub(
        r"^(?:left|right|lh|rh)\b[\s\-_:/]*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"^(?:l|r)[\s\-_:/]+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else str(label)


def _canonical_amygdala_label(label: str) -> str:
    text = str(label).strip()
    text = re.sub(
        r"^(?:left|right|lh|rh)\b[\s\-_:/]*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"^(?:l|r)[\s\-_:/]+", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\s*\((?:left|right|lh|rh|l|r)\)\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"[\s\-_:/]+(?:left|right|lh|rh|l|r)\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        raise ValueError(f"Could not canonicalize amygdala label: {label}")
    return text


def _indexed_amygdala_colors() -> list[tuple[float, float, float]]:
    """Return LUT colors in the positional order used by AMYGDALA_SHORT."""
    lut_values = list(_AMYGDALA_LUT_RGBA.values())
    if len(lut_values) != len(AMYGDALA_SHORT):
        raise ValueError(
            f"Color LUT length ({len(lut_values)}) != "
            f"AMYGDALA_SHORT length ({len(AMYGDALA_SHORT)})"
        )
    return [
        (rgba[0] / 255.0, rgba[1] / 255.0, rgba[2] / 255.0)
        for rgba in lut_values
    ]


def _amygdala_row_colors(
    labels: list[str],
    palette: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    colors = []
    for label in map(_canonical_amygdala_label, labels):
        try:
            colors.append(palette[AMYGDALA_SHORT.index(label)])
        except ValueError:
            print(f"[FIG3][WARN] Label '{label}' not in AMYGDALA_SHORT. Using gray.")
            colors.append((0.5, 0.5, 0.5))
    return colors


def _prefix_hippocampal_labels(labels: list[str], prefix: str) -> list[str]:
    display_labels = []
    for label in labels:
        text = re.sub(
            r"^(Left|Right|LH|RH)\b[\s\-_:/]*",
            "",
            str(label),
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\bSubiculum\b",
            "Sub.",
            text.strip(),
            flags=re.IGNORECASE,
        )
        display_labels.append(prefix + text)
    return display_labels


def _ipsilateral_blocks(
    pearson: np.ndarray,
    glasso: np.ndarray,
    amygdala_left: list[str],
    amygdala_right: list[str],
    hippocampus_left: list[str],
    hippocampus_right: list[str],
) -> list[dict]:
    n_amyg_left = len(amygdala_left)
    n_amyg_right = len(amygdala_right)
    n_hipp_left = len(hippocampus_left)
    n_hipp_right = len(hippocampus_right)
    expected = (
        n_amyg_left + n_amyg_right,
        n_hipp_left + n_hipp_right,
    )
    if pearson.shape != expected or glasso.shape != expected:
        raise ValueError(
            f"Matrix shape mismatch. Expected {expected}, got "
            f"Pearson {pearson.shape} and GLASSO {glasso.shape}."
        )

    blocks = []
    hemisphere_specs = (
        (
            slice(0, n_amyg_left),
            slice(0, n_hipp_left),
            amygdala_left,
            hippocampus_left,
            "Left",
            "LH_ipsi",
        ),
        (
            slice(n_amyg_left, n_amyg_left + n_amyg_right),
            slice(n_hipp_left, n_hipp_left + n_hipp_right),
            amygdala_right,
            hippocampus_right,
            "Right",
            "RH_ipsi",
        ),
    )
    for amyg_slice, hipp_slice, amyg_labels, hipp_labels, hemi_tag, file_tag in hemisphere_specs:
        blocks.append({
            "AH_pe": pearson[amyg_slice, hipp_slice],
            "AH_gl": glasso[amyg_slice, hipp_slice],
            "amyg_labels": [_strip_hemi_prefix(x) for x in amyg_labels],
            "hipp_labels": [_strip_hemi_prefix(x) for x in hipp_labels],
            "hemi_tag": hemi_tag,
            "file_tag": file_tag,
        })
    return blocks


def _parse_ap_bins(hippocampal_labels: list[str]) -> tuple[np.ndarray, list[int], dict[int, int]]:
    pattern = re.compile(r"AP[\s\-_]*([0-9]+)", flags=re.IGNORECASE)
    bins = []
    missing = []
    for index, label in enumerate(hippocampal_labels):
        match = pattern.search(str(label))
        if match is None:
            missing.append((index, str(label)))
        else:
            bins.append(int(match.group(1)))

    if missing:
        examples = ", ".join(f"{index}:{label}" for index, label in missing[:10])
        raise ValueError(f"Missing AP bin in hippocampal labels: {examples}")

    bins_array = np.asarray(bins, dtype=int)
    if np.any(bins_array < 1):
        raise ValueError(f"AP bins must be >= 1. Found: {np.unique(bins_array)}")

    unique_bins = sorted(np.unique(bins_array).tolist())
    bin_to_index = {value: index for index, value in enumerate(unique_bins)}
    return bins_array, unique_bins, bin_to_index


def _row_minmax_scale(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mat = np.asarray(mat, dtype=float)
    scaled = np.full_like(mat, np.nan, dtype=float)
    for row_index, row in enumerate(mat):
        valid = np.isfinite(row)
        if not np.any(valid):
            continue
        values = row[valid]
        value_min = np.min(values)
        value_max = np.max(values)
        if value_max - value_min <= eps:
            scaled[row_index, valid] = 0.5
        else:
            scaled[row_index, valid] = (
                values - value_min
            ) / (value_max - value_min)
    return scaled


def _positive_only(mat: np.ndarray) -> np.ndarray:
    values = np.asarray(mat, dtype=float).copy()
    values[values < 0] = np.nan
    return values


def _shared_positive_limits(matrices: list[np.ndarray]) -> tuple[float, float]:
    finite = [
        matrix[np.isfinite(matrix)]
        for matrix in matrices
        if np.any(np.isfinite(matrix))
    ]
    if not finite:
        return 0.0, 1.0
    vmax = max(float(np.nanpercentile(np.concatenate(finite), 99.0)), 1e-6)
    return 0.0, vmax


def _set_fixed_ticks(
    axis,
    direction: str,
    ticks: list[float],
    labels: list[str],
    data_max: float,
) -> None:
    start = float(ticks[0])
    requested_end = float(ticks[-1])
    actual_end = (
        requested_end
        if not np.isfinite(data_max) or data_max <= start
        else max(requested_end, data_max * 1.03)
    )
    if direction == "y":
        axis.set_ylim(start, actual_end)
        axis.set_yticks(ticks)
        axis.set_yticklabels(labels)
        axis.tick_params(
            axis="y",
            which="major",
            direction="out",
            pad=3,
            rotation=0,
        )
    else:
        axis.set_xlim(start, actual_end)
        axis.set_xticks(ticks)
        axis.set_xticklabels(labels)
        axis.tick_params(
            axis="x",
            which="major",
            direction="out",
            pad=4,
            rotation=0,
        )


def _plot_payload(
    raw_mat: np.ndarray,
    display_mat: np.ndarray,
    amygdala_labels: list[str],
    hippocampal_labels: list[str],
    ap_bins: np.ndarray,
    ap_colors: list[tuple[float, float, float]],
    amygdala_colors: list[tuple[float, float, float]],
) -> dict:
    # Manuscript heatmaps are flipped: amygdala columns, hippocampus rows.
    raw_plot = raw_mat.T
    display_plot = display_mat.T
    if raw_plot.shape != display_plot.shape:
        raise ValueError("Raw and display matrices differ after orientation.")
    if len(amygdala_labels) != raw_plot.shape[1]:
        raise ValueError("Amygdala label count does not match heatmap columns.")
    if len(hippocampal_labels) != raw_plot.shape[0]:
        raise ValueError("Hippocampal label count does not match heatmap rows.")
    if len(amygdala_colors) != raw_plot.shape[1]:
        raise ValueError("Amygdala color count does not match heatmap columns.")
    if len(ap_colors) != raw_plot.shape[0]:
        raise ValueError("AP color count does not match heatmap rows.")

    return {
        "raw": raw_plot,
        "disp": display_plot,
        "x_labels": amygdala_labels,
        "y_labels": hippocampal_labels,
        "top_colors": amygdala_colors,
        "right_colors": ap_colors,
        "ap_breaks": np.where(np.diff(ap_bins) != 0)[0] + 1,
    }


def _draw_panel(ax_col, ax_hm, ax_row, payload: dict, cmap, vmin: float, vmax: float) -> None:
    frame = pd.DataFrame(
        payload["disp"],
        index=payload["y_labels"],
        columns=payload["x_labels"],
    )
    ax_hm.set_facecolor("#e0e0e0")
    sns.heatmap(
        frame,
        ax=ax_hm,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar=False,
        linewidths=0.22,
        linecolor=(1, 1, 1, 0.40),
        xticklabels=False,
        yticklabels=False,
        square=False,
    )
    ax_hm.set_xticks(np.arange(len(payload["x_labels"])) + 0.5)
    ax_hm.set_xticklabels(
        payload["x_labels"],
        rotation=90,
        ha="center",
        va="top",
    )
    ax_hm.set_yticks(np.arange(len(payload["y_labels"])) + 0.5)
    ax_hm.set_yticklabels(payload["y_labels"], rotation=0)
    ax_hm.set_title("")
    ax_hm.set_xlabel("")
    ax_hm.set_ylabel("")
    ax_hm.tick_params(axis="both", length=0)

    for boundary in payload["ap_breaks"]:
        ax_hm.axhline(boundary, color="k", lw=0.45, alpha=0.22)

    # Marginal bars always summarize the same raw positive-only FC matrix.
    column_means = np.nanmean(payload["raw"], axis=0)
    row_means = np.nanmean(payload["raw"], axis=1)
    marginal_label = "Mean\n+FC"

    x_positions = np.arange(len(column_means)) + 0.5
    ax_col.bar(
        x_positions,
        column_means,
        width=1.0,
        color=payload["top_colors"],
        edgecolor=(0, 0, 0, 0.45),
        linewidth=0.35,
    )
    ax_col.set_xlim(ax_hm.get_xlim())
    ax_col.set_ylabel(marginal_label)
    ax_col.tick_params(axis="x", bottom=False, labelbottom=False)
    column_max = (
        float(np.nanmax(column_means))
        if np.any(np.isfinite(column_means))
        else 0.0
    )
    _set_fixed_ticks(
        ax_col,
        "y",
        TOP_BAR_TICKS,
        TOP_BAR_TICKLABELS,
        column_max,
    )
    ax_col.grid(
        axis="y",
        which="major",
        linestyle=":",
        linewidth=0.75,
        alpha=0.45,
        color="0.4",
    )
    ax_col.spines["left"].set_visible(True)
    ax_row.spines["right"].set_visible(False)
    ax_col.spines["bottom"].set_visible(False)
    ax_row.spines["top"].set_visible(False)
    ax_col.spines["left"].set_linewidth(0.8)
    ax_col.spines["left"].set_color("0.2")

    y_positions = np.arange(len(row_means)) + 0.5
    ax_row.barh(
        y_positions,
        row_means,
        height=1.0,
        color=payload["right_colors"],
        edgecolor=(0, 0, 0, 0.45),
        linewidth=0.35,
    )
    ax_row.set_ylim(ax_hm.get_ylim())
    ax_row.set_xlabel(marginal_label)
    ax_row.tick_params(axis="y", left=False, labelleft=False)
    row_max = (
        float(np.nanmax(row_means))
        if np.any(np.isfinite(row_means))
        else 0.0
    )
    _set_fixed_ticks(
        ax_row,
        "x",
        RIGHT_BAR_TICKS,
        RIGHT_BAR_TICKLABELS,
        row_max,
    )
    ax_row.grid(
        axis="x",
        which="major",
        linestyle=":",
        linewidth=0.75,
        alpha=0.45,
        color="0.4",
    )
    ax_row.spines["bottom"].set_visible(True)
    ax_row.spines["top"].set_visible(False)
    ax_row.spines["left"].set_visible(False)
    ax_row.spines["right"].set_visible(False)
    ax_col.spines["top"].set_linewidth(0.8)
    ax_col.spines["top"].set_color("0.2")


def _save_heatmap(
    raw_mat: np.ndarray,
    display_mat: np.ndarray,
    amygdala_labels: list[str],
    hippocampal_labels: list[str],
    ap_bins: np.ndarray,
    ap_colors: list[tuple[float, float, float]],
    amygdala_colors: list[tuple[float, float, float]],
    output_png: str,
    cmap,
    vmin: float,
    vmax: float,
) -> None:
    payload = _plot_payload(
        raw_mat,
        display_mat,
        amygdala_labels,
        hippocampal_labels,
        ap_bins,
        ap_colors,
        amygdala_colors,
    )
    fig = plt.figure(
        figsize=FIGSIZE,
        constrained_layout=False,
        facecolor="white",
    )
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 5.0],
        width_ratios=[7.0, 1.25],
        hspace=0.03,
        wspace=0.04,
    )
    ax_col = fig.add_subplot(grid[0, 0], facecolor="white")
    ax_hm = fig.add_subplot(
        grid[1, 0],
        sharex=ax_col,
        facecolor="white",
    )
    ax_row = fig.add_subplot(
        grid[1, 1],
        sharey=ax_hm,
        facecolor="white",
    )
    _draw_panel(ax_col, ax_hm, ax_row, payload, cmap, vmin, vmax)
    fig.subplots_adjust(
        left=0.10,
        right=0.975,
        bottom=0.22,
        top=0.965,
    )

    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    fig.savefig(
        output_png,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    base, _ = os.path.splitext(output_png)
    pdf_path = base + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[FIG3] Saved: {output_png}")
    print(f"[FIG3] Saved: {pdf_path}")


def make_row_c_heatmaps(
    *,
    pearson_z_stack: np.ndarray,
    glasso_z_stack: np.ndarray,
    amyg_rows_all: np.ndarray,
    hipp_rows_all: np.ndarray,
    amyg_labels_L: list[str],
    amyg_labels_R: list[str],
    hipp_labels_L: list[str],
    hipp_labels_R: list[str],
    out_heat: str,
    preferred_font: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[tuple[float, float, float]],
]:  
    """Create the fixed Figure 3C panels and return matrices reused downstream."""
    cmap = copy.copy(sns.color_palette("viridis", as_cmap=True))
    cmap.set_bad(color="#e0e0e0")
    _set_style(preferred_font)
    amygdala_palette = _indexed_amygdala_colors()

    # Matched estimand for both methods: subject mean of positive r, including
    # zeros. GLASSO support prevalence is reported but is not used as a mask.
    pearson_ah_z = pearson_z_stack[
        :, amyg_rows_all[:, None], hipp_rows_all
    ].astype(float)
    glasso_ah_z = glasso_z_stack[
        :, amyg_rows_all[:, None], hipp_rows_all
    ].astype(float)

    finite = np.isfinite(glasso_ah_z)
    denominator = finite.sum(axis=0).astype(float)
    denominator[denominator == 0] = np.nan
    any_prevalence = (
        finite & (glasso_ah_z != 0.0)
    ).sum(axis=0) / denominator
    positive_prevalence = (
        finite & (glasso_ah_z > 0.0)
    ).sum(axis=0) / denominator

    pearson_ah = mean_positive_r_from_z_stack(pearson_ah_z)
    glasso_ah = mean_positive_r_from_z_stack(glasso_ah_z)

    print("\n[ROWC] Aggregation mode: matched_unconditional_posmean")
    print("[ROWC] GLASSO support-mask mode: off")
    print("[ROWC] GLASSO AH 'ANY NON-ZERO' prevalence (positive or negative):")
    print(
        f"       min={np.nanmin(any_prevalence):.3f}, "
        f"max={np.nanmax(any_prevalence):.3f}"
    )
    print("[ROWC] GLASSO AH 'STRICTLY POSITIVE' prevalence:")
    print(
        f"       min={np.nanmin(positive_prevalence):.3f}, "
        f"p10={np.nanpercentile(positive_prevalence[np.isfinite(positive_prevalence)], 10):.3f}, "
        f"median={np.nanmedian(positive_prevalence):.3f}, "
        f"max={np.nanmax(positive_prevalence):.3f}"
    )
    print("[ROWC] No GLASSO support mask applied.\n")

    if not (
        len(amyg_labels_L)
        == len(amyg_labels_R)
        == len(AMYGDALA_SHORT)
    ):
        raise ValueError(
            "Amygdala label length mismatch: "
            f"len(amyg_labels_L)={len(amyg_labels_L)}, "
            f"len(amyg_labels_R)={len(amyg_labels_R)}, "
            f"len(AMYGDALA_SHORT)={len(AMYGDALA_SHORT)}"
        )

    amygdala_left = [f"L-{label}" for label in AMYGDALA_SHORT]
    amygdala_right = [f"R-{label}" for label in AMYGDALA_SHORT]
    hippocampus_left = _prefix_hippocampal_labels(hipp_labels_L, "L-")
    hippocampus_right = _prefix_hippocampal_labels(hipp_labels_R, "R-")
    blocks = _ipsilateral_blocks(
        pearson_ah,
        glasso_ah.copy(),
        amygdala_left,
        amygdala_right,
        hippocampus_left,
        hippocampus_right,
    )

    positive_matrices = [
        _positive_only(matrix)
        for block in blocks
        for matrix in (block["AH_pe"], block["AH_gl"])
    ]
    raw_vmin, raw_vmax = _shared_positive_limits(positive_matrices)

    for block in blocks:
        pearson = _positive_only(block["AH_pe"])
        glasso = _positive_only(block["AH_gl"])
        amygdala_labels = block["amyg_labels"]
        hippocampal_labels = block["hipp_labels"]
        file_tag = block["file_tag"]

        ap_bins, unique_ap_bins, bin_to_index = _parse_ap_bins(
            hippocampal_labels
        )
        ap_palette = sns.color_palette("Oranges", len(unique_ap_bins))
        ap_colors = [ap_palette[bin_to_index[value]] for value in ap_bins]
        amygdala_colors = _amygdala_row_colors(
            amygdala_labels,
            amygdala_palette,
        )

        raw_maps = {"Pearson": pearson, "GLASSO": glasso}
        for method, matrix in raw_maps.items():
            filename = (
                f"FIG3C_amyg_x_hipp_heatmap_{method}_{file_tag}_flipped_"
                f"{MODE_TAG}.png"
            )
            _save_heatmap(
                matrix,
                matrix,
                amygdala_labels,
                hippocampal_labels,
                ap_bins,
                ap_colors,
                amygdala_colors,
                os.path.join(out_heat, filename),
                cmap,
                raw_vmin,
                raw_vmax,
            )

        scaled_maps = {
            "Pearson": _row_minmax_scale(pearson),
            "GLASSO": _row_minmax_scale(glasso),
        }
        for method in ("Pearson", "GLASSO"):
            filename = (
                f"FIG3C_amyg_x_hipp_heatmap_{method}_rowScaled_"
                f"{file_tag}_flipped_{MODE_TAG}.png"
            )
            _save_heatmap(
                raw_maps[method],
                scaled_maps[method],
                amygdala_labels,
                hippocampal_labels,
                ap_bins,
                ap_colors,
                amygdala_colors,
                os.path.join(out_heat, filename),
                cmap,
                0.0,
                1.0,
            )

    print("\n[FIG3 ROW-C DONE]")
    print("  Heatmaps:", out_heat)
    
    return (
        pearson_ah,
        glasso_ah,
        any_prevalence,
        positive_prevalence,
        amygdala_palette,
    )
