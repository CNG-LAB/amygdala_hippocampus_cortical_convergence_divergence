#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot Pearson and GLASSO MSR heatmaps with joint FDR across methods.

The script loads saved MSR r-values and raw p-values from:
   - top10_union_pearson/
   - top10_union_glasso/
It pools all finite p-values into one family, applies FDR once, writes q-values
back into method-specific matrices, and saves:
   - q-value .npy files for each method
   - separate heatmaps for each method using joint-FDR stars

Prerequisites:
- Both method-specific scripts have produced:
    msr_rvals_top{TOP_PERCENT}.npy
    msr_raw_pvals_top{TOP_PERCENT}.npy
- The reference-map ordering is the same in both runs
- The r/p matrix shapes match across methods
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
from statsmodels.stats.multitest import fdrcorrection


# ============================================================
# CONFIG
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "utils"))
from project_config import ProjectPaths

PATHS = ProjectPaths.from_environment()
TOP_PERCENT = 10
OUTPUT_ROOT = str(PATHS.output_root / "group_level_gradients")

PEARSON_DIR = os.path.join(OUTPUT_ROOT, "top10_union_pearson")
GLASSO_DIR = os.path.join(OUTPUT_ROOT, "top10_union_glasso")

# Labels follow the ordering used in the MSR block.
MSR_SUBCORT_GRADS = [0, 1]
Y_LABELS = [f"AmygHipp G{g+1}" for g in MSR_SUBCORT_GRADS]

# Must match the exact reference_maps_config order used in the scripts
X_LABELS = [
    "CrtFC G1", "CrtFC G2", "CrtFC G3",
   # "Glas Ctx G1", "Glas Ctx G2", "Glas Ctx G3",
    "Shared.", "Dom."
]

# Plot settings
CORR_LABEL = "Kendall"
ALPHA_FDR = 0.05
FDR_METHOD = "indep"   # statsmodels fdrcorrection method
DPI = 600
ANNOT_SIZE = 22
LINEWIDTHS = 3
CMAP = "RdBu_r"
VMIN = -1
VMAX = 1
CENTER = 0


# ============================================================
# Statistical helpers
# ============================================================
def stars_from_q(q: float) -> str:
    if not np.isfinite(q):
        return ""
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return ""


def load_method_arrays(method_dir: str, top_percent: int):
    r_path = os.path.join(method_dir, f"msr_rvals_top{top_percent}.npy")
    p_path = os.path.join(method_dir, f"msr_raw_pvals_top{top_percent}.npy")

    if not os.path.exists(r_path):
        raise FileNotFoundError(f"Missing r-values file: {r_path}")
    if not os.path.exists(p_path):
        raise FileNotFoundError(f"Missing raw p-values file: {p_path}")

    r_vals = np.load(r_path)
    p_vals = np.load(p_path)

    if r_vals.shape != p_vals.shape:
        raise ValueError(
            f"Shape mismatch in {method_dir}: "
            f"r_vals {r_vals.shape} vs p_vals {p_vals.shape}"
        )

    return r_vals, p_vals, r_path, p_path


def make_joint_fdr_qvals(p_pear: np.ndarray, p_glas: np.ndarray,
                         alpha: float = 0.05, method: str = "indep"):
    all_p = []
    all_keys = []

    for meth_name, pmat in [("pearson", p_pear), ("glasso", p_glas)]:
        for i in range(pmat.shape[0]):
            for j in range(pmat.shape[1]):
                if np.isfinite(pmat[i, j]):
                    all_p.append(float(pmat[i, j]))
                    all_keys.append((meth_name, i, j))

    all_p = np.asarray(all_p, dtype=float)
    if all_p.size == 0:
        raise RuntimeError("No finite p-values found for joint FDR.")

    _, all_q = fdrcorrection(all_p, alpha=alpha, method=method)

    q_pear = np.full_like(p_pear, np.nan, dtype=float)
    q_glas = np.full_like(p_glas, np.nan, dtype=float)

    for q, (meth_name, i, j) in zip(all_q, all_keys):
        if meth_name == "pearson":
            q_pear[i, j] = q
        else:
            q_glas[i, j] = q

    return q_pear, q_glas


def build_annotations(r_vals: np.ndarray, q_vals: np.ndarray) -> np.ndarray:
    annot = np.empty_like(r_vals, dtype=object)
    for i in range(r_vals.shape[0]):
        for j in range(r_vals.shape[1]):
            r = r_vals[i, j]
            q = q_vals[i, j]
            annot[i, j] = "" if not np.isfinite(r) else f"{r:.2f}{stars_from_q(q)}"
    return annot


def plot_heatmap(r_vals: np.ndarray, q_vals: np.ndarray, x_labels, y_labels,
                 out_png: str, out_pdf: str, title: str = ""):
    if r_vals.shape != q_vals.shape:
        raise ValueError(f"r/q shape mismatch: {r_vals.shape} vs {q_vals.shape}")
    if r_vals.shape[0] != len(y_labels):
        raise ValueError(
            f"Number of y-labels ({len(y_labels)}) does not match matrix rows ({r_vals.shape[0]})."
        )
    if r_vals.shape[1] != len(x_labels):
        raise ValueError(
            f"Number of x-labels ({len(x_labels)}) does not match matrix cols ({r_vals.shape[1]})."
        )

    annot = build_annotations(r_vals, q_vals)

    fig_w = max(len(x_labels) * 2.0 + 2, 8)
    fig_h = max(len(y_labels) * 1.2 + 2.5, 4.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        r_vals,
        annot=annot,
        fmt="",
        cmap=CMAP,
        center=CENTER,
        vmin=VMIN,
        vmax=VMAX,
        square=False,
        linewidths=LINEWIDTHS,
        linecolor="white",
        xticklabels=x_labels,
        yticklabels=y_labels,
        ax=ax,
        annot_kws={"weight": "bold", "size": ANNOT_SIZE},
        cbar_kws={"label": CORR_LABEL, "shrink": 0.9},
    )

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    if title:
        ax.set_title(title, pad=18)

    plt.tight_layout()
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================
def main():
    font_family = "Lato"
    if font_family not in {font.name for font in fm.fontManager.ttflist}:
        fallback = "DejaVu Sans"
        print(f"[WARN] Font '{font_family}' not found. Falling back to '{fallback}'.")
        font_family = fallback
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [font_family],
        "axes.edgecolor": "black",
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    sns.set_context("talk", font_scale=1.4)
    plt.style.use("seaborn-v0_8-white")

    # Load
    r_pear, p_pear, r_pear_path, p_pear_path = load_method_arrays(PEARSON_DIR, TOP_PERCENT)
    r_glas, p_glas, r_glas_path, p_glas_path = load_method_arrays(GLASSO_DIR, TOP_PERCENT)

    print(f"[LOAD] Pearson r: {r_pear_path}")
    print(f"[LOAD] Pearson p: {p_pear_path}")
    print(f"[LOAD] GLASSO r: {r_glas_path}")
    print(f"[LOAD] GLASSO p: {p_glas_path}")

    if r_pear.shape != r_glas.shape:
        raise ValueError(
            f"Pearson and GLASSO r-value matrices have different shapes: "
            f"{r_pear.shape} vs {r_glas.shape}"
        )
    if p_pear.shape != p_glas.shape:
        raise ValueError(
            f"Pearson and GLASSO p-value matrices have different shapes: "
            f"{p_pear.shape} vs {p_glas.shape}"
        )

    # Joint FDR across both methods
    q_pear, q_glas = make_joint_fdr_qvals(
        p_pear=p_pear,
        p_glas=p_glas,
        alpha=ALPHA_FDR,
        method=FDR_METHOD,
    )

    # Save q-values separately
    q_pear_path = os.path.join(PEARSON_DIR, f"msr_jointFDR_qvals_top{TOP_PERCENT}.npy")
    q_glas_path = os.path.join(GLASSO_DIR,  f"msr_jointFDR_qvals_top{TOP_PERCENT}.npy")
    np.save(q_pear_path, q_pear)
    np.save(q_glas_path, q_glas)

    print(f"[SAVE] Pearson q-values: {q_pear_path}")
    print(f"[SAVE] GLASSO q-values: {q_glas_path}")

    print("\nPearson raw p-values:")
    print(np.array2string(p_pear, precision=4, suppress_small=False))
    print("\nPearson joint-FDR q-values:")
    print(np.array2string(q_pear, precision=4, suppress_small=False))

    print("\nGLASSO raw p-values:")
    print(np.array2string(p_glas, precision=4, suppress_small=False))
    print("\nGLASSO joint-FDR q-values:")
    print(np.array2string(q_glas, precision=4, suppress_small=False))

    # Plot separate heatmaps
    pear_png = os.path.join(PEARSON_DIR, f"heatmap_modular_top{TOP_PERCENT}_jointFDR.png")
    pear_pdf = os.path.join(PEARSON_DIR, f"heatmap_modular_top{TOP_PERCENT}_jointFDR.pdf")
    glas_png = os.path.join(GLASSO_DIR,  f"heatmap_modular_top{TOP_PERCENT}_jointFDR.png")
    glas_pdf = os.path.join(GLASSO_DIR,  f"heatmap_modular_top{TOP_PERCENT}_jointFDR.pdf")

    plot_heatmap(
        r_vals=r_pear,
        q_vals=q_pear,
        x_labels=X_LABELS,
        y_labels=Y_LABELS,
        out_png=pear_png,
        out_pdf=pear_pdf,
        title="Pearson",
    )

    plot_heatmap(
        r_vals=r_glas,
        q_vals=q_glas,
        x_labels=X_LABELS,
        y_labels=Y_LABELS,
        out_png=glas_png,
        out_pdf=glas_pdf,
        title="GLASSO",
    )

    print(f"[SAVE] Pearson heatmap: {pear_png}")
    print(f"[SAVE] GLASSO heatmap: {glas_png}")
    print("\nDone. Separate matrices kept separate; only FDR family was joint across both methods.")


if __name__ == "__main__":
    main()
