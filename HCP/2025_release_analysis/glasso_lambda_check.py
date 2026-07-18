#!/usr/bin/env python3
"""Summarize subject GLASSO regularization, density, and motion associations."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "utils"))
from project_config import ProjectPaths

PATHS = ProjectPaths.from_environment()
OUTPUT_DIR = PATHS.output_root / "glasso_diagnostics"
FD_CSV = PATHS.resource_root / "HCP_Group_MeanFD_Cleaned.csv"
MATRIX_NAME = "connectivity_matrix_REST_406x406_glasso_z.npy"
LAMBDA_NAME = "glasso_lambda1_REST.txt"


def read_subjects(path: Path) -> list[str]:
    subjects = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not subjects:
        raise ValueError("Subject list is empty")
    if len(set(subjects)) != len(subjects):
        raise ValueError("Subject list contains duplicate IDs")
    return subjects


def read_lambda(path: Path) -> float:
    text = path.read_text().strip().removeprefix("[").removesuffix("]")
    value = float(text)
    if value <= 0:
        raise ValueError(f"GLASSO lambda must be positive: {path}")
    return value


def edge_density(matrix: np.ndarray) -> float:
    if matrix.shape != (406, 406):
        raise ValueError(f"Expected a 406 x 406 matrix, got {matrix.shape}")
    off_diagonal = ~np.eye(406, dtype=bool)
    return float(np.count_nonzero(matrix[off_diagonal]) / off_diagonal.sum())


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def print_summary(name: str, values: np.ndarray) -> None:
    print(
        f"{name}: mean={values.mean():.6g}, sd={values.std(ddof=1):.6g}, "
        f"min={values.min():.6g}, max={values.max():.6g}"
    )


def save_histograms(lambdas: np.ndarray, log_lambdas: np.ndarray, densities: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(lambdas, bins=20, edgecolor="black")
    axes[0].set(title="Distribution of GLASSO lambda", xlabel="lambda", ylabel="Subjects")
    axes[1].hist(log_lambdas, bins=20, edgecolor="black")
    axes[1].set(title="Distribution of GLASSO log10(lambda)", xlabel="log10(lambda)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "lambda_histograms.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(densities, bins=20, edgecolor="black")
    ax.set(
        title="Distribution of GLASSO edge density",
        xlabel="Non-zero off-diagonal proportion",
        ylabel="Subjects",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "density_histogram.png", dpi=300)
    plt.close(fig)


def save_scatter(x: np.ndarray, y: np.ndarray, ylabel: str, filename: str) -> None:
    r = correlation(x, y)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, s=20, edgecolor="black", linewidth=0.5)
    ax.set(xlabel="MeanFD", ylabel=ylabel, title=f"{ylabel} vs MeanFD (r = {r:.2f})")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    subjects = read_subjects(PATHS.subject_list)

    lambdas = []
    densities = []
    for subject in subjects:
        func_dir = PATHS.work_root / subject / "func_native_2025"
        lambdas.append(read_lambda(func_dir / LAMBDA_NAME))
        densities.append(edge_density(np.load(func_dir / MATRIX_NAME)))

    lambdas = np.asarray(lambdas)
    densities = np.asarray(densities)
    log_lambdas = np.log10(lambdas)

    print_summary("lambda", lambdas)
    print_summary("log10(lambda)", log_lambdas)
    print_summary("density", densities)
    print(f"corr(lambda, density) = {correlation(lambdas, densities):.4f}")
    print(f"corr(log10(lambda), density) = {correlation(log_lambdas, densities):.4f}")

    save_histograms(lambdas, log_lambdas, densities)
    np.savez(
        OUTPUT_DIR / "glasso_lambda_density_stats.npz",
        subjects=np.asarray(subjects, dtype=object),
        lambda_vals=lambdas,
        density_vals=densities,
    )

    motion = pd.read_csv(FD_CSV, dtype={"Subject": str})
    required = {"Subject", "MeanFD"}
    if not required.issubset(motion.columns):
        raise ValueError(f"{FD_CSV} must contain {sorted(required)}")
    motion["Subject"] = motion["Subject"].str.strip()
    motion = motion.set_index("Subject").reindex(subjects)
    if motion["MeanFD"].isna().any():
        missing = motion.index[motion["MeanFD"].isna()].tolist()
        raise ValueError(f"Missing MeanFD for {len(missing)} subjects; first: {missing[0]}")
    mean_fd = motion["MeanFD"].to_numpy(dtype=float)

    print(f"corr(lambda, MeanFD) = {correlation(lambdas, mean_fd):.4f}")
    print(f"corr(log10(lambda), MeanFD) = {correlation(log_lambdas, mean_fd):.4f}")
    print(f"corr(density, MeanFD) = {correlation(densities, mean_fd):.4f}")
    save_scatter(mean_fd, lambdas, "GLASSO lambda", "scatter_lambda_vs_MeanFD.png")
    save_scatter(
        mean_fd,
        log_lambdas,
        "log10(GLASSO lambda)",
        "scatter_log10lambda_vs_MeanFD.png",
    )
    save_scatter(mean_fd, densities, "GLASSO edge density", "scatter_density_vs_MeanFD.png")


if __name__ == "__main__":
    main()
