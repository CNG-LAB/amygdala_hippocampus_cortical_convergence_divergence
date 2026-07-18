#!/usr/bin/env bash

# Copy this file to config.sh, edit the paths, then source it before running the
# pipeline. config.sh is intentionally ignored by git.

# HCP-YA 2025 inputs
export HCP_RFMRI_ZIP_ROOT="/path/to/HCP-YA_3T_rsfmri"
export HCP_STRUCTURAL_ROOT="/path/to/HCP-YA_2025/StructuralRecommended"
# Set separately only if segment_subregions outputs are stored elsewhere.
export AMYGDALA_SEGMENTATION_ROOT="${HCP_STRUCTURAL_ROOT}"

# Project derivatives
export HIPPAMYG_WORK_ROOT="/path/to/HCP_hippamyg"
export HIPPUNFOLD_ROOT="/path/to/hippunfold_outputs"
export HIPPUNFOLD_INPUT_ROOT="/path/to/hippunfold_bids_inputs"
export HIPPUNFOLD_RESOURCE_ROOT="/path/to/HippUnfold/resources"
export HIPPAMYG_OUTPUT_ROOT="/path/to/group_outputs"
export HIPPAMYG_RESOURCE_ROOT="/path/to/resources"
export HIPPAMYG_JOB_ROOT="/path/to/slurm_logs"

# Cohort files
export HIPPAMYG_SUBJECT_LIST="/path/to/subjects_ready.txt"
export HIPPAMYG_PENDING_SUBJECT_LIST="/path/to/subjects_not_ready.txt"
export HIPPAMYG_CANDIDATE_SUBJECT_LIST="/path/to/candidate_subjects_after_HCP_and_r227_QC.txt"
export HIPPAMYG_MOTION_THRESHOLD=0.2

# Restricted HCP demographic tables (not distributed with this repository)
export HCP_RESTRICTED_CSV="/path/to/RESTRICTED.csv"
export HCP_UNRESTRICTED_CSV="/path/to/unrestricted.csv"

# Software
export HIPPUNFOLD_IMAGE="/path/to/khanlab_hippunfold_1.5.1.sif"
export HIPPAMYG_CONDA_ENV="hippamyg"
export ACTFLOW_CONDA_ENV="actflow_env"
export CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"

# Parent directory containing the ActflowToolbox 0.3.2 checkout. The supplied
# actflow_env.yml installs its numerical dependencies, not the toolbox source.
export ACTFLOW_TOOLBOX_ROOT="/path/to/parent/of/ActflowToolbox"

# Thread count used by Workbench, NumPy/BLAS, and subject-level jobs.
export HIPPAMYG_N_THREADS=8
