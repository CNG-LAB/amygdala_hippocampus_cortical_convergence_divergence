#!/usr/bin/env bash
# Extract left and right cortical time series from the HCP-YA 2025 CIFTI file.

set -euo pipefail

SUBJECT="${1:?Usage: $0 <subject_id>}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=load_config.sh
source "${SCRIPT_DIR}/load_config.sh"

INPUT="${HIPPAMYG_WORK_ROOT}/${SUBJECT}/tmp/${SUBJECT}/MNINonLinear/Results/rfMRI_REST/rfMRI_REST_Atlas_MSMAll_hp2000_clean_rclean_tclean.dtseries.nii"
OUT_DIR="${HIPPAMYG_WORK_ROOT}/${SUBJECT}/func_native_2025"
LEFT="${OUT_DIR}/rfMRI_Cortex_Left_tclean.func.gii"
RIGHT="${OUT_DIR}/rfMRI_Cortex_Right_tclean.func.gii"

[[ -f "${INPUT}" ]] || { echo "Missing input: ${INPUT}" >&2; exit 2; }
mkdir -p "${OUT_DIR}"
[[ -f "${LEFT}" ]] || wb_command -cifti-separate "${INPUT}" COLUMN -metric CORTEX_LEFT "${LEFT}"
[[ -f "${RIGHT}" ]] || wb_command -cifti-separate "${INPUT}" COLUMN -metric CORTEX_RIGHT "${RIGHT}"
