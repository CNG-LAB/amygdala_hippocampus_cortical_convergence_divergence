#!/usr/bin/env bash
# Transform the cleaned HCP-YA 2025 volumetric BOLD series to the subject's
# 2 mm ACPC-aligned T1w grid using the supplied inverse nonlinear warp.

set -euo pipefail

SUBJECT="${1:?Usage: $0 <subject_id>}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=load_config.sh
source "${SCRIPT_DIR}/load_config.sh"

STRUCTURAL="${HCP_STRUCTURAL_ROOT}/${SUBJECT}"
T1W_DIR="${STRUCTURAL}/T1w"
MNI_DIR="${STRUCTURAL}/MNINonLinear"
FMRI_DIR="${HIPPAMYG_WORK_ROOT}/${SUBJECT}/tmp/${SUBJECT}/MNINonLinear/Results/rfMRI_REST"
REF_DIR="${HIPPAMYG_WORK_ROOT}/${SUBJECT}/refs"
OUT_DIR="${HIPPAMYG_WORK_ROOT}/${SUBJECT}/func_native_2025"
mkdir -p "${REF_DIR}" "${OUT_DIR}"

T1_NATIVE="${T1W_DIR}/T1w_acpc_dc_restore.nii.gz"
MASK_NATIVE="${T1W_DIR}/brainmask_fs.nii.gz"
REF_2MM="${REF_DIR}/T1w_acpc_dc_restore_2mm.nii.gz"
MASK_2MM="${REF_DIR}/brainmask_fs_2mm.nii.gz"
REF_BRAIN="${REF_DIR}/T1w_acpc_dc_restore_2mm_brain.nii.gz"
FMRI_IN="${FMRI_DIR}/rfMRI_REST_hp2000_clean_rclean_tclean.nii.gz"
WARP="${MNI_DIR}/xfms/standard2acpc_dc.nii.gz"
FNIRT_REF="${MNI_DIR}/T1w_restore.nii.gz"
OUT_BOLD="${OUT_DIR}/rfMRI_REST_inT1w_2mm_tclean_cubic.nii.gz"

for file in "${T1_NATIVE}" "${MASK_NATIVE}" "${FMRI_IN}" "${WARP}" "${FNIRT_REF}"; do
    [[ -f "${file}" ]] || { echo "Missing input: ${file}" >&2; exit 2; }
done

if [[ ! -f "${REF_2MM}" ]]; then
    flirt -interp spline -in "${T1_NATIVE}" -ref "${T1_NATIVE}" \
        -applyisoxfm 2 -out "${REF_2MM}"
fi
if [[ ! -f "${MASK_2MM}" ]]; then
    flirt -interp nearestneighbour -in "${MASK_NATIVE}" -ref "${REF_2MM}" \
        -applyisoxfm 2 -out "${MASK_2MM}"
fi
if [[ ! -f "${REF_BRAIN}" ]]; then
    fslmaths "${REF_2MM}" -mas "${MASK_2MM}" "${REF_BRAIN}"
fi

if [[ ! -f "${OUT_BOLD}" || "${HIPPAMYG_OVERWRITE:-0}" == 1 ]]; then
    wb_command -volume-warpfield-resample \
        "${FMRI_IN}" "${WARP}" "${REF_BRAIN}" CUBIC "${OUT_BOLD}" \
        -fnirt "${FNIRT_REF}"
    fslmaths "${OUT_BOLD}" -mas "${MASK_2MM}" "${OUT_BOLD}"
fi
