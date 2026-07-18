#!/usr/bin/env bash
# Build mass-conserving 2 mm amygdala masks and extract weighted time series.

set -euo pipefail

SUBJECT="${1:?Usage: $0 <subject_id>}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=load_config.sh
source "${SCRIPT_DIR}/load_config.sh"

export OMP_NUM_THREADS="${HIPPAMYG_N_THREADS}"
export OPENBLAS_NUM_THREADS="${HIPPAMYG_N_THREADS}"
export MKL_NUM_THREADS="${HIPPAMYG_N_THREADS}"

SEGMENTATION_ROOT="${AMYGDALA_SEGMENTATION_ROOT:-${HCP_STRUCTURAL_ROOT}}"
FS_MRI="${SEGMENTATION_ROOT}/${SUBJECT}/T1w/${SUBJECT}/mri"
WORK_DIR="${HIPPAMYG_WORK_ROOT}/${SUBJECT}"
REF_2MM="${WORK_DIR}/refs/T1w_acpc_dc_restore_2mm.nii.gz"
BOLD="${WORK_DIR}/func_native_2025/rfMRI_REST_inT1w_2mm_tclean_cubic.nii.gz"
REFS_DIR="${WORK_DIR}/refs"
MASK_DIR="${WORK_DIR}/masks_soft_2mm_mass"
TS_DIR="${WORK_DIR}/timeseries"
mkdir -p "${REFS_DIR}" "${MASK_DIR}" "${TS_DIR}"

LEFT_LABELS="${FS_MRI}/lh.hippoAmygLabels.mgz"
RIGHT_LABELS="${FS_MRI}/rh.hippoAmygLabels.mgz"
for file in "${LEFT_LABELS}" "${RIGHT_LABELS}" "${REF_2MM}" "${BOLD}"; do
    [[ -f "${file}" ]] || { echo "Missing input: ${file}" >&2; exit 2; }
done

IDS=(7001 7003 7005 7006 7007 7008 7009 7010 7015)
REF_FINE="${REFS_DIR}/T1w_acpc_dc_restore_0p3333mm_aligned.nii.gz"
if [[ ! -f "${REF_FINE}" ]]; then
    flirt -in "${REF_2MM}" -ref "${REF_2MM}" -applyisoxfm 0.3333333333 -out "${REF_FINE}"
fi

for hemisphere in L R; do
    source_labels="${LEFT_LABELS}"
    [[ "${hemisphere}" == R ]] && source_labels="${RIGHT_LABELS}"
    aligned_labels="${MASK_DIR}/labels_${hemisphere}_0p3333.nii.gz"
    if [[ ! -f "${aligned_labels}" ]]; then
        mri_vol2vol --mov "${source_labels}" --targ "${REF_FINE}" \
            --o "${aligned_labels}" --regheader --interp nearest
    fi
    for id in "${IDS[@]}"; do
        binary="${MASK_DIR}/bin_${hemisphere}_${id}_0p3333.nii.gz"
        [[ -f "${binary}" ]] || fslmaths "${aligned_labels}" \
            -thr "${id}" -uthr "${id}" -bin "${binary}"
    done
done

pool_args=()
[[ "${HIPPAMYG_OVERWRITE:-0}" == 1 ]] && pool_args+=(--force)
python "${SCRIPT_DIR}/pool_blockmean_exact_audit.py" \
    "${REF_2MM}" \
    "${MASK_DIR}/bin_*_*_0p3333.nii.gz" \
    "${MASK_DIR}/soft_amyg_{hemi}_{lab}_2mm_pool.nii.gz" \
    "${pool_args[@]}"

python "${SCRIPT_DIR}/extract_amygdala_timeseries.py" \
    "${BOLD}" "${MASK_DIR}" "${TS_DIR}/${SUBJECT}_amyg_softROIts_mass"
