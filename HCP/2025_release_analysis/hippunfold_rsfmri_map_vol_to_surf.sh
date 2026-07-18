#!/usr/bin/env bash
# Sample the cleaned T1w-space BOLD series on subject-specific HippUnfold surfaces.

set -euo pipefail

SUBJECT="${1:?Usage: $0 <subject_id>}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=load_config.sh
source "${SCRIPT_DIR}/load_config.sh"

BASE="${HIPPUNFOLD_ROOT}/${SUBJECT}/hippunfold/sub-${SUBJECT}"
FMRI="${HIPPAMYG_WORK_ROOT}/${SUBJECT}/func_native_2025/rfMRI_REST_inT1w_2mm_tclean_cubic.nii.gz"
OUT_DIR="${BASE}/func"
[[ -f "${FMRI}" ]] || { echo "Missing input: ${FMRI}" >&2; exit 2; }
mkdir -p "${OUT_DIR}"

for hemisphere in L R; do
    for label in hipp dentate; do
        surface="${BASE}/surf/sub-${SUBJECT}_hemi-${hemisphere}_space-T1w_den-2mm_label-${label}_midthickness.surf.gii"
        output="${OUT_DIR}/sub-${SUBJECT}_hemi-${hemisphere}_space-T1w_den-2mm_label-${label}_desc-REST_tclean.func.gii"
        [[ -f "${surface}" ]] || { echo "Missing surface: ${surface}" >&2; exit 2; }
        [[ -f "${output}" ]] || wb_command -volume-to-surface-mapping \
            "${FMRI}" "${surface}" "${output}" -trilinear
    done
done
