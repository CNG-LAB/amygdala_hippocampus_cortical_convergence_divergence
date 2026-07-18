#!/usr/bin/env bash
# Calculate mean relative RMS displacement and require all four HCP rfMRI runs.

set -euo pipefail

SUBJECT_LIST="${1:-${HIPPAMYG_CANDIDATE_SUBJECT_LIST:?Set HIPPAMYG_CANDIDATE_SUBJECT_LIST}}"
ZIP_ROOT="${HCP_RFMRI_ZIP_ROOT:?Set HCP_RFMRI_ZIP_ROOT}"
OUTPUT_ROOT="${HIPPAMYG_RESOURCE_ROOT:?Set HIPPAMYG_RESOURCE_ROOT}"
THRESHOLD="${HIPPAMYG_MOTION_THRESHOLD:-0.2}"
RUNS=(rfMRI_REST1_LR rfMRI_REST1_RL rfMRI_REST2_LR rfMRI_REST2_RL)

AUDIT_CSV="${OUTPUT_ROOT}/HCP_Group_MeanFD_Audit.csv"
CLEAN_CSV="${OUTPUT_ROOT}/HCP_Group_MeanFD_Cleaned.csv"
printf 'Subject,RunCount,MeanFD,PassMotion\n' > "${AUDIT_CSV}"
printf 'Subject,MeanFD\n' > "${CLEAN_CSV}"

while IFS= read -r subject; do
    [[ -z "${subject}" || "${subject}" == \#* ]] && continue
    archive="${ZIP_ROOT}/${subject}_Rest3TRecommended.zip"
    total=0
    count=0
    if [[ -f "${archive}" ]]; then
        for run in "${RUNS[@]}"; do
            path="${subject}/MNINonLinear/Results/${run}/Movement_RelativeRMS_mean.txt"
            value="$(unzip -p "${archive}" "${path}" 2>/dev/null || true)"
            if [[ -n "${value}" ]]; then
                total="$(awk -v x="${total}" -v y="${value}" 'BEGIN {print x + y}')"
                count=$((count + 1))
            fi
        done
    fi

    mean="nan"
    pass=0
    if [[ "${count}" -eq 4 ]]; then
        mean="$(awk -v x="${total}" 'BEGIN {printf "%.6f", x / 4}')"
        pass="$(awk -v x="${mean}" -v t="${THRESHOLD}" 'BEGIN {print x <= t}')"
    fi
    printf '%s,%s,%s,%s\n' "${subject}" "${count}" "${mean}" "${pass}" >> "${AUDIT_CSV}"
    if [[ "${pass}" -eq 1 ]]; then
        printf '%s,%s\n' "${subject}" "${mean}" >> "${CLEAN_CSV}"
    fi
done < "${SUBJECT_LIST}"
