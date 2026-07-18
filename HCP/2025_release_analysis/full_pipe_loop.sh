#!/usr/bin/env bash
# Submit the subject-level manuscript pipeline to Slurm.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=load_config.sh
source "${SCRIPT_DIR}/load_config.sh"

SUBJECT_LIST="${1:-${HIPPAMYG_PENDING_SUBJECT_LIST:?Set HIPPAMYG_PENDING_SUBJECT_LIST}}"
JOB_ROOT="${HIPPAMYG_JOB_ROOT:?Set HIPPAMYG_JOB_ROOT}"
mkdir -p "${JOB_ROOT}"

while IFS= read -r subject; do
    [[ -z "${subject}" || "${subject}" == \#* ]] && continue
    echo "Submitting ${subject}"
    sbatch \
        --job-name="hippamyg_${subject}" \
        --cpus-per-task="${HIPPAMYG_N_THREADS}" \
        --mem=72G \
        --time=10:00:00 \
        --output="${JOB_ROOT}/${subject}.out" \
        --error="${JOB_ROOT}/${subject}.err" \
        "${SCRIPT_DIR}/full_pipe_sbatch.sh" "${subject}"
done < "${SUBJECT_LIST}"
