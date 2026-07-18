#!/usr/bin/env bash
# Run HippUnfold 1.5.1 for one participant.

set -euo pipefail

SUBJECT="${1:?Usage: $0 <subject_id>}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=load_config.sh
source "${SCRIPT_DIR}/load_config.sh"

singularity run -e \
    "${HIPPUNFOLD_IMAGE:?Set HIPPUNFOLD_IMAGE}" \
    "${HIPPUNFOLD_INPUT_ROOT}" \
    "${HIPPUNFOLD_ROOT}/${SUBJECT}" \
    participant -p \
    --modality T1w \
    --cores "${HIPPAMYG_N_THREADS}" \
    --participant-label "${SUBJECT}" \
    --atlas multihist7 \
    --output-density 2mm \
    --force-output \
    -f \
    --latency-wait 8
