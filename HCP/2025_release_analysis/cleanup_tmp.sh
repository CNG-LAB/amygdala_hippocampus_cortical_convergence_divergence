#!/usr/bin/env bash
# Remove only the subject-specific temporary extraction directory.

set -euo pipefail

SUBJECT="${1:?Usage: $0 <subject_id>}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=load_config.sh
source "${SCRIPT_DIR}/load_config.sh"

TMP="${HIPPAMYG_WORK_ROOT}/${SUBJECT}/tmp"
[[ -d "${TMP}" ]] || exit 0

case "${TMP}" in
    "${HIPPAMYG_WORK_ROOT}/${SUBJECT}/tmp") rm -rf -- "${TMP}" ;;
    *) echo "Refusing to remove unexpected path: ${TMP}" >&2; exit 3 ;;
esac
