#!/usr/bin/env bash
# Verify and unpack the HCP-YA 2025 Rest3TRecommended archive for one subject.

set -euo pipefail

SUBJECT="${1:?Usage: $0 <subject_id>}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=load_config.sh
source "${SCRIPT_DIR}/load_config.sh"

ZIP="${HCP_RFMRI_ZIP_ROOT}/${SUBJECT}_Rest3TRecommended.zip"
MD5="${ZIP}.md5"
SUBJECT_DIR="${HIPPAMYG_WORK_ROOT}/${SUBJECT}"
TMP="${SUBJECT_DIR}/tmp"

[[ -f "${ZIP}" ]] || { echo "Missing archive: ${ZIP}" >&2; exit 2; }

if [[ -f "${MD5}" ]]; then
    checksum="$(awk 'NR == 1 {print $1}' "${MD5}")"
    [[ "${#checksum}" -ge 32 ]] || { echo "Invalid checksum file: ${MD5}" >&2; exit 2; }
    (cd "${HCP_RFMRI_ZIP_ROOT}" && echo "${checksum}  $(basename "${ZIP}")" | md5sum -c -)
fi

mkdir -p "${TMP}"
find "${TMP}" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
unzip -q -o "${ZIP}" -d "${TMP}"
