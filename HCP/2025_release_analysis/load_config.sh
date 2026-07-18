#!/usr/bin/env bash

# Source the project configuration once from every shell entry point.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${HIPPAMYG_CONFIG:-${SCRIPT_DIR}/config.sh}"

if [[ ! -r "${CONFIG_FILE}" ]]; then
    echo "Missing configuration: ${CONFIG_FILE}" >&2
    echo "Copy config.example.sh to config.sh and edit the paths." >&2
    exit 2
fi

# shellcheck source=/dev/null
source "${CONFIG_FILE}"

required_vars=(
    HCP_RFMRI_ZIP_ROOT
    HCP_STRUCTURAL_ROOT
    HIPPAMYG_WORK_ROOT
    HIPPUNFOLD_ROOT
    HIPPUNFOLD_INPUT_ROOT
    HIPPUNFOLD_RESOURCE_ROOT
    HIPPAMYG_OUTPUT_ROOT
    HIPPAMYG_RESOURCE_ROOT
)

for name in "${required_vars[@]}"; do
    if [[ -z "${!name:-}" ]]; then
        echo "Configuration variable ${name} is not set." >&2
        exit 2
    fi
done

export HIPPAMYG_N_THREADS="${HIPPAMYG_N_THREADS:-8}"
