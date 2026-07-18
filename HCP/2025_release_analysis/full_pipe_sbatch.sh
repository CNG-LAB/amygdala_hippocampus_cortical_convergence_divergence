#!/usr/bin/env bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=72G
#SBATCH --time=10:00:00
#SBATCH --job-name=hippamyg
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Run all subject-level stages used by the manuscript.

set -euo pipefail

SUBJECT="${1:?Usage: $0 <subject_id>}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=load_config.sh
source "${SCRIPT_DIR}/load_config.sh"

export OMP_NUM_THREADS="${HIPPAMYG_N_THREADS}"
export OPENBLAS_NUM_THREADS="${HIPPAMYG_N_THREADS}"
export MKL_NUM_THREADS="${HIPPAMYG_N_THREADS}"
export NUMEXPR_NUM_THREADS="${HIPPAMYG_N_THREADS}"

if [[ -r "${CONDA_SH:-}" ]]; then
    # shellcheck source=/dev/null
    source "${CONDA_SH}"
fi

bash "${SCRIPT_DIR}/unzip_hcp.sh" "${SUBJECT}"
bash "${SCRIPT_DIR}/cubic_registration_single_subj_2mm.sh" "${SUBJECT}"

HIPP_DONE="${HIPPUNFOLD_ROOT}/${SUBJECT}/hippunfold/sub-${SUBJECT}/coords/sub-${SUBJECT}_dir-PD_hemi-L_space-cropT1w_label-dentate_desc-laplace_coords.nii.gz"
if [[ ! -f "${HIPP_DONE}" ]]; then
    bash "${SCRIPT_DIR}/hippunfold_run_single_subj.sh" "${SUBJECT}"
fi

bash "${SCRIPT_DIR}/extract_cortex_surface_fmri.sh" "${SUBJECT}"
bash "${SCRIPT_DIR}/hippunfold_rsfmri_map_vol_to_surf.sh" "${SUBJECT}"

conda activate "${HIPPAMYG_CONDA_ENV}"
LABEL_L="${HIPPUNFOLD_ROOT}/${SUBJECT}/hippunfold/sub-${SUBJECT}/surf/sub-${SUBJECT}_hemi-L_space-unfold_den-2mm_label-hipp_DeKraker15.label.gii"
if [[ ! -f "${LABEL_L}" ]]; then
    python "${SCRIPT_DIR}/generate_dekrakerN.py" "${SUBJECT}"
    for hemisphere in L R; do
        BASE="${HIPPUNFOLD_ROOT}/${SUBJECT}/hippunfold/sub-${SUBJECT}"
        wb_command -volume-to-surface-mapping \
            "${BASE}/anat/sub-${SUBJECT}_hemi-${hemisphere}_space-unfold_label-hipp_atlas-multihist7_subfields-DeKraker15.nii.gz" \
            "${BASE}/surf/sub-${SUBJECT}_hemi-${hemisphere}_space-unfold_den-2mm_label-hipp_midthickness.surf.gii" \
            "${BASE}/surf/sub-${SUBJECT}_hemi-${hemisphere}_space-unfold_den-2mm_label-hipp_DeKraker15.label.gii" \
            -enclosing
    done
fi

bash "${SCRIPT_DIR}/amyg_resample_softmask_extract_ts.sh" "${SUBJECT}"
conda activate "${ACTFLOW_CONDA_ENV}"
python "${SCRIPT_DIR}/build_full_glasser_pearson_and_glasso_fc.py" "${SUBJECT}"

# Temporary HCP package contents are removed only after every stage succeeds.
bash "${SCRIPT_DIR}/cleanup_tmp.sh" "${SUBJECT}"
