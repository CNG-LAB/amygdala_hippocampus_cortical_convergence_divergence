# Selective convergence and graded divergence of hippocampal and amygdala subregions using functional connectivity


Code accompanying *Selective convergence and graded divergence of hippocampal
and amygdala subregions across the human cortex*.

doi: https://doi.org/10.64898/2026.07.06.736898

This repository contains the subject-level processing and group analyses used
for the manuscript's 722-participant HCP Young Adult 2025 sample. The pipeline
constructs 406 x 406 Pearson and graphical-lasso (GLASSO) functional
connectivity matrices, then produces the reported dominance, sharedness,
seed-preference, hippocampus-amygdala coupling, reliability, threshold
sensitivity, and joint-gradient results.

HCP data, restricted demographic tables, subject identifiers, FreeSurfer
derivatives, and atlas/template resources are not redistributed. Users must
obtain those inputs under their respective data-use and software terms.

## Analysis definition

Each subject matrix contains, in order:

1. 9 left amygdala subnuclei;
2. 9 right amygdala subnuclei;
3. 15 left hippocampal parcels;
4. 15 right hippocampal parcels; and
5. 358 Glasser cortical parcels.

The hippocampal parcels are five subfields (subiculum, CA1, CA2, CA3, and CA4)
divided into three longitudinal bins. Glasser parcels 120 and 300 are excluded.
The four HCP resting-state runs contribute 4,800 time points per subject.

The cortical reference gradients are computed from the group-average Pearson
cortex-to-cortex matrix in this same HCP sample. They are sample-derived
gradients.

## Repository contents

- `HCP/2025_release_analysis/` contains the executable subject pipeline and
  manuscript analysis entry points.
- `utils/` contains numerical, labeling, plotting, configuration, Figure 3C,
  motion-audit, and split-half modules used by those entry points.
- `hippamyg.yml` and `actflow_env.yml` define the two pinned Conda environments.
- `software_versions.txt` records the external software versions used.
- `HCP/2025_release_analysis/config.example.sh` documents every local path and
  software setting.

The main entry points are:

| Script | Purpose |
|---|---|
| `full_pipe_loop.sh` | Submit the subject pipeline to Slurm for a subject list. |
| `full_pipe_sbatch.sh` | Run all required processing for one subject. |
| `build_full_glasser_pearson_and_glasso_fc.py` | Build the subject Pearson and GLASSO matrices. |
| `hippunfold_check_surface_mapping.py` | Optional single-subject surface-mapping QC. |
| `hipp_majority_voting_dekrak15_nifti.py` | Build fixed-cohort hippocampal visualization labels. |
| `compute_cortical_gradients.py` | Compute the sample-derived cortical reference gradients. |
| `amyg_hipp_top_connections_paper.py` | Run the primary dominance/sharedness, seed-level, coupling, and split-half analyses. |
| `correlate_different_thresholds.py` | Create the 5%/10%/15% robustness comparisons. |
| `hipp_amyg_gradients_top10_paper.py` | Run the Pearson or GLASSO joint-gradient analysis. |
| `jointfdr_corr_matrices_gradients.py` | Apply the joint FDR correction across gradient correspondence tests. |
| `glasso_lambda_check.py` | Reproduce the GLASSO lambda, density, and motion diagnostics. |
| `demographics_age_gender.py` | Summarize age and sex for the final cohort. |
| `create_amygdala_svg.py` | Generate the amygdala component of the Figure 1 schematic. |
| `utils/schematic_hipp_unfolded.py` | Generate the unfolded hippocampal schematic. |
| `utils/plot_yeo7_glasser.py` | Generate the Glasser/Yeo-7 cortical schematic. |

The remaining scripts in `HCP/2025_release_analysis/` are called by the
subject pipeline and should be kept beside these entry points.

## Software requirements

Create both pinned Conda environments:

```bash
conda env create -f hippamyg.yml
conda env create -f actflow_env.yml
```

The subject pipeline activates `hippamyg` for hippocampal labeling and
`actflow_env` for subject-level connectivity estimation. Group analyses and
figure generation use `hippamyg`.

`actflow_env.yml` contains the numerical dependencies required by
ActflowToolbox but does not install the toolbox source. Obtain
ActflowToolbox 0.3.2 separately and set `ACTFLOW_TOOLBOX_ROOT` in
`config.sh` to the directory containing the `ActflowToolbox/` package.

The external software versions used for the analysis are recorded in
`software_versions.txt`:

- HippUnfold 1.5.1;
- hippunfold_toolbox commit
  `fbd5d76a9f59f616fd9a4416c4785d5083868ab8`;
- DataLad 1.1.5;
- FreeSurfer 7.4.1;
- FSL 6.0.7.11;
- ActflowToolbox 0.3.2; and
- Connectome Workbench 2.1.0.

FSL, FreeSurfer, Connectome Workbench, Singularity/Apptainer, Slurm,
`unzip`, and `md5sum` must be available on compute-node `PATH`.
HippUnfold is run from the 1.5.1 container specified in `config.sh`.
Slurm is required only by the supplied batch launcher.

After creating `config.sh`, verify the relevant commands and both Python
environments:

```bash
for command in flirt fslmaths mri_vol2vol wb_command singularity unzip md5sum; do
    command -v "${command}" || exit 1
done

source HCP/2025_release_analysis/config.sh

conda run -n hippamyg python -c "import brainspace, nibabel, nilearn, neuromaps, numpy, pandas, scipy, statsmodels, surfplot"

PYTHONPATH="${ACTFLOW_TOOLBOX_ROOT}" conda run -n actflow_env python -c "from ActflowToolbox.connectivity_estimation import graphicalLassoCV; print('Actflow import OK')"
```

The optional `create_amygdala_svg.py` schematic generator additionally
requires `scikit-image`, `shapely`, and `svgwrite`. These packages are not
part of the supplied numerical-analysis environments. Install them only if
regenerating that schematic:

```bash
conda install -n hippamyg -c conda-forge scikit-image shapely svgwrite
```

## Required input layout

### HCP functional and structural data

The subject pipeline expects:

```text
${HCP_RFMRI_ZIP_ROOT}/${SUBJECT}_Rest3TRecommended.zip
${HCP_RFMRI_ZIP_ROOT}/${SUBJECT}_Rest3TRecommended.zip.md5   # optional

${HCP_STRUCTURAL_ROOT}/${SUBJECT}/T1w/T1w_acpc_dc_restore.nii.gz
${HCP_STRUCTURAL_ROOT}/${SUBJECT}/T1w/brainmask_fs.nii.gz
${HCP_STRUCTURAL_ROOT}/${SUBJECT}/MNINonLinear/T1w_restore.nii.gz
${HCP_STRUCTURAL_ROOT}/${SUBJECT}/MNINonLinear/xfms/standard2acpc_dc.nii.gz
```

The resting-state archive must contain the HCP-YA 2025 cleaned concatenated
resting-state files used by `unzip_hcp.sh`,
`cubic_registration_single_subj_2mm.sh`, and
`extract_cortex_surface_fmri.sh`.

### FreeSurfer amygdala segmentations

Run FreeSurfer's hippocampal/amygdala subregion segmentation before this
pipeline. The expected files are:

```text
${AMYGDALA_SEGMENTATION_ROOT}/${SUBJECT}/T1w/${SUBJECT}/mri/lh.hippoAmygLabels.mgz
${AMYGDALA_SEGMENTATION_ROOT}/${SUBJECT}/T1w/${SUBJECT}/mri/rh.hippoAmygLabels.mgz
```

Set `AMYGDALA_SEGMENTATION_ROOT` separately when these derivatives are not
stored under `HCP_STRUCTURAL_ROOT`.

### HippUnfold inputs and resources

`HIPPUNFOLD_INPUT_ROOT` must be a BIDS-compatible T1w input tree accepted by
HippUnfold 1.5.1. Group plotting additionally requires:

```text
${HIPPUNFOLD_RESOURCE_ROOT}/canonical_surfs/tpl-avg_space-canonical_den-2mm_label-hipp_midthickness.surf.gii
```

### Project resources

Place the following files under `HIPPAMYG_RESOURCE_ROOT`:

| File | Required content |
|---|---|
| `glasser.csv` | Concatenated left/right fsLR-32k vertex labels using integer Glasser IDs 1-360. Its length must match the concatenated cortical GIFTIs. |
| `atlas-Glasser_dseg.tsv` | One row per Glasser parcel, including `index` and `community_yeo7` columns. |
| `glasser-360_conte69_lh.label.gii` | Left fsLR-32k Glasser label GIFTI. |
| `glasser-360_conte69_rh.label.gii` | Right fsLR-32k Glasser label GIFTI. |
| `lh.AmygLabels.mgz`, `rh.AmygLabels.mgz` | Left/right label volumes used for group amygdala rendering. |
| `lh.hippoAmygLabels.mgz`, `rh.hippoAmygLabels.mgz` | Label volumes used by the Figure 1 amygdala schematic. |

`Glasser32k_dist_L.npy` and `Glasser32k_dist_R.npy` are optional caches in the
same directory. If they are absent, the Moran spectral randomization helper
computes them from the Glasser GIFTIs and neuromaps fsLR surfaces. Supplying
the cached arrays is preferable on compute nodes.

The motion audit described below creates
`HCP_Group_MeanFD_Cleaned.csv` in this resource directory. The historical
filename and `MeanFD` column are retained for compatibility, but the stored
quantity is the four-run mean of HCP `Movement_RelativeRMS_mean.txt`, not
Power-style framewise displacement.

## Configuration

Create the untracked local configuration:

```bash
cp HCP/2025_release_analysis/config.example.sh \
   HCP/2025_release_analysis/config.sh
```

Edit every `/path/to/...` value in `config.sh`, then load it:

```bash
source HCP/2025_release_analysis/config.sh
source HCP/2025_release_analysis/load_config.sh
```

The supplied environments are named `hippamyg` and `actflow_env`, matching
the defaults in `config.example.sh`. Set `ACTFLOW_TOOLBOX_ROOT` to the
directory containing the ActflowToolbox 0.3.2 checkout.

The three cohort variables have distinct roles:

- `HIPPAMYG_CANDIDATE_SUBJECT_LIST`: candidates entering the four-run motion
  audit;
- `HIPPAMYG_PENDING_SUBJECT_LIST`: subjects to submit to the Slurm subject
  pipeline; and
- `HIPPAMYG_SUBJECT_LIST`: the final subject list used
  by every group analysis.

Each list is a plain text file containing one HCP subject ID per line. Empty
lines and lines beginning with `#` are ignored by the group loaders.

## Cohort and motion audit

Construct the candidate list after applying the criteria described in the manuscript, then run:

```bash
source HCP/2025_release_analysis/config.sh
bash utils/get_FD_HCP.sh
```

The script requires all four HCP runs and applies
`HIPPAMYG_MOTION_THRESHOLD` (0.2 by default). It writes an all-candidate audit
table and a passing-subject table under `HIPPAMYG_RESOURCE_ROOT`. After all
remaining data-availability/QC criteria are applied, set
`HIPPAMYG_SUBJECT_LIST` to the final ordered list.

## Run order

Start every new shell with:

```bash
source HCP/2025_release_analysis/config.sh
conda activate "${HIPPAMYG_CONDA_ENV}"
```

### 1. Subject-level processing

Submit all subjects listed in `HIPPAMYG_PENDING_SUBJECT_LIST`:

```bash
bash HCP/2025_release_analysis/full_pipe_loop.sh
```

Alternatively, pass a list explicitly:

```bash
bash HCP/2025_release_analysis/full_pipe_loop.sh /path/to/subjects.txt
```

For a single subject without the launcher:

```bash
bash HCP/2025_release_analysis/full_pipe_sbatch.sh 100206
```

For each subject, this performs archive verification/extraction, cubic BOLD
mapping to the 2-mm T1w grid, HippUnfold processing, cortical and hippocampal
surface mapping, fractional-mask amygdala extraction, and Pearson/GLASSO
matrix construction. Temporary extracted HCP data are removed only after all
stages succeed.

Optional HippUnfold surface-mapping QC can be run after a subject finishes:

```bash
python HCP/2025_release_analysis/hippunfold_check_surface_mapping.py 100206
```

### 2. Check cohort metadata and GLASSO diagnostics

After all final-cohort matrices exist:

```bash
python HCP/2025_release_analysis/demographics_age_gender.py
python HCP/2025_release_analysis/glasso_lambda_check.py
```

The demographic command requires the restricted and unrestricted HCP tables
configured in `config.sh`. The GLASSO diagnostic requires the motion-audit CSV.

### 3. Create cohort hippocampal display labels

```bash
python HCP/2025_release_analysis/hipp_majority_voting_dekrak15_nifti.py
```

This writes left/right majority-voted unfolded NIfTI maps and canonical-surface
label arrays under `HIPPAMYG_OUTPUT_ROOT`. Figure 1, 3, and 4 visualizations use these
outputs.

### 4. Compute the cortical reference gradients

```bash
python HCP/2025_release_analysis/compute_cortical_gradients.py
```

This writes `cortex_intrinsic_gradients.npy`,
`cortex_intrinsic_eigenvalues.npy`, and `kept_sorted.npy` under:

```text
${HIPPAMYG_OUTPUT_ROOT}/group_level_gradients/top10_union_pearson/
```

This stage must finish before either primary group analysis uses the
cortical gradients.

### 5. Run the primary dominance/sharedness analysis

```bash
python HCP/2025_release_analysis/amyg_hipp_top_connections_paper.py
```

The default 10% run creates the primary group arrays, Figures 2-3 components,
seed-level summaries, strength corroboration, cross-estimator comparisons, and
split-half reliability outputs.

### 6. Run threshold sensitivity

```bash
HIPPAMYG_TOP_PERCENT=5 \
python HCP/2025_release_analysis/amyg_hipp_top_connections_paper.py

HIPPAMYG_TOP_PERCENT=15 \
python HCP/2025_release_analysis/amyg_hipp_top_connections_paper.py

python HCP/2025_release_analysis/correlate_different_thresholds.py
```

Non-10% runs stop after writing the threshold-specific dominance/sharedness
maps needed for the supplementary robustness figure.

### 7. Run both joint-gradient analyses

```bash
HIPPAMYG_METHOD=pearson \
python HCP/2025_release_analysis/hipp_amyg_gradients_top10_paper.py

HIPPAMYG_METHOD=glasso \
python HCP/2025_release_analysis/hipp_amyg_gradients_top10_paper.py
```

The joint-gradient script loads the cortical gradients from step 4, it does not
recompute them.

### 8. Apply joint FDR correction

```bash
python HCP/2025_release_analysis/jointfdr_corr_matrices_gradients.py
```

Run this only after both estimator-specific joint-gradient runs have saved
their raw Moran spectral randomization results.

### 9. Generate Figure 1 schematics

```bash
python HCP/2025_release_analysis/create_amygdala_svg.py
python utils/schematic_hipp_unfolded.py
python utils/plot_yeo7_glasser.py
```

The hippocampal schematic requires the majority labels from step 3. The amygdala SVG command also requires the optional schematic dependencies listed above.

The group-level dependency order is therefore:

```text
subject matrices
    -> majority hippocampal labels
    -> sample-derived cortical gradients
    -> 10% dominance/sharedness analysis
    -> 5% and 15% sensitivity analyses
    -> Pearson and GLASSO joint gradients
    -> joint FDR correction
```

## Reproducibility-critical settings

- Every subject matrix must be 406 x 406 and arise from 406 nodes x 4,800 time
  points.
- Hippocampal functional GIFTI data are converted to float64 before parcel
  averaging.
- A hippocampal label is rejected when more than 30% of its vertex-by-time
  values are NaN, reproducing the criterion used for the manuscript analysis.
- Amygdala weighted sums use float32 accumulation by default. Changing
  `AMYG_ACC_DTYPE` to `float64` changes the numerical result.
- GLASSO uses ActflowToolbox 0.3.2 with four blocked folds and its default
  lambda grid.
- GLASSO group means treat exact regularization-induced zero edges as absent;
  Pearson group means retain true zeros.
- The 10% cortical selection retains 36 of the 358 available cortical parcels
  per seed.
- Random seeds and diffusion-embedding settings are fixed in the analysis
  scripts.

For a clean reproduction, use empty work and output roots. Several expensive
subject stages reuse existing derivatives when files already exist.
`HIPPAMYG_OVERWRITE=1` forces regeneration for the stages that explicitly
support it, but it does not override every cache in the pipeline.

## Runtime controls

| Variable | Default | Effect |
|---|---:|---|
| `HIPPAMYG_TOP_PERCENT` | `10` | Select 5%, 10%, or 15% dominance/sharedness masks. |
| `HIPPAMYG_METHOD` | `pearson` | Select the joint-gradient estimator (`pearson` or `glasso`). |
| `HIPPAMYG_RUN_MSR` | `1` | Set to `0` only for a diagnostic run that intentionally skips Moran spectral randomization. |
| `HIPPAMYG_OVERWRITE` | `0` | Recompute supported cached subject derivatives when set to `1`. |
| `HIPPAMYG_N_THREADS` | `8` | Set Workbench, BLAS, and subject-job thread count. |
| `AMYG_ACC_DTYPE` | `float32` | Select amygdala weighted-sum accumulation precision. |
| `HIPPAMYG_HEMISPHERE` | `L` | Select the hemisphere for the unfolded hippocampal schematic. |
