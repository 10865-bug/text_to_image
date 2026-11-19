#!/bin/bash
# Prompts 1-40 metrics rerun: FK, SVDD, URGE, pure guidance, SDXL base.
# TwoQuery is always skipped. ImageReward guides; CLIP and HPS are scored after.
# SVDD is about 50 min/prompt, so keep it in its own job.
#   SKIP_SVDD=1 sbatch --job-name=p140_rest slurm/run_table5.slurm
#   SKIP_FK=1 SKIP_URGE=1 SKIP_BASE=1 SKIP_SDXL=1 sbatch --job-name=p140_svdd slurm/run_table5.slurm

#SBATCH --job-name=t2i_base
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

if [[ -z "${SLURM_SUBMIT_DIR:-}" ]]; then
  echo "ERROR: SLURM_SUBMIT_DIR unset — run from text_to_image/: sbatch slurm/run_table5.slurm" >&2
  exit 1
fi

if [[ -f "${SLURM_SUBMIT_DIR}/generate_table5_twoquery.py" ]]; then
  T2I_ROOT="${SLURM_SUBMIT_DIR}"
elif [[ -f "${SLURM_SUBMIT_DIR}/text_to_image/generate_table5_twoquery.py" ]]; then
  T2I_ROOT="${SLURM_SUBMIT_DIR}/text_to_image"
else
  echo "ERROR: cannot find generate_table5_twoquery.py under ${SLURM_SUBMIT_DIR}" >&2
  exit 1
fi

export T2I_ROOT
cd "${T2I_ROOT}"

JOB_TAG="${SLURM_JOB_ID:-local}"
PROMPT_OFFSET="${PROMPT_OFFSET:-0}"
MAX_PROMPTS="${MAX_PROMPTS:-40}"
SKIP_FK="${SKIP_FK:-0}"
SKIP_SVDD="${SKIP_SVDD:-0}"
SKIP_URGE="${SKIP_URGE:-0}"
SKIP_BASE="${SKIP_BASE:-0}"
SKIP_SDXL="${SKIP_SDXL:-0}"
BASE_SD15_IMAGES="${BASE_SD15_IMAGES:-0}"
RESULTS_DIR="${T2I_ROOT}/samples_table5_twoquery/run-${JOB_TAG}-p140-metrics"
mkdir -p "${RESULTS_DIR}"
exec >"${RESULTS_DIR}/slurm.log" 2>&1

export PYTHON_BIN="${PYTHON_BIN:-python}"
source "${T2I_ROOT}/slurm/common_env.sh"

export PYTHONPATH="${T2I_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

SKIP_ARGS=()
[[ "${SKIP_FK}" == 1 ]] && SKIP_ARGS+=(--skip-fk)
[[ "${SKIP_SVDD}" == 1 ]] && SKIP_ARGS+=(--skip-svdd)
[[ "${SKIP_URGE}" == 1 ]] && SKIP_ARGS+=(--skip-urge)
[[ "${SKIP_BASE}" == 1 ]] && SKIP_ARGS+=(--skip-base)
[[ "${SKIP_SDXL}" == 1 ]] && SKIP_ARGS+=(--skip-sdxl)

echo "[table5] prompts 1-40 offset=${PROMPT_OFFSET} max=${MAX_PROMPTS} skip fk=${SKIP_FK} svdd=${SKIP_SVDD} urge=${SKIP_URGE} base=${SKIP_BASE} sdxl=${SKIP_SDXL} sd15_base_images=${BASE_SD15_IMAGES}"

set -x
"${PYTHON_BIN}" generate_table5_twoquery.py \
  --output-dir "${RESULTS_DIR}" \
  --prompt-file "${T2I_ROOT}/t2i_prompts_40.txt" \
  --prompt-offset "${PROMPT_OFFSET}" \
  --prompt-index-base "${PROMPT_OFFSET}" \
  --max-prompts "${MAX_PROMPTS}" \
  --metrics "ImageReward,Clip-Score,HumanPreference" \
  --num-particles 4 \
  --time-steps 100 \
  --eta 1.0 \
  --seed 42 \
  --svdd-m 8 \
  --svdd-mode smc \
  --sd15-id "${SD15_MODEL_ID:-runwayml/stable-diffusion-v1-5}" \
  --skip-twoquery \
  --sd15-base-images "${BASE_SD15_IMAGES}" \
  "${SKIP_ARGS[@]}"
