#!/bin/bash
# TwoQuery-MH on t2i_prompts_40.txt (GenEval two-object 1-40).
#   cd /path/to/Fk-Diffusion-Steering-main/text_to_image
#   TWOQ_K=8 TWOQ_J=10 PROMPT_OFFSET=0 MAX_PROMPTS=10 sbatch slurm/run_table5_twoquery.slurm
#   TWOQ_K=4 TWOQ_J=10 TIME_STEPS=50 sbatch slurm/run_table5_twoquery.slurm

#SBATCH --job-name=t2i_2qmh
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

if [[ -z "${SLURM_SUBMIT_DIR:-}" ]]; then
  echo "ERROR: SLURM_SUBMIT_DIR unset — run from text_to_image/: sbatch slurm/run_table5_twoquery.slurm" >&2
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

TWOQ_K="${TWOQ_K:-2}"
TWOQ_J="${TWOQ_J:-10}"
TIME_STEPS="${TIME_STEPS:-100}"
PROMPT_OFFSET="${PROMPT_OFFSET:-0}"
MAX_PROMPTS="${MAX_PROMPTS:-40}"
JOB_TAG="${SLURM_JOB_ID:-local}"
RESULTS_DIR="${T2I_ROOT}/samples_table5_twoquery/run-${JOB_TAG}-k${TWOQ_K}-j${TWOQ_J}-s${TIME_STEPS}-off${PROMPT_OFFSET}"
mkdir -p "${RESULTS_DIR}"
exec >"${RESULTS_DIR}/slurm.log" 2>&1

export PYTHON_BIN="${PYTHON_BIN:-python}"
source "${T2I_ROOT}/slurm/common_env.sh"

export PYTHONPATH="${T2I_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[table5] TwoQuery-MH k=${TWOQ_K} J=${TWOQ_J} steps=${TIME_STEPS} offset=${PROMPT_OFFSET} max=${MAX_PROMPTS} particles=1"

set -x
"${PYTHON_BIN}" generate_table5_twoquery.py \
  --output-dir "${RESULTS_DIR}" \
  --prompt-file "${T2I_ROOT}/t2i_prompts_40.txt" \
  --prompt-offset "${PROMPT_OFFSET}" \
  --prompt-index-base "${PROMPT_OFFSET}" \
  --max-prompts "${MAX_PROMPTS}" \
  --metrics "ImageReward,Clip-Score,HumanPreference" \
  --num-particles 4 \
  --time-steps "${TIME_STEPS}" \
  --eta 1.0 \
  --seed 42 \
  --twoq-k "${TWOQ_K}" \
  --twoq-j "${TWOQ_J}" \
  --sd15-id "${SD15_MODEL_ID:-runwayml/stable-diffusion-v1-5}" \
  --skip-sdxl \
  --skip-base \
  --skip-fk \
  --skip-svdd \
  --skip-urge
