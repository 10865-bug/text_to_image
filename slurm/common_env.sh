# Cluster environment for the Table 5 T2I job (no MuJoCo).
# Optional conda env. Set CONDA_ENV to the environment that has the requirements.

set -euo pipefail

if [[ -n "${MODULE_LOAD:-}" ]]; then
  # shellcheck disable=SC2086
  module load ${MODULE_LOAD}
fi

# Leave CONDA_ENV empty to use the python already on PATH.
_t2i_find_conda_sh() {
  local base=""
  if command -v conda >/dev/null 2>&1; then
    base="$(conda info --base 2>/dev/null || true)"
  fi
  if [[ -z "${base}" ]]; then
    local cand
    for cand in \
      "${HOME}/miniconda3" \
      "${HOME}/anaconda3" \
      "${HOME}/.miniconda3" \
      "${HOME}/.conda" \
      "/software/anaconda3" \
      "/software/miniconda3"
    do
      if [[ -f "${cand}/etc/profile.d/conda.sh" ]]; then
        base="${cand}"
        break
      fi
    done
  fi
  if [[ -n "${base}" && -f "${base}/etc/profile.d/conda.sh" ]]; then
    printf '%s\n' "${base}/etc/profile.d/conda.sh"
  fi
}

if [[ -n "${CONDA_ENV:-}" ]]; then
  _conda_sh="$(_t2i_find_conda_sh || true)"
  if [[ -n "${_conda_sh}" ]]; then
    # shellcheck disable=SC1090
    source "${_conda_sh}"
    set +e
    conda activate "${CONDA_ENV}"
    _act=$?
    set -e
    if [[ "${_act}" -eq 0 ]]; then
      if [[ -z "${PYTHON_BIN:-}" || "${PYTHON_BIN}" == "python" ]]; then
        PYTHON_BIN="$(command -v python)"
      fi
      echo "[env] conda activate ${CONDA_ENV} ok"
    else
      echo "[env] WARNING: conda activate ${CONDA_ENV} failed (${_act})" >&2
    fi
  fi
  if [[ -z "${PYTHON_BIN:-}" || "${PYTHON_BIN}" == "python" ]]; then
    for cand in \
      "${HOME}/.conda/envs/${CONDA_ENV}/bin/python" \
      "${HOME}/miniconda3/envs/${CONDA_ENV}/bin/python" \
      "${HOME}/anaconda3/envs/${CONDA_ENV}/bin/python"
    do
      if [[ -x "${cand}" ]]; then
        PYTHON_BIN="${cand}"
        break
      fi
    done
  fi
fi

export PYTHONUNBUFFERED=1
export PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ "${PYTHON_BIN}" == */python || "${PYTHON_BIN}" == */python3 ]]; then
  export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
  _env_lib="$(cd "$(dirname "${PYTHON_BIN}")/../lib" && pwd)"
  if [[ -d "${_env_lib}" ]]; then
    export LD_LIBRARY_PATH="${_env_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
fi

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

echo "[env] T2I_ROOT=${T2I_ROOT:-}"
echo "[env] CONDA_ENV=${CONDA_ENV:-} CONDA_PREFIX=${CONDA_PREFIX:-}"
echo "[env] PYTHON_BIN=${PYTHON_BIN} ($(command -v "${PYTHON_BIN}" || true))"
"${PYTHON_BIN}" - <<'PY'
import sys
print("[env] python", sys.version.split()[0])
try:
    import torch
    print("[env] torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
except Exception as exc:
    print("[env] torch import failed:", exc, file=sys.stderr)
    raise
PY
