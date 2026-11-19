#!/bin/bash
# Login-node install for Table 5 T2I. Does not reinstall torch.
set -euo pipefail
T2I_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${T2I_ROOT}"
export PYTHONPATH="${T2I_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
"${PYTHON_BIN}" -m pip install -U pip
"${PYTHON_BIN}" -m pip install -r "${T2I_ROOT}/requirements_t2i.txt"
echo "[setup] python=${PYTHON_BIN}"
"${PYTHON_BIN}" - <<'PY'
import importlib
for m in ("torch", "diffusers", "transformers", "PIL", "clip", "hpsv2"):
    importlib.import_module(m)
    print("ok", m)
from fkd_diffusers.rewards import ensure_hpsv2_bpe
print("hpsv2 bpe", ensure_hpsv2_bpe())
print("ImageReward / HPSv2 checkpoint download is deferred until first score")
PY
