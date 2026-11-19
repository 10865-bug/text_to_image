# Text-to-image sampling

Reproduction code for the Stable Diffusion v1.5 text-to-image comparison.
ImageReward is the guidance reward. Clip-Score and HPS v2.1 are computed on the reported image.
The prompt list is `t2i_prompts_40.txt` (40 GenEval two-object prompts).
Every run uses 100 DDIM steps and `eta = 1` unless noted.

## Setup

Install PyTorch for your CUDA version first, then:

```bash
pip install -r requirements_t2i.txt
pip install git+https://github.com/openai/CLIP.git
```

The first run downloads Stable Diffusion v1.5, ImageReward, CLIP, and HPS v2.1.

## Methods

All of these are launched by `generate_table5_twoquery.py`.

| Method | What it runs |
|---|---|
| FK-Steering | 4 particles, ImageReward, max potential |
| SVDD | 4 particles, 8 candidates per step |
| URGE | 4 particles |
| TwoQuery | centered Stein pool of size `K`, outer MH, `J = 10` |

`J = 10` means 11 complete trajectories. TwoQuery reports the image kept by independent Metropolis--Hastings. FK-Steering, SVDD, and URGE report the particle with the highest ImageReward.

## Commands

From this directory:

```bash
# TwoQuery, K=2
python generate_table5_twoquery.py \
  --skip-sdxl --skip-fk --skip-svdd --skip-urge --skip-base \
  --twoq-k 2 --twoq-j 10 --time-steps 100 --max-prompts 40 \
  --output-dir samples_twoquery_k2

# SVDD, N=4 particles, 8 candidates per step
python generate_table5_twoquery.py \
  --skip-sdxl --skip-fk --skip-twoquery --skip-urge --skip-base \
  --svdd-mode smc --svdd-m 8 --num-particles 4 --time-steps 100 --max-prompts 40 \
  --output-dir samples_svdd

# FK-Steering, N=4
python generate_table5_twoquery.py \
  --skip-sdxl --skip-svdd --skip-urge --skip-twoquery --skip-base \
  --num-particles 4 --time-steps 100 --max-prompts 40 \
  --output-dir samples_fk

# URGE, N=4
python generate_table5_twoquery.py \
  --skip-sdxl --skip-fk --skip-svdd --skip-twoquery --skip-base \
  --num-particles 4 --time-steps 100 --max-prompts 40 \
  --output-dir samples_urge
```

Each output directory contains `summary.json`. The reported number for a metric is the mean in that file.

The same commands run on a cluster. From this directory, with the environment already activated:

```bash
sbatch slurm/run_table5_twoquery.slurm
TWOQ_K=4 sbatch slurm/run_table5_twoquery.slurm
SKIP_SDXL=1 SKIP_BASE=1 sbatch slurm/run_table5.slurm
```

The scripts do not name an account, a partition, or a GPU model. `sbatch` uses the account and partition already configured for the submitting user, and each script requests one GPU. A site that requires those choices takes them on the command line; they override the script:

```bash
sbatch --account=def-name --partition=gpu --gres=gpu:a100:1 slurm/run_table5_twoquery.slurm
```

The job runs `python` from the environment that `sbatch` receives. Activate that environment before submitting. To activate a named conda environment inside the job instead, set `CONDA_ENV` to its name.
