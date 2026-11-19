"""T2I figure: FK / SVDD / URGE (N=4 SMC) and TwoQuery-MH (single chain).

FK / URGE DDIM settings are copied from generate_samples_for_paper.py
(time_steps=100, eta=1.0, num_particles=4, lmbda=2.0, ImageReward, max
potential, resample 20/20/80). TwoQuery does not use particle SMC: one
trajectory, Stein pool width k, outer independent MH with J=10 as in D4RL
2q_mh. k is 2 or 4 (passed per job).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from typing import List

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if os.path.join(HERE, "fkd_diffusers") not in sys.path:
    sys.path.insert(0, os.path.join(HERE, "fkd_diffusers"))


DEFAULT_PROMPT_FILE = os.path.join(HERE, "t2i_prompts_40.txt")
TWOQ_J = 10
GRID_PROMPTS = 8
# Table-4 metrics. Guidance stays ImageReward; CLIP / HPS are post-hoc on saved images.
DEFAULT_EVAL_METRICS = ["ImageReward", "Clip-Score", "HumanPreference"]
METRIC_ALIASES = {
    "clip": "Clip-Score",
    "clip-score": "Clip-Score",
    "clip-score-only": "Clip-Score-only",
    "hps": "HumanPreference",
    "humanpreference": "HumanPreference",
    "imagereward": "ImageReward",
}

PAPER_FKD = dict(
    lmbda=2.0,
    use_smc=True,
    adaptive_resampling=True,
    resample_frequency=20,
    resampling_t_start=20,
    resampling_t_end=80,
    guidance_reward_fn="ImageReward",
    metric_to_chase=None,
    time_steps=100,
    num_particles=4,
    potential_type="max",
)

def urge_resample_indices(start: int, end: int, frequency: int) -> List[int]:
    """Solver steps that actually multinomial-resample.

    URGE's first ``resample`` call only stores the baseline reward. That call
    is solver step 0, so step 0 is never a resampling step.
    """
    scheduled = list(range(int(start), int(end) + 1, max(int(frequency), 1)))
    return [step for step in scheduled if step != 0]


ROW_LABELS = {
    "fk_sd15": "FK-Steering\nSDv1.5, N=4",
    "svdd_sd15": "SVDD\nSDv1.5, N=4",
    "urge_sd15": "URGE\nSDv1.5, N=4",
    "guidance_sd15": "Pure guidance\nSDv1.5",
    "base_sdxl": "Base Model\nSDXL",
    "twoquery_k2_sd15": "TwoQuery-MH k=2\n1 chain",
    "twoquery_k4_sd15": "TwoQuery-MH k=4\n1 chain",
}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def slug(prompt: str) -> str:
    return prompt.replace(" ", "_").replace(".", "").replace(",", "")[:80]


def load_prompts(path: str, max_prompts: int, offset: int = 0) -> List[str]:
    with open(path, encoding="utf-8") as handle:
        prompts = [line.strip() for line in handle if line.strip() and not line.startswith("#")]
    if offset:
        prompts = prompts[int(offset) :]
    if max_prompts and max_prompts > 0:
        prompts = prompts[: int(max_prompts)]
    if not prompts:
        raise SystemExit(f"no prompts in {path} (offset={offset}, max={max_prompts})")
    return prompts


def parse_metrics(raw: str) -> List[str]:
    metrics = []
    for part in (raw or "").split(","):
        name = part.strip()
        if not name:
            continue
        metrics.append(METRIC_ALIASES.get(name.lower(), name))
    if not metrics:
        raise SystemExit("empty --metrics")
    return metrics


def _as_float_list(values) -> List[float]:
    import numpy as np
    import torch

    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().reshape(-1).tolist()
    return [float(x) for x in np.asarray(values, dtype=np.float64).reshape(-1)]


def eval_metrics(prompt: str, images, metrics: List[str]) -> dict:
    from fks_utils import do_eval

    raw = do_eval(
        prompt=[prompt] * len(images),
        images=images,
        metrics_to_compute=list(metrics),
    )
    return {name: _as_float_list(payload["result"]) for name, payload in raw.items()}


def sort_by_imagereward(prompt: str, images):
    import numpy as np
    from fks_utils import do_eval

    results = do_eval(
        prompt=[prompt] * len(images),
        images=images,
        metrics_to_compute=["ImageReward"],
    )
    scores = np.array(results["ImageReward"]["result"], dtype=np.float64)
    order = np.argsort(scores)[::-1]
    images = [images[int(i)] for i in order]
    scores = scores[order]
    return images, scores.tolist()


def imagereward_one(prompt: str, image) -> float:
    _, scores = sort_by_imagereward(prompt, [image])
    return float(scores[0])


def load_sd15(repo_id: str):
    import torch
    from fks_utils import get_model

    if repo_id == "runwayml/stable-diffusion-v1-5":
        return get_model("stable-diffusion-v1-5")
    from diffusers import DDIMScheduler
    from fkd_diffusers.fkd_pipeline_sd import FKDStableDiffusion

    pipe = FKDStableDiffusion.from_pretrained(
        repo_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    return pipe


def load_sdxl(repo_id: str):
    import torch
    from fks_utils import get_model

    if repo_id == "stabilityai/stable-diffusion-xl-base-1.0":
        return get_model("stable-diffusion-xl")
    from diffusers import DDIMScheduler
    from fkd_diffusers.fkd_pipeline_sdxl import FKDStableDiffusionXL

    pipe = FKDStableDiffusionXL.from_pretrained(repo_id, torch_dtype=torch.float16)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    return pipe


def run_pipeline(pipeline, prompt: str, fkd_args: dict, eta: float, time_steps: int, n_images: int):
    if fkd_args.get("use_smc", False):
        out = pipeline(
            prompt,
            num_inference_steps=time_steps,
            eta=eta,
            fkd_args=fkd_args,
        )
    else:
        out = pipeline(
            [prompt] * n_images,
            num_inference_steps=time_steps,
            eta=eta,
            fkd_args=fkd_args,
        )
    if hasattr(out, "images"):
        images = out.images
    else:
        images = out[0]
    return list(images)


def run_twoquery_mh(
    pipeline,
    prompt: str,
    paper: dict,
    eta: float,
    steps: int,
    twoq_k: int,
    twoq_j: int,
    twoq_tau,
    seed: int,
):
    import torch
    from fkd_diffusers.twoq_core import independent_mh_reduce

    n_prop = 1 + max(int(twoq_j), 1)
    images = []
    log_rs = []
    for j in range(n_prop):
        seed_everything(int(seed) * 10007 + j)
        tq_args = deepcopy(paper)
        tq_args["steering_type"] = "TWOQUERY"
        tq_args["use_smc"] = True
        tq_args["num_particles"] = 1
        tq_args["twoq_k"] = int(twoq_k)
        if twoq_tau is not None:
            tq_args["twoq_tau"] = float(twoq_tau)
        print(f"[table5] TwoQuery-MH chain {j + 1}/{n_prop} k={twoq_k} :: {prompt}")
        imgs = run_pipeline(pipeline, prompt, tq_args, eta, steps, 1)
        image = imgs[0]
        images.append(image)
        handler = getattr(pipeline, "_fkd", None)
        tau = float(getattr(handler, "tau", 0.5) or 0.5)
        if handler is not None and hasattr(handler, "accumulated_log_r"):
            ell = handler.accumulated_log_r(1).reshape(-1)[0].item()
        else:
            ell = 0.0
        terminal = imagereward_one(prompt, image) / max(tau, 1e-8)
        log_rs.append(float(ell) + float(terminal))

    log_stack = torch.tensor(log_rs, dtype=torch.float32).view(n_prop, 1)
    idx = torch.arange(n_prop, dtype=torch.float32).view(n_prop, 1, 1)
    gen = torch.Generator()
    gen.manual_seed(int(seed) * 17 + 1)
    picked, _ = independent_mh_reduce(idx, log_stack, generator=gen)
    chosen_i = int(picked.reshape(-1)[0].item())
    return images[chosen_i], images, log_rs, chosen_i, n_prop


def run_svdd_mh(
    pipeline,
    prompt: str,
    paper: dict,
    eta: float,
    steps: int,
    svdd_m: int,
    svdd_j: int,
    svdd_tau,
    seed: int,
):
    """Same outer MH as TwoQuery. Each chain is one particle with i.i.d. SVDD noise."""
    import torch
    from fkd_diffusers.twoq_core import independent_mh_reduce

    n_prop = 1 + max(int(svdd_j), 1)
    images = []
    log_rs = []
    for j in range(n_prop):
        seed_everything(int(seed) * 10007 + j)
        svdd_args = deepcopy(paper)
        svdd_args["steering_type"] = "SVDD"
        svdd_args["use_smc"] = True
        svdd_args["num_particles"] = 1
        svdd_args["svdd_m"] = int(svdd_m)
        svdd_args["svdd_mh"] = True
        if svdd_tau is not None:
            svdd_args["twoq_tau"] = float(svdd_tau)
        print(f"[table5] SVDD-MH chain {j + 1}/{n_prop} m={svdd_m} :: {prompt}")
        imgs = run_pipeline(pipeline, prompt, svdd_args, eta, steps, 1)
        image = imgs[0]
        images.append(image)
        handler = getattr(pipeline, "_fkd", None)
        tau = float(getattr(handler, "tau", 0.5) or 0.5)
        if handler is not None and hasattr(handler, "accumulated_log_r"):
            ell = handler.accumulated_log_r(1).reshape(-1)[0].item()
        else:
            ell = 0.0
        terminal = imagereward_one(prompt, image) / max(tau, 1e-8)
        log_rs.append(float(ell) + float(terminal))

    log_stack = torch.tensor(log_rs, dtype=torch.float32).view(n_prop, 1)
    idx = torch.arange(n_prop, dtype=torch.float32).view(n_prop, 1, 1)
    gen = torch.Generator()
    gen.manual_seed(int(seed) * 17 + 1)
    picked, _ = independent_mh_reduce(idx, log_stack, generator=gen)
    chosen_i = int(picked.reshape(-1)[0].item())
    return images[chosen_i], images, log_rs, chosen_i, n_prop


def make_grid(output_dir: str, prompts: List[str]) -> str:
    from PIL import Image

    plt = _plt()
    rows = [(key, ROW_LABELS[key]) for key in ROW_LABELS if os.path.isdir(os.path.join(output_dir, key))]
    show = prompts[:GRID_PROMPTS]
    if not rows or not show:
        return ""
    n_r, n_c = len(rows), len(show)
    fig, axes = plt.subplots(n_r, n_c, figsize=(n_c * 3.2, n_r * 3.4), dpi=160)
    if n_r == 1:
        axes = [axes]
    for r, (key, label) in enumerate(rows):
        for c, prompt in enumerate(show):
            ax = axes[r][c] if n_c > 1 else axes[r]
            path = os.path.join(output_dir, key, slug(prompt), "best.png")
            if os.path.isfile(path):
                ax.imshow(Image.open(path))
            ax.axis("off")
            if r == 0:
                short = prompt.replace("a photo of ", "")
                ax.set_title(short, fontsize=7)
            if c == 0:
                ax.set_ylabel(label, fontsize=8)
    fig.tight_layout()
    grid_path = os.path.join(output_dir, "table5_grid.png")
    fig.savefig(grid_path, bbox_inches="tight")
    plt.close(fig)
    return grid_path


def _ks_statistic(left: List[float], right: List[float]):
    """Two-sample Kolmogorov-Smirnov statistic on the real line.

    This is the distance between the empirical CDFs of a scalar pushforward,
    the same functional as the critic-Q KS in the paper once both samples are
    in hand. It does not change which image was generated or selected.
    """
    a = sorted(float(x) for x in left)
    b = sorted(float(x) for x in right)
    if len(a) < 2 or len(b) < 2:
        return None
    grid = sorted(set(a) | set(b))

    def cdf(sample: List[float], threshold: float) -> float:
        lo, hi = 0, len(sample)
        while lo < hi:
            mid = (lo + hi) // 2
            if sample[mid] <= threshold:
                lo = mid + 1
            else:
                hi = mid
        return lo / len(sample)

    return max(abs(cdf(a, t) - cdf(b, t)) for t in grid)


def _best_by_prompt(recs: List[dict], metric: str) -> dict:
    out = {}
    for rec in recs:
        idx = rec.get("geneval_two_object_index")
        best = rec.get("best") or {}
        if idx is None or metric not in best:
            continue
        out[int(idx)] = float(best[metric])
    return out


def _all_scores(recs: List[dict], metric: str) -> List[float]:
    vals = []
    for rec in recs:
        proposals = (rec.get("proposal_metrics") or {}).get(metric)
        if proposals:
            vals.extend(float(x) for x in proposals)
            continue
        for value in (rec.get("metrics") or {}).get(metric) or []:
            vals.append(float(value))
    return vals


def pushforward_ks(records: dict, metrics: List[str]) -> dict:
    """KS of each metric's pushforward, from scores already computed.

    ``best`` compares the one reported image per prompt, on prompts shared by
    the pair. ``particles`` compares every scored image. ``selected_vs_particles``
    compares a method's reported images with its own full particle cloud, and
    is omitted when the method saved only one image per prompt.
    """
    methods = list(records)
    result = {"best": {}, "particles": {}, "selected_vs_particles": {}}
    for metric in metrics:
        best_maps = {name: _best_by_prompt(recs, metric) for name, recs in records.items()}
        clouds = {name: _all_scores(recs, metric) for name, recs in records.items()}
        best_pairs = {}
        particle_pairs = {}
        for i, left in enumerate(methods):
            for right in methods[i + 1 :]:
                shared = sorted(set(best_maps[left]) & set(best_maps[right]))
                stat = _ks_statistic(
                    [best_maps[left][idx] for idx in shared],
                    [best_maps[right][idx] for idx in shared],
                )
                if stat is not None:
                    best_pairs[f"{left} vs {right}"] = {"ks": stat, "n": len(shared)}
                stat = _ks_statistic(clouds[left], clouds[right])
                if stat is not None:
                    particle_pairs[f"{left} vs {right}"] = {
                        "ks": stat,
                        "n_left": len(clouds[left]),
                        "n_right": len(clouds[right]),
                    }
        selected = {}
        for name in methods:
            chosen = list(best_maps[name].values())
            cloud = clouds[name]
            if len(cloud) <= len(chosen):
                continue
            stat = _ks_statistic(chosen, cloud)
            if stat is None:
                continue
            selected[name] = {
                "ks": stat,
                "n_selected": len(chosen),
                "n_particles": len(cloud),
            }
        result["best"][metric] = best_pairs
        result["particles"][metric] = particle_pairs
        result["selected_vs_particles"][metric] = selected
    return result


def write_summary(output_dir: str, records: dict, metrics: List[str]) -> None:
    summary = {}
    for key, recs in records.items():
        block = {"n": len(recs)}
        for metric in metrics:
            vals = []
            for rec in recs:
                best = rec.get("best") or {}
                if metric in best:
                    vals.append(float(best[metric]))
                elif metric == "ImageReward":
                    scores = rec.get("imagereward_sorted") or []
                    if scores:
                        vals.append(float(scores[0]))
            if vals:
                mean = sum(vals) / len(vals)
                var = sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1)
                block[metric] = {"mean": mean, "std": var ** 0.5}
        summary[key] = block
    summary["pushforward_ks"] = pushforward_ks(records, metrics)
    path = os.path.join(output_dir, "summary.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print("[table5] summary", json.dumps(summary, indent=2))


def prefetch(args) -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    from huggingface_hub import snapshot_download

    print("[prefetch] SDv1.5", args.sd15_id)
    snapshot_download(args.sd15_id)
    print("[prefetch] ImageReward")
    from fkd_diffusers.image_reward_utils import rm_load

    rm_load("ImageReward-v1.0", device="cpu")
    print("[prefetch] CLIP ViT-L/14")
    import clip

    clip.load("ViT-L/14", device="cpu", download_root=".")
    print("[prefetch] HPSv2.1")
    import hpsv2
    from PIL import Image
    from fkd_diffusers.rewards import ensure_hpsv2_bpe

    ensure_hpsv2_bpe()
    dummy = Image.new("RGB", (224, 224), color=(128, 128, 128))
    hpsv2.score(dummy, "warmup", hps_version="v2.1")
    print("[prefetch] done")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="samples_table5_twoquery")
    p.add_argument("--prompt-file", default=DEFAULT_PROMPT_FILE)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--time-steps", type=int, default=100)
    p.add_argument("--num-particles", type=int, default=4, help="FK / SVDD / URGE particle count")
    p.add_argument("--twoq-k", type=int, default=2)
    p.add_argument("--twoq-j", type=int, default=TWOQ_J)
    p.add_argument("--svdd-m", type=int, default=4)
    p.add_argument("--svdd-j", type=int, default=TWOQ_J)
    p.add_argument(
        "--svdd-mode",
        choices=("mh", "smc"),
        default="mh",
        help="mh: one chain, outer MH like TwoQuery; smc: N particles, best of N",
    )
    p.add_argument("--twoq-tau", type=float, default=None, help="default 1/lmbda = 0.5")
    p.add_argument("--sd15-id", default=os.environ.get("SD15_MODEL_ID", "runwayml/stable-diffusion-v1-5"))
    p.add_argument("--sdxl-id", default=os.environ.get("SDXL_MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--prefetch-only", action="store_true")
    p.add_argument("--skip-sdxl", action="store_true")
    p.add_argument("--skip-fk", action="store_true")
    p.add_argument("--skip-svdd", action="store_true")
    p.add_argument("--skip-urge", action="store_true")
    p.add_argument("--skip-twoquery", action="store_true")
    p.add_argument("--skip-base", action="store_true")
    p.add_argument(
        "--sd15-base-images",
        type=int,
        default=0,
        help="unguided SD1.5 images per prompt, kept for the KS reference; 0 skips",
    )
    p.add_argument("--max-prompts", type=int, default=40)
    p.add_argument(
        "--prompt-offset",
        type=int,
        default=0,
        help="skip this many non-comment lines in --prompt-file (0-based)",
    )
    p.add_argument(
        "--metrics",
        default=",".join(DEFAULT_EVAL_METRICS),
        help="comma-separated eval metrics; guidance is always ImageReward",
    )
    p.add_argument(
        "--prompt-index-base",
        type=int,
        default=0,
        help="added to 1-based prompt_idx when recording GenEval two-object index",
    )
    p.add_argument("--resample-frequency", type=int, default=None)
    p.add_argument("--resample-t-start", type=int, default=None)
    p.add_argument("--resample-t-end", type=int, default=None)
    p.add_argument(
        "--adaptive-resampling",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="default keeps the paper setting; pass --no-adaptive-resampling for a fixed SMC count",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.prefetch_only:
        prefetch(args)
        return

    metrics = parse_metrics(args.metrics)
    prompts = load_prompts(args.prompt_file, args.max_prompts, offset=args.prompt_offset)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    cfg = vars(args).copy()
    cfg["n_prompts"] = len(prompts)
    cfg["eval_metrics"] = metrics
    cfg["twoquery_num_particles"] = 1
    cfg["twoquery_mh_j"] = int(args.twoq_j)
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(output_dir, "prompts.json"), "w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=2)

    seed_everything(args.seed)
    import torch
    from PIL import Image
    from fks_utils import do_eval

    device = args.device if torch.cuda.is_available() else "cpu"
    print("[table5] warmup eval", metrics)
    do_eval(
        prompt=["warmup"],
        images=[Image.new("RGB", (224, 224))],
        metrics_to_compute=metrics,
    )

    paper = deepcopy(PAPER_FKD)
    paper["time_steps"] = int(args.time_steps)
    paper["num_particles"] = int(args.num_particles)
    if args.resample_frequency is not None:
        paper["resample_frequency"] = int(args.resample_frequency)
    if args.resample_t_start is not None:
        paper["resampling_t_start"] = int(args.resample_t_start)
    if args.resample_t_end is not None:
        paper["resampling_t_end"] = int(args.resample_t_end)
    if args.adaptive_resampling is not None:
        paper["adaptive_resampling"] = bool(args.adaptive_resampling)
    resample_idx = urge_resample_indices(
        paper["resampling_t_start"],
        paper["resampling_t_end"],
        paper["resample_frequency"],
    )
    cfg["resample_frequency"] = paper["resample_frequency"]
    cfg["resampling_t_start"] = paper["resampling_t_start"]
    cfg["resampling_t_end"] = paper["resampling_t_end"]
    cfg["adaptive_resampling"] = paper["adaptive_resampling"]
    cfg["urge_resample_indices"] = resample_idx
    cfg["urge_n_resamples"] = len(resample_idx)
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(
        "[table5] URGE resample"
        f" freq={paper['resample_frequency']}"
        f" window={paper['resampling_t_start']}..{paper['resampling_t_end']}"
        f" adaptive={paper['adaptive_resampling']}"
        f" n={len(resample_idx)} steps={resample_idx}"
    )
    n = int(args.num_particles)
    eta = float(args.eta)
    steps = int(args.time_steps)
    records = {}

    def dump_method(key: str, prompt: str, prompt_idx: int, images, extra=None):
        images, ir_scores = sort_by_imagereward(prompt, images)
        extra_metrics = [m for m in metrics if m != "ImageReward"]
        scored = {"ImageReward": ir_scores}
        if extra_metrics:
            scored.update(eval_metrics(prompt, images, extra_metrics))
        d = os.path.join(output_dir, key, slug(prompt))
        os.makedirs(d, exist_ok=True)
        images[0].save(os.path.join(d, "best.png"))
        rec = {
            "prompt": prompt,
            "prompt_idx": prompt_idx,
            "geneval_two_object_index": int(args.prompt_index_base) + prompt_idx + 1,
            "method": key,
            "imagereward_sorted": ir_scores,
            "metrics": scored,
            "best": {name: float(vals[0]) for name, vals in scored.items() if vals},
        }
        if extra:
            rec.update(extra)
        records.setdefault(key, []).append(rec)
        with open(os.path.join(d, "scores.json"), "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)

    print(f"[table5] {len(prompts)} prompts from {args.prompt_file}")
    print("[table5] loading SDv1.5", args.sd15_id)
    pipe = load_sd15(args.sd15_id).to(device)
    for prompt_idx, prompt in enumerate(prompts):
        # Offset is already applied in load_prompts, so local 0 is not prompt 0.
        seed = int(args.prompt_offset) + prompt_idx
        if not args.skip_fk:
            seed_everything(seed)
            fk_args = deepcopy(paper)
            fk_args["steering_type"] = "FK"
            fk_args["use_smc"] = True
            fk_args["num_particles"] = n
            print(f"[table5] FK SDv1.5 N={n} :: {prompt}")
            images = run_pipeline(pipe, prompt, fk_args, eta, steps, n)
            dump_method("fk_sd15", prompt, prompt_idx, images)
        if not args.skip_svdd:
            if args.svdd_mode == "mh":
                print(
                    f"[table5] SVDD-MH SDv1.5 particles=1 m={args.svdd_m} J={args.svdd_j} :: {prompt}"
                )
                chosen, proposals, log_rs, chosen_i, n_prop = run_svdd_mh(
                    pipe,
                    prompt,
                    paper,
                    eta,
                    steps,
                    int(args.svdd_m),
                    int(args.svdd_j),
                    args.twoq_tau,
                    seed,
                )
                proposal_metrics = eval_metrics(prompt, proposals, metrics)
                prop_dir = os.path.join(output_dir, "svdd_sd15", slug(prompt), "mh_proposals")
                os.makedirs(prop_dir, exist_ok=True)
                for j, image in enumerate(proposals):
                    image.save(os.path.join(prop_dir, f"proposal_{j:02d}.png"))
                dump_method(
                    "svdd_sd15",
                    prompt,
                    prompt_idx,
                    [chosen],
                    extra={
                        "svdd_m": int(args.svdd_m),
                        "svdd_j": int(args.svdd_j),
                        "svdd_mode": "mh",
                        "n_proposals": n_prop,
                        "mh_index": chosen_i,
                        "mh_log_r": log_rs,
                        "num_particles": 1,
                        "proposal_metrics": proposal_metrics,
                    },
                )
            else:
                seed_everything(seed)
                svdd_args = deepcopy(paper)
                svdd_args["steering_type"] = "SVDD"
                svdd_args["use_smc"] = True
                svdd_args["num_particles"] = n
                svdd_args["svdd_m"] = int(args.svdd_m)
                svdd_args["svdd_mh"] = False
                print(f"[table5] SVDD SMC SDv1.5 N={n} M={args.svdd_m} :: {prompt}")
                images = run_pipeline(pipe, prompt, svdd_args, eta, steps, n)
                dump_method(
                    "svdd_sd15",
                    prompt,
                    prompt_idx,
                    images,
                    extra={"svdd_m": int(args.svdd_m), "svdd_mode": "smc", "num_particles": n},
                )
        if not args.skip_urge:
            seed_everything(seed)
            urge_args = deepcopy(paper)
            urge_args["steering_type"] = "URGE"
            urge_args["use_smc"] = True
            urge_args["num_particles"] = n
            print(
                f"[table5] URGE SDv1.5 N={n}"
                f" resamples={len(resample_idx)} :: {prompt}"
            )
            images = run_pipeline(pipe, prompt, urge_args, eta, steps, n)
            dump_method(
                "urge_sd15",
                prompt,
                prompt_idx,
                images,
                extra={
                    "n_resamples": len(resample_idx),
                    "resample_indices": resample_idx,
                    "resample_frequency": paper["resample_frequency"],
                    "adaptive_resampling": paper["adaptive_resampling"],
                    "urge_drift": "tweedie_reward",
                },
            )
        if not args.skip_twoquery:
            twoq_k = int(args.twoq_k)
            key = f"twoquery_k{twoq_k}_sd15"
            print(
                f"[table5] TwoQuery-MH SDv1.5 particles=1 k={twoq_k} J={args.twoq_j} :: {prompt}"
            )
            chosen, proposals, log_rs, chosen_i, n_prop = run_twoquery_mh(
                pipe,
                prompt,
                paper,
                eta,
                steps,
                twoq_k,
                int(args.twoq_j),
                args.twoq_tau,
                seed,
            )
            proposal_metrics = eval_metrics(prompt, proposals, metrics)
            prop_dir = os.path.join(output_dir, key, slug(prompt), "mh_proposals")
            os.makedirs(prop_dir, exist_ok=True)
            for j, image in enumerate(proposals):
                image.save(os.path.join(prop_dir, f"proposal_{j:02d}.png"))
            dump_method(
                key,
                prompt,
                prompt_idx,
                [chosen],
                extra={
                    "twoq_k": twoq_k,
                    "twoq_j": int(args.twoq_j),
                    "n_proposals": n_prop,
                    "mh_index": chosen_i,
                    "mh_log_r": log_rs,
                    "num_particles": 1,
                    "proposal_metrics": proposal_metrics,
                },
            )
        if not args.skip_base:
            seed_everything(seed)
            g_args = deepcopy(paper)
            g_args["steering_type"] = "G"
            g_args["use_smc"] = False
            g_args["num_particles"] = 1
            print(f"[table5] pure guidance SDv1.5 :: {prompt}")
            images = run_pipeline(pipe, prompt, g_args, eta, steps, 1)
            dump_method(
                "guidance_sd15",
                prompt,
                prompt_idx,
                images,
                extra={"steering_type": "G", "num_particles": 1},
            )
        n_base = int(args.sd15_base_images)
        if n_base > 0:
            base_args = deepcopy(paper)
            base_args["steering_type"] = "Base"
            base_args["use_smc"] = False
            base_args["num_particles"] = 1
            images = []
            print(f"[table5] unguided SD1.5 base x{n_base} :: {prompt}")
            for chunk in range(0, n_base, 8):
                seed_everything(seed * 10007 + chunk + 1)
                batch_n = min(8, n_base - chunk)
                images.extend(run_pipeline(pipe, prompt, base_args, eta, steps, batch_n))
            ordered, _ = sort_by_imagereward(prompt, images)
            sample_dir = os.path.join(output_dir, "base_sd15", slug(prompt))
            os.makedirs(sample_dir, exist_ok=True)
            for i, image in enumerate(ordered):
                image.save(os.path.join(sample_dir, f"sample_{i:02d}.png"))
            dump_method(
                "base_sd15",
                prompt,
                prompt_idx,
                ordered,
                extra={"model": "SD1.5", "steering_type": "Base", "num_images": len(ordered)},
            )
    del pipe
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    if not args.skip_sdxl:
        print("[table5] loading SDXL", args.sdxl_id)
        pipe_xl = load_sdxl(args.sdxl_id).to(device)
        for prompt_idx, prompt in enumerate(prompts):
            seed = int(args.prompt_offset) + prompt_idx
            seed_everything(seed)
            xl_args = deepcopy(paper)
            xl_args["steering_type"] = "Base"
            xl_args["use_smc"] = False
            xl_args["num_particles"] = 1
            print(f"[table5] SDXL base :: {prompt}")
            images = run_pipeline(pipe_xl, prompt, xl_args, eta, steps, 1)
            dump_method(
                "base_sdxl",
                prompt,
                prompt_idx,
                images,
                extra={"model": "SDXL", "steering_type": "Base", "num_particles": 1},
            )
        del pipe_xl
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    with open(os.path.join(output_dir, "scores.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    write_summary(output_dir, records, metrics)
    grid = make_grid(output_dir, prompts)
    if grid:
        print("[table5] wrote", grid)


if __name__ == "__main__":
    main()
