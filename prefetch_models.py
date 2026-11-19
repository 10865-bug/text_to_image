"""CPU-only download of SDv1.5 (+ optional SDXL), ImageReward, CLIP.

Does not fetch .ckpt / full-precision / non-EMA copies. Default is SDv1.5 only.
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

SD15_IGNORE = [
    "*.ckpt",
    "*.msgpack",
    "*non_ema*",
    "v1-5-pruned*",
    "*.bin",
    "text_encoder/model.safetensors",
    "unet/diffusion_pytorch_model.safetensors",
    "vae/diffusion_pytorch_model.safetensors",
    "safety_checker/*",
]

SDXL_ALLOW = [
    "model_index.json",
    "scheduler/*",
    "tokenizer/*",
    "tokenizer_2/*",
    "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors",
    "text_encoder_2/config.json",
    "text_encoder_2/model.fp16.safetensors",
    "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.fp16.safetensors",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sd15-id", default=os.environ.get("SD15_MODEL_ID", "runwayml/stable-diffusion-v1-5"))
    p.add_argument("--sdxl-id", default=os.environ.get("SDXL_MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0"))
    p.add_argument("--with-sdxl", action="store_true")
    p.add_argument("--skip-imagereward", action="store_true")
    p.add_argument("--skip-clip", action="store_true")
    p.add_argument("--skip-hps", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    from huggingface_hub import snapshot_download

    print("[prefetch] SDv1.5 fp16 only", args.sd15_id)
    snapshot_download(args.sd15_id, ignore_patterns=SD15_IGNORE)
    if args.with_sdxl:
        print("[prefetch] SDXL fp16 only", args.sdxl_id)
        snapshot_download(args.sdxl_id, allow_patterns=SDXL_ALLOW)
    else:
        print("[prefetch] skip SDXL")
    if not args.skip_imagereward:
        print("[prefetch] ImageReward")
        from fkd_diffusers.image_reward_utils import rm_load

        rm_load("ImageReward-v1.0", device="cpu")
    if not args.skip_clip:
        print("[prefetch] CLIP ViT-L/14")
        import clip

        clip.load("ViT-L/14", device="cpu", download_root=".")
    if not args.skip_hps:
        print("[prefetch] HPSv2.1")
        import hpsv2
        from PIL import Image
        from fkd_diffusers.rewards import ensure_hpsv2_bpe

        ensure_hpsv2_bpe()
        dummy = Image.new("RGB", (224, 224), color=(128, 128, 128))
        hpsv2.score(dummy, "warmup", hps_version="v2.1")
    print("[prefetch] done")


if __name__ == "__main__":
    main()
