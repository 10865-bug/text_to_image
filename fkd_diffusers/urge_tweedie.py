"""URGE proposal: DDIM mean plus V^2 ∇G, with the Girsanov weight of that same drift.

The previous run set G = log p_t, so ∇G was the Tweedie score -ε/sqrt(1-ᾱ).
That vector is order-1 in every latent coordinate. The Girsanov quadratic term
sums those coordinates, ½ σ² ||∇G||² ≈ 900 at the first step, and N=4 collapsed.
The weight itself matched the drift; it was not the old CLIP martingale.

G is now the same plug-in used to approximate Doob's h: ImageReward on the
Tweedie clean image, with the UNet epsilon held fixed,

    x̂_0 = (x - sqrt(1-ᾱ) ε) / sqrt(ᾱ),    G = r(x̂_0).

The extra drift is u = σ² ∇_x G, and

    log β = (r_new - r_old) - (u/σ)·ξ - ½ ||u/σ||²

uses that u and the same ξ that enters x = μ + u + σ ξ.
"""

from __future__ import annotations

from typing import Callable, List, Sequence

import torch
import torch.nn.functional as F

from fkd_diffusers.twoq_stein import ddim_mean_and_x0, ddim_std_dev, recover_epsilon


def prompt_list_for_particles(prompt, n: int) -> List[str]:
    if isinstance(prompt, (list, tuple)):
        if len(prompt) == n:
            return list(prompt)
        if len(prompt) == 0:
            return [""] * n
        repeats = (n + len(prompt) - 1) // len(prompt)
        return (list(prompt) * repeats)[:n]
    return [str(prompt)] * n


def _timestep_int(timestep) -> int:
    if torch.is_tensor(timestep):
        return int(timestep.reshape(-1)[0].detach().cpu().item())
    return int(timestep)


def alpha_bar(scheduler, timestep) -> float:
    t_int = _timestep_int(timestep)
    a = scheduler.alphas_cumprod[t_int]
    if torch.is_tensor(a):
        return float(a.detach().float().reshape(-1)[0].item())
    return float(a)


def gaussian_girsanov_log(u: torch.Tensor, xi: torch.Tensor, sigma: float) -> torch.Tensor:
    """log N(μ, σ²I) - log N(μ+u, σ²I) at μ+u+σξ."""
    if sigma <= 0.0:
        return torch.zeros(u.shape[0], device=u.device, dtype=torch.float32)
    u_flat = u.reshape(u.shape[0], -1).float()
    xi_flat = xi.reshape(xi.shape[0], -1).float()
    sig = float(sigma)
    coef = (u_flat * xi_flat).sum(dim=1) / sig
    quad = 0.5 * u_flat.pow(2).sum(dim=1) / (sig * sig)
    return (-coef - quad).to(dtype=torch.float32)


def _image_reward_model(device):
    from fkd_diffusers.rewards import REWARDS_DICT

    if REWARDS_DICT.get("ImageReward", None) is None:
        from fkd_diffusers.image_reward_utils import rm_load

        REWARDS_DICT["ImageReward"] = rm_load("ImageReward-v1.0", device=device)
    model = REWARDS_DICT["ImageReward"]
    target = torch.device(device)
    if next(model.parameters()).device != target:
        model.to(target)
    # score() sends images to this attribute, not to the parameter device.
    model.device = target
    model.eval()
    return model


def _ensure_vae_fp32(vae):
    """SD1.5 marks the decoder force_upcast because fp16 overflows on x̂_0.

    This backward was decoding in fp16, so a finite reward gradient turned into
    NaN a few steps in and the guided latent was non-finite.
    """
    if next(vae.parameters()).dtype != torch.float32:
        vae.to(dtype=torch.float32)
    return vae


def _decode_unit_interval(vae, x0: torch.Tensor) -> torch.Tensor:
    vae = _ensure_vae_fp32(vae)
    sf = float(vae.config.scaling_factor)
    decoded = vae.decode((x0.float() / sf)).sample
    return (decoded.float() / 2.0 + 0.5).clamp(0.0, 1.0)


def _reward_on_pixels(model, pixels: torch.Tensor, prompts: Sequence[str]) -> torch.Tensor:
    text = model.blip.tokenizer(
        list(prompts),
        padding="max_length",
        truncation=True,
        max_length=35,
        return_tensors="pt",
    )
    device = pixels.device
    image = F.interpolate(pixels, size=(224, 224), mode="bicubic", align_corners=False)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device, dtype=image.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device, dtype=image.dtype).view(1, 3, 1, 1)
    image = (image - mean) / std
    enc_dtype = next(model.blip.visual_encoder.parameters()).dtype
    rewards = model.score_gard(
        text.input_ids.to(device),
        text.attention_mask.to(device),
        image.to(dtype=enc_dtype),
    )
    return rewards.float().reshape(-1)


def tweedie_reward_and_grad(pipe, latents: torch.Tensor, eps: torch.Tensor, alpha: float, prompts: Sequence[str]):
    """r(x̂_0) and ∇_x r, one particle at a time. ε is constant in the Tweedie map."""
    model = _image_reward_model(latents.device)
    sqrt_a = max(float(alpha), 1e-12) ** 0.5
    sqrt_oma = max(1.0 - float(alpha), 0.0) ** 0.5
    values = []
    grads = []
    for i in range(int(latents.shape[0])):
        with torch.enable_grad():
            x = latents[i : i + 1].detach().float().requires_grad_(True)
            e = eps[i : i + 1].detach().float()
            x0 = (x - sqrt_oma * e) / sqrt_a
            pixels = _decode_unit_interval(pipe.vae, x0)
            reward = _reward_on_pixels(model, pixels, [prompts[i]])
            grad = torch.autograd.grad(reward.sum(), x, retain_graph=False, create_graph=False)[0]
        if grad is None:
            raise RuntimeError("[URGE] ImageReward gradient did not reach the latent")
        if (not torch.isfinite(grad).all()) or (not torch.isfinite(reward).all()):
            print(
                "[URGE] non-finite ImageReward grad after fp32 decode; "
                "this particle keeps the unguided DDIM mean"
            )
            grad = torch.zeros_like(grad)
            reward = torch.nan_to_num(reward.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        values.append(reward.detach().reshape(-1))
        grads.append(grad.detach())
    return torch.cat(values, dim=0), torch.cat(grads, dim=0)


def apply_urge_tweedie_step(
    *,
    pipe,
    latents: torch.Tensor,
    noise_pred: torch.Tensor,
    t,
    eta: float,
    prompt,
    B: int,
    K: int,
    sampling_idx: int,
    reward_fn: Callable[[torch.Tensor, List[str]], torch.Tensor],
    generator=None,
) -> torch.Tensor:
    """Resample the arrival state, then step with u = σ² ∇r(x̂_0)."""
    del reward_fn, generator
    scheduler = pipe.scheduler
    mu, x0_parent = ddim_mean_and_x0(scheduler, noise_pred, t, latents, float(eta))
    std = ddim_std_dev(scheduler, t, float(eta))
    sigma = float(std.detach().float().reshape(-1)[0].item())
    alpha = alpha_bar(scheduler, t)
    eps = recover_epsilon(scheduler, latents, noise_pred, x0_parent, t)
    prompts = prompt_list_for_particles(prompt, int(latents.shape[0]))
    rewards, grad_g = tweedie_reward_and_grad(pipe, latents, eps, alpha, prompts)
    rewards = rewards.to(device=latents.device, dtype=torch.float32)
    grad_g = grad_g.to(device=latents.device, dtype=torch.float32)

    urge = pipe._fkd
    pending = getattr(urge, "_pending_girsanov", None)
    next_parts = []
    gir_parts = []
    u_over_sigma = []

    for b in range(int(B)):
        sl = slice(b * K, (b + 1) * K)
        mu_b = mu[sl]
        g_b = grad_g[sl]
        gir_in = None if pending is None else pending[sl]
        _, indices = urge.resample(
            sampling_idx=int(sampling_idx),
            latents=latents[sl],
            rewards=rewards[sl],
            girsanov_log=gir_in,
            guidance_grad=None,
        )
        mu_b = mu_b.index_select(0, indices).float()
        g_b = g_b.index_select(0, indices).float()
        u = (sigma * sigma) * g_b
        if sigma > 1e-8:
            xi = torch.randn(mu_b.shape, device=mu_b.device, dtype=torch.float32)
            nxt = mu_b + u + sigma * xi
            gir = gaussian_girsanov_log(u, xi, sigma)
            u_over_sigma.append((u.reshape(K, -1).norm(dim=1) / sigma).detach())
        else:
            nxt = mu_b
            gir = torch.zeros(K, device=mu_b.device, dtype=torch.float32)
            u_over_sigma.append(torch.zeros(K, device=mu_b.device))
        if (not torch.isfinite(nxt).all()) or (not torch.isfinite(gir).all()):
            print(
                f"[URGE] non-finite guided step at solver index {sampling_idx}; "
                "dropping the drift and keeping the DDIM mean"
            )
            nxt = torch.nan_to_num(mu_b, nan=0.0, posinf=0.0, neginf=0.0)
            gir = torch.zeros(K, device=mu_b.device, dtype=torch.float32)
        stored = nxt.to(dtype=latents.dtype)
        if not torch.isfinite(stored).all():
            limit = float(torch.finfo(torch.float16).max)
            stored = nxt.clamp(-limit, limit).to(dtype=latents.dtype)
        next_parts.append(stored)
        gir_parts.append(gir.to(device=latents.device, dtype=torch.float32))

    urge._pending_girsanov = torch.cat(gir_parts, dim=0)
    if int(sampling_idx) in (0, 1, 50, 99):
        ratio = torch.cat(u_over_sigma, dim=0)
        quad = 0.5 * ratio.pow(2)
        print(
            f"[URGE] step={sampling_idx} sigma={sigma:.4e} alpha={alpha:.4f} "
            f"mean||u||/sigma={float(ratio.mean()):.4f} "
            f"mean_quad={float(quad.mean()):.4f} "
            f"mean_girsanov={float(urge._pending_girsanov.mean()):.4f}"
        )
    return torch.cat(next_parts, dim=0)
