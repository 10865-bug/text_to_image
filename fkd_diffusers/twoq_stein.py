"""TwoQuery Stein step on the official DDIM kernel (FK / URGE T2I stack).

Does not change FK or URGE. Operates on a single chain (no particle SMC).
At each reverse step, draw a k-point centered Stein pool, score each
candidate with ImageReward on a Tweedie x0_hat that reuses the current-step
epsilon (no extra UNet NFE), and select J ∝ H(Y_i). Outer independent MH
is applied by generate_table5_twoquery.py across J+1 independent chains.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Union

import torch

from fkd_diffusers.twoq_core import (
    centered_stein_noises,
    posterior_select_and_log_increment,
    select_pool_child,
    stein_candidates,
)


def _as_int_timestep(t) -> int:
    if torch.is_tensor(t):
        return int(t.reshape(-1)[0].detach().cpu().item())
    return int(t)


def _expand_like(v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    while v.ndim < x.ndim:
        v = v.reshape(*v.shape, *([1] * (x.ndim - v.ndim)))
    return v


def _randn_like_pool(
    k: int,
    batch_shape,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]],
    device,
    dtype,
) -> torch.Tensor:
    gen = None if isinstance(generator, list) else generator
    try:
        return centered_stein_noises(
            k, tuple(batch_shape), generator=gen, device=device, dtype=dtype
        )
    except RuntimeError:
        z = centered_stein_noises(
            k, tuple(batch_shape), generator=gen, device="cpu", dtype=torch.float32
        )
        return z.to(device=device, dtype=dtype)


def _rand_uniform(n: int, generator, device, dtype) -> torch.Tensor:
    gen = None if isinstance(generator, list) else generator
    try:
        return torch.rand(n, generator=gen, device=device, dtype=dtype)
    except RuntimeError:
        return torch.rand(n, generator=gen, device="cpu", dtype=torch.float32).to(
            device=device, dtype=dtype
        )


def ddim_std_dev(scheduler, timestep, eta: float) -> torch.Tensor:
    """DDIM stochastic scale std_dev_t = eta * sqrt(variance)."""
    t_int = _as_int_timestep(timestep)
    n_train = int(scheduler.config.num_train_timesteps)
    n_inf = int(scheduler.num_inference_steps)
    prev_t = t_int - n_train // n_inf
    if hasattr(scheduler, "_get_variance"):
        variance = scheduler._get_variance(t_int, prev_t)
    else:
        alphas = scheduler.alphas_cumprod
        a_t = alphas[t_int]
        if prev_t >= 0:
            a_prev = alphas[prev_t]
        else:
            a_prev = scheduler.final_alpha_cumprod
        variance = (1 - a_prev) / torch.clamp(1 - a_t, min=1e-12) * (1 - a_t / torch.clamp(a_prev, min=1e-12))
    if not torch.is_tensor(variance):
        variance = torch.tensor(float(variance))
    variance = torch.clamp(variance.to(dtype=torch.float32), min=0.0)
    return (float(eta) * torch.sqrt(variance)).to(dtype=torch.float32)


def ddim_mean_and_x0(
    scheduler,
    noise_pred: torch.Tensor,
    t,
    latents: torch.Tensor,
    eta: float,
):
    """Official DDIM mean (noise = 0) and predicted x0. Advances scheduler state."""
    zeros = torch.zeros_like(latents)
    try:
        out = scheduler.step(
            model_output=noise_pred,
            timestep=t,
            sample=latents,
            eta=float(eta),
            generator=None,
            variance_noise=zeros,
            return_dict=True,
        )
    except TypeError:
        out = scheduler.step(
            model_output=noise_pred,
            timestep=t,
            sample=latents,
            eta=0.0,
            return_dict=True,
        )
    mu = out.prev_sample
    x0 = getattr(out, "pred_original_sample", None)
    if x0 is None:
        x0 = torch.zeros_like(latents)
    return mu, x0


def recover_epsilon(scheduler, latents: torch.Tensor, noise_pred: torch.Tensor, x0: torch.Tensor, t) -> torch.Tensor:
    pred_type = str(getattr(scheduler.config, "prediction_type", "epsilon"))
    if pred_type == "epsilon":
        return noise_pred
    t_int = _as_int_timestep(t)
    a_t = scheduler.alphas_cumprod[t_int].to(device=latents.device, dtype=latents.dtype)
    beta_t = (1.0 - a_t).clamp(min=1e-12)
    return (latents - _expand_like(a_t.sqrt(), latents) * x0) / _expand_like(beta_t.sqrt(), latents)


def x0_hat_from_next_latent(scheduler, y: torch.Tensor, eps: torch.Tensor, t) -> torch.Tensor:
    """Tweedie x0 at the previous noise level, reusing the current-step epsilon."""
    t_int = _as_int_timestep(t)
    n_train = int(scheduler.config.num_train_timesteps)
    n_inf = int(scheduler.num_inference_steps)
    prev_t = t_int - n_train // n_inf
    alphas = scheduler.alphas_cumprod
    if prev_t >= 0:
        a_prev = alphas[prev_t]
    else:
        a_prev = scheduler.final_alpha_cumprod
    a_prev = a_prev.to(device=y.device, dtype=y.dtype)
    alpha = torch.sqrt(a_prev.clamp(min=1e-12))
    sigma = torch.sqrt((1.0 - a_prev).clamp(min=0.0))
    alpha = _expand_like(alpha, y)
    sigma = _expand_like(sigma, y)
    return (y - sigma * eps) / alpha


class TwoQueryStein:
    """Single-chain Stein / 2Q transition (Algorithm 1).

    This is not SMC: there is one latent trajectory. Outer independent MH
    (Algorithm 2) is applied by the caller across J independent chains.
    """

    def __init__(
        self,
        *,
        k: int = 2,
        tau: float = 0.5,
        device=None,
    ) -> None:
        self.k = max(int(k), 2)
        self.tau = max(float(tau), 1e-8)
        self.device = device
        self.log_r: Optional[torch.Tensor] = None

    def accumulated_log_r(self, n: int, device=None, dtype=None) -> torch.Tensor:
        if self.log_r is None:
            return torch.zeros(n, device=device, dtype=dtype or torch.float32)
        return self.log_r.detach().reshape(-1).to(dtype=dtype or torch.float32)

    def step(
        self,
        *,
        latents: torch.Tensor,
        noise_pred: torch.Tensor,
        t,
        scheduler,
        eta: float,
        generator,
        prompts: Sequence[str],
        decode_x0_fn: Callable[[torch.Tensor], torch.Tensor],
        reward_fn: Callable[[torch.Tensor, Sequence[str]], torch.Tensor],
    ) -> torch.Tensor:
        mu, x0_parent = ddim_mean_and_x0(scheduler, noise_pred, t, latents, eta)
        std = ddim_std_dev(scheduler, t, eta).to(device=latents.device, dtype=latents.dtype)
        n = int(latents.shape[0])
        if float(std.max().detach().cpu()) < 1e-8:
            if self.log_r is None:
                self.log_r = torch.zeros(n, device=latents.device, dtype=torch.float32)
            return mu

        eps = recover_epsilon(scheduler, latents, noise_pred, x0_parent, t)
        z = _randn_like_pool(self.k, latents.shape, generator, latents.device, latents.dtype)
        y = stein_candidates(mu, std, z)
        x0 = x0_hat_from_next_latent(scheduler, y, eps.unsqueeze(0), t)
        if bool(getattr(scheduler.config, "clip_sample", False)):
            clip_range = float(getattr(scheduler.config, "clip_sample_range", 1.0))
            x0 = x0.clamp(-clip_range, clip_range)

        x0_flat = x0.reshape(self.k * n, *x0.shape[2:])
        images = decode_x0_fn(x0_flat)
        prompts_flat = list(prompts) * self.k
        rewards = reward_fn(images, prompts_flat)
        if not torch.is_tensor(rewards):
            rewards = torch.as_tensor(rewards, device=latents.device, dtype=torch.float32)
        else:
            rewards = rewards.to(device=latents.device, dtype=torch.float32)
        log_h = (rewards.view(self.k, n) / self.tau).to(dtype=torch.float32)

        u = _rand_uniform(n, generator, log_h.device, log_h.dtype)
        idx, _, log_inc = posterior_select_and_log_increment(log_h, u=u)
        chosen = select_pool_child(y, idx)
        if self.log_r is None:
            self.log_r = log_inc
        else:
            self.log_r = self.log_r + log_inc
        return chosen.to(dtype=latents.dtype)
