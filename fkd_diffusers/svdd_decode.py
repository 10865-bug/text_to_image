"""SVDD-PM on the official DDIM kernel (FK / URGE T2I stack).

Does not change FK or URGE. At each reverse step, for every particle draw M
i.i.d. kernel samples (not Stein-centered), score ImageReward on a Tweedie
x0_hat that reuses the current-step epsilon.

``mh=False`` keeps the original particle SMC step: softmax, then one
multinomial draw per particle, no path weight.

``mh=True`` uses TwoQuery's local selection and log increment
(log S - log H), and the caller runs the same outer independent MH.
The only remaining difference from TwoQuery is that the pool noise is i.i.d.
N(0, I) rather than centered Stein noise.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Union

import torch

from fkd_diffusers.twoq_core import posterior_select_and_log_increment, select_pool_child
from fkd_diffusers.twoq_stein import (
    ddim_mean_and_x0,
    ddim_std_dev,
    recover_epsilon,
    x0_hat_from_next_latent,
    _expand_like,
    _rand_uniform,
)


def _randn_iid(
    m: int,
    batch_shape,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]],
    device,
    dtype,
) -> torch.Tensor:
    gen = None if isinstance(generator, list) else generator
    try:
        return torch.randn((m, *batch_shape), generator=gen, device=device, dtype=dtype)
    except RuntimeError:
        z = torch.randn((m, *batch_shape), generator=gen, device="cpu", dtype=torch.float32)
        return z.to(device=device, dtype=dtype)


class SVDDSoftValue:
    def __init__(
        self,
        *,
        m: int = 4,
        tau: float = 0.5,
        mh: bool = False,
        device=None,
    ) -> None:
        self.m = max(int(m), 2)
        self.tau = max(float(tau), 1e-8)
        self.mh = bool(mh)
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
            if self.mh and self.log_r is None:
                self.log_r = torch.zeros(n, device=latents.device, dtype=torch.float32)
            return mu

        eps = recover_epsilon(scheduler, latents, noise_pred, x0_parent, t)
        z = _randn_iid(self.m, latents.shape, generator, latents.device, latents.dtype)
        y = mu.unsqueeze(0) + _expand_like(std, mu).unsqueeze(0) * z
        x0 = x0_hat_from_next_latent(scheduler, y, eps.unsqueeze(0), t)
        if bool(getattr(scheduler.config, "clip_sample", False)):
            clip_range = float(getattr(scheduler.config, "clip_sample_range", 1.0))
            x0 = x0.clamp(-clip_range, clip_range)

        x0_flat = x0.reshape(self.m * n, *x0.shape[2:])
        images = decode_x0_fn(x0_flat)
        prompts_flat = list(prompts) * self.m
        rewards = reward_fn(images, prompts_flat)
        if not torch.is_tensor(rewards):
            rewards = torch.as_tensor(rewards, device=latents.device, dtype=torch.float32)
        else:
            rewards = rewards.to(device=latents.device, dtype=torch.float32)
        if self.mh:
            log_h = (rewards.view(self.m, n) / self.tau).to(dtype=torch.float32)
            u = _rand_uniform(n, generator, log_h.device, log_h.dtype)
            idx, _, log_inc = posterior_select_and_log_increment(log_h, u=u)
            chosen = select_pool_child(y, idx)
            if self.log_r is None:
                self.log_r = log_inc
            else:
                self.log_r = self.log_r + log_inc
            return chosen.to(dtype=latents.dtype)

        w = torch.softmax(rewards.view(self.m, n) / self.tau, dim=0).transpose(0, 1)
        gen = None if isinstance(generator, list) else generator
        try:
            idx = torch.multinomial(w, num_samples=1, generator=gen).squeeze(-1)
        except RuntimeError:
            idx = torch.multinomial(w.cpu(), num_samples=1, generator=gen).squeeze(-1).to(w.device)
        batch = torch.arange(n, device=latents.device)
        return y[idx, batch].to(dtype=latents.dtype)
