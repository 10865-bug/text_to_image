"""
Feynman-Kac Diffusion (FKD) steering mechanism implementation.
"""

import math
import torch
from enum import Enum
import numpy as np
from typing import Callable, Optional, Tuple, Dict
import logging
import random
import matplotlib.pyplot as plt


class PotentialType(Enum):
    DIFF = "diff"
    MAX = "max"
    ADD = "add"
    RT = "rt"



class FKD:
    """
    FKD steering for diffusion pipelines.
    - Uses MAX/ADD/DIFF/RT potentials.
    - Interval resampling with terminal correction.
    - Matches current pipeline contracts: reward_fn(x_imgs_tensor)->1D[K] tensor; latent_to_decode_fn returns tensor.
    """

    def __init__(
        self,
        *,
        potential_type: PotentialType,
        lmbda: float,
        num_particles: int,
        adaptive_resampling: bool,
        resample_frequency: int,
        resampling_t_start: int,
        resampling_t_end: int,
        time_steps: int,
        reward_fn: Callable[[torch.Tensor], torch.Tensor],
        reward_min_value: float = 0.0,  
        latent_to_decode_fn: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
        device: torch.device = torch.device("cuda"),
        replacement: bool = True,
        **kwargs,
    ) -> None:
        self.potential_type = PotentialType(potential_type)
        self.lmbda = float(lmbda)
        self.num_particles = int(num_particles)
        self.adaptive_resampling = bool(adaptive_resampling)
        self.resample_frequency = int(resample_frequency)
        self.resampling_t_start = int(resampling_t_start)
        self.resampling_t_end = int(resampling_t_end)
        self.time_steps = int(time_steps)

        self.reward_fn = reward_fn
        self.latent_to_decode_fn = latent_to_decode_fn
        self.device = device
        self.replacement = bool(replacement)

        self.population_rs = torch.ones(self.num_particles, device=self.device, dtype=torch.float32) * float(reward_min_value)
        self.product_of_potentials = torch.ones(self.num_particles, device=self.device, dtype=torch.float32)

        steps = list(range(self.resampling_t_start, self.resampling_t_end + 1, max(1, self.resample_frequency)))
        terminal = self.time_steps - 1
        if terminal not in steps:
            steps.append(terminal)
        self._resample_steps = set(int(s) for s in steps)

    def resample(
        self,
        *,
        sampling_idx: int,
        latents: torch.Tensor,          # [K, C, H, W] latents
        x0_preds: torch.Tensor,         # pixel-space [K, 3, H, W] in [0,1]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if sampling_idx not in self._resample_steps:
            return latents, None

        population_images = self.latent_to_decode_fn(x0_preds) 
        if not torch.is_tensor(population_images):
            raise RuntimeError("[FKD] latent_to_decode_fn must return torch.Tensor")
        # ---- Debug: inspect image signal ----
        img_mean = population_images.mean().item()
        img_std = population_images.std().item()
        print(f"[FKD Debug] step={sampling_idx}: image mean={img_mean:.4f}, std={img_std:.4f}")

        rs_candidates = self.reward_fn(population_images)
        if not torch.is_tensor(rs_candidates):
            rs_candidates = torch.as_tensor(rs_candidates, device=self.device, dtype=torch.float32)
        else:
            rs_candidates = rs_candidates.to(self.device, dtype=torch.float32)
        # ---- Debug: inspect reward signal ----
        print(f"[FKD Debug] step={sampling_idx}: reward min={rs_candidates.min().item():.4f}, "
            f"max={rs_candidates.max().item():.4f}, mean={rs_candidates.mean().item():.4f}, "
            f"std={rs_candidates.std().item():.4f}")

        if rs_candidates.dim() != 1 or int(rs_candidates.shape[0]) != self.num_particles:
            raise RuntimeError(f"[FKD] rewards must be 1D [K], got {tuple(rs_candidates.shape)}")

        if self.potential_type == PotentialType.MAX:
            rs_eff = torch.maximum(rs_candidates, self.population_rs)
            w = torch.exp(self.lmbda * rs_eff)
        elif self.potential_type == PotentialType.ADD:
            rs_eff = rs_candidates + self.population_rs
            w = torch.exp(self.lmbda * rs_eff)
        elif self.potential_type == PotentialType.DIFF:
            diffs = rs_candidates - self.population_rs
            w = torch.exp(self.lmbda * diffs)
            rs_eff = rs_candidates  
        elif self.potential_type == PotentialType.RT:
            rs_eff = rs_candidates
            w = torch.exp(self.lmbda * rs_eff)
        else:
            raise ValueError(f"potential_type {self.potential_type} not recognized")
        
        if sampling_idx == self.time_steps - 1 and self.potential_type in (PotentialType.MAX, PotentialType.ADD, PotentialType.RT):
            w = torch.exp(self.lmbda * rs_candidates) / (self.product_of_potentials + 1e-12)

        w = torch.clamp(w, 0.0, 1e10)
        w = torch.nan_to_num(w, nan=0.0, posinf=1e10, neginf=0.0)

        if self.adaptive_resampling or sampling_idx == self.time_steps - 1:
            normalized_w = w / (w.sum() + 1e-12)
            ess = 1.0 / (normalized_w.pow(2).sum() + 1e-12)

            if ess < 0.5 * self.num_particles:
                indices = torch.multinomial(w, num_samples=self.num_particles, replacement=self.replacement)
                resampled_latents = latents.index_select(0, indices)
                self.population_rs = rs_candidates.index_select(0, indices) 
                resampled_images = population_images.index_select(0, indices)
                self.product_of_potentials = self.product_of_potentials.index_select(0, indices) * w.index_select(0, indices)
            else:
                resampled_images = population_images
                resampled_latents = latents
                self.population_rs = rs_candidates
        else:
            indices = torch.multinomial(w, num_samples=self.num_particles, replacement=self.replacement)
            resampled_latents = latents.index_select(0, indices)
            self.population_rs = rs_candidates.index_select(0, indices) 
            resampled_images = population_images.index_select(0, indices)
            self.product_of_potentials = self.product_of_potentials.index_select(0, indices) * w.index_select(0, indices)

        return resampled_latents, resampled_images


class FKD_G:
    """
    FKG: FK steering with gradient-guided proposal (FK + G minimal integration).
    - Matches FKD interface: resample(sampling_idx, latents, x0_preds) -> (resampled_latents, resampled_images)
    - Rewards are computed internally via reward_fn(x_imgs_tensor) -> 1D [K] tensor (same as FKD).
    - Uses MAX/ADD/DIFF/RT potentials and terminal correction (same semantics as FKD).
    - No external rewards, no importance ratio, no proposal providers.
    """

    def __init__(
        self,
        *,
        potential_type: PotentialType,
        lmbda: float,
        num_particles: int,
        adaptive_resampling: bool,
        resample_frequency: int,
        resampling_t_start: int,
        resampling_t_end: int,
        time_steps: int,
        reward_fn: Callable[[torch.Tensor], torch.Tensor],
        reward_min_value: float = 0.0,
        latent_to_decode_fn: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
        device: torch.device = torch.device("cuda"),
        replacement: bool = True,
        **kwargs,
    ) -> None:
        # Configuration
        self.potential_type = PotentialType(potential_type)
        self.lmbda = float(lmbda)
        self.num_particles = int(num_particles)
        self.adaptive_resampling = bool(adaptive_resampling)
        self.resample_frequency = int(resample_frequency)
        self.resampling_t_start = int(resampling_t_start)
        self.resampling_t_end = int(resampling_t_end)
        self.time_steps = int(time_steps)

        # External interfaces
        self.reward_fn = reward_fn
        self.latent_to_decode_fn = latent_to_decode_fn

        # Runtime
        self.device = device
        self.replacement = bool(replacement)

        # Population state
        self.population_rs = torch.ones(self.num_particles, device=self.device, dtype=torch.float32) * float(reward_min_value)
        self.product_of_potentials = torch.ones(self.num_particles, device=self.device, dtype=torch.float32)

        # Precompute resampling steps and include terminal step
        steps = list(range(self.resampling_t_start, self.resampling_t_end + 1, max(1, self.resample_frequency)))
        terminal = self.time_steps - 1
        if terminal not in steps:
            steps.append(terminal)
        self._resample_steps = set(int(s) for s in steps)

    def resample(
        self,
        *,
        sampling_idx: int,
        latents: torch.Tensor,          # [K, C, H, W] latents (after G displacement in pipeline)
        x0_preds: torch.Tensor,         # [K, 3, H, W] pixel-space in [0,1]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Skip if not a resample step
        if sampling_idx not in self._resample_steps:
            return latents, None

        # Decode/convert pixel predictions and compute rewards
        population_images = self.latent_to_decode_fn(x0_preds)
        if not torch.is_tensor(population_images):
            raise RuntimeError("[FKG] latent_to_decode_fn must return torch.Tensor")

        rs_candidates = self.reward_fn(population_images)
        if not torch.is_tensor(rs_candidates):
            rs_candidates = torch.as_tensor(rs_candidates, device=self.device, dtype=torch.float32)
        else:
            rs_candidates = rs_candidates.to(self.device, dtype=torch.float32)

        if rs_candidates.dim() != 1 or int(rs_candidates.shape[0]) != self.num_particles:
            raise RuntimeError(f"[FKG] rewards must be 1D [K], got {tuple(rs_candidates.shape)}")

        # Potentials (same semantics as FKD)
        if self.potential_type == PotentialType.MAX:
            rs_eff = torch.maximum(rs_candidates, self.population_rs)
            w = torch.exp(self.lmbda * rs_eff)
        elif self.potential_type == PotentialType.ADD:
            rs_eff = rs_candidates + self.population_rs
            w = torch.exp(self.lmbda * rs_eff)
        elif self.potential_type == PotentialType.DIFF:
            diffs = rs_candidates - self.population_rs
            w = torch.exp(self.lmbda * diffs)
            rs_eff = rs_candidates
        elif self.potential_type == PotentialType.RT:
            rs_eff = rs_candidates
            w = torch.exp(self.lmbda * rs_eff)
        else:
            raise ValueError(f"potential_type {self.potential_type} not recognized")

        # Terminal correction for MAX/ADD/RT
        if sampling_idx == self.time_steps - 1 and self.potential_type in (PotentialType.MAX, PotentialType.ADD, PotentialType.RT):
            w = torch.exp(self.lmbda * rs_candidates) / (self.product_of_potentials + 1e-12)

        # Sanitize weights
        w = torch.clamp(w, 0.0, 1e10)
        w = torch.nan_to_num(w, nan=0.0, posinf=1e10, neginf=0.0)

        # Adaptive resampling or terminal step
        if self.adaptive_resampling or sampling_idx == self.time_steps - 1:
            normalized_w = w / (w.sum() + 1e-12)
            ess = 1.0 / (normalized_w.pow(2).sum() + 1e-12)

            if ess < 0.5 * self.num_particles:
                indices = torch.multinomial(w, num_samples=self.num_particles, replacement=self.replacement)
                resampled_latents = latents.index_select(0, indices)
                self.population_rs = rs_candidates.index_select(0, indices)
                resampled_images = population_images.index_select(0, indices)
                self.product_of_potentials = self.product_of_potentials.index_select(0, indices) * w.index_select(0, indices)
            else:
                resampled_images = population_images
                resampled_latents = latents
                self.population_rs = rs_candidates
        else:
            indices = torch.multinomial(w, num_samples=self.num_particles, replacement=self.replacement)
            resampled_latents = latents.index_select(0, indices)
            self.population_rs = rs_candidates.index_select(0, indices)
            resampled_images = population_images.index_select(0, indices)
            self.product_of_potentials = self.product_of_potentials.index_select(0, indices) * w.index_select(0, indices)

        return resampled_latents, resampled_images


class URGE:
    """
    Unified URGE steering class implementing three-term reweighting:

      - Martingale term (orange):    - V(t) * <grad, dW_t>
      - Drift term (red):            - 0.5 * V(t)^2 * ||grad||^2 * dt
      - Reward increment (blue):     + (r( X_hat_{t+dt} ) - r( X_hat_t ))

    Design:
      - External rewards preferred (strictly 1D [K]); in typical use, supply ImageReward scores.
      - External gradient preferred (guidance_grad [K,3,H,W]); e.g., CLIP-based gradient on decoded images.
      - Internal autograd path is available only if decoded_images is provided and reward_fn is differentiable.

    Behavior:
      - Maintains log_weights and weights internally; uses ESS decision and multinomial resampling.
      - Returns (resampled_latents, indices).
    """

    def __init__(
        self,
        *,
        num_particles: int,
        adaptive_resampling: bool,
        resample_frequency: int,
        resampling_t_start: int,
        resampling_t_end: int,
        time_steps: int,
        reward_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        latent_to_decode_fn: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
        noise_sampler_fn: Optional[Callable[[Tuple[int, ...], int, torch.device, torch.dtype], torch.Tensor]] = None,
        sigma_provider_fn: Optional[Callable[[int], float]] = None,
        dt_provider_fn: Optional[Callable[[int], float]] = None,
        device: torch.device = torch.device("cuda"),
        ess_threshold_ratio: float = 0.5,
        use_last_step_resample: bool = False,
        eta: float = 1.0,
        replacement: bool = True,
        debug: bool = False,
        **kwargs,
    ):
        # Hyperparameters
        self.num_particles = int(num_particles)
        self.adaptive_resampling = bool(adaptive_resampling)
        self.resample_frequency = int(resample_frequency)
        self.resampling_t_start = int(resampling_t_start)
        self.resampling_t_end = int(resampling_t_end)
        self.time_steps = int(time_steps)

        # Functions/providers
        self.reward_fn = reward_fn
        self.latent_to_decode_fn = latent_to_decode_fn
        self.noise_sampler_fn = noise_sampler_fn
        self.sigma_provider_fn = sigma_provider_fn
        self.dt_provider_fn = dt_provider_fn

        # Runtime
        self.device = device
        self.ess_threshold_ratio = float(ess_threshold_ratio)
        self.use_last_step_resample = bool(use_last_step_resample)
        self.eta = float(max(0.0, min(1.0, eta)))
        self.replacement = bool(replacement)
        self.debug = bool(debug)

        # State
        self.population_rs: Optional[torch.Tensor] = None
        self.log_weights = torch.zeros(self.num_particles, device=self.device)
        self.weights = torch.ones(self.num_particles, device=self.device) / float(self.num_particles)
        self.last_indices: Optional[torch.Tensor] = None
        # Log Girsanov factor of the transition that produced the current latents.
        # Consumed on the next resample() call, together with r_arrival - r_departure.
        self._pending_girsanov: Optional[torch.Tensor] = None

        # Precompute resampling steps (explicit set). Optionally include terminal step.
        steps = list(range(self.resampling_t_start, self.resampling_t_end + 1, max(1, self.resample_frequency)))
        if self.use_last_step_resample and (self.time_steps - 1) not in steps:
            steps.append(self.time_steps - 1)
        self._resampling_steps = set(int(s) for s in steps)

    def _default_sigma(self, sampling_idx: int) -> float:
        return 1.0

    def _default_dt(self, sampling_idx: int) -> float:
        return 1.0 / max(1, self.time_steps)

    def _sample_noise(self, shape: Tuple[int, ...], sampling_idx: int, dtype: torch.dtype) -> torch.Tensor:
        if self.noise_sampler_fn is not None:
            return self.noise_sampler_fn(shape, sampling_idx, self.device, dtype)
        return torch.randn(shape, device=self.device, dtype=dtype)

    def _compute_reward_and_grad(self, decoded_images: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Internal autograd path: compute rewards and pixel gradients via reward_fn when decoded_images is provided.
        Returns (rewards[K], guidance_grad or None), both on GPU, float32, contiguous.
        Use only when reward_fn is differentiable; otherwise prefer external rewards and external guidance_grad.
        """
        decoded_images_fp32 = decoded_images.to(self.device, dtype=torch.float32).detach().requires_grad_(True)

        try:
            if self.reward_fn is None:
                raise RuntimeError("reward_fn is None; cannot compute autograd reward.")

            rewards = self.reward_fn(decoded_images_fp32)
            if not torch.is_tensor(rewards):
                rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)

            # Enforce 1D [K]
            if rewards.dim() == 2 and rewards.shape[1] == 1:
                rewards = rewards.squeeze(1)
            elif rewards.dim() != 1:
                raise RuntimeError(f"[URGE] reward_fn must return 1D [K]; got {tuple(rewards.shape)}")

            grad_outputs = torch.ones_like(rewards, device=self.device, dtype=rewards.dtype)
            grads = torch.autograd.grad(
                outputs=rewards,
                inputs=decoded_images_fp32,
                grad_outputs=grad_outputs,
                retain_graph=False,
                create_graph=False,
            )[0]

            return rewards.detach().to(self.device, dtype=torch.float32).contiguous(), grads.detach().to(self.device, dtype=torch.float32).contiguous()

        except Exception as e:

            try:
                if self.reward_fn is None:
                    raise RuntimeError("reward_fn is None")

                rewards = self.reward_fn(decoded_images.to(self.device, dtype=torch.float32))
                if not torch.is_tensor(rewards):
                    rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)

                if rewards.dim() == 2 and rewards.shape[1] == 1:
                    rewards = rewards.squeeze(1)
                elif rewards.dim() != 1:
                    raise RuntimeError(f"[URGE] reward_fn must return 1D [K]; got {tuple(rewards.shape)}")

            except Exception:
                rewards = torch.zeros(self.num_particles, device=self.device, dtype=torch.float32)

            return rewards.detach().to(self.device, dtype=torch.float32).contiguous(), None

    def resample(
        self,
        *,
        sampling_idx: int,
        latents: torch.Tensor,                         # [K,C,H,W]
        decoded_images: Optional[torch.Tensor] = None, # [K,3,H,W] (used only when internal gradient path is needed)
        rewards: Optional[torch.Tensor] = None,        # [K] (external source preferred, e.g., ImageReward at current step)
        rewards_next: Optional[torch.Tensor] = None,   # [K] (optional: next-step rewards to form blue increment)
        guidance_grad: Optional[torch.Tensor] = None,  # [K,3,H,W] (external gradient, e.g., CLIP on decoded_images)
        V_t: Optional[float] = None,
        dt: Optional[float] = None,
        noise_for_martingale: Optional[torch.Tensor] = None,
        girsanov_log: Optional[torch.Tensor] = None,    # [K] log(dP/dP_G) of the transition into these latents
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Unified contract:
          - Input: latents [K,C,H,W].
                   Prefer external rewards (strictly 1D [K]) and external guidance_grad [K,3,H,W].
                   If guidance_grad is None and decoded_images+reward_fn are provided and differentiable,
                   use internal autograd path to obtain (rewards, guidance_grad).
          - Output: (resampled_latents, indices).
        """
        # Shape checks
        if latents.dim() != 4:
            raise RuntimeError(f"[URGE] latents must be 4D [K,C,H,W], got {tuple(latents.shape)}")
        K = int(latents.shape[0])
        if K != self.num_particles:
            raise RuntimeError(f"[URGE] K={K} != num_particles={self.num_particles}")

        # Rewards source and gradient
        if rewards is not None:
            rs_candidates = rewards.to(self.device, dtype=torch.float32).contiguous()
        else:
            if decoded_images is None:
                raise RuntimeError("[URGE] Either rewards or decoded_images must be provided")
            # Internal path (use only if reward_fn is differentiable)
            with torch.enable_grad():
                rs_candidates, auto_grad = self._compute_reward_and_grad(decoded_images)
            if guidance_grad is None:
                guidance_grad = auto_grad

        if rs_candidates.dim() != 1 or int(rs_candidates.shape[0]) != self.num_particles:
            raise RuntimeError(f"[URGE] rewards must be 1D [K], got {tuple(rs_candidates.shape)}")

        # First call: initialize baseline rewards and carry-through
        if self.population_rs is None:
            self.population_rs = rs_candidates.detach().to(self.device)

            return latents, torch.arange(self.num_particles, device=self.device, dtype=torch.long)

        # Providers
        V_t_val = float(V_t) if V_t is not None else (
            float(self.sigma_provider_fn(sampling_idx)) if self.sigma_provider_fn else self._default_sigma(sampling_idx)
        )
        dt_val = float(dt) if dt is not None else (
            float(self.dt_provider_fn(sampling_idx)) if self.dt_provider_fn else self._default_dt(sampling_idx)
        )
        dt_val = abs(dt_val)

        # Reward increment (blue): prefer explicit rewards_next; otherwise use current minus previous
        if rewards_next is not None:
            reward_inc = (rewards_next.to(self.device, dtype=torch.float32) - rs_candidates).contiguous()
        else:
            reward_inc = (rs_candidates - self.population_rs).to(self.device, dtype=torch.float32)

        # Guided URGE: girsanov_log is log(dP/dP_G) for the same drift and noise
        # that produced these latents. It replaces the pixel-space martingale.
        if girsanov_log is not None:
            g_log = girsanov_log.to(self.device, dtype=torch.float32).reshape(-1)
            if int(g_log.shape[0]) != self.num_particles:
                raise RuntimeError(
                    f"[URGE] girsanov_log must be [K]={self.num_particles}, got {tuple(g_log.shape)}"
                )
            delta_logw = self.eta * reward_inc + g_log
        else:
            # Legacy path: pixel CLIP gradient with an independent dW.
            martingale_term = torch.zeros(self.num_particles, device=self.device, dtype=torch.float32)
            drift_term = torch.zeros(self.num_particles, device=self.device, dtype=torch.float32)

            if guidance_grad is not None:
                gg = guidance_grad.to(self.device, dtype=torch.float32)
                gg_flat = gg.view(self.num_particles, -1)

                if noise_for_martingale is not None:
                    dW = noise_for_martingale.to(self.device, dtype=gg.dtype)
                else:
                    dW = self._sample_noise(gg.shape, sampling_idx, gg.dtype)

                ip = (gg_flat * dW.view(self.num_particles, -1)).sum(dim=1)
                martingale_term = - V_t_val * ip
                grad_sq_norm = (gg_flat.pow(2).sum(dim=1))
                drift_term = - 0.5 * (V_t_val ** 2) * grad_sq_norm * dt_val

            delta_logw = self.eta * (reward_inc + martingale_term + drift_term)
        self.log_weights = self.log_weights + delta_logw

        # Normalize to avoid overflow and derive weights
        logw_shifted = self.log_weights - torch.max(self.log_weights)
        w = torch.exp(logw_shifted)
        w = torch.nan_to_num(w, nan=0.0, posinf=1e10, neginf=0.0)
        w_sum = float(w.sum().item())
        if w_sum <= 0.0 or (not torch.isfinite(w).all()):
            w = torch.ones_like(w)
            w_sum = float(w.sum().item())
        self.weights = (w / (w_sum + 1e-12)).contiguous()

        # ESS decision and resampling schedule
        ess = 1.0 / (torch.sum(self.weights * self.weights) + 1e-12)
        ess_threshold = max(1.0, self.ess_threshold_ratio * float(self.num_particles))
        is_resample_step = (sampling_idx in self._resampling_steps)
        is_last = (sampling_idx == self.time_steps - 1)

        # Resampling decision
        should_resample = False
        if is_resample_step:
            should_resample = True
        if self.adaptive_resampling and (float(ess.item()) < ess_threshold) and not is_resample_step:
            should_resample = True
        if self.use_last_step_resample and is_last:
            should_resample = True

        # Carry-through when not resampling
        if not should_resample:
            self.population_rs = rs_candidates.detach().to(self.device)
            return latents, torch.arange(self.num_particles, device=self.device, dtype=torch.long)

        # Multinomial resampling
        indices = (
            torch.tensor([0], device=self.device, dtype=torch.long)
            if self.num_particles == 1
            else torch.multinomial(self.weights, num_samples=self.num_particles, replacement=self.replacement)
        )

        if indices.dim() != 1 or indices.dtype != torch.long or int(indices.shape[0]) != self.num_particles:
            raise RuntimeError(f"[URGE] invalid indices: shape={tuple(indices.shape)} dtype={indices.dtype}")

        latents = latents.contiguous()
        resampled_latents = latents.index_select(0, indices)

        # Update state along resampled ancestry
        self.last_indices = indices
        self.population_rs = rs_candidates.index_select(0, indices)

        # Reset weights after resampling to avoid runaway accumulation and path bias
        self.log_weights = torch.zeros_like(self.log_weights)
        self.weights = torch.ones(self.num_particles, device=self.device, dtype=torch.float32) / float(self.num_particles)

        return resampled_latents, indices



if __name__ == "__main__":

    # set seed for reproducibility
    random.seed(0)
    torch.manual_seed(0)

    # 1x1 pixel images
    num_particles = 8
    x0s = torch.rand(num_particles, 1, 1)

    # reward darker images
    reward_function = lambda x: -0.5 * x.sum(dim=(1, 2))

    # Define the FKD steering mechanism
    fkds = FKD(
        potential_type=PotentialType.DIFF,
        lmbda=10.0,
        num_particles=num_particles,
        adaptive_resampling=False,
        resample_frequency=1,
        resampling_t_start=-1,
        resampling_t_end=100,
        time_steps=100,
        reward_fn=lambda x: reward_function(x),
        device=torch.device("cpu"),
    )

    # Define the sampling index
    sampling_idx = 0

    # Perform resampling
    resampled_latents, resampled_images = fkds.resample(
        sampling_idx=sampling_idx,
        latents=x0s,
        x0_preds=x0s,
    )

    plt.rc("text", usetex=False)

    fig, axs = plt.subplots(2, num_particles, figsize=(num_particles * 2, 4))

    axs[0, 0].set_title("Initial")
    axs[1, 0].set_title("Resampled")

    for i in range(num_particles):
        axs[0, i].imshow(x0s[i].detach().numpy(), cmap="gray", vmin=0, vmax=1)
        axs[1, i].imshow(
            resampled_images[i].detach().numpy(), cmap="gray", vmin=0, vmax=1
        )
        axs[0, i].axis("off")
        axs[1, i].axis("off")

    out_path = "resampled_examples.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    print("Saved resampled examples to:", out_path)