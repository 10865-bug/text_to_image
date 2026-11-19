"""Stein / 2Q-Doob helpers (paper Algorithms 1–2).

Paper: "Stein Guidance without Gradients" (ICLR 2027 source).

At each reverse step the paper draws a K-point centered Gaussian Stein pool
    Z_i = sqrt(K/(K-1)) (G_i - mean_j G_j),   G_i iid N(0, I),
    Y_i = mu + sigma * Z_i,                   sum_i Z_i = 0,
queries a_i = log H(Y_i), selects J proportional to exp(a_i), and accumulates
    ℓ ← ℓ + logsumexp(a) - log K - a^J
    = ℓ + log S - log H^J,   S = K^{-1} sum_i H(Y_i).

K = 2 recovers the original antithetic pair Y^± = mu ± sigma Z.
These helpers are batched PyTorch and do not depend on D4RL.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


def centered_stein_noises(
    k: int,
    batch_shape: Tuple[int, ...],
    generator: Optional[torch.Generator] = None,
    device=None,
    dtype=None,
) -> torch.Tensor:
    """Paper Eq. (13): jointly Gaussian Z_i with standard marginals and zero sum.

    Returns Z with shape ``(k, *batch_shape)``. Requires ``k >= 2``.
    """
    k = int(k)
    if k < 2:
        raise ValueError(f"Stein pool width k must be >= 2, got {k}")
    g = torch.randn((k, *batch_shape), generator=generator, device=device, dtype=dtype)
    return math.sqrt(k / (k - 1)) * (g - g.mean(dim=0, keepdim=True))


def stein_candidates(
    mean: torch.Tensor,
    sigma: torch.Tensor,
    z: torch.Tensor,
) -> torch.Tensor:
    """Y_i = mean + sigma * Z_i. ``z`` has leading pool axis, shape (k, *mean.shape)."""
    if sigma.ndim < mean.ndim:
        sigma = sigma.view(*sigma.shape, *([1] * (mean.ndim - sigma.ndim)))
    return mean.unsqueeze(0) + sigma.unsqueeze(0) * z


def antithetic_euler_pair(
    mean: torch.Tensor,
    sigma: torch.Tensor,
    z: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Y^+ = mean + sigma * Z, Y^- = mean - sigma * Z (K = 2 special case)."""
    if sigma.ndim < mean.ndim:
        sigma = sigma.view(*sigma.shape, *([1] * (mean.ndim - sigma.ndim)))
    y_plus = mean + sigma * z
    y_minus = mean - sigma * z
    return y_plus, y_minus


def logsumexp_pair(a_plus: torch.Tensor, a_minus: torch.Tensor) -> torch.Tensor:
    return torch.logaddexp(a_plus, a_minus)


def posterior_select_and_log_increment(
    log_h: torch.Tensor,
    u: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select J ∝ exp(a_i) and return the Algorithm-1 log increment.

    Parameters
    ----------
    log_h:
        log H(Y_i), shape (k, B).
    u:
        Uniform(0, 1) of shape (B,); sampled if omitted.

    Returns
    -------
    idx : (B,) selected pool index
    a_selected : (B,) log H of the selected child
    log_increment : (B,) logsumexp(a) - log k - a^J
    """
    if log_h.ndim < 2:
        raise ValueError(f"log_h must be (k, B), got {tuple(log_h.shape)}")
    k = log_h.shape[0]
    if k < 2:
        raise ValueError(f"Stein pool width k must be >= 2, got {k}")
    if u is None:
        u = torch.rand(log_h.shape[1], device=log_h.device, dtype=log_h.dtype)
    u = u.clamp(0.0, 1.0 - 1e-8)
    probs = torch.softmax(log_h, dim=0)
    cdf = probs.cumsum(dim=0)
    idx = (cdf >= u.unsqueeze(0)).to(dtype=torch.int64).argmax(dim=0)
    a_selected = log_h.gather(0, idx.unsqueeze(0)).squeeze(0)
    log_increment = torch.logsumexp(log_h, dim=0) - math.log(float(k)) - a_selected
    return idx, a_selected, log_increment


def posterior_sign_and_log_increment(
    a_plus: torch.Tensor,
    a_minus: torch.Tensor,
    u: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """K = 2 wrapper: choose + iff J = 0. Same increment as the pair formula."""
    log_h = torch.stack((a_plus, a_minus), dim=0)
    idx, a_selected, log_increment = posterior_select_and_log_increment(log_h, u=u)
    return idx == 0, a_selected, log_increment


def select_child(
    y_plus: torch.Tensor,
    y_minus: torch.Tensor,
    choose_plus: torch.Tensor,
) -> torch.Tensor:
    mask = choose_plus.reshape(*choose_plus.shape, *([1] * (y_plus.ndim - choose_plus.ndim)))
    return torch.where(mask, y_plus, y_minus)


def select_pool_child(y: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """``y`` is (k, B, ...), ``idx`` is (B,)."""
    batch = torch.arange(idx.shape[0], device=y.device)
    return y[idx, batch]


def add_terminal_log_value(log_r: torch.Tensor, log_h_terminal: torch.Tensor) -> torch.Tensor:
    """log R_n = ℓ + log H_n(X_n)."""
    return log_r + log_h_terminal


def product_log_ratio(
    log_h_plus: torch.Tensor,
    log_h_minus: torch.Tensor,
    log_h_selected: torch.Tensor,
    log_h_terminal: torch.Tensor,
) -> torch.Tensor:
    """Closed-form K = 2 ratio. Inputs over time in the leading dim: (n, B)."""
    step = logsumexp_pair(log_h_plus, log_h_minus) - math.log(2.0) - log_h_selected
    return step.sum(dim=0) + log_h_terminal


def product_log_ratio_pool(
    log_h: torch.Tensor,
    log_h_selected: torch.Tensor,
    log_h_terminal: torch.Tensor,
) -> torch.Tensor:
    """Closed-form ratio for a K-pool. ``log_h`` is (n, k, B)."""
    k = log_h.shape[1]
    step = torch.logsumexp(log_h, dim=1) - math.log(float(k)) - log_h_selected
    return step.sum(dim=0) + log_h_terminal


def independent_mh_accept(
    log_r_current: torch.Tensor,
    log_r_proposal: torch.Tensor,
    u: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Independent MH: accept if log U ≤ min{0, log R' − log R}."""
    if u is None:
        u = torch.rand_like(log_r_current)
    log_alpha = torch.clamp(log_r_proposal - log_r_current, max=0.0)
    return torch.log(torch.clamp(u, min=1e-12)) <= log_alpha


def independent_mh_reduce(
    samples: torch.Tensor,
    log_r: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run outer independent MH along axis 0 of a stack of independent proposals.

    samples: (M, B, ...)
    log_r:   (M, B)
    Returns the last accepted sample (B, ...) and its log R (B,).
    """
    if samples.shape[0] == 1:
        return samples[0], log_r[0]
    current = samples[0]
    current_log_r = log_r[0]
    for j in range(1, samples.shape[0]):
        u = torch.rand(current_log_r.shape, device=current_log_r.device, generator=generator)
        accept = independent_mh_accept(current_log_r, log_r[j], u=u)
        mask = accept.reshape(*accept.shape, *([1] * (current.ndim - accept.ndim)))
        current = torch.where(mask, samples[j], current)
        current_log_r = torch.where(accept, log_r[j], current_log_r)
    return current, current_log_r


def snis_weights(log_r: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Self-normalized importance weights, stable in log space."""
    return torch.softmax(log_r, dim=dim)


def snis_sample_index(
    log_r: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample one index per column from Categorical(softmax(log R)).

    log_r: (M, B) -> index (B,)
    """
    weights = snis_weights(log_r, dim=0).transpose(0, 1)  # (B, M)
    return torch.multinomial(weights, num_samples=1, generator=generator).squeeze(-1)
