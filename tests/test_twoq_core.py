"""Sanity checks for Algorithm 1 (ratio) and the k-pool Stein helpers."""

from __future__ import annotations

import math
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fkd_diffusers.twoq_core import (
    add_terminal_log_value,
    antithetic_euler_pair,
    centered_stein_noises,
    independent_mh_accept,
    independent_mh_reduce,
    posterior_select_and_log_increment,
    posterior_sign_and_log_increment,
    product_log_ratio,
    product_log_ratio_pool,
    select_child,
    select_pool_child,
    snis_weights,
    stein_candidates,
)


def test_ratio_matches_product_formula():
    torch.manual_seed(0)
    n, b = 7, 16
    a_plus = torch.randn(n, b)
    a_minus = torch.randn(n, b)
    u = torch.rand(n, b)
    choose, a_sel, inc = posterior_sign_and_log_increment(a_plus, a_minus, u=u)
    log_r = inc.sum(dim=0)
    log_h_n = torch.randn(b)
    log_r = add_terminal_log_value(log_r, log_h_n)
    closed = product_log_ratio(a_plus, a_minus, a_sel, log_h_n)
    assert torch.allclose(log_r, closed, atol=1e-5)
    assert choose.dtype == torch.bool
    assert choose.shape == (n, b)


def test_stein_pool_is_centered_and_k2_matches_pair():
    torch.manual_seed(0)
    z = centered_stein_noises(4, (8, 3))
    assert z.shape == (4, 8, 3)
    assert torch.allclose(z.sum(dim=0), torch.zeros(8, 3), atol=1e-5)

    mean = torch.tensor([[1.0, -2.0]])
    sigma = torch.tensor([[0.3]])
    z2 = centered_stein_noises(2, (1, 2))
    y = stein_candidates(mean, sigma, z2)
    assert torch.allclose(y.mean(dim=0), mean, atol=1e-5)
    assert torch.allclose(y[0] - mean, mean - y[1], atol=1e-5)
    idx = torch.tensor([0])
    assert torch.allclose(select_pool_child(y, idx), y[0])

    a = torch.tensor([[0.4], [-1.2]])
    u = torch.tensor([0.1])
    idx2, a_sel, inc = posterior_select_and_log_increment(a, u=u)
    choose, a_sel_pair, inc_pair = posterior_sign_and_log_increment(a[0], a[1], u=u)
    assert idx2.shape == (1,)
    assert torch.equal(choose, idx2 == 0)
    assert torch.allclose(a_sel, a_sel_pair)
    assert torch.allclose(inc, inc_pair)

    log_h = torch.randn(5, 3, 7)
    log_sel = log_h[:, 1, :]
    log_hn = torch.randn(7)
    closed = product_log_ratio_pool(log_h, log_sel, log_hn)
    step = torch.logsumexp(log_h, dim=1) - math.log(3.0) - log_sel
    assert torch.allclose(closed, step.sum(0) + log_hn, atol=1e-5)


def test_mh_accepts_better_and_snis_normalized():
    cur = torch.tensor([0.0])
    better = torch.tensor([2.0])
    worse = torch.tensor([-4.0])
    assert bool(independent_mh_accept(cur, better, u=torch.tensor([0.9])).item())
    assert not bool(independent_mh_accept(cur, worse, u=torch.tensor([0.5])).item())

    samples = torch.stack([torch.zeros(2, 3), torch.ones(2, 3)], dim=0)
    log_r = torch.tensor([[0.0, 0.0], [50.0, -50.0]])
    picked, picked_r = independent_mh_reduce(samples, log_r)
    assert picked.shape == (2, 3)
    assert picked_r.shape == (2,)

    w = snis_weights(torch.tensor([0.0, 0.0, 0.0]), dim=0)
    assert math.isclose(float(w.sum()), 1.0, abs_tol=1e-6)

    mean = torch.tensor([[1.0, -2.0]])
    sigma = torch.tensor([[0.3]])
    z = torch.tensor([[0.5, -1.0]])
    y_plus, y_minus = antithetic_euler_pair(mean, sigma, z)
    assert torch.allclose(0.5 * (y_plus + y_minus), mean)
    chosen = select_child(y_plus, y_minus, torch.tensor([True]))
    assert torch.allclose(chosen, y_plus)


if __name__ == "__main__":
    test_ratio_matches_product_formula()
    test_stein_pool_is_centered_and_k2_matches_pair()
    test_mh_accepts_better_and_snis_normalized()
    print("ok")
