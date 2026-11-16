"""CPU oracles for the Braindecode helpers. Run: pytest tests/ -q"""
import numpy as np
import torch
from scipy.signal import hilbert

from hpu_complex import braindecode as bd


def test_analytic_signal_matches_scipy():
    torch.manual_seed(0)
    x = torch.randn(3, 200)
    got = bd.analytic_signal(x)
    ref = torch.tensor(hilbert(x.numpy(), axis=-1))
    assert (got - ref).abs().max() < 1e-4


def test_aggregate_matches_loop_mean():
    torch.manual_seed(0)
    preds = torch.randn(10, 3)
    trials = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
    got = bd.aggregate_trial_predictions(preds, trials, 3)
    ref = torch.stack([preds[trials == t].mean(0) for t in range(3)])
    assert (got - ref).abs().max() < 1e-6


def test_renorm_matches_torch():
    torch.manual_seed(0)
    x = torch.randn(5, 4, 8) * 3
    got = bd.renorm(x, maxnorm=1.0, dim=0)
    ref = torch.renorm(x, p=2, dim=0, maxnorm=1.0)
    assert (got - ref).abs().max() < 1e-5


def test_patch_is_noop_without_hpu():
    assert bd.patch_braindecode() is False
    assert bd.fft_available() is False
