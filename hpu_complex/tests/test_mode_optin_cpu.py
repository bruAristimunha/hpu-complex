"""CPU checks: the dispatch mode is opt-in, not global (regression gate).

enable() must not enter ComplexMode; complex_mode() scopes it. Run:
python -m pytest hpu_complex/tests/test_mode_optin_cpu.py -q
"""
import torch
from torch.utils._python_dispatch import _get_current_dispatch_mode_stack

import hpu_complex as H
from hpu_complex.mode import ComplexMode


def _mode_active():
    return any(isinstance(m, ComplexMode)
               for m in _get_current_dispatch_mode_stack())


def test_enable_does_not_enter_mode():
    H.enable()
    assert not _mode_active()


def test_complex_mode_scopes_and_computes():
    assert not _mode_active()
    a = torch.randn(4, 8, dtype=torch.complex64)
    b = torch.randn(4, 8, dtype=torch.complex64)
    with H.complex_mode():
        assert _mode_active()
        got = a * b + a / b
        # Scalar operands arrive unwrapped at dispatch-mode level;
        # they must fall through, not crash _pair (regression).
        assert (a * 1j - (a.cpu() * 1j)).abs().max() < 1e-5
        assert (a - 1j - (a.cpu() - 1j)).abs().max() < 1e-5
    assert not _mode_active()
    assert (got - (a * b + a / b)).abs().max() < 1e-5


def test_lazy_env_does_not_break_enable_or_cpu_complex(monkeypatch):
    monkeypatch.setenv("PT_HPU_LAZY_MODE", "1")
    H.enable()
    assert not _mode_active()
    a = torch.randn(4, 8, dtype=torch.complex64, device="cpu")
    b = torch.randn(4, 8, dtype=torch.complex64, device="cpu")
    assert torch.isfinite(a * b + torch.view_as_complex(torch.randn(4, 8, 2))).all()
    r = torch.randn(2, 32)
    assert torch.isfinite(H.rfft(r)).all()
    assert torch.isfinite(H.stft(r, 16, 8)).all()
