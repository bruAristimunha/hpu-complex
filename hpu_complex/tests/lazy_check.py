"""On-device lazy-mode (PT_HPU_LAZY_MODE=1) verification for Issue #3.

Run on Gaudi2 via:
  PT_HPU_LAZY_MODE=1 python3 hpu_complex/tests/lazy_check.py

Verifies:
1. hpu_complex.enable() does not fail or warn in lazy mode.
2. Real-only float32 lazy ops (fwd + bwd + mark_step) work cleanly.
3. Device-agnostic float-pair API (hpu_complex.rfft, hpu_complex.stft) works in lazy mode.
4. CPU complex64 ops still work while lazy mode is enabled.
5. All complex64 / torch.fft / linalg.eig(float32) HPU paths raise clean RuntimeErrors
   without process aborts.
"""
import sys
import warnings

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu
from habana_frameworks.torch.utils.internal import is_lazy
import torch

import hpu_complex as H


def check_real_only_lazy_safety():
    torch.manual_seed(0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        H.enable()
    assert not caught, f"Unexpected warnings in enable() during lazy mode: {[str(w.message) for w in caught]}"

    xc = torch.randn(4, 4, requires_grad=True)
    xh = xc.detach().clone().to("hpu").requires_grad_()
    yc = (xc @ xc.T).sum()
    yh = (xh @ xh.T).sum()
    yc.backward()
    yh.backward()
    htcore.mark_step()
    hthpu.synchronize()
    assert float((yh.detach().cpu() - yc.detach()).abs()) < 1e-4
    assert float((xh.grad.cpu() - xc.grad).abs().max()) < 1e-4
    print("lazy_check real_only_lazy_safety: OK")


def check_float_pair_api_lazy():
    torch.manual_seed(0)
    xc = torch.randn(2, 128, requires_grad=True)
    xh = xc.detach().clone().to("hpu").requires_grad_()
    spec_c = H.rfft(xc)
    spec_h = H.rfft(xh)
    stft_c = H.stft(xc, 32, 16)
    stft_h = H.stft(xh, 32, 16)
    loss_c = spec_c.abs().sum() + stft_c.abs().sum()
    loss_h = spec_h.abs().sum() + stft_h.abs().sum()
    loss_c.backward()
    loss_h.backward()
    htcore.mark_step()
    hthpu.synchronize()
    assert float((spec_h.detach().cpu() - spec_c.detach()).abs().max()) < 1e-4
    assert float((stft_h.detach().cpu() - stft_c.detach()).abs().max()) < 1e-4
    assert float((xh.grad.cpu() - xc.grad).abs().max()) < 1e-4
    print("lazy_check float_pair_api: OK")


def check_cpu_complex_ops_unaffected():
    a = torch.randn(4, 4, dtype=torch.complex64)
    b = torch.randn(4, 4, dtype=torch.complex64)
    c = a @ b + torch.fft.rfft(torch.randn(4, 8)).sum()
    assert c.device.type == "cpu" and torch.isfinite(c).all()
    print("lazy_check cpu_complex_ops_unaffected: OK")


def check_lazy_complex_guards():
    hpu = torch.device("hpu")
    real_pair = torch.randn(4, 4, 2, device=hpu)
    real_re = torch.randn(4, 4, device=hpu)
    real_im = torch.randn(4, 4, device=hpu)
    real_sig = torch.randn(8, 22, 512, device=hpu)
    real_mat = torch.randn(4, 4, device=hpu)
    cpu_c64 = torch.randn(4, 4, dtype=torch.complex64)
    htcore.mark_step()

    cases = [
        ("zeros_c64", lambda: torch.zeros(4, 4, dtype=torch.complex64, device=hpu)),
        ("ones_c64", lambda: torch.ones(4, 4, dtype=torch.complex64, device=hpu)),
        ("to_hpu_then_cpu", lambda: cpu_c64.to(hpu).cpu()),
        ("view_as_complex", lambda: torch.view_as_complex(real_pair)),
        ("torch.complex", lambda: torch.complex(real_re, real_im)),
        ("torch.fft.rfft", lambda: torch.fft.rfft(real_sig)),
        ("torch.linalg.eig(f32)", lambda: torch.linalg.eig(real_mat)),
        ("torch.linalg.eigvals(f32)", lambda: torch.linalg.eigvals(real_mat)),
        ("real.to(complex64)", lambda: real_mat.to(torch.complex64)),
        ("real.view(complex64)", lambda: real_pair.view(torch.complex64)),
        ("torch.polar", lambda: torch.polar(real_re.abs(), real_im)),
    ]
    for name, fn in cases:
        try:
            fn()
        except RuntimeError as exc:
            msg = str(exc)
            assert "PT_HPU_LAZY_MODE=0" in msg, (name, msg)
            print(f"lazy_check guard {name}: OK ({msg.splitlines()[0]})")
        else:
            raise AssertionError(f"Expected RuntimeError for {name} in lazy mode")


def main():
    if not is_lazy():
        print("lazy_check: SKIP (PT_HPU_LAZY_MODE is 0; run with PT_HPU_LAZY_MODE=1)")
        return
    check_real_only_lazy_safety()
    check_float_pair_api_lazy()
    check_cpu_complex_ops_unaffected()
    check_lazy_complex_guards()
    print("LAZY_CHECK_ALL_OK")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
