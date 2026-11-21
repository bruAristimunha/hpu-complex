"""On-device lazy-mode (PT_HPU_LAZY_MODE=1) verification for Issue #3.

Run on Gaudi2 via:
  PT_HPU_LAZY_MODE=1 python3 hpu_complex/tests/lazy_check.py

Verifies:
1. hpu_complex.enable() does not fail or warn in lazy mode.
2. Real-only float32 lazy ops (fwd + bwd + mark_step) work cleanly.
3. Device-agnostic float-pair API (hpu_complex.rfft, hpu_complex.stft) works in lazy mode.
4. CPU complex64 ops still work while lazy mode is enabled.
5. complex64 runs in lazy mode (lazy views of the Habana tensor behind each complex storage): factories, .to,
   view_as_complex, torch.complex, torch.fft, eigvals, products, in-place writes through .real, against the CPU.
6. What lazy mode still refuses raises a clean RuntimeError (no process abort): complex64 under
   torch.inference_mode, and permuted complex64 tensors of rank 5 or more.
7. green_ops_check.py in lazy mode: a real tensor times a Python complex scalar (x * 2j used to abort the process
   inside Habana), complex cat / exp / neg / masked_fill_ / linalg.vector_norm, with gradients.
8. The Habana tensors held by lazy complex storages are released (no growth over 100 steps).
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


def _err(a, b):
    a, b = a.detach().cpu(), b.detach().cpu()
    assert a.shape == b.shape and a.dtype == b.dtype, (a.shape, b.shape, a.dtype, b.dtype)
    if a.is_complex():
        a, b = torch.view_as_real(a.resolve_conj()), torch.view_as_real(b.resolve_conj())
    return (a.double() - b.double()).abs().max().item() if a.numel() else 0.0


def check_lazy_complex64_runs():
    """complex64 in lazy mode: the entry points the old guard refused now compute on lazy views, against the CPU."""
    hpu = torch.device("hpu")
    torch.manual_seed(0)
    pair, re, im = torch.randn(4, 4, 2), torch.randn(4, 4), torch.randn(4, 4)
    sig, mat, c64 = torch.randn(8, 22, 512), torch.randn(4, 4), torch.randn(4, 4, dtype=torch.complex64)
    cases = [
        ("zeros_c64", lambda d: torch.zeros(4, 4, dtype=torch.complex64, device=d), 0),
        ("ones_c64", lambda d: torch.ones(4, 4, dtype=torch.complex64, device=d), 0),
        ("full_c64", lambda d: torch.full((2, 3), 1 - 2j, dtype=torch.complex64, device=d), 0),
        ("scalar_tensor_c64", lambda d: torch.scalar_tensor(1 + 2j, dtype=torch.complex64, device=d), 0),
        ("to_hpu_then_cpu", lambda d: c64.to(d), 0),
        ("view_as_complex", lambda d: torch.view_as_complex(pair.to(d)), 0),
        ("torch.complex", lambda d: torch.complex(re.to(d), im.to(d)), 0),
        ("torch.fft.rfft", lambda d: torch.fft.rfft(sig.to(d)), 2e-4),
        ("irfft(rfft)", lambda d: torch.fft.irfft(torch.fft.rfft(sig.to(d)), n=512), 1e-5),
        ("torch.linalg.eigvals(f32).abs", lambda d: torch.linalg.eigvals(mat.to(d)).abs().sort(-1).values, 1e-5),
        ("real.to(complex64)", lambda d: mat.to(d).to(torch.complex64), 0),
        ("real.view(complex64)", lambda d: pair.to(d).view(torch.complex64), 0),
        ("torch.polar", lambda d: torch.polar(re.to(d).abs(), im.to(d)), 1e-6),
        # entries ~1.3e5 (257 bins of |X|^2 ~ 512): scaled to compare relatively
        ("complex @ complex.mH (card GEMM), relative", lambda d: (lambda z: z @ z.mH / 1.3e5)(torch.fft.rfft(sig.to(d))[:4]),
         1e-6),
        ("z.real written in place", lambda d: (lambda z: (z.real.mul_(2), z)[1])(c64.to(d).clone()), 0),
        ("torch.no_grad() rfft().real", lambda d: _no_grad(lambda: torch.fft.rfft(sig.to(d)).real), 2e-4),
    ]
    for name, fn, tol in cases:
        got = fn(hpu)
        htcore.mark_step()
        e = _err(got, fn("cpu"))
        assert e <= tol, (name, e)
        print(f"lazy_check runs {name}: OK (max|err| {e:.1e})")


def _no_grad(fn):
    with torch.no_grad():
        return fn()


def check_lazy_complex_guards():
    """What lazy mode still refuses, with a clean RuntimeError (no process abort)."""
    hpu = torch.device("hpu")
    sig = torch.randn(8, 512, device=hpu)
    z5 = torch.randn(2, 3, 4, 5, 6, dtype=torch.complex64).to(hpu)
    htcore.mark_step()
    cases = [
        ("complex64 under torch.inference_mode", lambda: _inference(lambda: torch.fft.rfft(sig).real.sum()),
         "torch.no_grad()"),
        ("permuted 5-D complex64", lambda: z5.permute(0, 2, 1, 3, 4) * 2, "rank 5 or more"),
    ]
    for name, fn, match in cases:
        try:
            fn()
            htcore.mark_step()
        except RuntimeError as exc:
            msg = str(exc)
            assert match in msg and "PT_HPU_LAZY_MODE=0" in msg, (name, msg)
            print(f"lazy_check guard {name}: OK ({msg.splitlines()[0][:120]})")
        else:
            raise AssertionError(f"Expected RuntimeError for {name} in lazy mode")


def _inference(fn):
    with torch.inference_mode():
        return fn()


def check_green_ops():
    """The eager card checks of green_ops_check.py, in lazy mode (real x complex scalar, cat/exp/neg/masked_fill_/norm)."""
    import math
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from green_ops_check import add_checks

    failed = []

    def check(name, fn, atol):
        try:
            e = fn()
            htcore.mark_step()
            ok = e <= atol
        except Exception as exc:  # noqa: BLE001
            e, ok = f"{type(exc).__name__}: {str(exc).splitlines()[0][:150]}", False
        print(f"lazy_check green_ops {name}: {'OK' if ok else 'FAIL'} ({e})")
        if not ok:
            failed.append(name)

    def worst(*errors):
        return math.nan if any(math.isnan(x) for x in errors) else max(errors)

    def grad_err(f, x, relative=False, reference=None):
        xc, xh = x.clone().requires_grad_(), x.to("hpu").requires_grad_()
        yc, yh = (reference or f)(xc), f(xh)
        yc.backward(torch.ones_like(yc))
        yh.backward(torch.ones_like(yh))
        htcore.mark_step()
        return worst(*(_err(g, r) / (r.detach().abs().max().item() if relative else 1.0)
                       for g, r in ((yh, yc), (xh.grad, xc.grad))))

    add_checks(check, _err, grad_err, worst, sync=htcore.mark_step)
    assert not failed, failed


def check_lazy_storages_drain():
    """Each lazy complex64 storage holds its Habana tensor; they are released with the complex tensors."""
    x = torch.randn(4, 64, device="hpu")
    before, peak = torch.ops.hpu_complex.lazy_storages_alive(), 0
    for _ in range(100):
        z = torch.fft.rfft(x) * (1 + 1j)
        peak = max(peak, torch.ops.hpu_complex.lazy_storages_alive())
        v = z.abs().sum()
        del z
        htcore.mark_step()
    v.item()
    after = torch.ops.hpu_complex.lazy_storages_alive()
    assert peak > before and after == before, (before, peak, after)
    print(f"lazy_check storages drain: OK ({before} -> {after} after 100 steps, {peak} alive at most)")


def main():
    if not is_lazy():
        print("lazy_check: SKIP (PT_HPU_LAZY_MODE is 0; run with PT_HPU_LAZY_MODE=1)")
        return
    check_real_only_lazy_safety()
    check_float_pair_api_lazy()
    check_cpu_complex_ops_unaffected()
    check_lazy_complex64_runs()
    check_lazy_complex_guards()
    check_green_ops()
    check_lazy_storages_drain()
    print("LAZY_CHECK_ALL_OK")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
