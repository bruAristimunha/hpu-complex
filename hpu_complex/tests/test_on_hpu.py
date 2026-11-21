"""On-device checks: python3 -m pytest hpu_complex/tests/test_on_hpu.py, or, in the Gaudi image (no pytest),
python3 hpu_complex/tests/run_on_hpu.py. Skipped on machines without habana_frameworks.

The float FFT core (hpu_complex.rfft/stft_magnitude) on HPU against torch.fft on the CPU, its gradients and
torch.compile; complex arithmetic and clip_grad_norm_ with and without the dispatch mode.
"""
try:
    import pytest
except ImportError:  # the Gaudi image: run_on_hpu.py imports this module directly
    pytest = None
else:
    pytest.importorskip("habana_frameworks.torch")
import habana_frameworks.torch.core as htcore  # noqa: E402
from habana_frameworks.torch.utils.internal import is_lazy  # noqa: E402
import torch  # noqa: E402

import hpu_complex as H  # noqa: E402


def _skip_if_lazy(name):
    if not is_lazy():
        return False
    if pytest is not None:
        pytest.skip(f"{name} requires eager mode (PT_HPU_LAZY_MODE=0)")
    print(f"{name}: SKIP (lazy mode)")
    return True


def test_core_fft_on_hpu():
    # Gaudi computes float64 as float32 (syn Double->float): test in float32
    # against a float64 CPU reference, relative to peak magnitude.
    torch.manual_seed(0)
    for n in (200, 1024, 1920):
        x = torch.randn(4, n)
        got = H.magnitude(H.rfft(x.to("hpu"))).cpu()
        ref = torch.fft.rfft(x.double()).abs()
        rel = ((got - ref.float()).abs().max() / ref.max()).item()
        print(f"n={n} on-HPU magnitude rel_err={rel:.2e}")
        assert rel < 1e-3, n


def test_stft_and_grad_on_hpu():
    x = torch.randn(2, 4000, device="hpu", requires_grad=True)
    mag = H.stft_magnitude(x, 512, 256)
    assert mag.device.type == "hpu"
    mag.sum().backward()
    assert torch.isfinite(x.grad).all()
    print(f"stft_magnitude {tuple(mag.shape)} grad ok")


def test_complex_without_mode_and_clip_grad():
    if _skip_if_lazy("test_complex_without_mode_and_clip_grad"):
        return
    # Default enable(): no dispatch mode entered, yet complex forward
    # (the extension's PrivateUse1 kernels) and clip_grad_norm_ (0-d
    # total_norm: regression for the stack({a,b}, -2) crash in
    # float_addsub) work.
    H.enable()
    a = torch.complex(torch.randn(4, 8, device="hpu"),
                      torch.randn(4, 8, device="hpu"))
    b = torch.complex(torch.randn(4, 8, device="hpu"),
                      torch.randn(4, 8, device="hpu"))
    ac, bc = a.cpu(), b.cpu()
    for got, ref in [(a * b, ac * bc), (a / b, ac / bc), (a + b, ac + bc)]:
        err = (got.cpu() - ref).abs().max() / ref.abs().max()
        assert err < 1e-4, err
    assert torch.isfinite(a.abs()).all()
    m = torch.nn.Sequential(
        torch.nn.Conv1d(22, 32, 5), torch.nn.Flatten(),
        torch.nn.Linear(32 * 10, 4)).to("hpu")
    loss = m(torch.randn(8, 22, 14, device="hpu")).sum()
    loss.backward()
    n = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    torch.hpu.synchronize()
    assert torch.isfinite(n)
    with H.complex_mode():  # same clip inside the scoped mode
        loss = m(torch.randn(8, 22, 14, device="hpu")).sum()
        loss.backward()
        n = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        torch.hpu.synchronize()
        assert torch.isfinite(n)
    print("complex without mode + clip_grad_norm_ (both mode states): OK")


def test_compile_attempt():
    if _skip_if_lazy("test_compile_attempt"):
        return
    H.enable()
    f = torch.compile(lambda t: H.magnitude(H.rfft(t)), backend="hpu_backend")
    x = torch.randn(2, 1024, device="hpu")
    y = f(x)
    torch.hpu.synchronize()
    assert y.device.type == "hpu" and y.shape == (2, 513)
    print(f"compile via core ops: OK {tuple(y.shape)} {y.device}")


def _max_err(a, b):
    a, b = a.detach().cpu(), b.detach().cpu()
    if a.is_complex():
        a = torch.view_as_real(a.resolve_conj())
        b = torch.view_as_real(b.to(a.dtype if not b.is_complex() else b.dtype).resolve_conj())
    return (a.double() - b.double()).abs().max().item()


def test_complex_strided_and_inplace_views_on_hpu():
    if _skip_if_lazy("test_complex_strided_and_inplace_views_on_hpu"):
        return
    H.enable()
    torch.manual_seed(0)
    hpu = torch.device("hpu")
    z = torch.complex(torch.randn(3, 5, 5), torch.randn(3, 5, 5))
    r = torch.randn(3, 5, 5)
    s = torch.complex(torch.randn(3, 1, 5), torch.randn(3, 1, 5))

    for name, view_fn in (
        ("slice", lambda t: t[1:, 1:, 1:]),
        ("transpose", lambda t: t.transpose(1, 2)),
        ("diagonal", lambda t: t.diagonal(dim1=-2, dim2=-1)),
    ):
        zh, zc = view_fn(z.to(hpu)), view_fn(z)
        for got, ref in (
            (zh * zh, zc * zc),
            (zh / (zh.abs() + 1), zc / (zc.abs() + 1)),
            (zh + 1.5, zc + 1.5),
            (zh - zh, zc - zc),
            (zh.abs(), zc.abs()),
            (zh.sgn(), zc.sgn()),
            (torch.view_as_real(zh).contiguous(), torch.view_as_real(zc)),
        ):
            assert _max_err(got, ref) < 1e-5, (name, _max_err(got, ref))

        for mutate in (
            lambda b: view_fn(b).fill_(1.25 - 0.75j),
            lambda b: view_fn(b).zero_(),
            lambda b: view_fn(b).mul_(2.0 + 0.5j),
            lambda b: view_fn(b).add_(0.5 - 1.0j),
            lambda b: view_fn(b).copy_(view_fn(z.to(b.device)).conj()),
        ):
            bc, bh = z.clone(), z.to(hpu)
            mutate(bc)
            mutate(bh)
            torch.hpu.synchronize()
            assert _max_err(bh, bc) < 1e-5, (name, _max_err(bh, bc))

    zh_exp, zc_exp = s[1:].to(hpu).expand(2, 5, 5), s[1:].expand(2, 5, 5)
    assert _max_err(zh_exp * 2.0, zc_exp * 2.0) < 1e-5
    assert _max_err(zh_exp + z[1:].to(hpu), zc_exp + z[1:]) < 1e-5
    assert _max_err(zh_exp.abs(), zc_exp.abs()) < 1e-5
    assert _max_err(zh_exp.sgn(), zc_exp.sgn()) < 1e-5
    assert _max_err(z.to(hpu) * torch.ones((), device=hpu).expand(3, 5, 5),
                    z * torch.ones(()).expand(3, 5, 5)) < 1e-5
    assert _max_err(z.to(hpu) / torch.full((), 2.0, device=hpu).expand(3, 5, 5),
                    z / torch.full((), 2.0).expand(3, 5, 5)) < 1e-5
    assert _max_err(z.to(hpu)[2, 3].sum(), z[2, 3].sum()) < 1e-5
    assert _max_err(z.to(hpu) + r.to(hpu), z + r) < 1e-5
    print("complex <6D strided, expanded & in-place views: OK")


def test_complex_linalg_backward_on_hpu():
    if _skip_if_lazy("test_complex_linalg_backward_on_hpu"):
        return
    H.enable()
    torch.manual_seed(0)
    hpu = torch.device("hpu")

    def cplx(*shape):
        return torch.complex(torch.randn(*shape), torch.randn(*shape))

    a_h = cplx(4, 8, 8)
    herm = (a_h @ a_h.mH + 8 * torch.eye(8, dtype=torch.complex64)).detach()
    sq = cplx(4, 8, 8)
    rect = cplx(4, 8, 6)

    cases = [
        ("eigh", lambda t: (lambda w, V: w.sum() + V.abs().sum())(*torch.linalg.eigh(t)), herm, 1e-4),
        ("svd", lambda t: (lambda U, S, Vh: U.abs().sum() + S.sum() + Vh.abs().sum())(
            *torch.linalg.svd(t, full_matrices=False)), rect, 1e-4),
        ("qr", lambda t: (lambda Q, R: Q.real.sum() + R[:, :2].abs().sum())(*torch.linalg.qr(t)), rect, 1e-4),
        ("eig", lambda t: (lambda w, V: w.abs().sum() + V.abs().sum())(*torch.linalg.eig(t)), sq, 1e-4),
        ("matrix_exp", lambda t: torch.linalg.matrix_exp(t).real.sum() +
         torch.linalg.matrix_exp(t)[:, :2, :2].abs().sum(), sq, 1e-4),
        ("matrix_exp_mH", lambda t: torch.linalg.matrix_exp(t.mH).abs().sum(), sq, 1e-4),
    ]
    for name, fn, inp, tol in cases:
        xc = inp.clone().requires_grad_()
        xh = inp.clone().to(hpu).requires_grad_()
        yc, yh = fn(xc), fn(xh)
        yc.backward()
        yh.backward()
        torch.hpu.synchronize()
        assert _max_err(yh, yc) / max(yc.detach().abs().max().item(), 1.0) < tol, (name, "fwd", _max_err(yh, yc))
        rel = _max_err(xh.grad, xc.grad) / max(xc.grad.abs().max().item(), 1e-12)
        assert rel < tol, (name, "bwd", rel)

    for hh_fn in (
        lambda a, tau: torch.linalg.householder_product(a, tau).abs().sum(),
        lambda a, tau: torch.linalg.householder_product(a, tau)[:, :4, :4].real.sum(),
    ):
        ac, tc = cplx(4, 8, 6).requires_grad_(), cplx(4, 6).requires_grad_()
        ah, th = ac.detach().clone().to(hpu).requires_grad_(), tc.detach().clone().to(hpu).requires_grad_()
        yc, yh = hh_fn(ac, tc), hh_fn(ah, th)
        yc.backward()
        yh.backward()
        torch.hpu.synchronize()
        # Against complex128 (the exact gradient), next to the CPU's own complex64 error on the same input: some draws
        # are ill-conditioned (CPU complex64 itself 1e-4 off), where a fixed 5e-4 bound against the CPU's complex64 is
        # a coin toss (main 4.0e-4, this branch 5.7e-4 on one draw; both at the CPU's error on 24 random draws).
        a64, t64 = ac.detach().to(torch.complex128).requires_grad_(), tc.detach().to(torch.complex128).requires_grad_()
        hh_fn(a64, t64).backward()
        ea, et = (_max_err(g, r.grad) / r.grad.abs().max().item() for g, r in ((ah.grad, a64), (th.grad, t64)))
        ca, ct = (_max_err(g, r.grad) / r.grad.abs().max().item() for g, r in ((ac.grad, a64), (tc.grad, t64)))
        print(f"householder_product bwd rel err vs complex128: card {ea:.1e} {et:.1e}, CPU complex64 {ca:.1e} {ct:.1e}",
              flush=True)
        assert ea < max(5e-4, 10 * ca) and et < max(5e-4, 10 * ct), ("householder_product bwd", ea, et, ca, ct)
    print("complex64 linalg 6-op fwd+bwd: OK")


def test_real_eig_and_eigvals_on_hpu():
    if _skip_if_lazy("test_real_eig_and_eigvals_on_hpu"):
        return
    H.enable()
    torch.manual_seed(42)
    hpu = torch.device("hpu")

    base_2d = torch.randn(8, 8)
    base_3d = torch.randn(4, 8, 8)
    base_4d = torch.randn(2, 3, 6, 6)

    # Forward-only eigvals (requires_grad=False -> aten::_linalg_eigvals).
    for name, base, vf in (
        ("2D (8,8)", base_2d, lambda t: t),
        ("3D (4,8,8)", base_3d, lambda t: t),
        ("4D (2,3,6,6)", base_4d, lambda t: t),
        (".mT", base_3d, lambda t: t.mT),
        ("[:, 1:7, 1:7]", base_3d, lambda t: t[:, 1:7, 1:7]),
    ):
        with torch.no_grad():
            wc = torch.linalg.eigvals(vf(base))
            wh = torch.linalg.eigvals(vf(base.detach().clone().to(hpu)))
        torch.hpu.synchronize()
        assert wh.dtype == torch.complex64, (name, wh.dtype)
        assert _max_err(wh, wc) < 1e-4, (name, _max_err(wh, wc))

    # Forward + backward for real float32 linalg.eig and linalg.eigvals.
    cases = [
        ("eig 2D", base_2d, lambda t: t,
         lambda t: (lambda w, V: w.abs().sum() + V.abs().sum())(*torch.linalg.eig(t))),
        ("eig 3D", base_3d, lambda t: t,
         lambda t: (lambda w, V: w.abs().sum() + V.abs().sum())(*torch.linalg.eig(t))),
        ("eig 3D gauge-inv", base_3d, lambda t: t,
         lambda t: (lambda w, V: w[:, :4].real.sum() + (V[:, :4, :4] * V[:, :4, :4].conj()).real.sum())(
             *torch.linalg.eig(t))),
        ("eig 4D", base_4d, lambda t: t,
         lambda t: (lambda w, V: w.abs().sum() + V.abs().sum())(*torch.linalg.eig(t))),
        ("eig .mT", base_3d, lambda t: t.mT,
         lambda t: (lambda w, V: w.abs().sum() + V.abs().sum())(*torch.linalg.eig(t))),
        ("eig slice", base_3d, lambda t: t[:, 1:7, 1:7],
         lambda t: (lambda w, V: w.abs().sum() + V.abs().sum())(*torch.linalg.eig(t))),
        ("eigvals 2D", base_2d, lambda t: t, lambda t: torch.linalg.eigvals(t).abs().sum()),
        ("eigvals 3D", base_3d, lambda t: t, lambda t: torch.linalg.eigvals(t).abs().sum()),
        ("eigvals 4D", base_4d, lambda t: t, lambda t: torch.linalg.eigvals(t).abs().sum()),
        ("eigvals .mT", base_3d, lambda t: t.mT, lambda t: torch.linalg.eigvals(t).abs().sum()),
        ("eigvals slice", base_3d, lambda t: t[:, 1:7, 1:7], lambda t: torch.linalg.eigvals(t).abs().sum()),
    ]
    for name, base, vf, loss_fn in cases:
        bc = base.detach().clone().requires_grad_()
        bh = base.detach().clone().to(hpu).requires_grad_()
        yc, yh = loss_fn(vf(bc)), loss_fn(vf(bh))
        yc.backward()
        yh.backward()
        torch.hpu.synchronize()
        assert bh.grad.dtype == torch.float32, (name, bh.grad.dtype)
        fwd_rel = _max_err(yh, yc) / max(yc.detach().abs().max().item(), 1.0)
        bwd_rel = _max_err(bh.grad, bc.grad) / max(bc.grad.abs().max().item(), 1e-12)
        assert fwd_rel < 1e-4, (name, "fwd", fwd_rel)
        assert bwd_rel < 1e-4, (name, "bwd", bwd_rel)

    # float64 -> complex128 on HPU (Issue #6): host memory, computed on the CPU in float64. The reference input is
    # the float64 tensor as the card holds it (float32 precision), so results match exactly.
    x64 = torch.randn(4, 4, dtype=torch.float64, device=hpu)
    x64_held = x64.cpu()
    for name, fn in (
        ("eig(f64)", lambda t: torch.linalg.eig(t)[0]),
        ("eigvals(f64)", lambda t: torch.linalg.eigvals(t)),
    ):
        wh, wc = fn(x64), fn(x64_held)
        assert wh.dtype == torch.complex128 and wh.device.type == "hpu", (name, wh.dtype, wh.device)
        assert _max_err(wh, wc) < 1e-12, (name, _max_err(wh, wc))
    print("real float32 eig/eigvals fwd+bwd & float64 -> complex128: OK")


def test_complex_d2h_nondense_and_h2d_broadcast_copy_on_hpu():
    if _skip_if_lazy("test_complex_d2h_nondense_and_h2d_broadcast_copy_on_hpu"):
        return
    H.enable()
    torch.manual_seed(0)
    hpu = torch.device("hpu")
    z = torch.complex(torch.randn(3, 5, 5), torch.randn(3, 5, 5))
    z5 = torch.complex(torch.randn(2, 3, 4, 5, 6), torch.randn(2, 3, 4, 5, 6))

    for vf in (lambda t: t.diagonal(dim1=-2, dim2=-1), lambda t: t.narrow(-1, 1, 3), lambda t: t[:, ::2, 1::2]):
        for conj_src in (False, True):
            ref = torch.full_like(z, 99.0 + 99.0j)
            got = torch.full_like(z, 99.0 + 99.0j)
            vf(ref).copy_(vf(z).conj() if conj_src else vf(z))
            vf(got).copy_(vf(z.to(hpu)).conj() if conj_src else vf(z.to(hpu)))
            assert _max_err(got, ref) == 0.0
        ref_r = torch.full_like(z.real, 99.0)
        got_r = torch.full_like(z.real, 99.0)
        vf(ref_r).copy_(vf(z).real)
        vf(got_r).copy_(vf(z.to(hpu)))
        assert _max_err(got_r, ref_r) == 0.0

    ref5 = torch.full_like(z5.real, 99.0)
    got5 = torch.full_like(z5.real, 99.0)
    ref5.copy_(z5.real)
    got5.copy_(z5.to(hpu))
    assert _max_err(got5, ref5) == 0.0

    for src in (torch.tensor(1.5 - 2.25j), torch.tensor(3.25), torch.complex(torch.randn(5), torch.randn(5))):
        bc = torch.zeros_like(z)
        bh = torch.zeros_like(z, device=hpu)
        bc[1:].copy_(src)
        bh[1:].copy_(src)
        torch.hpu.synchronize()
        assert _max_err(bh, bc) == 0.0
    print("complex64 D2H non-dense & H2D broadcast copy_: OK")


def test_lazy_mode_guard_and_real_safety():
    H.enable()
    hpu = torch.device("hpu")
    # Real-only float32 fwd+bwd must stay untouched in both eager and lazy modes.
    xc = torch.randn(4, 4, requires_grad=True)
    xh = xc.detach().clone().to(hpu).requires_grad_()
    (xc @ xc.T).sum().backward()
    (xh @ xh.T).sum().backward()
    if is_lazy():
        htcore.mark_step()
    torch.hpu.synchronize()
    assert _max_err(xh.grad, xc.grad) < 1e-4

    # Float-pair core API works in both eager and lazy modes.
    sig = torch.randn(2, 64, device=hpu)
    spec = H.rfft(sig)
    stft = H.stft(sig, 32, 16)
    if is_lazy():
        htcore.mark_step()
    torch.hpu.synchronize()
    assert spec.shape == (2, 33, 2) and stft.shape[0] == 2

    if not is_lazy():
        # Verify the lazy-mode guard is wired and toggleable via set_lazy_mode: in lazy mode it refuses complex64
        # under torch.inference_mode (complex64 itself runs there, on lazy views).
        torch.ops.hpu_complex.set_lazy_mode(True)
        try:
            try:
                with torch.inference_mode():
                    torch.zeros(2, 2, dtype=torch.complex64, device=hpu)
            except RuntimeError as exc:
                assert "PT_HPU_LAZY_MODE=0" in str(exc)
            else:
                raise AssertionError("Expected RuntimeError when lazy guard is active")
        finally:
            torch.ops.hpu_complex.set_lazy_mode(False)
        print("lazy mode guard (toggled) + real safety: OK")
        return

    real_pair = torch.randn(4, 4, 2, device=hpu)
    real_re = torch.randn(4, 4, device=hpu)
    real_im = torch.randn(4, 4, device=hpu)
    real_sig = torch.randn(4, 32, device=hpu)
    real_mat = torch.randn(4, 4, device=hpu)
    cpu_c64 = torch.randn(4, 4, dtype=torch.complex64)
    htcore.mark_step()
    for name, fn in (
        ("zeros_c64", lambda: torch.zeros(4, 4, dtype=torch.complex64, device=hpu)),
        ("ones_c64", lambda: torch.ones(4, 4, dtype=torch.complex64, device=hpu)),
        ("to_hpu_then_cpu", lambda: cpu_c64.to(hpu).cpu()),
        ("view_as_complex", lambda: torch.view_as_complex(real_pair)),
        ("torch.complex", lambda: torch.complex(real_re, real_im)),
        ("torch.fft.rfft", lambda: torch.fft.rfft(real_sig)),
        ("torch.linalg.eig(f32)", lambda: torch.linalg.eig(real_mat)),
        ("torch.linalg.eigvals(f32)", lambda: torch.linalg.eigvals(real_mat)),
    ):
        out = fn()  # complex64 runs in lazy mode now (lazy_check.py compares the values with the CPU)
        htcore.mark_step()
        for t in out if isinstance(out, tuple) else (out,):
            assert torch.isfinite(torch.view_as_real(t.cpu())).all(), name
    try:
        with torch.inference_mode():
            torch.fft.rfft(real_sig)
    except RuntimeError as exc:
        assert "PT_HPU_LAZY_MODE=0" in str(exc), str(exc)
    else:
        raise AssertionError("Expected RuntimeError for complex64 under inference_mode in lazy mode")
    print("lazy mode complex64 + guard + real safety: OK")
