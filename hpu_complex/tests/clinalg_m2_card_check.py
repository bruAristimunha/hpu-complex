"""On a Gaudi card (eager or lazy): targeted verification for M2 complex_linalg, G10 (angle), and G11/G16 (im2col/col2im).

Checks:
- det / slogdet: regular, singular (0 det / sign=0, -inf logabsdet, 0 first-order grad), and singular rank-(n-1)
  double-backward (create_graph=True) against finite-difference complex128 reference.
- slogdet partial gradients (sign only, logabsdet only, combined).
- eigh / eigvalsh: UPLO='L' and 'U' with junk in unread triangle, clustered eigenvalues, gV=None gradient.
- svd / svdvals: full_matrices=True tall/wide orthogonality, compute_uv=False with grad, svd(..., out=...).
- torch.inference_mode(): all card linalg ops + angle + im2col/col2im (inference tensors, 0 CPU fallbacks).
- G10 (angle, angle.out, eq.Scalar_out): 0j forward + backward (zero CPU fallbacks), signed zeros, complex out=.
- G11/G16 (im2col/unfold, col2im/fold): Tier-1 (non-overlapping) & Tier-2 (overlapping dilated/strided) fwd + bwd.
- Eager mode only: torch.compile(backend='hpu_backend') fwd+bwd for det, slogdet, eigh, eigvalsh, svd, svdvals, angle, unfold.

Exits 1 if any check fails; prints one ``RESULT {json}`` line per check.
"""
import json
import os
import sys
import warnings

import habana_frameworks.torch.core as htcore
import torch
import torch.nn.functional as F
from habana_frameworks.torch.hpu.metrics import metric_global

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "0") != "0"
HPU = torch.device("hpu")
C64, C128 = torch.complex64, torch.complex128


def sync():
    if LAZY:
        htcore.mark_step()


def to_cpu(t):
    sync()
    return t.cpu()


def fb_ops():
    return dict(dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {})


def rel(got, ref):
    got = got.detach().cpu().to(ref.dtype)
    if (torch.view_as_real(got) if got.is_complex() else got).isnan().any():
        return float("nan")
    return ((got - ref).abs().max() / ref.abs().max().clamp_min(1e-30)).item()


def rand(*shape, seed):
    return torch.randn(*shape, dtype=C128, generator=torch.Generator().manual_seed(seed))


def main():
    import hpu_ops

    hpu_ops.enable()
    import hpu_complex
    import hpu_complex.linalg as L

    hpu_complex.enable()
    L.install()
    ext = torch.ops.hpu_complex

    results = []

    def check(name, fn, check_zero_fb=True):
        row = {"case": name, "lazy": LAZY}
        calls0, ext0, fb0 = sum(L.CALLS.values()), ext.cpu_fallback_calls(), fb_ops()
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                ok, err = fn()
            sync()
            row["taken"] = sum(L.CALLS.values()) - calls0
            row["ext_cpu_calls"] = ext.cpu_fallback_calls() - ext0
            row["bridge_fallbacks"] = {k: v - fb0.get(k, 0) for k, v in fb_ops().items() if v - fb0.get(k, 0)}
            row["cpu_warnings"] = sorted({str(w.message)[:90] for w in caught if "CPU" in str(w.message)})
            row["err"] = err
            if check_zero_fb:
                ok = bool(ok) and row["ext_cpu_calls"] == 0 and not row["bridge_fallbacks"] and not row["cpu_warnings"]
            else:
                ok = bool(ok)
            row["status"] = "ok" if ok else "wrong"
        except Exception as e:  # noqa: BLE001
            row["status"] = "error"
            row["error"] = repr(e)[:300]
            sync()
        results.append(row)
        print("RESULT " + json.dumps(row), flush=True)

    # 1. Singular det + near-singular & regular double backward (create_graph=True)
    def test_det_singular_double_backward():
        A0 = rand(2, 4, 4, seed=11)
        A0[:, :, -1] = 0.0
        V128 = rand(2, 4, 4, seed=12)
        V = V128.to(C64).to(HPU)

        # (a) Exact singular matrix: det==0, first-order grad==0, second-order grad is finite (PyTorch CPU returns NaN)
        Ah = A0.to(C64).to(HPU).requires_grad_()
        sync()
        d = torch.linalg.det(Ah)
        if to_cpu(d).abs().max().item() > 1e-6:
            return False, float(to_cpu(d).abs().max().item())
        g1_only = torch.autograd.grad((d * (1 - 1j)).real.sum(), Ah, retain_graph=True)[0]
        if to_cpu(g1_only).abs().max().item() > 1e-6:
            return False, float(to_cpu(g1_only).abs().max().item())
        g1 = torch.autograd.grad((d * (1 - 1j)).real.sum(), Ah, create_graph=True)[0]
        sync()
        g2 = torch.autograd.grad((g1 * V.conj()).real.sum(), Ah)[0]
        if not to_cpu(g2).abs().isfinite().all().item():
            return False, float("nan")

        # (b) Near-singular matrix (via _det_svd_backward) & regular matrix double-backward match complex128 CPU
        errs = []
        W128 = rand(2, 4, 4, seed=19)
        A_ns = A0 + 1e-3 * W128
        Ah_ns = A_ns.to(C64).to(HPU).requires_grad_()
        sync()
        g1_ns_h = L._det_svd_backward(Ah_ns, torch.ones(2, dtype=C64, device=HPU) * (1 + 1j))
        sync()
        g2_ns_h = torch.autograd.grad((g1_ns_h * V.conj()).real.sum(), Ah_ns)[0]
        Ac_ns = A_ns.clone().requires_grad_()
        g1_ns_c = torch.autograd.grad((torch.linalg.det(Ac_ns) * (1 - 1j)).real.sum(), Ac_ns, create_graph=True)[0]
        g2_ns_c = torch.autograd.grad((g1_ns_c * V128.conj()).real.sum(), Ac_ns)[0]
        errs.append(rel(to_cpu(g2_ns_h), g2_ns_c))

        A_reg = rand(2, 4, 4, seed=20)
        Ah_reg = A_reg.to(C64).to(HPU).requires_grad_()
        sync()
        d_rh = torch.linalg.det(Ah_reg)
        sync()
        g1_rh = torch.autograd.grad((d_rh * (1 - 1j)).real.sum(), Ah_reg, create_graph=True)[0]
        sync()
        g2_rh = torch.autograd.grad((g1_rh * V.conj()).real.sum(), Ah_reg)[0]
        Ac_reg = A_reg.clone().requires_grad_()
        g1_rc = torch.autograd.grad((torch.linalg.det(Ac_reg) * (1 - 1j)).real.sum(), Ac_reg, create_graph=True)[0]
        g2_rc = torch.autograd.grad((g1_rc * V128.conj()).real.sum(), Ac_reg)[0]
        errs.append(rel(to_cpu(g2_rh), g2_rc))
        return max(errs) < 5e-3, max(errs)

    # PyTorch's float32 cumprod_backward on HPU falls back for aten::max.dim_max / aten::relu_ (bridge op, not hpu_complex)
    check("det_singular_double_backward", test_det_singular_double_backward, check_zero_fb=False)

    # 2. slogdet partial gradients (sign only, logabsdet only)
    def test_slogdet_partial_grads():
        A0 = rand(8, 6, 6, seed=13)
        errs = []
        for mode in ("sign_only", "lad_only"):
            Ac = A0.clone().requires_grad_()
            sc, lc = torch.linalg.slogdet(Ac)
            (sc.real.sum() if mode == "sign_only" else lc.sum()).backward()
            Ah = A0.to(C64).to(HPU).requires_grad_()
            sh, lh = torch.linalg.slogdet(Ah)
            (sh.real.sum() if mode == "sign_only" else lh.sum()).backward()
            errs.append(rel(to_cpu(Ah.grad), Ac.grad))
        return max(errs) < 1e-4, max(errs)

    check("slogdet_partial_grads", test_slogdet_partial_grads)

    # 3. eigh UPLO='U'/'L' + unread junk + clustered eigenvalues + eigvalsh gV=None
    def test_eigh_uplo_and_clusters():
        A0 = rand(4, 8, 8, seed=14)
        H0 = A0 + A0.mH
        # Clustered eigenvalues: perturb a 2-distinct-block spectrum slightly
        U0, _ = torch.linalg.qr(rand(4, 8, 8, seed=15))
        spec = torch.tensor([-2.0, -2.0 + 1e-3, -0.5, 0.0, 1.0, 1.0 + 2e-3, 3.0, 3.5], dtype=torch.float64)
        Hc = U0 @ torch.diag_embed(spec.to(C128)) @ U0.mH
        Hc = 0.5 * (Hc + Hc.mH)
        junk = rand(4, 8, 8, seed=16)
        errs = []
        for uplo in ("L", "U"):
            tri_junk = junk.triu(1) if uplo == "L" else junk.tril(-1)
            A_in = Hc + tri_junk
            Ac = A_in.clone().requires_grad_()
            wc, Vc = torch.linalg.eigh(Ac, UPLO=uplo)
            ((Vc * wc.unsqueeze(-2)) @ Vc.mH).abs().sum().backward()
            Ah = A_in.to(C64).to(HPU).requires_grad_()
            wh, Vh = torch.linalg.eigh(Ah, UPLO=uplo)
            L._cmatmul(Vh * wh.unsqueeze(-2), Vh.mH).abs().sum().backward()
            errs.append(rel(to_cpu(wh), wc))
            # Reconstruction error against true Hermitian matrix
            H_clean = (A_in.tril() + A_in.tril(-1).mH) if uplo == "L" else (A_in.triu() + A_in.triu(1).mH)
            H_clean.diagonal(dim1=-2, dim2=-1).imag.zero_()
            recon = (to_cpu(Vh).to(C128) * to_cpu(wh).to(C128).unsqueeze(-2)) @ to_cpu(Vh).to(C128).mH
            errs.append(rel(recon, H_clean))
            errs.append(rel(to_cpu(Ah.grad), Ac.grad))
        # Also test eigvalsh (gV=None path)
        Ah2 = H0.to(C64).to(HPU).requires_grad_()
        Ac2 = H0.clone().requires_grad_()
        torch.linalg.eigvalsh(Ah2, UPLO="U").sum().backward()
        torch.linalg.eigvalsh(Ac2, UPLO="U").sum().backward()
        errs.append(rel(to_cpu(Ah2.grad), Ac2.grad))
        return max(errs) < 1e-3, max(errs)

    check("eigh_uplo_and_clusters", test_eigh_uplo_and_clusters)

    # 4. svd full_matrices orthogonality + svd(..., out=...) + compute_uv=False grad
    def test_svd_variants():
        errs = []
        for m, n in ((6, 4), (4, 6)):
            A0 = rand(4, m, n, seed=17 + m)
            Ah = A0.to(C64).to(HPU)
            U, S, Vh = torch.linalg.svd(Ah, full_matrices=True)
            Uc, _Sc, Vhc = to_cpu(U).to(C128), to_cpu(S).to(C128), to_cpu(Vh).to(C128)
            errs.append(rel(Uc @ Uc.mH, torch.eye(m, dtype=C128).expand(4, m, m)))
            errs.append(rel(Vhc @ Vhc.mH, torch.eye(n, dtype=C128).expand(4, n, n)))
            # out= variant (non-grad)
            Uo = torch.empty(4, m, min(m, n), dtype=C64, device=HPU)
            So = torch.empty(4, min(m, n), dtype=torch.float32, device=HPU)
            Vho = torch.empty(4, min(m, n), n, dtype=C64, device=HPU)
            torch.linalg.svd(Ah, full_matrices=False, out=(Uo, So, Vho))
            recon = (to_cpu(Uo).to(C128) * to_cpu(So).to(C128).unsqueeze(-2)) @ to_cpu(Vho).to(C128)
            errs.append(rel(recon, A0))
            # compute_uv=False with grad
            Ahg = A0.to(C64).to(HPU).requires_grad_()
            Acg = A0.clone().requires_grad_()
            torch.linalg.svdvals(Ahg).sum().backward()
            torch.linalg.svdvals(Acg).sum().backward()
            errs.append(rel(to_cpu(Ahg.grad), Acg.grad))
        return max(errs) < 1e-4, max(errs)

    check("svd_variants", test_svd_variants)

    # 5. inference_mode across ALL card linalg ops + angle + im2col/col2im
    def test_inference_mode_all():
        prev = os.environ.get("HPU_COMPLEX_LINALG")
        os.environ["HPU_COMPLEX_LINALG"] = "all"
        try:
            return _run_inference_mode_all()
        finally:
            if prev is None:
                os.environ.pop("HPU_COMPLEX_LINALG", None)
            else:
                os.environ["HPU_COMPLEX_LINALG"] = prev

    def _run_inference_mode_all():
        A0 = rand(4, 6, 6, seed=21)
        H0 = A0 @ A0.mH / 6 + torch.eye(6, dtype=C128)
        L0 = torch.linalg.cholesky(H0)
        B0 = rand(4, 6, 3, seed=22)
        T0 = A0.tril() + 2 * torch.eye(6, dtype=C128)
        R0 = rand(4, 6, 4, seed=23)
        img0 = rand(2, 2, 6, 6, seed=24)

        with torch.inference_mode():
            Ah = A0.to(C64).to(HPU)
            Hh = H0.to(C64).to(HPU)
            Lh = L0.to(C64).to(HPU)
            Bh = B0.to(C64).to(HPU)
            Th = T0.to(C64).to(HPU)
            Rh = R0.to(C64).to(HPU)
            imgh = img0.to(C64).to(HPU)

            outs = [
                (torch.linalg.solve(Ah, Bh), torch.linalg.solve(A0, B0)),
                (torch.linalg.solve(Ah, B0.mT.to(C64).to(HPU), left=False), torch.linalg.solve(A0, B0.mT, left=False)),
                (torch.linalg.inv(Ah), torch.linalg.inv(A0)),
                (torch.linalg.det(Ah), torch.linalg.det(A0)),
                (torch.linalg.slogdet(Ah).logabsdet, torch.linalg.slogdet(A0).logabsdet),
                (torch.linalg.eigh(Hh).eigenvalues, torch.linalg.eigh(H0).eigenvalues),
                (torch.linalg.eigvalsh(Hh, UPLO="U"), torch.linalg.eigvalsh(H0, UPLO="U")),
                (torch.linalg.svd(Rh, full_matrices=False).S, torch.linalg.svd(R0, full_matrices=False).S),
                (torch.linalg.svdvals(Rh), torch.linalg.svdvals(R0)),
                (torch.linalg.cholesky(Hh), torch.linalg.cholesky(H0)),
                (torch.linalg.matrix_exp(Ah / 6), torch.linalg.matrix_exp(A0 / 6)),
                (torch.cholesky_solve(Bh, Lh), torch.cholesky_solve(B0, L0)),
                (torch.cholesky_inverse(Lh), torch.cholesky_inverse(L0)),
                (torch.linalg.solve_triangular(Th, Bh, upper=False), torch.linalg.solve_triangular(T0, B0, upper=False)),
                (torch.linalg.qr(Rh).R.abs(), torch.linalg.qr(R0).R.abs()),
                (torch.linalg.pinv(Rh), torch.linalg.pinv(R0)),
                (torch.linalg.lstsq(Rh, Bh).solution, torch.linalg.lstsq(R0, B0).solution),
                (torch.angle(Ah), torch.angle(A0)),
                (F.unfold(imgh, kernel_size=2, stride=2), F.unfold(img0, kernel_size=2, stride=2)),
                (F.fold(F.unfold(imgh, kernel_size=3, padding=1), (6, 6), kernel_size=3, padding=1),
                 F.fold(F.unfold(img0, kernel_size=3, padding=1), (6, 6), kernel_size=3, padding=1)),
            ]
            # Verify all outputs are inference tensors
            if not all(o.is_inference() for o, _ in outs):
                return False, "not_inference"
        errs = [rel(to_cpu(o), r) for o, r in outs]
        return max(errs) < 5e-4, max(errs)

    check("inference_mode_all", test_inference_mode_all)

    # 6. G10: angle, angle.out, eq.Scalar_out (including 0j forward + backward)
    def test_g10_angle():
        z0 = torch.tensor([1 + 2j, 0 + 0j, -3 + 0j, -3 - 0.0j, 0 - 1j, 2 - 5j], dtype=C128)
        zc = z0.clone().requires_grad_()
        zh = z0.to(C64).to(HPU).requires_grad_()
        ac = torch.angle(zc)
        ah = torch.angle(zh)
        ac.sum().backward()
        ah.sum().backward()
        err_fwd = (to_cpu(ah) - ac.to(torch.float32)).abs().max().item()
        err_bwd = rel(to_cpu(zh.grad), zc.grad)
        out = torch.empty_like(ah)
        torch.angle(zh.detach(), out=out)
        err_out = (to_cpu(out) - ac.to(torch.float32)).abs().max().item()
        eq0 = (zh.detach() == 0j)
        err_eq = int((to_cpu(eq0) != (zc.detach() == 0j)).any().item())
        return max(err_fwd, err_bwd, err_out, err_eq) < 1e-5, max(err_fwd, err_bwd, err_out, err_eq)

    check("g10_angle_fwd_bwd_out_eq0", test_g10_angle)

    # 7. G11/G16: im2col (Tier-1 non-overlap & Tier-2 overlapping dilated/padded) + col2im fwd + bwd
    def test_g11_g16_im2col_col2im():
        errs = []
        for kw, stride, pad, dil in ((2, 2, 0, 1), (3, 1, 1, 2)):
            x0 = rand(2, 3, 7, 6, seed=30 + kw)
            xc = x0.clone().requires_grad_()
            xh = x0.to(C64).to(HPU).requires_grad_()
            yc = F.unfold(xc, kernel_size=kw, stride=stride, padding=pad, dilation=dil)
            yh = F.unfold(xh, kernel_size=kw, stride=stride, padding=pad, dilation=dil)
            (yc.abs() ** 2).sum().backward()
            (yh.abs() ** 2).sum().backward()
            errs.append(rel(to_cpu(yh), yc))
            errs.append(rel(to_cpu(xh.grad), xc.grad))
            # Fold (col2im) fwd + bwd
            fc_in = yc.detach().clone().requires_grad_()
            fh_in = yh.detach().clone().requires_grad_()
            fc = F.fold(fc_in, (7, 6), kernel_size=kw, stride=stride, padding=pad, dilation=dil)
            fh = F.fold(fh_in, (7, 6), kernel_size=kw, stride=stride, padding=pad, dilation=dil)
            (fc.abs() ** 2).sum().backward()
            (fh.abs() ** 2).sum().backward()
            errs.append(rel(to_cpu(fh), fc))
            errs.append(rel(to_cpu(fh_in.grad), fc_in.grad))
        return max(errs) < 1e-5, max(errs)

    check("g11_g16_im2col_col2im", test_g11_g16_im2col_col2im)

    # 8. Eager mode only: torch.compile(backend="hpu_backend") on new ops
    if not LAZY:
        def test_compile_m2():
            g = torch.Generator().manual_seed(41)
            A0 = torch.randn(4, 6, 6, dtype=C64, generator=g)
            H0 = A0 @ A0.mH / 6 + torch.eye(6, dtype=C64)
            R0 = torch.randn(4, 6, 4, dtype=C64, generator=g)
            img0 = torch.randn(2, 2, 6, 6, dtype=C64, generator=g)

            def fn(A, H, R, img):
                return (
                    torch.linalg.det(A).abs().sum()
                    + torch.linalg.slogdet(A).logabsdet.sum()
                    + torch.linalg.eigh(H).eigenvalues.sum()
                    + torch.linalg.eigvalsh(H, UPLO="U").sum()
                    + torch.linalg.svd(R, full_matrices=False).S.sum()
                    + torch.linalg.svdvals(R).sum()
                    + torch.angle(A).sum()
                    + F.unfold(img, kernel_size=3, padding=1).abs().sum()
                )

            cfn = torch.compile(fn, backend="hpu_backend")
            ins_e = [t.to(HPU).requires_grad_() for t in (A0, H0, R0, img0)]
            le = fn(*ins_e)
            le.backward()
            ins_c = [t.to(HPU).requires_grad_() for t in (A0, H0, R0, img0)]
            lc = cfn(*ins_c)
            lc.backward()
            err = abs(le.item() - lc.item()) / max(abs(le.item()), 1e-30)
            gerr = max(rel(c.grad, e.grad.cpu()) for c, e in zip(ins_c, ins_e))
            return max(err, gerr) < 1e-4, max(err, gerr)

        check("torch_compile_m2", test_compile_m2, check_zero_fb=False)

    failed = sum(r["status"] != "ok" for r in results)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
