"""On a Gaudi card (eager or lazy): M6D-complex routes — eig/eigvals past n=32 (ceig_big), TPC qr, pinv/lstsq >32.

One ``RESULT {json}`` line per case; exits 1 if any case fails (wrong values, bridge CPU fallbacks, or
"runs on the CPU" warnings).
"""
import json
import os
import sys
import warnings

import habana_frameworks.torch.core as htcore
import torch
from habana_frameworks.torch.hpu.metrics import metric_global

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "0") != "0"
HPU = torch.device("hpu")
C64, C128 = torch.complex64, torch.complex128
FAILED = []


def sync():
    if LAZY:
        htcore.mark_step()


def fb_ops():
    return dict(dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {})


def check(name, fn):
    fb0 = fb_ops()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ok, err = fn()
        sync()
        fb = {k: v - fb0.get(k, 0) for k, v in fb_ops().items() if v - fb0.get(k, 0)}
        cw = [str(w.message)[:80] for w in caught if "runs on the CPU" in str(w.message)]
        ok = bool(ok) and not fb and not cw
        print("RESULT " + json.dumps({"case": name, "lazy": LAZY, "err": err, "fb": fb, "cw": cw,
                                      "status": "ok" if ok else "wrong"}), flush=True)
        if not ok:
            FAILED.append(name)
    except Exception as e:  # noqa: BLE001
        print("RESULT " + json.dumps({"case": name, "lazy": LAZY, "status": "error", "error": repr(e)[:300]}), flush=True)
        FAILED.append(name)
        sync()


def rel(got, ref):
    got = got.detach().cpu().to(ref.dtype)
    return ((got - ref).abs().max() / ref.abs().max().clamp_min(1e-30)).item()


def main():
    import hpu_ops

    hpu_ops.enable()
    import hpu_complex
    import hpu_complex.linalg as L

    hpu_complex.enable()
    L.install()
    g = torch.Generator().manual_seed(7)

    def eig_case(n):
        A_c = torch.randn(2, n, n, dtype=C64, generator=g)
        Lv, V = torch.linalg.eig(A_c.to(HPU))
        A128, V128, L128 = A_c.to(C128), V.cpu().to(C128), Lv.cpu().to(C128)
        resid = (A128 @ V128 - V128 * L128.unsqueeze(-2)).abs().max().item()
        return resid <= 2e-3 * A_c.abs().max().item(), resid

    def eigvals_case(n):
        A_c = torch.randn(n, n, dtype=C64, generator=g)
        got = torch.linalg.eigvals(A_c.to(HPU)).cpu().to(C128)
        ref = torch.linalg.eigvals(A_c.to(C128))
        k_g = (got.real * 1e6 + got.imag).argsort()
        k_r = (ref.real * 1e6 + ref.imag).argsort()
        err = (got[k_g] - ref[k_r]).abs().max().item()
        return err <= 2e-3, err

    def qr_case(shape, mode, rank_def=False, grad=False):
        A_c = torch.randn(*shape, dtype=C64, generator=g)
        if rank_def:
            A_c[..., :, 1] = 0
        A_h = A_c.to(HPU)
        if grad:
            A_h = A_h.detach().requires_grad_()
        Q, R = torch.linalg.qr(A_h, mode)
        if grad:
            (Q.abs().sum() + R.abs().sum()).backward()
            A_r = A_c.clone().requires_grad_()
            Qr, Rr = torch.linalg.qr(A_r, mode)
            (Qr.abs().sum() + Rr.abs().sum()).backward()
            return torch.allclose(A_h.grad.cpu(), A_r.grad, rtol=1e-3, atol=1e-4), \
                (A_h.grad.cpu() - A_r.grad).abs().max().item()
        if mode == "r":
            return bool(torch.isfinite(R.cpu()).all()), 0.0
        resid = (Q.cpu().to(C128) @ R.cpu().to(C128) - A_c.to(C128)).abs().max().item()
        return resid <= 5e-5 * A_c.abs().max().clamp_min(1.0).item() and bool(torch.isfinite(R.cpu()).all()), resid

    def pinv_case():
        A_c = torch.randn(48, 40, dtype=C64, generator=g)
        got = torch.linalg.pinv(A_c.to(HPU))
        err = rel(got, torch.linalg.pinv(A_c.to(C128)).to(C64))
        return err <= 5e-4, err

    def lstsq_case():
        A_c = torch.randn(48, 40, dtype=C64, generator=g)
        B_c = torch.randn(48, 3, dtype=C64, generator=g)
        got = torch.linalg.lstsq(A_c.to(HPU), B_c.to(HPU)).solution
        err = rel(got, torch.linalg.lstsq(A_c.to(C128), B_c.to(C128)).solution.to(C64))
        return err <= 5e-4, err

    check("eig_n48", lambda: eig_case(48))
    check("eig_n64", lambda: eig_case(64))
    check("eigvals_n48", lambda: eigvals_case(48))
    check("qr_reduced_32x16", lambda: qr_case((32, 16), "reduced"))
    check("qr_complete_24x16", lambda: qr_case((24, 16), "complete"))
    check("qr_r_16x8", lambda: qr_case((16, 8), "r"))
    check("qr_rankdef_16x8", lambda: qr_case((16, 8), "reduced", rank_def=True))
    check("qr_grad_12x8", lambda: qr_case((12, 8), "reduced", grad=True))
    def fwd_routes_case():
        # Forward-only calls of the formerly grad-scoped routes: on the card now, no CPU fallback.
        A_c = torch.randn(8, 6, dtype=C64, generator=g)
        tau_c = torch.randn(6, dtype=C64, generator=g)
        hp = torch.linalg.householder_product(A_c.to(HPU), tau_c.to(HPU))
        err_hp = rel(hp, torch.linalg.householder_product(A_c.to(C128), tau_c.to(C128)).to(C64))
        T_c = A_c[:6].triu() + 6 * torch.eye(6, dtype=C64)
        B_c = torch.randn(6, 3, dtype=C64, generator=g)
        tri = torch.linalg.solve_triangular(T_c.to(HPU), B_c.to(HPU), upper=True)
        err_tri = rel(tri, torch.linalg.solve_triangular(T_c.to(C128), B_c.to(C128), upper=True).to(C64))
        return max(err_hp, err_tri) <= 5e-4, max(err_hp, err_tri)

    def solve_left_false_case():
        A_c = torch.randn(12, 12, dtype=C64, generator=g) + 12 * torch.eye(12, dtype=C64)
        B_c = torch.randn(3, 12, dtype=C64, generator=g)
        got = torch.linalg.solve(A_c.to(HPU), B_c.to(HPU), left=False)
        err = rel(got, torch.linalg.solve(A_c.to(C128), B_c.to(C128), left=False).to(C64))
        return err <= 5e-4, err

    check("pinv_48x40", pinv_case)
    check("lstsq_48x40", lstsq_case)
    check("fwd_routes_householder_tri", fwd_routes_case)
    check("solve_left_false", solve_left_false_case)
    print("FAILED" if FAILED else "ALL_OK", FAILED, flush=True)
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
