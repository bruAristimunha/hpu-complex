"""On a Gaudi card: complex64 linalg on the card (hpu_complex.linalg) against the CPU route, eager or lazy.
    PT_HPU_LAZY_MODE=0|1 python3 hpu_complex/tests/clinalg_card_check.py [--ops solve,inv] [--n 4,16,22,32] [--batch 32]

For each op and size: the forward and the gradient of a gauge-invariant loss against CPU complex128 on the same values,
next to CPU complex64's own error (pass: within 10x of it, plus 1e-6 for exact cases); hpu_complex's CPU-fallback
calls, the bridge's cpu_fallback ops and "runs on the CPU" warnings during the card call; and the mean of --reps synced
calls (forward, and forward + backward) on the card and on the CPU route (hpu_complex.linalg uninstalled), in
alternating rounds in this process. One ``RESULT {json}`` line per case; exits 1 if any case fails. A case the
module declines (beyond the real kernels' sizes, or, with --default-routes, an op not routed in this mode) fails
with taken = 0.
"""
import argparse
import json
import os
import statistics
import sys
import time
import warnings

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu
import torch
from habana_frameworks.torch.hpu.metrics import metric_global

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "0") != "0"
HPU = torch.device("hpu")
C64, C128 = torch.complex64, torch.complex128


def sync():
    if LAZY:
        htcore.mark_step()
    hthpu.synchronize()


def fb_ops():
    return dict(dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {})


def rand(*shape, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, dtype=C128, generator=g)


def hpd(b, n, seed):
    A = rand(b, n, n, seed=seed)
    return A @ A.mH / n + torch.eye(n, dtype=C128)


def tri(b, n, seed, upper=False):
    A = rand(b, n, n, seed=seed)
    T = A.triu() if upper else A.tril()
    return T + 2 * torch.eye(n, dtype=C128)  # condition about 10-100


def cases(b, n):
    """name -> (function of complex tensors, inputs (complex128 CPU), loss of the outputs)."""
    k = max(1, n // 2)
    L = torch.linalg.cholesky(hpd(b, n, 3))
    out = {
        "solve": (lambda A, B: torch.linalg.solve(A, B), [rand(b, n, n, seed=1), rand(b, n, 3, seed=2)]),
        "solve_left_false": (lambda A, B: torch.linalg.solve(A, B, left=False), [rand(b, n, n, seed=1),
                                                                                rand(b, 3, n, seed=2)]),
        "inv": (lambda A: torch.linalg.inv(A), [rand(b, n, n, seed=1)]),
        "solve_triangular": (lambda A, B: torch.linalg.solve_triangular(A, B, upper=True),
                             [tri(b, n, 4, upper=True), rand(b, n, 3, seed=2)]),
        "solve_triangular_right_unit": (
            lambda A, B: torch.linalg.solve_triangular(A, B, upper=False, left=False, unitriangular=True),
            [tri(b, n, 4), rand(b, 3, n, seed=2)]),
        "cholesky": (lambda A: torch.linalg.cholesky(A), [hpd(b, n, 3)]),
        "cholesky_upper": (lambda A: torch.linalg.cholesky(A, upper=True), [hpd(b, n, 3)]),
        "cholesky_solve": (lambda B, L: torch.cholesky_solve(B, L), [rand(b, n, 3, seed=2), L]),
        "matrix_exp": (lambda A: torch.linalg.matrix_exp(A), [rand(b, n, n, seed=1) / n ** 0.5]),
        "householder_product": (lambda A, tau: torch.linalg.householder_product(A, tau),
                                [*torch.geqrf(rand(b, n, max(1, n - 1), seed=5))]),
        # eigh: compared and differentiated through (w, |V|^2), which do not depend on the eigenvectors' phases
        "eigh": (lambda A: _gauge_free(torch.linalg.eigh(A)), [_herm(b, n, 6)]),
        "eigvalsh": (lambda A: torch.linalg.eigvalsh(A), [_herm(b, n, 6)]),
        "pinv": (lambda A: torch.linalg.pinv(A), [rand(b, n, max(1, n - 2), seed=7)]),
        "cholesky_inverse": (lambda L: torch.cholesky_inverse(L), [L]),
        "cholesky_inverse_upper": (lambda U: torch.cholesky_inverse(U, upper=True), [L.mH.contiguous()]),
        "cholesky_solve_upper": (lambda B, U: torch.cholesky_solve(B, U, upper=True),
                                 [rand(b, n, 3, seed=2), L.mH.contiguous()]),
        "solve_triangular_lower_left": (lambda A, B: torch.linalg.solve_triangular(A, B, upper=False),
                                        [tri(b, n, 4), rand(b, n, 3, seed=2)]),
        # qr: compared and differentiated through (Q Q^H, |R|), which do not depend on the signs of diag(R)
        "qr": (lambda A: _qr_gauge_free(torch.linalg.qr(A)), [rand(b, n, n, seed=9)]),
        "qr_tall": (lambda A: _qr_gauge_free(torch.linalg.qr(A)), [rand(b, n, max(1, n // 2), seed=9)]),
        "hpd_slogdet": (_hpd_logabsdet, [hpd(b, n, 3)]),
        "det": (lambda A: torch.linalg.det(A), [rand(b, n, n, seed=1)]),
        "slogdet": (lambda A: torch.linalg.slogdet(A), [rand(b, n, n, seed=1)]),
        "slogdet_phase": (lambda A: _slogdet_phase(torch.linalg.slogdet(A)), [rand(b, n, n, seed=1)]),
        "svd": (lambda A: _svd_gauge_free(torch.linalg.svd(A, full_matrices=False)), [rand(b, n, max(1, n - 2), seed=7)]),
        "svd_full": (lambda A: _svd_gauge_free(torch.linalg.svd(A, full_matrices=True)),
                     [rand(b, n, max(1, n - 2), seed=7)]),
        "svd_wide": (lambda A: _svd_gauge_free(torch.linalg.svd(A, full_matrices=False)),
                     [rand(b, max(1, n - 2), n, seed=7)]),
        "svd_wide_full": (lambda A: _svd_gauge_free(torch.linalg.svd(A, full_matrices=True)),
                          [rand(b, max(1, n - 2), n, seed=7)]),
        "svdvals": (lambda A: torch.linalg.svdvals(A), [rand(b, n, max(1, n - 2), seed=7)]),
        "lstsq": (lambda A, B: torch.linalg.lstsq(A, B).solution, [rand(b, n, max(1, n - 2), seed=7),
                                                                   rand(b, n, 3, seed=8)]),
    }
    del k
    return out


def _slogdet_phase(res):
    return (res.sign * (1 + 2j)).real + res.logabsdet


def _svd_gauge_free(res):
    U, S, Vh = res
    k = S.shape[-1]
    return (U[..., :k] * S.unsqueeze(-2)) @ Vh[..., :k, :], S


def _qr_gauge_free(qr):
    Q, R = qr
    return Q @ Q.mH, R.abs()


def _hpd_logabsdet(A):
    """hpu_complex.linalg.hpd_slogdet on the card (installed); torch.linalg.slogdet elsewhere (CPU, CPU route)."""
    import hpu_complex.linalg as L

    if A.device.type == "hpu" and L._UNDO is not None:
        return L.hpd_slogdet(A).logabsdet
    return torch.linalg.slogdet(A).logabsdet


def _herm(b, n, seed):
    A = rand(b, n, n, seed=seed)
    return A + A.mH


def _gauge_free(wv):
    w, V = wv
    return w, V.abs() ** 2


def loss_of(out):
    outs = out if isinstance(out, (tuple, list)) else (out,)
    return sum((o.abs() ** 2).sum() for o in outs if o.is_floating_point() or o.is_complex())


def rel(got, ref):
    got = got.detach().cpu().to(ref.dtype)
    return ((got - ref).abs().max() / ref.abs().max().clamp_min(1e-30)).item()


def run(f, inputs, grad):
    xs = [x.clone().requires_grad_(grad) for x in inputs]
    out = f(*xs)
    if grad:
        loss_of(out).backward()
    return out, [x.grad for x in xs] if grad else None


def time_calls(f, inputs, grad, reps):
    xs = [x.clone().requires_grad_(grad) for x in inputs]
    keep = []

    def once():
        out = f(*xs)
        if grad:
            g = torch.autograd.grad(loss_of(out), xs)
            keep.append(g)
        keep.append(out)

    for _ in range(3):
        once()
    sync()
    t = time.perf_counter()
    for _ in range(reps):
        once()
    sync()
    return (time.perf_counter() - t) / reps * 1e3


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ops", default="")
    parser.add_argument("--n", default="4,16,22,32")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--no-time", action="store_true")
    parser.add_argument("--default-routes", action="store_true",
                        help="only the routes taken by default (hpu_complex.linalg.ROUTES); else every op "
                             "(HPU_COMPLEX_LINALG=all)")
    args = parser.parse_args()
    if not args.default_routes:
        os.environ["HPU_COMPLEX_LINALG"] = "all"

    import hpu_ops

    hpu_ops.enable()
    import hpu_complex
    import hpu_complex.linalg as L

    hpu_complex.enable()
    ext = torch.ops.hpu_complex
    failed = 0
    wanted = set(filter(None, args.ops.split(",")))
    for n in map(int, args.n.split(",")):
        for name, (f, inputs) in cases(args.batch, n).items():
            if wanted and name not in wanted:
                continue
            row = {"op": name, "n": n, "batch": args.batch, "lazy": LAZY}
            try:
                ref, ref_g = run(f, inputs, True)
                c64, c64_g = run(f, [x.to(C64) for x in inputs], True)
                outs = lambda o: o if isinstance(o, (tuple, list)) else (o,)  # noqa: E731
                cpu_err = max(rel(o, r) for o, r in zip(outs(c64), outs(ref)) if r.is_complex() or r.is_floating_point())
                cpu_gerr = max(rel(g, r) for g, r in zip(c64_g, ref_g) if r is not None)
                hin = [x.to(C64).to(HPU) for x in inputs]
                L.install()
                calls0, ext0, fb0 = sum(L.CALLS.values()), ext.cpu_fallback_calls(), fb_ops()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    out, grads = run(f, hin, True)
                    sync()
                row["taken"] = sum(L.CALLS.values()) - calls0
                row["ext_cpu_calls"] = ext.cpu_fallback_calls() - ext0
                fb1 = fb_ops()
                row["bridge_fallbacks"] = {k: v - fb0.get(k, 0) for k, v in fb1.items() if v - fb0.get(k, 0)}
                row["cpu_warnings"] = sorted({str(w.message)[:90] for w in caught if "CPU" in str(w.message)})
                err = max(rel(o, r) for o, r in zip(outs(out), outs(ref)) if r.is_complex() or r.is_floating_point())
                gerr = max(rel(g, r) for g, r in zip(grads, ref_g) if r is not None)
                row.update(err=err, cpu_c64_err=cpu_err, grad_err=gerr, cpu_c64_grad_err=cpu_gerr)
                ok = err <= 10 * cpu_err + 1e-6 and gerr <= 10 * cpu_gerr + 1e-6
                ok = ok and row["taken"] >= 1 and row["ext_cpu_calls"] == 0 and not row["bridge_fallbacks"]
                if not args.no_time:
                    times = {"card_fwd": [], "cpu_route_fwd": [], "card_fwd_bwd": [], "cpu_route_fwd_bwd": []}
                    for r in range(args.rounds):
                        order = [("card", L.install), ("cpu_route", L.uninstall)]
                        for label, switch in (order if r % 2 == 0 else order[::-1]):
                            switch()
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                times[f"{label}_fwd"].append(time_calls(f, hin, False, args.reps))
                                times[f"{label}_fwd_bwd"].append(time_calls(f, hin, True, args.reps))
                    row["ms_median"] = {k: round(statistics.median(v), 3) for k, v in times.items()}
                    row["ms_all"] = {k: [round(x, 3) for x in v] for k, v in times.items()}
                    row["speedup_fwd"] = round(row["ms_median"]["cpu_route_fwd"] / row["ms_median"]["card_fwd"], 2)
                    row["speedup_fwd_bwd"] = round(row["ms_median"]["cpu_route_fwd_bwd"]
                                                   / row["ms_median"]["card_fwd_bwd"], 2)
                L.install()
                row["status"] = "ok" if ok else "wrong"
            except Exception as e:  # noqa: BLE001  (the failure is the result)
                row.update(status="error", error=repr(e)[:300])
                L.install()
            failed += row["status"] != "ok"
            print("RESULT " + json.dumps(row), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
