"""On a Gaudi card: adversarial inputs for the second round of complex64 linalg on the card (hpu_complex.linalg):
structured QR inputs (Habana's qr NaN trap), extreme diagonal scales, offset / transposed / broadcast views, n = 1,
NaN input to eigvalsh, junk in the unread triangle, and torch.compile(backend="hpu_backend") (declined: torch's route).
    PT_HPU_LAZY_MODE=0|1 python3 hpu_complex/tests/clinalg_adversarial2_check.py
One ``RESULT {json}`` line per case; exits 1 if any case fails."""
import json
import os
import sys
import warnings

import habana_frameworks.torch.core as htcore
import torch
from habana_frameworks.torch.hpu.metrics import metric_global

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "0") != "0"
os.environ["HPU_COMPLEX_LINALG"] = "all"
C64, C128 = torch.complex64, torch.complex128


def fb_ops():
    return dict(dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {})


def cpu(t):
    if LAZY:
        htcore.mark_step()
    return t.cpu()


def rand(*shape, seed):
    return torch.randn(*shape, dtype=C128, generator=torch.Generator().manual_seed(seed))


def rel(got, ref):
    got = got.to(ref.dtype)
    if (torch.view_as_real(got) if got.is_complex() else got).isnan().any():
        return float("nan")
    return ((got - ref).abs().max() / ref.abs().max().clamp_min(1e-30)).item()


def main():
    import hpu_ops

    hpu_ops.enable()
    import hpu_complex
    import hpu_complex.linalg as L

    hpu_complex.enable()
    ext = torch.ops.hpu_complex
    warnings.simplefilter("ignore")
    tri = rand(4, 6, 6, seed=1).tril() + 3 * torch.eye(6, dtype=C128) * (1 + 1j)
    big = rand(3, 9, 9, seed=2)
    hpd = rand(4, 5, 5, seed=3)
    hpd = hpd @ hpd.mH / 5 + torch.eye(5, dtype=C128)
    lfac = torch.linalg.cholesky(hpd)
    eye = torch.eye(4, dtype=C128).expand(2, 4, 4).contiguous()
    cases = {
        "qr_identity": (lambda A: torch.linalg.qr(A), [eye], lambda Q, R, A: rel(Q.to(C128) @ R.to(C128), A), True),
        "qr_triangular_real_diag": (lambda A: torch.linalg.qr(A), [rand(2, 5, 5, seed=4).triu().real.to(C128)
                                                                   + 2 * torch.eye(5, dtype=C128)],
                                    lambda Q, R, A: rel(Q.to(C128) @ R.to(C128), A), True),
        "qr_n1": (lambda A: torch.linalg.qr(A), [rand(3, 1, 1, seed=5)], lambda Q, R, A: rel(Q.to(C128) @ R.to(C128), A),
                  True),
        "tri_scale_1e18": (lambda A, B: torch.linalg.solve_triangular(A, B, upper=False), [tri * 1e18, rand(4, 6, 2, seed=6)],
                           None, False),
        "tri_scale_1e-18": (lambda A, B: torch.linalg.solve_triangular(A, B, upper=False),
                            [tri * 1e-18, rand(4, 6, 2, seed=6)], None, False),
        "tri_offset_transposed_view": (lambda A, B: torch.linalg.solve_triangular(A[1:3, 2:8, 1:7].mT, B, upper=True),
                                       [big.tril() + 3 * torch.eye(9, dtype=C128), rand(6, 3, seed=7)], None, False),
        "tri_broadcast_n1_right": (lambda A, B: torch.linalg.solve_triangular(A, B, upper=True, left=False),
                                   [rand(1, 1, seed=8) + 2, rand(5, 3, 1, seed=9)], None, False),
        "cholesky_solve_broadcast_view": (lambda B, L_: torch.cholesky_solve(B[..., ::2], L_), [rand(2, 1, 5, 6, seed=10), lfac],
                                          None, False),
        # the reference reads only the triangle too, so junk above it changes neither
        "cholesky_inverse_junk_upper": (lambda L_: torch.cholesky_inverse(L_),
                                        [lfac + torch.triu(torch.ones_like(lfac), 1)], None, False),
        "eigvalsh_upper_view": (lambda A: torch.linalg.eigvalsh(A.mT, UPLO="U"), [hpd], None, False),
    }
    failed = 0
    for name, (f, inputs, metric, pair) in cases.items():
        row = {"case": name, "lazy": LAZY}
        try:
            ref = f(*[x.to(C64).to(C128) for x in inputs])
            calls0, ext0, fb0 = sum(L.CALLS.values()), ext.cpu_fallback_calls(), fb_ops()
            out = f(*[x.to(C64).to("hpu") for x in inputs])
            outs = [cpu(o) for o in (out if isinstance(out, tuple) else (out,))]
            row["taken"] = sum(L.CALLS.values()) - calls0
            row["ext_cpu_calls"] = ext.cpu_fallback_calls() - ext0
            row["bridge_fallbacks"] = {k: v - fb0.get(k, 0) for k, v in fb_ops().items() if v - fb0.get(k, 0)}
            if pair:
                err = metric(outs[0], outs[1], inputs[0].to(C64).to(C128))
            else:
                err = rel(outs[0], ref if not isinstance(ref, tuple) else ref[0])
            row["err"] = float(f"{err:.1e}")
            ok = err <= 1e-4 and row["taken"] >= 1 and row["ext_cpu_calls"] == 0 and not row["bridge_fallbacks"]
        except Exception as e:  # noqa: BLE001
            row["error"] = repr(e)[:200]
            ok = False
        row["status"] = "ok" if ok else "wrong"
        failed += not ok
        print("RESULT " + json.dumps(row), flush=True)
    # NaN input to eigvalsh: the others finite (reported: whether hpu-ops' real kernel gives NaN for that matrix)
    H = hpd.to(C64)
    H[0, 2, 1] = complex(float("nan"), 0)
    try:
        w = cpu(torch.linalg.eigvalsh(H.to("hpu")))
        row = {"case": "eigvalsh_nan", "lazy": LAZY, "nan_first": bool(w[0].isnan().any()),
               "others_finite": bool(torch.isfinite(w[1:]).all())}
        ok = row["others_finite"]
    except Exception as e:  # noqa: BLE001
        row, ok = {"case": "eigvalsh_nan", "lazy": LAZY, "error": repr(e)[:200]}, False
    row["status"] = "ok" if ok else "wrong"
    failed += not ok
    print("RESULT " + json.dumps(row), flush=True)
    # torch.compile: declined (tracing), torch's route; values right
    if not LAZY:
        def g(A, B, L_):
            return torch.linalg.solve_triangular(A, B, upper=False) + torch.cholesky_solve(B, L_)

        args = [tri[:, :5, :5].contiguous(), rand(4, 5, 2, seed=11), lfac]
        try:
            got = cpu(torch.compile(g, backend="hpu_backend")(*[x.to(C64).to("hpu") for x in args]))
            err = rel(got, g(*[x.to(C64).to(C128) for x in args]))
            row = {"case": "compiled", "err": float(f"{err:.1e}")}
            ok = err <= 1e-4
        except Exception as e:  # noqa: BLE001
            row, ok = {"case": "compiled", "error": repr(e)[:200]}, False
        row["status"] = "ok" if ok else "wrong"
        failed += not ok
        print("RESULT " + json.dumps(row), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
