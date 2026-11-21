"""On a Gaudi card: semantics of the second round of complex64 linalg on the card (hpu_complex.linalg) that
clinalg_card_check.py does not time: cholesky ``info`` against torch's on non-PD inputs, the batch == 2n shapes
(Habana's NaN trap) for cholesky / cholesky_inverse / cholesky_solve / qr / hpd_slogdet, and no CPU fallback.
    PT_HPU_LAZY_MODE=0|1 python3 hpu_complex/tests/clinalg_card2_check.py
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


def hpd(b, n, seed):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(b, n, n, dtype=C128, generator=g)
    return A @ A.mH / n + torch.eye(n, dtype=C128)


def cpu(t):
    if LAZY:
        htcore.mark_step()
    return t.cpu()


def rel(got, ref):
    got = got.to(ref.dtype)
    if torch.view_as_real(got).isnan().any() if got.is_complex() else got.isnan().any():
        return float("nan")
    return ((got - ref).abs().max() / ref.abs().max().clamp_min(1e-30)).item()


def main():
    import hpu_ops

    hpu_ops.enable()
    import hpu_complex
    import hpu_complex.linalg as L

    hpu_complex.enable()
    ext = torch.ops.hpu_complex
    failed = 0
    warnings.simplefilter("ignore")

    def report(row, ok):
        nonlocal failed
        row["status"] = "ok" if ok else "wrong"
        failed += not ok
        print("RESULT " + json.dumps(row), flush=True)

    for b, n in ((32, 4), (32, 16), (8, 4), (16, 8), (64, 32), (3, 1)):
        H = hpd(b, n, n).to(C64)
        bad = H.clone()
        for i in range(min(b, n)):
            bad[i, i, i] = -10.0  # matrix i fails at complex order i + 1
        if b > n:
            bad[n, n - 1, n - 1] = float("nan")
        Lc = torch.linalg.cholesky(H.to(C128))
        B = torch.randn(b, n, 3, dtype=C64)
        calls0, ext0, fb0 = sum(L.CALLS.values()), ext.cpu_fallback_calls(), fb_ops()
        Lh, info_ok = torch.linalg.cholesky_ex(H.to("hpu"))
        _, info_bad = torch.linalg.cholesky_ex(bad.to("hpu"))
        _, info_upper = torch.linalg.cholesky_ex(bad.mH.contiguous().to("hpu"), upper=True)
        inv = torch.cholesky_inverse(Lh)
        X = torch.cholesky_solve(B.to("hpu"), Lh)
        Q, R = torch.linalg.qr(H.to("hpu"))
        lad = L.hpd_slogdet(H.to("hpu")).logabsdet
        w = torch.linalg.eigvalsh(H.to("hpu")) if 2 * n <= 32 else None
        res = {k: cpu(v) for k, v in dict(L=Lh, info_ok=info_ok, info_bad=info_bad, info_upper=info_upper, inv=inv,
                                           X=X, Q=Q, R=R, lad=lad).items()}
        if w is not None:
            res["w"] = cpu(w)
        fb1 = fb_ops()
        row = {"b": b, "n": n, "lazy": LAZY, "taken": sum(L.CALLS.values()) - calls0,
               "ext_cpu_calls": ext.cpu_fallback_calls() - ext0,
               "bridge_fallbacks": {k: v - fb0.get(k, 0) for k, v in fb1.items() if v - fb0.get(k, 0)}}
        ref_bad = torch.linalg.cholesky_ex(bad).info
        ref_upper = torch.linalg.cholesky_ex(bad.mH.contiguous(), upper=True).info
        row["info_bad"] = res["info_bad"].tolist()[:8]
        row["info_matches"] = (res["info_bad"].tolist() == ref_bad.tolist()
                               and res["info_upper"].tolist() == ref_upper.tolist()
                               and not res["info_ok"].any().item())
        errs = {"L": rel(res["L"], Lc), "inv": rel(res["inv"], torch.cholesky_inverse(Lc)),
                "X": rel(res["X"], torch.cholesky_solve(B.to(C128), Lc)),
                "QR": rel(res["Q"].to(C128) @ res["R"].to(C128), H.to(C128)),
                "lad": rel(res["lad"].double(), torch.linalg.slogdet(H.to(C128)).logabsdet)}
        if w is not None:
            errs["w"] = rel(res["w"].double(), torch.linalg.eigvalsh(H.to(C128)))
        row["err"] = {k: float(f"{v:.1e}") for k, v in errs.items()}
        ok = row["info_matches"] and all(v <= 5e-4 for v in errs.values())
        ok = ok and row["ext_cpu_calls"] == 0 and not row["bridge_fallbacks"]
        report(row, ok)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
