"""Which (batch shape, order) give wrong results from Habana's float32 cholesky_fwd, cholesky_inverse_fwd and qr_fwd on
Gaudi2 (bridge 1.21.4), no hpu-ops, no hpu-complex. Also the two workarounds for cholesky: a size-1 dim before the
matrix ((..., b, 1, n, n)) and one identity matrix appended to the batch.
    PT_HPU_LAZY_MODE=0|1 python3 repro/habana_linalg_shape_sweep.py
One line per case that is wrong (NaN, or error > 1e-3 relative), then a summary line per op."""
import habana_frameworks.torch.core as htcore
import torch

LAZY = __import__("os").environ.get("PT_HPU_LAZY_MODE", "0") != "0"


def spd(batch, n, seed):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(*batch, n, n, generator=g)
    return A @ A.mT / n + torch.eye(n)


def rel(got, ref):
    got = got.cpu()
    if got.isnan().any():
        return float("nan")
    return ((got - ref).abs().max() / ref.abs().max()).item()


def cholesky(S, how):
    h = S.to("hpu")
    if how == "unsqueeze":
        L, info = torch.linalg.cholesky_ex(h.unsqueeze(-3))
        L = L.squeeze(-3)
    elif how == "pad":
        eye = torch.eye(S.shape[-1], device="hpu").expand(*S.shape[:-3], 1, *S.shape[-2:])
        L, info = torch.linalg.cholesky_ex(torch.cat([h, eye], -3))
        L = L.narrow(-3, 0, S.shape[-3])
    else:
        L, info = torch.linalg.cholesky_ex(h)
    if LAZY:
        htcore.mark_step()
    return L


def main():
    orders = [2, 4, 8, 16, 32, 44, 64]
    bad = {}
    total = {}
    for n in orders:
        shapes = sorted({(b,) for b in (1, 4, 32, n - 1, n, n + 1, 2 * n) if b >= 1}
                        | {(1, n), (2, n), (n, 1), (2, max(1, n // 2)), (n, n)})
        for batch in shapes:
            S = spd(batch, n, n)
            L = torch.linalg.cholesky(S)
            cases = {
                "cholesky": lambda: rel(cholesky(S, ""), L),
                "cholesky_unsqueeze": lambda: rel(cholesky(S, "unsqueeze"), L) if len(batch) >= 1 else 0.0,
                "cholesky_pad": lambda: rel(cholesky(S, "pad"), L),
                "cholesky_inverse": lambda: rel(torch.cholesky_inverse(L.to("hpu")), torch.cholesky_inverse(L)),
                "qr_R_abs": lambda: rel(torch.linalg.qr(S.to("hpu")).R.abs(), torch.linalg.qr(S).R.abs()),
            }
            for name, f in cases.items():
                total[name] = total.get(name, 0) + 1
                try:
                    e = f()
                except Exception as ex:  # noqa: BLE001
                    e = f"error {repr(ex)[:120]}"
                if isinstance(e, str) or not e <= 1e-3:
                    bad.setdefault(name, []).append((batch, n))
                    print(f"WRONG {name} batch={batch} n={n} err={e}", flush=True)
    for name in total:
        print(f"SUMMARY lazy={LAZY} {name}: {len(bad.get(name, []))}/{total[name]} wrong: {bad.get(name, [])}",
              flush=True)


if __name__ == "__main__":
    main()
