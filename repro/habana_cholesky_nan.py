"""Habana's float32 linalg.cholesky on Gaudi2 (bridge 1.21.4), no hpu-ops, no hpu-complex: a batch of b matrices of
order n with b == n (the dim before the matrix dims equal to n) returns NaN with info = 0; b != n is right, and so is
the same batch viewed as (b, 1, n, n), the workaround in hpu_complex.linalg.
    PT_HPU_LAZY_MODE=0|1 python3 repro/habana_cholesky_nan.py [--sweep]
--sweep: every b in (1, 4, 32) and n in 1..130 (the cases that are wrong: b == n only). Default: b == n, nearby
b, and a 2-D batch whose total is n."""
import sys

import habana_frameworks.torch.core as htcore  # noqa: F401
import torch


def case(batch, n):
    g = torch.Generator().manual_seed(n)
    A = torch.randn(*batch, n, n, generator=g)
    S = A @ A.mT / n + torch.eye(n)
    ref = torch.linalg.cholesky(S)
    L, info = torch.linalg.cholesky_ex(S.to("hpu"))
    L, info = L.cpu(), info.cpu()
    return bool(L.isnan().any()), int(info.max()), (L - ref).abs().nan_to_num(9.0).max().item()


if "--sweep" in sys.argv:
    grid = [((b,), n) for b in (1, 4, 32) for n in range(1, 131)]
else:
    grid = [((n,), n) for n in (2, 3, 4, 8, 16, 22, 32, 44, 64)] + [((n + 1,), n) for n in (4, 32)] + \
           [((n - 1,), n) for n in (4, 32)] + [((2, 16), 32), ((4, 8), 32), ((2, 2), 4), ((1, 32), 32), ((32, 1), 32), ((4, 1), 4)]
for batch, n in grid:
    nan, info, err = case(batch, n)
    if nan or err > 1e-4 or "--sweep" not in sys.argv:
        print(f"batch={batch} n={n} nan={nan} info_max={info} err={err:.1e}", flush=True)
