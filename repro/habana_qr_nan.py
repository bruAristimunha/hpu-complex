"""Habana's float32 linalg.qr on Gaudi2 (bridge 1.21.4), no hpu-ops, no hpu-complex: NaN Q and R when a Householder
column has a zero tail (the identity, a diagonal or triangular matrix, a first column that is a multiple of e_0) or the
matrix is rank-deficient; LAPACK's Householder (tau = 0 for a zero tail) gives the exact factors. Eager and lazy.
    PT_HPU_LAZY_MODE=0|1 python3 repro/habana_qr_nan.py
Workaround in hpu_complex.linalg: the QR of W A for a fixed dense orthogonal W (the same R; Q = W^T Q'); a
rank-deficient matrix stays NaN."""
import os

import habana_frameworks.torch.core as htcore
import torch

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "0") != "0"


def cpu(t):
    if LAZY:
        htcore.mark_step()
    return t.cpu()


cases = {
    "eye(2)": torch.eye(2).expand(3, 2, 2).contiguous(),
    "eye(8)": torch.eye(8).expand(3, 8, 8).contiguous(),
    "diag(2, 3)": torch.diag(torch.tensor([2.0, 3.0])).expand(3, 2, 2).contiguous(),
    "upper triangular 6x6": torch.randn(3, 6, 6).triu() + 3 * torch.eye(6),
    "first column 2 e_0": torch.cat([torch.tensor([[[2.0], [0.0], [0.0], [0.0]]]).expand(3, 4, 1),
                                     torch.randn(3, 4, 3)], -1),
    "ones(4, 4) (rank 1)": torch.ones(3, 4, 4),
    "randn(6, 6) (control)": torch.randn(3, 6, 6),
}
for name, A in cases.items():
    Q, R = torch.linalg.qr(A.to("hpu"))
    Q, R = cpu(Q), cpu(R)
    print(f"{name}: NaN={bool(Q.isnan().any() or R.isnan().any())} |QR - A| = {(Q @ R - A).abs().max().item():.1e}",
          flush=True)
