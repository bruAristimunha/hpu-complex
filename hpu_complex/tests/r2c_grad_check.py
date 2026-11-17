"""On one Gaudi: gradients of _fft_r2c (the one-GEMM backward) against CPU float64 autograd, per n and norm.

Prints the max abs error relative to the largest reference gradient, for rfft (onesided), fft of a real
input (full spectrum: the c2c path), eager and compiled.
"""
import habana_frameworks.torch.core  # noqa: F401
import torch

import hpu_complex

hpu_complex.enable()
hpu = torch.device("hpu")
torch.manual_seed(0)
worst = 0.0


def grad_error(fn, x, g, hpu_fn=None):
    """Relative max error of d<fn(x), g>/dx on the HPU (float32) against the CPU (float64)."""
    xr = x.clone().requires_grad_()
    (ref,) = torch.autograd.grad(fn(xr), xr, g)
    xh = x.float().to(hpu).requires_grad_()
    (got,) = torch.autograd.grad((hpu_fn or fn)(xh), xh, g.to(torch.complex64).to(hpu))
    return ((got.cpu().double() - ref).abs().max() / ref.abs().max()).item()


for norm in ("backward", "forward", "ortho"):
    def rfft(t, norm=norm):
        return torch.fft.rfft(t, norm=norm)

    for n in (8, 9, 16, 200):
        x = torch.randn(5, 7, n, dtype=torch.float64)
        e_half = grad_error(rfft, x, torch.randn(5, 7, n // 2 + 1, dtype=torch.complex128))
        e_full = grad_error(lambda t: torch.fft.fft(t, norm=norm), x, torch.randn(5, 7, n, dtype=torch.complex128))
        worst = max(worst, e_half, e_full)
        print(f"R2C_GRAD norm={norm:8s} n={n:3d} rfft {e_half:.1e}  fft(real) {e_full:.1e}", flush=True)
    e = grad_error(rfft, torch.randn(5, 7, 200, dtype=torch.float64), torch.randn(5, 7, 101, dtype=torch.complex128),
                   hpu_fn=torch.compile(rfft, backend="hpu_backend"))
    worst = max(worst, e)
    print(f"R2C_GRAD norm={norm:8s} n=200 rfft compiled {e:.1e}", flush=True)
print(f"R2C_GRAD worst relative error {worst:.1e} {'OK' if worst < 1e-5 else 'FAIL'}", flush=True)
