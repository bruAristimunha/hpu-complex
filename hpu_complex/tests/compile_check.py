"""Compiled rfft(...).abs() on one Gaudi, forward + backward vs CPU, printing the full error if tracing fails."""
import traceback

import habana_frameworks.torch.core  # noqa: F401
import torch

import hpu_complex

hpu_complex.enable()
hpu = torch.device("hpu")
torch.manual_seed(0)
x = torch.randn(6, 30, 200)


def f(t):
    return torch.fft.rfft(t, norm="forward").abs()


try:
    g = torch.compile(f, backend="hpu_backend")
    xh = x.to(hpu).requires_grad_()
    xc = x.clone().requires_grad_()
    yh, yc = g(xh), f(xc)
    yh.sum().backward()
    yc.sum().backward()
    torch.hpu.synchronize()
    print("COMPILED out err", (yh.detach().cpu() - yc.detach()).abs().max().item(),
          "grad err", (xh.grad.cpu() - xc.grad).abs().max().item(), flush=True)
except Exception:
    traceback.print_exc()
    print("COMPILED FAILED", flush=True)
