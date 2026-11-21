"""Probe: complex64 householder_product backward on the card (eager) against complex128 on the CPU, 12 seeds."""
import statistics

import torch

import hpu_complex

hpu_complex.enable()
hpu = torch.device("hpu")
fns = (lambda a, tau: torch.linalg.householder_product(a, tau).abs().sum(),
       lambda a, tau: torch.linalg.householder_product(a, tau)[:, :4, :4].real.sum())


def grads(a0, t0, fn, dev, dtype):
    a = a0.detach().to(dtype).to(dev).clone().requires_grad_()
    t = t0.detach().to(dtype).to(dev).clone().requires_grad_()
    fn(a, t).backward()
    return a.grad.cpu().to(torch.complex128), t.grad.cpu().to(torch.complex128)


errs = {"card": [], "cpu c64": []}
for seed in range(12):
    g = torch.Generator().manual_seed(seed)
    a0, t0 = torch.randn(4, 8, 6, dtype=torch.complex64, generator=g), torch.randn(4, 6, dtype=torch.complex64, generator=g)
    for fn in fns:
        ra, rt = grads(a0, t0, fn, "cpu", torch.complex128)
        for name, dev in (("card", hpu), ("cpu c64", "cpu")):
            ga, gt = grads(a0, t0, fn, dev, torch.complex64)
            errs[name].append(max(((ga - ra).abs().max() / ra.abs().max()).item(),
                                  ((gt - rt).abs().max() / rt.abs().max()).item()))
for k, v in errs.items():
    print(f"HH {k}: median {statistics.median(v):.2e} max {max(v):.2e}", flush=True)
