"""Detail of round-2 breaks: which part differs (one Gaudi)."""
import os
import sys
import warnings

import habana_frameworks.torch.core as htcore
import torch

import hpu_complex

hpu_complex.enable()
warnings.simplefilter("ignore")
H = torch.device("hpu")
LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"


def sync():
    if LAZY:
        htcore.mark_step()
    torch.hpu.synchronize()


def rnd(*shape, seed=0):
    return torch.randn(shape, dtype=torch.complex128, generator=torch.Generator().manual_seed(seed))


def show(name, got, ref):
    got = got.detach().cpu().resolve_conj().resolve_neg()
    err = (got - ref).abs().max().item() if got.numel() else 0.0
    print(f"{name}: maxerr {err:.3g}", flush=True)
    if err > 1e-12:
        print("   hpu", got.flatten()[:6].tolist(), "\n   cpu", ref.flatten()[:6].tolist(), flush=True)


which = sys.argv[1] if len(sys.argv) > 1 else "all"
if which in ("all", "grad"):
    terms = {
        "real[1:].pow(2)": lambda z: z.real[1:].pow(2).sum(),
        "imag.t().exp()": lambda z: z.imag.t().exp().sum(),
        "imag.exp()": lambda z: z.imag.exp().sum(),
        "view_as_real[...,0]": lambda z: torch.view_as_real(z)[..., 0].sum(),
        "view_as_real[...,1]": lambda z: torch.view_as_real(z)[..., 1].sum(),
        "real.sum()": lambda z: z.real.sum(),
        "imag.sum()": lambda z: z.imag.sum(),
        "imag*2 sum": lambda z: (z.imag * 2).sum(),
    }
    for name, f in terms.items():
        grads = []
        for dev in ("cpu", H):
            z = rnd(3, 4).to(dev).requires_grad_()
            f(z).backward()
            sync() if dev != "cpu" else None
            grads.append(z.grad)
        show("grad " + name, grads[1], grads[0])
if which in ("all", "mask"):
    x = rnd(3, 4)
    m = x.real > 0
    steps = {
        "z[m] = 1j": lambda z, d: z.__setitem__(m.to(d), 1j),
        "z[idx] *= 2": lambda z, d: z.__setitem__(torch.tensor([0, 2], device=d),
                                                   z[torch.tensor([0, 2], device=d)] * 2),
        "masked_fill_": lambda z, d: z.masked_fill_((x.imag > 1).to(d), 0),
        "index_add_": lambda z, d: z.index_add_(0, torch.tensor([1, 1], device=d), rnd(2, 4, seed=3).to(d)),
    }
    for name, f in steps.items():
        outs = []
        for d in ("cpu", H):
            z = x.clone().to(d)
            f(z, d)
            sync() if d != "cpu" else None
            outs.append(z)
        show("mask " + name, outs[1], outs[0])
    for d in ("cpu", H):
        z = x.clone().to(d)
        r = z[m.to(d)]
        print("read z[m]", d, r.detach().cpu()[:3].tolist() if r.numel() else r.shape)
if which in ("all", "nan"):
    x = torch.randn(4, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(0)).float().double()
    x[0, 0] = float("nan")
    print("eigvals nan cpu", torch.linalg.eigvals(x).tolist())
    print("eigvals nan hpu", torch.linalg.eigvals(x.to(H)).cpu().tolist())
    print("eigvals nan hpu-input .cpu()", x.to(H).cpu().tolist()[0])
if which == "mask2":  # the probe's bool_mask_and_index, step by step on one tensor
    x = rnd(3, 4)
    m = x.real > 0
    zs = {}
    for d in ("cpu", H):
        z = x.clone().to(d)
        trace = []
        z[m.to(d)] = 1j
        trace.append(z.clone())
        z[torch.tensor([0, 2], device=d)] *= 2
        trace.append(z.clone())
        z.masked_fill_((x.imag > 1).to(d), 0)
        trace.append(z.clone())
        z.index_add_(0, torch.tensor([1, 1], device=d), rnd(2, 4, seed=3).to(d))
        trace.append(z.clone())
        trace += [z[m.to(d)], z[[0, 2]]]
        sync() if d != "cpu" else None
        zs[str(d)] = trace
    for i, name in enumerate(["z[m]=1j", "z[idx]*=2", "masked_fill_", "index_add_", "read z[m]", "read z[[0,2]]"]):
        show("seq " + name, zs["hpu"][i], zs["cpu"][i])
