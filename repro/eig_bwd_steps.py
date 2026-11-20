"""Probe: where does the card lose precision in linalg_eig's backward of a REAL input? One step at a time.
    PT_HPU_LAZY_MODE=0 python3 repro/eig_bwd_steps.py

Runs torch's linalg_eig_backward formula (FunctionsManual.cpp) as a chain of steps three times, each step from its
own previous result: on the card (float32), on the CPU (float32) and on the CPU (float64). Per step: the card's
error vs the CPU on the SAME inputs (is that op itself precise on the card?), and both float32 chains vs float64
(how much the conditioning amplifies rounding). Then the autograd gradients vs CPU float64.
"""
import habana_frameworks.torch.core  # noqa: F401
import habana_frameworks.torch.hpu as hthpu
import torch

import hpu_complex

hpu_complex.enable()
hpu = torch.device("hpu")
g = torch.Generator().manual_seed(0)
X = torch.randn(32, 22, 22, generator=g)


def chain(x):
    """The steps of linalg_eig_backward for loss = |L|.sum() + |V|.sum(), as (name, value) pairs."""
    L, V = torch.linalg.eig(x)
    gL, gV = torch.sgn(L), torch.sgn(V)
    out = [("eig L", L), ("eig V", V)]
    Vh = V.mH
    VhgV = Vh @ gV
    out.append(("VhgV = V^H gV", VhgV))
    t2 = Vh @ (V * torch.real(VhgV.diagonal(0, -2, -1)).unsqueeze(-2))
    out.append(("V^H (V real(diag))", t2))
    gA = VhgV - t2
    out.append(("difference (cancellation)", gA))
    E = L.conj().unsqueeze(-2) - L.conj().unsqueeze(-1)
    E.diagonal(0, -2, -1).fill_(1)
    gA = gA / E
    gA.diagonal(0, -2, -1).copy_(gL.conj())
    out.append(("/ eigengaps", gA))
    gAVh = gA @ Vh
    out.append(("gA V^H", gAVh))
    res = torch.linalg.solve(Vh, gAVh)
    out.append(("solve(V^H, .)", res))
    out.append(("real(.) = grad", torch.real(res)))
    return out


def rel(a, ref):
    a, ref = a.detach().cpu().to(torch.complex128), ref.detach().to(torch.complex128)
    return ((a - ref).abs().max() / ref.abs().max()).item()


card = chain(X.to(hpu))
hthpu.synchronize()
cpu32 = chain(X)
cpu64 = chain(X.double())
print(f"min |eigengap| {(lambda L: (L.unsqueeze(-1) - L.unsqueeze(-2)).abs().add(torch.eye(22) * 1e9).min().item())(cpu64[0][1]):.2e}")
print(f"{'step':28s} {'card vs CPU f32':>16s} {'card vs f64':>12s} {'CPU f32 vs f64':>15s}")
for (name, h), (_, c), (_, d) in zip(card, cpu32, cpu64):
    print(f"{name:28s} {rel(h, c):16.1e} {rel(h, d):12.1e} {rel(c, d):15.1e}", flush=True)


def grad(x):
    x = x.detach().requires_grad_()
    (lambda w, v: w.abs().sum() + v.abs().sum())(*torch.linalg.eig(x))[()].backward()
    return x.grad


g64, g32, gh = grad(X.double()), grad(X), grad(X.to(hpu))
hthpu.synchronize()
print(f"autograd grad vs CPU float64: card {rel(gh, g64):.1e}   CPU float32 {rel(g32, g64):.1e}")
print(f"card autograd vs card chain {rel(gh, card[-1][1].cpu()):.1e}")

# Over many inputs: is the card's float32 gradient systematically worse than the CPU's, or input by input?
rows = []
for seed in range(20):
    Xs = torch.randn(32, 22, 22, generator=torch.Generator().manual_seed(100 + seed))
    r64, r32, rh = grad(Xs.double()), grad(Xs), grad(Xs.to(hpu))
    hthpu.synchronize()
    rows.append((rel(rh, r64), rel(r32, r64)))
card_e = torch.tensor([r[0] for r in rows])
cpu_e = torch.tensor([r[1] for r in rows])
print(f"20 batches, grad rel err vs float64: card median {card_e.median():.1e} max {card_e.max():.1e} | "
      f"CPU float32 median {cpu_e.median():.1e} max {cpu_e.max():.1e} | card worse in {(card_e > cpu_e).sum().item()}/20")
