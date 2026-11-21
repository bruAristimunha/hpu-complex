"""Fix round 2 (team J): bisect the bool_mask_and_index difference on the card, step by step.

PT_HPU_LAZY_MODE=0|1 python3 tests/probe_fix_r2_j.py
"""
import os
import warnings

import habana_frameworks.torch.core as htcore
import torch

import hpu_complex

warnings.simplefilter("ignore")
hpu_complex.enable()
H = torch.device("hpu")
LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"


def sync():
    if LAZY:
        htcore.mark_step()
    torch.hpu.synchronize()


def err(g, r):
    return (g.detach().cpu().resolve_conj() - r).abs().max().item() if g.numel() else 0.0


g = torch.Generator().manual_seed(0)
x = torch.randn(3, 4, dtype=torch.complex128, generator=g)
src = torch.randn(2, 4, dtype=torch.complex128, generator=torch.Generator().manual_seed(3))
m = x.real > 0
steps = {
    "setitem bool mask = 1j": lambda t, mm, i: t.__setitem__(mm, 1j),
    "getitem int index": lambda t, mm, i: t[i] * 1,
    "index *= 2": lambda t, mm, i: t.__setitem__(i, t[i] * 2),
    "masked_fill_": lambda t, mm, i: t.masked_fill_((x.imag > 1).to(t.device), 0),
    "index_add_": lambda t, mm, i: t.index_add_(0, torch.tensor([1, 1], device=t.device), src.to(t.device)),
    "getitem bool mask": lambda t, mm, i: t[mm] * 1,
    "getitem list": lambda t, mm, i: t[[0, 2]] * 1,
    "index_put_ tensor value": lambda t, mm, i: t.index_put_((mm,), torch.full((int(m.sum()),), 2j,
                                                                             dtype=torch.complex128, device=t.device)),
}
print(f"PROBE lazy={LAZY}", flush=True)
for name, fn in steps.items():
    ref = x.clone()
    rout = fn(ref, m, torch.tensor([0, 2]))
    z = x.to(H)
    hout = fn(z, m.to(H), torch.tensor([0, 2], device=H))
    sync()
    out_err = err(hout, rout) if isinstance(rout, torch.Tensor) and rout is not ref else 0.0
    print(f"STEP {name:28s} self err {err(z, ref):.3g} result err {out_err:.3g}", flush=True)
