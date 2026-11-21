"""Habana bug reproducer (no hpu_complex): contiguous() of a permuted rank-6 op result reads wrong values.

A Habana op on a permuted rank-6 view keeps the permuted strides in its result. That result copies right to the
CPU, but its contiguous(), .float() of a bool one and r + 0 return wrong values (max diff 1-4), with no error.
Measured on Gaudi2, bridge 1.21.4 / torch 2.6, eager mode (PT_HPU_LAZY_MODE=0):

    python3 repro/rank6_permuted_contiguous.py

Workaround (hpu_complex's dense_copy): view the storage as one contiguous block in stride order and permute it
back, then call contiguous() on that view.
"""
import habana_frameworks.torch.core  # noqa: F401
import torch

hpu = torch.device("hpu")
x = torch.randn(2, 2, 3, 2, 2, 3, 2, 2, generator=torch.Generator().manual_seed(0))
sel = lambda t: t[1, 1:].permute(0, 2, 1, 3, 4, 5, 6)[..., 0]  # noqa: E731  rank 6, offset, not dense


def row(name, got, want):
    d = (got.cpu().double() - want.double()).abs().max().item()
    print(f"{name:44s} maxdiff {d:.3g} {'OK' if d == 0 else 'WRONG'}", flush=True)
    return d == 0


r_h, r_c = sel(x.to(hpu)) * 1, sel(x) * 1
print("strides of the HPU result", r_h.stride(), "CPU", r_c.stride())
ok = [
    row("(sel(x) * 1) to the CPU", r_h, r_c),
    row("(sel(x) * 1) + 0", r_h + 0, r_c + 0),
    row("(sel(x) * 1).contiguous()", r_h.contiguous(), r_c.contiguous()),
    row("(sel(x) > 0).contiguous()", (sel(x.to(hpu)) > 0).contiguous(), (sel(x) > 0).contiguous()),
    row("(sel(x) > 0).float()", (sel(x.to(hpu)) > 0).float(), (sel(x) > 0).float()),
    row("sel(x).contiguous() (the view itself)", sel(x.to(hpu)).contiguous(), sel(x).contiguous()),
]
order = sorted(range(r_h.dim()), key=lambda d: -r_h.stride(d))
block = r_h.as_strided([r_h.size(d) for d in order], [r_h.stride(d) for d in order], r_h.storage_offset())
row("workaround: stride-order block, permuted back", block.permute([order.index(d) for d in range(6)]).contiguous(),
    r_c.contiguous())
print("HABANA_BUG_REPRODUCED" if not all(ok) else "NOT_REPRODUCED")
