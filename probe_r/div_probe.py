"""Probe: complex division (Smith) at extremes on the card against the exact value rounded to complex64."""
import os

import habana_frameworks.torch.core as htcore
import torch

import hpu_complex

hpu_complex.enable()
hpu = torch.device("hpu")
xd = torch.tensor([1.0, -2.0, 0.5])
wd = torch.tensor([1e20j, 1e-30j, 3e37 + 1e37j, 1e-30 + 1e-30j, 3 + 4j, -2j], dtype=torch.complex64)
zd = torch.tensor([1 + 1j, -3e38 + 1j, 1e-30 - 2e-30j, 2.5, 1j, 7 - 7j], dtype=torch.complex64)
for name, got, ref in (
        ("x/1e20j", xd.to(hpu) / 1e20j, xd.double() / 1e20j), ("x/1e-30j", xd.to(hpu) / 1e-30j, xd.double() / 1e-30j),
        ("z/w", zd.to(hpu) / wd.to(hpu), zd.to(torch.complex128) / wd.to(torch.complex128)),
        ("x0/w", xd[:1].to(hpu) / wd.to(hpu), xd[:1].double() / wd.to(torch.complex128))):
    if os.environ.get("PT_HPU_LAZY_MODE") == "1":
        htcore.mark_step()
    print(name, got.cpu().tolist(), "\n   ref", ref.to(torch.complex64).tolist())
