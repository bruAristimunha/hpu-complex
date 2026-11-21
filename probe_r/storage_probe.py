"""Probe: the storage cases of the review fixes, one at a time, on the card (eager or lazy)."""
import os
import warnings

import habana_frameworks.torch.core as htcore
import torch

import hpu_complex

hpu_complex.enable()
hpu = torch.device("hpu")
LAZY = os.environ.get("PT_HPU_LAZY_MODE", "0") != "0"
warnings.simplefilter("ignore")


def sync():
    if LAZY:
        htcore.mark_step()
    torch.hpu.synchronize()


def run(name, fn):
    try:
        print(name, "->", fn(), flush=True)
    except Exception as e:  # noqa: BLE001
        print(name, "-> RAISE", type(e).__name__, str(e).splitlines()[0][:160], flush=True)


def grow():
    base = torch.tensor([11, 22, 33], dtype=torch.complex64, device=hpu)
    torch.neg(torch.tensor([7, 8, 9, 10], dtype=torch.complex64, device=hpu), out=base[1:])
    sync()
    return base.cpu().tolist()


def grow_plain():
    out = torch.tensor([5, 6], dtype=torch.complex64, device=hpu)
    torch.neg(torch.tensor([7, 8, 9, 10], dtype=torch.complex64, device=hpu), out=out)
    sync()
    return out.cpu().tolist()


def set_detach():
    y = torch.tensor([1, 2], dtype=torch.complex64, device=hpu)
    alias = y.view(-1)
    y.set_()
    sync()
    a = alias.cpu().tolist()
    y.resize_(3)
    sync()
    y.fill_(99)
    sync()
    return a, alias.cpu().tolist(), y.cpu().tolist()


def failed_set():
    x = torch.tensor([2 + 0j], device=hpu)
    try:
        x.set_(torch.zeros(1, dtype=torch.complex64, device=hpu).untyped_storage(), 0, (2,), (1,))
    except RuntimeError as e:
        print("   raised:", str(e)[:80])
    sync()
    return x.cpu().tolist()


run("grow out=base[1:]", grow)
run("grow plain out", grow_plain)
run("set_() detach", set_detach)
run("failed set_", failed_set)
