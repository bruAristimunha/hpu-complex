"""Round 2b detail: device mismatch per dtype, .to(hpu) strides, save/load of a host float64 (one Gaudi)."""
import io
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


part = sys.argv[1]
if part.startswith("mismatch"):
    dt = {"mismatch_c128": torch.complex128, "mismatch_c64": torch.complex64, "mismatch_f32": torch.float32}[part]
    x = torch.ones(3, 4, dtype=dt).to(H)
    try:
        y = x + torch.ones(3, 4, dtype=dt)
        sync()
        print(f"{part}: NO RAISE, result {y.device} {y.dtype} sum {y.sum().item()}", flush=True)
    except RuntimeError as e:
        print(f"{part}: raises {str(e).splitlines()[0][:120]}", flush=True)
if part == "strides":
    g = torch.Generator().manual_seed(0)
    base = torch.randn(6, 8, dtype=torch.complex128, generator=g)
    for name, x in {"slice_step_mT": base[1::2, ::3].mT, "mT": base.mT, "offset": base[1:, 2:], "expanded": base[:1].expand(3, 8)}.items():
        y = x.to(H)
        print(f"strides {name}: cpu {x.stride()} clone {x.clone().stride()} hpu {y.stride()} values_equal "
              f"{torch.equal(y.cpu(), x)}", flush=True)
        f = x.to(torch.complex64).to(H)
        print(f"  complex64 same view: hpu {f.stride()}", flush=True)
if part == "save":
    g = torch.Generator().manual_seed(0)
    x = torch.randn(3, 4, dtype=torch.complex128, generator=g)
    r = x.to(H).abs() * (1 + 1e-12)
    ref = x.abs() * (1 + 1e-12)
    buf = io.BytesIO()
    torch.save({"r": r, "z": x.to(H)}, buf)
    buf.seek(0)
    b = torch.load(buf)
    tag = lambda t: torch._C._dispatch_keys(t).has(torch._C.DispatchKey.PrivateUse1)  # noqa: E731
    print(f"save r before: {r.dtype} {r.device} tagged {tag(r)} err {(r.cpu() - ref).abs().max().item():.3g}", flush=True)
    print(f"load r after: {b['r'].dtype} {b['r'].device} tagged {tag(b['r'])} err {(b['r'].cpu() - ref).abs().max().item():.3g}", flush=True)
    print(f"load z after: {b['z'].dtype} {b['z'].device} tagged {tag(b['z'])} err {(b['z'].cpu() - x).abs().max().item():.3g}", flush=True)
    m = torch.nn.Module()
    m.register_buffer("r", r)
    sd = io.BytesIO()
    torch.save(m.state_dict(), sd)
    sd.seek(0)
    m.load_state_dict(torch.load(sd))  # copy_ into the host buffer: exact?
    print(f"state_dict round trip into host buffer: err {(m.r.cpu() - ref).abs().max().item():.3g}", flush=True)
