"""Round 2b detail 4 (one Gaudi): float64 Habana tensor times a Python complex scalar; save/load device in lazy mode."""
import io
import sys
import warnings

import habana_frameworks.torch.core as htcore
import torch

import hpu_complex

hpu_complex.enable()
warnings.simplefilter("ignore")
H = torch.device("hpu")
part = sys.argv[1]
x = torch.randn(3, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(0)).float().double()
if part == "f64_times_1j":
    y = x.to(H) * 1j
    htcore.mark_step()
    print(f"{part}: {y.dtype} err {(y.cpu() - x * 1j).abs().max().item():.3g}", flush=True)
if part == "f64_times_2j_tensor":
    y = x.to(H) * torch.tensor(2j, dtype=torch.complex128)
    htcore.mark_step()
    print(f"{part}: {y.dtype} err {(y.cpu() - x * 2j).abs().max().item():.3g}", flush=True)
if part == "f32_times_1j":
    y = x.float().to(H) * 1j
    htcore.mark_step()
    print(f"{part}: {y.dtype} err {(y.cpu() - x.float() * 1j).abs().max().item():.3g}", flush=True)
if part == "f64_to_c128_then_mul":
    y = x.to(H).to(torch.complex128) * 1j
    htcore.mark_step()
    print(f"{part}: {y.dtype} err {(y.cpu() - x * 1j).abs().max().item():.3g}", flush=True)
if part == "save_device":
    for name, t in {"habana_f32": torch.ones(2, device=H), "c128": torch.ones(2, dtype=torch.complex128, device=H),
                    "host_f64": torch.ones(2, dtype=torch.complex128, device=H).real.clone()}.items():
        b = io.BytesIO()
        torch.save(t, b)
        b.seek(0)
        u = torch.load(b)
        print(f"save_device {name}: saved {t.device} loaded {u.device} {u.dtype}", flush=True)
