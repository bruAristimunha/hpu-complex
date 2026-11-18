"""BIOT's spectrum on one Gaudi, compiled fwd+bwd: torch.stft(...).abs() via the extension vs the real-DFT matmuls."""
import math
import time

import habana_frameworks.torch.core  # noqa: F401
import torch

import hpu_complex

hpu_complex.enable()
hpu = torch.device("hpu")
torch.manual_seed(0)
N_FFT, HOP = 200, 100
x = torch.randn(3000, 6000, device=hpu)  # (batch x channels, time), a third of a micro-batch


def plugin(t):
    return torch.stft(t, N_FFT, HOP, center=False, onesided=True, return_complex=True,
                      window=torch.ones(N_FFT, device=t.device)).abs()


k = torch.arange(N_FFT // 2 + 1, dtype=torch.float64)
j = torch.arange(N_FFT, dtype=torch.float64)
ang = (torch.outer(j, k) % N_FFT) * (2 * math.pi / N_FFT)
cos, sin = torch.cos(ang).float().to(hpu), (-torch.sin(ang)).float().to(hpu)


def real_dft(t):
    frames = t.unfold(-1, N_FFT, HOP)
    return torch.sqrt((frames @ cos).square() + (frames @ sin).square()).transpose(-1, -2)


for name, f in (("plugin stft.abs", plugin), ("real-DFT matmuls", real_dft)):
    g = torch.compile(f, backend="hpu_backend")
    xs = x.clone().requires_grad_()
    times = []
    for i in range(8):
        torch.hpu.synchronize()
        t0 = time.perf_counter()
        g(xs).sum().backward()
        torch.hpu.synchronize()
        if i >= 3:
            times.append(time.perf_counter() - t0)
    print(f"STFT_BENCH {name:18s} {1e3 * sorted(times)[len(times) // 2]:.1f} ms fwd+bwd", flush=True)
