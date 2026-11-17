"""Eager float-only loop (CBraMod-sized), with or without the extension: python3 float_tax.py [--ext]. Prints ms/iter."""
import sys
import time

import habana_frameworks.torch.core  # noqa: F401
import torch

if "--ext" in sys.argv:
    import hpu_complex

    hpu_complex.enable()
hpu = torch.device("hpu")
torch.manual_seed(0)
x = torch.randn(100, 90 * 30, 200, device=hpu)
w = torch.randn(200, 200, device=hpu)
buf = torch.zeros(100, 90 * 30, 200, device=hpu)


def step():
    y = torch.matmul(x, w) + x
    y = y.abs() * 0.5 - 1.0
    buf[:, :1000].copy_(y[:, :1000])
    m = torch.ones(100, 90, dtype=torch.bool).to(hpu, non_blocking=True)
    return (buf.sum() + m.sum()).item()


for _ in range(5):
    step()
times = []
for _ in range(30):
    torch.hpu.synchronize()
    t = time.perf_counter()
    step()
    torch.hpu.synchronize()
    times.append(time.perf_counter() - t)
times.sort()
print(f"FLOAT_TAX ext={'--ext' in sys.argv} median {1e3 * times[len(times) // 2]:.2f} ms/iter", flush=True)
