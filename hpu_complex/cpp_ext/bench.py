"""Benchmark: ext FFT vs CPU round-trip vs python core, on Gaudi.
Run in pod: python3 hpu_complex/cpp_ext/bench.py
"""
import pathlib
import sys
import time

import habana_frameworks.torch.core  # noqa: F401
import torch

here = pathlib.Path(__file__).parent
sys.path.insert(0, str(here.parent.parent))
import hpu_complex as H
from hpu_complex.mode import MODE

# One load, through the package: a second copy of the extension under another
# name would register a second PrivateUse1 fallback and abort the process.
H.enable()
MODE.__enter__()
print("BENCH LOADED", flush=True)


def timeit(fn, warmup=3, reps=10):
    for _ in range(warmup):
        fn()
    torch.hpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.hpu.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3  # ms


def cpu_roundtrip(x):
    return torch.fft.rfft(x.cpu()).to("hpu")


print(f"{'shape':>14} {'ext':>9} {'compiled':>9} {'cpu-trip':>9}  (ms/call)")
for rows, n in [(1024, 512), (1024, 1024), (1024, 1920), (256, 6000),
                (8192, 512)]:
    x = torch.randn(rows, n, device="hpu")
    xc = x.cpu()
    def cell(fn):
        try:
            return f"{timeit(fn):9.2f}"
        except Exception as e:
            return f"{type(e).__name__[:9]:>9}"
    cf = torch.compile(lambda t: torch.fft.rfft(t), backend="hpu_backend")
    try:
        cf(x)
    except Exception:
        pass
    print(f"{rows}x{n:<7} {cell(lambda: torch.fft.rfft(x))} "
          f"{cell(lambda: cf(x))} {cell(lambda: cpu_roundtrip(xc))}",
          flush=True)

# STFT front-end (the training workload from the issue)
x = torch.randn(64, 16000, device="hpu")
def cell5(fn):
    try:
        return f"{timeit(fn, reps=5):.2f}"
    except Exception as e:
        return f"{type(e).__name__}: {str(e)[:60]}"
print(f"stft 64x16000 n_fft=512: ext={cell5(lambda: torch.stft(x, 512, 256, return_complex=True))} ms  "
      f"pycore-mag={cell5(lambda: H.stft_magnitude(x, 512, 256))} ms", flush=True)
print("BENCH_DONE")

# accuracy at large n (dense float32 accumulation)
for n in (4096, 6000):
    x = torch.randn(8, n)
    got = torch.fft.rfft(x.to("hpu"))
    ref = torch.fft.rfft(x.double())
    rel = ((torch.view_as_real(got).cpu() - torch.view_as_real(ref).float()).abs().max() / ref.abs().max()).item()
    print(f"accuracy n={n}: rel_err={rel:.2e}", flush=True)
print("ACC_DONE")
