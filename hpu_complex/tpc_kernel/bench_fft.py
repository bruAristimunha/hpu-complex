"""TPC fused FFT vs GEMM extension vs CPU round-trip."""
import pathlib
import sys
import time

import torch
from torch.utils.cpp_extension import load

sys.path.insert(0, "/voyager/ceph/users/bpinto/hpu-probe")
exec(open("/voyager/ceph/users/bpinto/hpu-probe/hpu_complex/tpc_kernel/run_fft_test.py").read().split("torch.manual_seed")[0])
import hpu_complex

hpu_complex.enable()


def timeit(fn, warmup=2, reps=5):
    for _ in range(warmup):
        fn()
    torch.hpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.hpu.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3


print(f"{'shape':>12} {'tpc':>9} {'ext':>9} (ms/call)", flush=True)
for rows, n in [(1024, 512), (1024, 1024), (8192, 512), (256, 4096)]:
    x = torch.randn(rows, n, device="hpu")
    twid = make_twiddles(n)
    t_tpc = timeit(lambda: torch.ops.custom_op.fft_r2c(x, twid))
    t_ext = timeit(lambda: torch.fft.rfft(x))
    print(f"{rows}x{n:<6} {t_tpc:9.2f} {t_ext:9.2f}", flush=True)
print("TPC_BENCH_DONE")
