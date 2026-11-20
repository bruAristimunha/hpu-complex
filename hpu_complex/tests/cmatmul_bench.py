"""On a Gaudi card: complex64 bmm on the card (hpu_complex's GEMM kernel) vs the CPU round trip it replaces.
    PT_HPU_LAZY_MODE=0 python3 hpu_complex/tests/cmatmul_bench.py [--calls 50]

Same process, same inputs. The baseline is what hpu_complex's CPU fallback did for bmm: copy both operands to the
CPU, bmm there, copy the result back (``torch.bmm(a.cpu(), b.cpu()).to(hpu)``). Mean ms of ``--calls`` synced
calls after warm-up, outputs kept alive; bridge ``cpu_fallback`` ops and "runs on the CPU" warnings per variant.
"""
import argparse
import json
import time
import warnings

import habana_frameworks.torch.core  # noqa: F401
import habana_frameworks.torch.hpu as hthpu
import torch
from habana_frameworks.torch.hpu.metrics import metric_global

import hpu_complex


def fb():
    return dict(dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {})


def timed(fn, calls):
    for _ in range(5):
        fn()
    hthpu.synchronize()
    f0 = fb()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        keep = []
        t = time.perf_counter()
        for _ in range(calls):
            keep.append(fn())
        hthpu.synchronize()
        ms = (time.perf_counter() - t) / calls * 1e3
    f1 = fb()
    return {"ms": round(ms, 3), "bridge_fallback": {k: f1[k] - f0.get(k, 0) for k in f1 if f1[k] - f0.get(k, 0)},
            "cpu_warnings": sorted({str(w.message)[:90] for w in caught if "runs on the CPU" in str(w.message)})}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calls", type=int, default=50)
    parser.add_argument("--sweep", action="store_true", help="square (32, n, n, n) and (b, 22, 22, 22) shapes")
    args = parser.parse_args()
    hpu_complex.enable()
    hpu = torch.device("hpu")
    torch.manual_seed(0)
    shapes = ((32, 22, 22, 22), (32, 22, 22, 1000), (220, 22, 400, 22), (32, 64, 64, 64))
    if args.sweep:
        shapes = tuple((32, n, n, n) for n in (8, 16, 22, 32, 40, 48, 64)) + tuple(
            (b, 22, 22, 22) for b in (1, 8, 128, 512)) + ((1, 64, 64, 64), (1, 128, 128, 128))
    for shape in shapes:
        b, m, k, n = shape
        a = torch.complex(torch.randn(b, m, k), torch.randn(b, m, k)).to(hpu)
        c = torch.complex(torch.randn(b, k, n), torch.randn(b, k, n)).to(hpu)
        ref = torch.bmm(a.cpu(), c.cpu())
        got = torch.bmm(a, c)
        hthpu.synchronize()
        row = {"shape (b,m,k,n)": shape,
               "rel_err": float(((got.cpu() - ref).abs().max() / ref.abs().max()).item()),
               "hpu_gemm": timed(lambda: torch.bmm(a, c), args.calls),
               "cpu_roundtrip": timed(lambda: torch.bmm(a.cpu(), c.cpu()).to(hpu), args.calls)}
        row["macs"] = b * m * k * n
        row["speedup"] = round(row["cpu_roundtrip"]["ms"] / row["hpu_gemm"]["ms"], 2)
        print("RESULT " + json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
