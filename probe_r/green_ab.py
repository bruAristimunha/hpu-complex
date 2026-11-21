"""Probe: Green train step time per kernel variant, in one process (variants switched at runtime, rotated rounds)."""
import copy
import json
import os
import statistics
import sys
import time

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu
import torch

from spd_diag import protocol as P
from spd_diag.cache import open_cache
from spd_diag.models import MODELS

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"
P._enable_device("hpu")
hpu = torch.device("hpu")
cache = open_cache("BNCI2014_001")
sd = cache.subject(cache.subjects[0], "X")
X = torch.as_tensor(sd.X[:32], dtype=torch.float32)
X = X / X.std()
y = torch.as_tensor(sd.y[:32]).long()
name = next(k for k in MODELS if k.lower() == "green")
torch.manual_seed(0)
model = copy.deepcopy(P.build(name, X.shape[1], cache.info, None)).to(hpu)
opt = torch.optim.Adam(model.parameters(), 1e-3)
xb, yb = X.to(hpu), y.to(hpu)


def sync():
    if LAZY:
        htcore.mark_step()
    hthpu.synchronize()


def step():
    opt.zero_grad(set_to_none=True)
    out = model(xb)
    out = out[0] if isinstance(out, tuple) else out
    torch.nn.functional.cross_entropy(out, yb).backward()
    if LAZY:
        htcore.mark_step()
    opt.step()
    if LAZY:
        htcore.mark_step()




def main():
    variants = sys.argv[1].split("/")
    steps, rounds = int(sys.argv[2]), int(sys.argv[3])
    for v in variants:
        torch.ops.hpu_complex.set_ablate(v)
        for _ in range(5):
            step()
    sync()
    times = {v: [] for v in variants}
    for r in range(rounds):
        for v in variants[r % len(variants):] + variants[:r % len(variants)]:
            torch.ops.hpu_complex.set_ablate(v)
            for _ in range(2):
                step()
            sync()
            t = time.perf_counter()
            for _ in range(steps):
                step()
            sync()
            times[v].append(round((time.perf_counter() - t) / steps * 1e3, 2))
    print("AB " + json.dumps({"lazy": LAZY, "median": {v: statistics.median(t) for v, t in times.items()}, "all": times}))



if __name__ == "__main__":
    main()
