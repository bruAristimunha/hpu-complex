"""Round 2b detail 5 (one Gaudi): eigvals of a matrix with a NaN, float32 vs float64, vs the CPU."""
import sys
import warnings

import habana_frameworks.torch.core as htcore
import torch

import hpu_complex

hpu_complex.enable()
warnings.simplefilter("ignore")
dt = {"f32": torch.float32, "f64": torch.float64}[sys.argv[1]]
x = torch.randn(4, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(0)).float().to(dt)
x[0, 0] = float("nan")
print(f"{sys.argv[1]} cpu eigvals {torch.linalg.eigvals(x).tolist()}", flush=True)
print(f"{sys.argv[1]} cpu eigvals again {torch.linalg.eigvals(x.clone()).tolist()}", flush=True)
h = x.to("hpu")
print(f"{sys.argv[1]} hpu input back nan at {torch.isnan(h.cpu()).nonzero().tolist()}", flush=True)
L = torch.linalg.eigvals(h)
htcore.mark_step()
print(f"{sys.argv[1]} hpu eigvals {L.cpu().tolist()}", flush=True)
