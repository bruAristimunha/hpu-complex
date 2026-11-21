"""Probe: aten op counts (nested calls included) of one Green train step, from torch.profiler (CPU events)."""
import json
import sys

import torch
from torch.profiler import ProfilerActivity, profile

sys.argv = [sys.argv[0], "x", "1", "1"]
import green_ab as g  # noqa: E402  (builds the model, defines step())

for _ in range(3):
    g.step()
g.sync()
with profile(activities=[ProfilerActivity.CPU]) as prof:
    g.step()
    g.sync()
counts = {e.key: e.count for e in prof.key_averages() if e.key.startswith("aten::")}
print("PROF " + json.dumps({"total": sum(counts.values()), "counts": counts}))
