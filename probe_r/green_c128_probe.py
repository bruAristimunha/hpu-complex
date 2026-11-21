"""Probe: which op of a Green train step yields a complex128 HPU tensor (dtype trace with a TorchDispatchMode)."""
import copy

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

from spd_diag import protocol as P
from spd_diag.cache import open_cache
from spd_diag.models import MODELS

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
seen = {}


class Trace(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if any(isinstance(t, torch.Tensor) and t.dtype in (torch.complex128, torch.float64) and t.device.type == "hpu"
               for t in tree_leaves(out)):
            ins = tuple((t.dtype, tuple(t.shape), t.device.type) for t in tree_leaves((args, kwargs))
                        if isinstance(t, torch.Tensor))
            seen.setdefault(str(func), ins)
        return out


with Trace():
    loss = torch.nn.functional.cross_entropy(model(X.to(hpu)), y.to(hpu))
    loss.backward()
for k, v in list(seen.items())[:15]:
    print("C128/F64", k, v, flush=True)
print("host_in_use", torch.ops.hpu_complex.host_in_use())
