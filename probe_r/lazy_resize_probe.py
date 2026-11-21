"""Probe: Habana's own float resize_ in lazy mode, with a view taken before (does the view follow the storage?)."""
import habana_frameworks.torch.core as htcore
import torch

hpu = torch.device("hpu")
r = torch.arange(4, dtype=torch.float32, device=hpu)
v = r[:2]
a = r.as_strided((2,), (1,), 2)
ptr = 0
r.resize_(8)
htcore.mark_step()
print("same StorageImpl data_ptr", ptr == r.untyped_storage().data_ptr(), "prefix", r.cpu()[:4].tolist())
r.fill_(5)
htcore.mark_step()
print("view sees fill:", v.cpu().tolist(), a.cpu().tolist(), "r:", r.cpu().tolist())
o = torch.zeros(2, device=hpu)
w = o[:1]
torch.neg(torch.ones(6, device=hpu), out=o)
htcore.mark_step()
print("out= growth of a float tensor: view", w.cpu().tolist(), "out", o.cpu().tolist())
