"""Detail 2: the zero gradient of z.real[1:] — Habana float64 copies from short-lived host memory."""
import os
import warnings

import habana_frameworks.torch.core as htcore
import torch

import hpu_complex

hpu_complex.enable()
warnings.simplefilter("ignore")
H = torch.device("hpu")
LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"
F64 = torch.float64


def sync():
    if LAZY:
        htcore.mark_step()
    torch.hpu.synchronize()


def rnd(*shape, seed=0):
    return torch.randn(shape, dtype=torch.complex128, generator=torch.Generator().manual_seed(seed))


x = rnd(3, 4)
z = x.to(H)
ref = (x.real[1:] * 2)


def check(name, got, want):
    sync()
    g = got.detach().cpu()
    print(f"{name}: maxerr {(g - want).abs().max().item():.3g} dtype {got.dtype} tagged "
          f"{torch._C._dispatch_keys(got).has(torch._C.DispatchKey.PrivateUse1)}", flush=True)


# 1. Habana float64 slice copy_ from a host temporary that dies right after
h = torch.zeros(3, 4, dtype=F64, device=H)
h[1:].copy_(z.real[1:] * 2)
check("habana f64 slice.copy_(host temp)", h[1:], ref)
# 2. same, temporary kept alive
t = z.real[1:] * 2
h2 = torch.zeros(3, 4, dtype=F64, device=H)
h2[1:].copy_(t)
check("habana f64 slice.copy_(host kept)", h2[1:], ref)
# 3. full copy from temp
h3 = torch.zeros(2, 4, dtype=F64, device=H)
h3.copy_(z.real[1:] * 2)
check("habana f64 copy_(host temp)", h3, ref)
# 4. many temps in a row (allocator reuse)
outs = []
for i in range(20):
    o = torch.empty(2, 4, dtype=F64, device=H)
    o.copy_(z.real[1:] * (i + 1))
    outs.append(o)
sync()
worst = max(((outs[i].cpu() - x.real[1:] * (i + 1)).abs().max().item() for i in range(20)))
print(f"20 copies from host temps: worst maxerr {worst:.3g}", flush=True)
# 5. the autograd path step by step
zz = x.to(H).requires_grad_()
r = zz.real
s = r[1:]
y = s.pow(2).sum()
gs, = torch.autograd.grad(y, s, retain_graph=True)
check("grad wrt real[1:]", gs, 2 * x.real[1:])
gr, = torch.autograd.grad(y, r, retain_graph=True)
check("grad wrt real", gr, torch.cat([torch.zeros(1, 4, dtype=F64), 2 * x.real[1:]]))
gz, = torch.autograd.grad(y, zz)
check("grad wrt z", gz, torch.complex(torch.cat([torch.zeros(1, 4, dtype=F64), 2 * x.real[1:]]),
                                      torch.zeros(3, 4, dtype=F64)))
# 6. precision: a gradient value not representable in float32
zz = x.to(H).requires_grad_()
(zz.real[1:] * (1 + 1e-12)).sum().backward()
want = torch.zeros(3, 4, dtype=torch.complex128)
want[1:] = 1 + 1e-12
check("grad 1+1e-12 through real[1:]", zz.grad, want)
