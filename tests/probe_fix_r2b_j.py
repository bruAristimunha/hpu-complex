"""Fix round 2 (team J): bool_mask_and_index chained, and compile_eig_backward per backend, on the card.

PT_HPU_LAZY_MODE=0|1 python3 tests/probe_fix_r2b_j.py [mask|grad]
"""
import os
import sys
import warnings

import habana_frameworks.torch.core as htcore
import torch

import hpu_complex

warnings.simplefilter("ignore")
hpu_complex.enable()
H = torch.device("hpu")
LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"
C128, F64 = torch.complex128, torch.float64


def sync():
    if LAZY:
        htcore.mark_step()
    torch.hpu.synchronize()


def rnd(*shape, dtype=C128, seed=0):
    t = torch.randn(shape, dtype=dtype, generator=torch.Generator().manual_seed(seed))
    return t.float().double() if dtype == F64 else t


def err(g, r):
    g = g.detach().cpu().resolve_conj()
    return (g - r).abs().max().item() if g.numel() else 0.0


def mask():
    x = rnd(3, 4)
    for d in ("cpu", H):
        z = x.clone().to(d)  # x.to("cpu") is x itself: the CPU pass must not change x
        m = (x.real > 0).to(d)
        trace = []
        z[m] = 1j
        trace.append(z.clone())
        z[torch.tensor([0, 2], device=d)] *= 2
        trace.append(z.clone())
        z.masked_fill_((x.imag > 1).to(d), 0)
        trace.append(z.clone())
        z.index_add_(0, torch.tensor([1, 1], device=d), rnd(2, 4, seed=3).to(d))
        trace.append(z.clone())
        trace += [z[m], z[[0, 2]]]
        if d == "cpu":
            ref = trace
        else:
            sync()
            for i, (g, r) in enumerate(zip(trace, ref)):
                print(f"MASK step {i} err {err(g, r):.3g} shapes {tuple(g.shape)} {tuple(r.shape)}", flush=True)


def grad():
    def f(x):
        return torch.linalg.eigvals(x).abs().sum()

    x0 = rnd(4, 4, dtype=F64)
    xc = x0.clone().requires_grad_()
    f(xc).backward()
    for name in ("plain", "eager", "aot_eager", "hpu_backend"):
        torch._dynamo.reset()
        fn = f if name == "plain" else torch.compile(f, backend=name)
        x = x0.to(H).requires_grad_()
        out = fn(x)
        out.backward()
        sync()
        print(f"GRAD {name:12s} loss err {err(out, f(x0)):.3g} grad err {err(x.grad, xc.grad):.3g} "
              f"grad dtype {x.grad.dtype} keys_pu1 "
              f"{torch._C._dispatch_keys(x.grad).has(torch._C.DispatchKey.PrivateUse1)}", flush=True)


def graph():
    """Which nodes of the hpu_backend graphs touch complex/float64 and how they are placed."""
    import hpu_complex as hc
    from habana_frameworks.torch.dynamo.compile_backend import passes

    orig = hc._touches_complex

    def logged(node):
        r = orig(node)
        print(f"NODE {node.graph.__hash__() % 1000:3d} {str(node.target)[:40]:40s} dtypes {hc._node_dtypes(node)} "
              f"in {[hc._node_dtypes(n) for n in node.all_input_nodes]} eager={r}", flush=True)
        return r

    hc._touches_complex = logged

    def f(x):
        return torch.linalg.eigvals(x).abs().sum()

    x = rnd(4, 4, dtype=F64).to(H).requires_grad_()
    torch.compile(f, backend="hpu_backend")(x).backward()
    sync()


def gradval():
    """The compiled gradient next to torch's: which values come back wrong."""
    def f(x):
        return torch.linalg.eigvals(x).abs().sum()

    x0 = rnd(3, 3, dtype=F64)
    xc = x0.clone().requires_grad_()
    f(xc).backward()
    x = x0.to(H).requires_grad_()
    torch.compile(f, backend="hpu_backend")(x).backward()
    sync()
    print("GRADVAL torch\n", xc.grad, "\nGRADVAL compiled\n", x.grad.cpu(), flush=True)
    z = x0.to(C128).requires_grad_()
    f(z).backward()
    print("GRADVAL complex grad of the same loss (torch)\n", z.grad, flush=True)


def gradsplit():
    """Where the compiled eig gradient goes wrong: forward outputs, or the backward graph."""
    x0 = rnd(3, 3, dtype=F64)
    ref_L, ref_V = torch.linalg.eig(x0)
    variants = {
        "fwd eig V": lambda x: torch.linalg.eig(x)[1],
        "fwd eig L": lambda x: torch.linalg.eig(x)[0],
        "c128 in: eigvals.abs.sum": lambda x: torch.linalg.eigvals(x).abs().sum(),
    }
    for name, fn in variants.items():
        torch._dynamo.reset()
        out = torch.compile(fn, backend="hpu_backend")(x0.to(H))
        sync()
        print(f"SPLIT {name:26s} err {err(out, fn(x0)):.3g}", flush=True)
    for name, fn in {"c128 leaf": lambda z: torch.linalg.eigvals(z).abs().sum(),
                     "f64 leaf, loss outside": None}.items():
        torch._dynamo.reset()
        if fn is not None:
            zc = x0.to(C128).requires_grad_()
            fn(zc).backward()
            z = x0.to(C128).to(H).requires_grad_()
            torch.compile(fn, backend="hpu_backend")(z).backward()
        else:
            zc = x0.clone().requires_grad_()
            torch.linalg.eigvals(zc).abs().sum().backward()
            z = x0.to(H).requires_grad_()
            torch.compile(lambda x: torch.linalg.eigvals(x), backend="hpu_backend")(z).abs().sum().backward()
        sync()
        print(f"SPLIT grad {name:21s} err {err(z.grad, zc.grad):.3g}", flush=True)


{"mask": mask, "grad": grad, "graph": graph, "gradval": gradval, "gradsplit": gradsplit}[sys.argv[1]]()
