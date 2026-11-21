"""Which torch.compile(hpu_backend) case segfaults, with and without hpu_complex (one Gaudi, one variant per process).

python3 tests/probe_adv_compile_j.py <variant> [--no-ext]
variants: eigvals_f64, eigvals_f32, c128_input, c128_mul_only, f64_abs_only, eigvals_f64_graph_break
"""
import sys
import warnings

import habana_frameworks.torch.core as htcore  # noqa: F401
import torch

warnings.simplefilter("ignore")
if "--no-ext" not in sys.argv:
    import hpu_complex

    hpu_complex.enable()
H = torch.device("hpu")
v = sys.argv[1]
g = torch.Generator().manual_seed(0)
x64 = torch.randn(4, 5, 5, dtype=torch.float64, generator=g).float().double()
z = torch.randn(3, 4, dtype=torch.complex128, generator=g)


def eigvals_break(x):
    L = torch.linalg.eigvals(x)
    torch._dynamo.graph_break()
    return L.abs().sum(-1) * 2


fns = {
    "eigvals_f64": (lambda x: torch.linalg.eigvals(x).abs().sum(-1) * 2, x64),
    "eigvals_f32": (lambda x: torch.linalg.eigvals(x).abs().sum(-1) * 2, x64.float()),
    "eigvals_only_f64": (lambda x: torch.linalg.eigvals(x), x64),
    "c128_input": (lambda t: (t * t.conj()).real.sum(-1) + t.abs().sum(-1), z),
    "c128_mul_only": (lambda t: t * t, z),
    "f64_abs_only": (lambda t: t.abs() * 2, z),
    "eigvals_f64_graph_break": (eigvals_break, x64),
}
f, inp = fns[v]
ref = f(inp)
print(f"{v}: start", flush=True)
out = torch.compile(f, backend="hpu_backend")(inp.to(H))
out = out.cpu()
err = (out.to(ref.dtype) - ref).abs().max().item()
print(f"{v}: OK out {out.dtype} ref {ref.dtype} maxerr {err:.3g}", flush=True)
