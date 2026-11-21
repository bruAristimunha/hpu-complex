"""On a Gaudi card (eager mode): complex64 linalg under torch.compile(backend="hpu_backend") with hpu_complex.linalg.
    python3 hpu_complex/tests/clinalg_compile_probe.py
Traced calls are declined (torch's route): checks that compiled solve / inv / cholesky / matrix_exp forward and
backward still match eager, twice in a row, and prints how many calls the card kernels took and the CPU fallbacks."""
import json
import os
import warnings

import habana_frameworks.torch.core as htcore  # noqa: F401
import torch

import hpu_ops

hpu_ops.enable()
import hpu_complex  # noqa: E402
import hpu_complex.linalg as L  # noqa: E402

hpu_complex.enable()
H = torch.device("hpu")
g = torch.Generator().manual_seed(0)
A0 = torch.randn(8, 6, 6, dtype=torch.complex64, generator=g)
B0 = torch.randn(8, 6, 2, dtype=torch.complex64, generator=g)
S0 = A0 @ A0.mH / 6 + torch.eye(6)


TERMS = {"solve": lambda A, B, S: torch.linalg.solve(A, B).abs().sum(),
         "inv": lambda A, B, S: torch.linalg.inv(A).abs().sum(),
         "cholesky": lambda A, B, S: torch.linalg.cholesky(S).abs().sum(),
         "matrix_exp": lambda A, B, S: torch.linalg.matrix_exp(A / 6).abs().sum()}
ONLY = [t for t in os.environ.get("ONLY", "").split(",") if t] or list(TERMS)


def f(A, B, S):
    return sum(TERMS[t](A, B, S) for t in ONLY)


rows = {}
for name, fn in (("eager", f), ("compiled", torch.compile(f, backend="hpu_backend"))):
    for run in range(2):
        xs = [t.to(H).requires_grad_() for t in (A0, B0, S0)]
        calls, ext = sum(L.CALLS.values()), torch.ops.hpu_complex.cpu_fallback_calls()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loss = fn(*xs)
            loss.backward()
        rows[f"{name}{run}"] = {"loss": loss.item(), "grads": [x.grad.cpu() if x.grad is not None else torch.zeros(x.shape, dtype=x.dtype) for x in xs],
                                "taken": sum(L.CALLS.values()) - calls,
                                "ext_cpu_calls": torch.ops.hpu_complex.cpu_fallback_calls() - ext}
ref = rows["eager0"]
out = {}
for k, r in rows.items():
    out[k] = {"loss_rel": abs(r["loss"] - ref["loss"]) / abs(ref["loss"]),
              "grad_rel": max(((a - b).abs().max() / b.abs().max().clamp_min(1e-30)).item() for a, b in zip(r["grads"], ref["grads"])),
              "taken": r["taken"], "ext_cpu_calls": r["ext_cpu_calls"]}
ok = all(v["loss_rel"] < 1e-5 and v["grad_rel"] < 1e-4 for v in out.values())
print("RESULT " + json.dumps({"status": "ok" if ok else "wrong", "only": ONLY,
                              "linalg": os.environ.get("HPU_COMPLEX_LINALG", "1"), **out}))
