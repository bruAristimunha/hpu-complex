"""On a Gaudi card: real HPU tensors and Python complex scalars under torch.inference_mode, each case in its own process.
    PT_HPU_LAZY_MODE=0|1 python3 hpu_complex/tests/inference_check.py [--case NAME]

torch.inference_mode excludes the autograd keys, so ``x * 2j`` skips AutogradHPU, where hpu_complex promoted it, and
reached Habana's float mul: in lazy mode it aborts the process ("value cannot be converted to type double without
overflow", then a device critical error). An abort kills only its own subprocess here and fails the case. Each case
compares with the CPU and checks that results made under inference_mode are inference tensors (torch's rule).
The ``c64_*`` cases are complex64 tensors made under inference_mode (lazy mode refused those before, the guard).
Exits 1 if a case fails; prints ``CASE <name> -> <outcome>`` lines and ``inference_check: N/M OK``.
"""
import argparse
import os
import subprocess
import sys

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "0") != "0"


def _scalar_cases(torch):
    return {
        "mul": lambda x: x * 2j,
        "rmul_0d": lambda x: (1 - 2j) * x[0],
        "add": lambda x: x + 1j,
        "sub": lambda x: x - 1j,
        "rsub": lambda x: 1j - x,
        "div": lambda x: x / 2j,
        "exp_mul": lambda x: torch.exp(x * 2j),
        "tensor_0d": lambda x: x * torch.tensor(2j),
        "gabor": lambda x: torch.exp(2j * torch.pi * x[:, None] * x[None, :4]).real,
        "c64_rfft_abs": lambda x: torch.fft.rfft(x).abs(),
        "c64_complex_real": lambda x: torch.complex(x, -x).real * 3,
        "c64_view_as_real": lambda x: torch.view_as_real(torch.complex(x, x * 0.5) * (1 + 1j)),
        "c64_mm": lambda x: (torch.complex(x[:4, None] * x[None, :4], x[:4, None]) @
                             torch.complex(x[None, :4] + x[:4, None], x[4:8, None] * 0)).imag,
    }


def run_case(name):
    import habana_frameworks.torch.core as htcore
    import habana_frameworks.torch.hpu as hthpu
    import torch

    import hpu_complex

    hpu_complex.enable()
    cases = _scalar_cases(torch)
    torch.manual_seed(0)
    xc = torch.linspace(-1, 1, 16)
    x = xc.to("hpu")
    normal = torch.complex(x, x).mul(1j)  # a normal complex64 tensor, made before inference_mode
    htcore.mark_step()
    try:
        if name == "inference_tensor_outside":
            with torch.inference_mode():
                xi = xc.to("hpu") * 1
            y, ref = xi * 2j, xc * 2j
            inference_ok = True
        elif name == "normal_complex_inside":
            with torch.inference_mode():
                y = (normal.real * 2j + normal[:3].sum()).real
            ref = ((torch.complex(xc, xc) * 1j).real * 2j + (torch.complex(xc, xc) * 1j)[:3].sum()).real
            inference_ok = y.is_inference()
        elif name == "inplace_normal_inside":  # in-place on a normal complex64 tensor under inference_mode
            z, zc = normal.clone(), torch.complex(xc, xc) * 1j
            with torch.inference_mode():
                z.mul_(2j)
                z.real.add_(1)
                z[:2] = 3 - 1j
            zc.mul_(2j)
            zc.real.add_(1)
            zc[:2] = 3 - 1j
            y, ref, inference_ok = torch.view_as_real(z), torch.view_as_real(zc), not z.is_inference()
        elif name == "inference_complex_outside":  # complex64 made under inference_mode, used after it
            with torch.inference_mode():
                zi = torch.complex(x, x * 0.5)
            htcore.mark_step()
            zr = torch.complex(xc, xc * 0.5)
            y = torch.view_as_real(zi * (2 - 1j)) + zi.imag[:, None] + torch.fft.fft(zi).abs()[:, None]
            ref = torch.view_as_real(zr * (2 - 1j)) + zr.imag[:, None] + torch.fft.fft(zr).abs()[:, None]
            inference_ok = not y.is_inference()
            try:
                zi.add_(1)
                inference_ok = False  # torch: no in-place update of an inference tensor outside inference_mode
            except RuntimeError as e:
                inference_ok &= "nference" in str(e)
        elif name == "green_eval":  # spd_learn's Green, an eval forward under inference_mode
            import copy

            from spd_diag import protocol as P
            from spd_diag.cache import open_cache
            from spd_diag.models import MODELS

            P._enable_device("hpu")
            cache = open_cache("BNCI2014_001")
            sd = cache.subject(cache.subjects[0], "X")
            X = torch.as_tensor(sd.X[:8], dtype=torch.float32)
            X = X / X.std()
            model_cpu = P.build(next(k for k in MODELS if k.lower() == "green"), X.shape[1], cache.info, None).eval()
            model = copy.deepcopy(model_cpu).to("hpu")
            with torch.inference_mode():
                out = model(X.to("hpu"))
                y = out[0] if isinstance(out, tuple) else out
                htcore.mark_step()
                out = model_cpu(X)
                ref = out[0] if isinstance(out, tuple) else out
            inference_ok = y.is_inference()
        elif name == "train_after_eval":  # an eval pass under inference_mode, then a training step
            w = torch.nn.Parameter(torch.tensor([0.5], device="hpu"))
            with torch.inference_mode():
                torch.exp(x * w * 2j).real.sum()
            htcore.mark_step()
            loss = torch.exp(x * w * 2j).real.sum()
            loss.backward()
            htcore.mark_step()
            wc = torch.tensor([0.5], requires_grad=True)
            torch.exp(xc * wc * 2j).real.sum().backward()
            y, ref, inference_ok = w.grad, wc.grad, not w.grad.is_inference()
        else:
            with torch.inference_mode():
                y = cases[name](x)
                htcore.mark_step()
            ref = cases[name](xc)
            inference_ok = y.is_inference()
        htcore.mark_step()
        hthpu.synchronize()
        err = (y.cpu() - ref).abs().max().item()
        tol = 2e-3 if name == "green_eval" else 1e-4  # Green: float32 eigh on the card vs the CPU
        ok = err < tol * max(1.0, ref.abs().max().item()) and y.dtype == ref.dtype and inference_ok
        out = (f"{'ok' if ok else 'WRONG'} dtype={y.dtype} inference={y.is_inference()} max|err|={err:.1e}")
    except Exception as e:  # noqa: BLE001  (the failure is the result)
        out = f"raises {type(e).__name__}: {str(e).splitlines()[0][:160]}"
    print(f"CASE {name} -> {out}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case")
    args = parser.parse_args()
    if args.case:
        run_case(args.case)
        return 0
    import torch

    names = list(_scalar_cases(torch)) + ["inference_tensor_outside", "normal_complex_inside", "inplace_normal_inside",
                                          "inference_complex_outside", "green_eval", "train_after_eval"]
    passed = 0
    for name in names:
        try:
            p = subprocess.run([sys.executable, __file__, "--case", name], capture_output=True, text=True,
                               env=dict(os.environ), timeout=600)
            lines = [ln for ln in p.stdout.splitlines() if ln.startswith("CASE")]
            if lines and p.returncode == 0:
                line = lines[-1]
            else:
                tail = [ln for ln in (p.stderr or "").splitlines() if "what()" in ln or "Error" in ln][:2]
                line = f"CASE {name} -> process died rc={p.returncode} {tail}"
        except subprocess.TimeoutExpired:
            line = f"CASE {name} -> timeout"
        print(line, flush=True)
        passed += " -> ok " in line
    print(f"inference_check ({'lazy' if LAZY else 'eager'}): {passed}/{len(names)} OK", flush=True)
    return 0 if passed == len(names) else 1


if __name__ == "__main__":
    sys.exit(main())
