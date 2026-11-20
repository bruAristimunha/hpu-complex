"""Issue #3 probe: a real HPU tensor times a Python complex scalar (``x * 2j``) in lazy mode.

    PT_HPU_LAZY_MODE=1 python3 repro/lazy_complex_scalar.py            # every case, one subprocess each
    PT_HPU_LAZY_MODE=1 python3 repro/lazy_complex_scalar.py --case mul [--no-ext]

spd_learn's Green builds its Gabor wavelets as ``torch.exp(2j * pi * f * tt)`` with real HPU tensors ``f``,
``tt``. In lazy mode this aborts the process (``value cannot be converted to type double without overflow``,
then a Synapse device critical error) instead of raising. Each case prints ``CASE <name> <ext> -> <outcome>``.
"""
import argparse
import os
import subprocess
import sys

CASES = {
    "mul": lambda x: x * 2j,
    "rmul_0d": lambda x: 2j * x[0],
    "add": lambda x: x + 1j,
    "exp_mul": lambda x: __import__("torch").exp(x * 2j),
    "to_c64": lambda x: x.to(__import__("torch").complex64),
}


def run_case(name, ext):
    import torch
    import habana_frameworks.torch.core as htcore
    import habana_frameworks.torch.hpu as hthpu

    if ext:
        import hpu_complex
        hpu_complex.enable()
    x = torch.linspace(-1, 1, 8, device="hpu")
    htcore.mark_step()
    try:
        y = CASES[name](x)
        htcore.mark_step()
        hthpu.synchronize()
        out = f"ok dtype={y.dtype} max|err|={(y.cpu() - CASES[name](x.cpu())).abs().max().item():.1e}"
    except Exception as e:  # noqa: BLE001
        out = f"raises {type(e).__name__}: {str(e).splitlines()[0][:150]}"
    print(f"CASE {name} {'ext' if ext else 'no-ext'} -> {out}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case")
    parser.add_argument("--no-ext", action="store_true")
    args = parser.parse_args()
    if args.case:
        run_case(args.case, not args.no_ext)
        return
    for ext in (False, True):
        for name in CASES:
            cmd = [sys.executable, __file__, "--case", name] + ([] if ext else ["--no-ext"])
            p = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ), timeout=300)
            line = [ln for ln in p.stdout.splitlines() if ln.startswith("CASE")]
            if line:
                print(line[-1], flush=True)
            else:
                tail = [ln for ln in (p.stderr or "").splitlines() if "what()" in ln or "Error" in ln][:2]
                print(f"CASE {name} {'ext' if ext else 'no-ext'} -> process died rc={p.returncode} {tail}",
                      flush=True)


if __name__ == "__main__":
    main()
