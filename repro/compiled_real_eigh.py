"""Probe (does NOT reproduce, measured): spd_learn Green under torch.compile(backend="hpu_backend") on Gaudi2.
    PT_HPU_LAZY_MODE=0 python3 repro/compiled_real_eigh.py [--ext]

Compiled Green (hpu_complex + hpu_ops enabled, bridge 1.21.4) fails at its graph's float32 ``aten._linalg_eigh``
node (LogEig) with "synNodeCreateWithId failed for node: strided_view with synStatus 1 [Invalid argument]";
eager Green trains. These smaller cases try to isolate it: float32 eigh compiled on a covariance, on a contiguous
graph input, after a matmul, and fed by ``.real`` of a complex covariance (``--ext``). All of them pass on Gaudi2
(2026-11-20), so the trigger is elsewhere in Green's graph (Shrinkage / TraceNorm / hpu_ops' patches). One
``CASE <name> -> <outcome>`` line per case, each in a fresh process.
"""
import argparse
import subprocess
import sys

CASES = ("eager", "compiled", "compiled_contiguous_input", "compiled_from_matmul", "compiled_real_of_complex",
         "compiled_real_of_complex_contiguous")


def run(case, ext):
    import habana_frameworks.torch.core  # noqa: F401
    import torch

    if ext:
        import hpu_complex
        hpu_complex.enable()
    torch.manual_seed(0)
    a = torch.randn(32, 22, 100, device="hpu")

    def from_cov(t):  # a covariance, then its eigenvalues: Green's LogEig input
        c = t @ t.mT / t.shape[-1] + 1e-3 * torch.eye(t.shape[-2], device=t.device)
        return torch.linalg.eigh(c)[0].log().sum()

    def from_input(c):
        return torch.linalg.eigh(c)[0].log().sum()

    cov = (a @ a.mT / 100 + 1e-3 * torch.eye(22, device="hpu")).contiguous()
    try:
        if case == "eager":
            out = from_cov(a)
        elif case == "compiled":
            out = torch.compile(from_cov, backend="hpu_backend")(a)
        elif case == "compiled_contiguous_input":
            out = torch.compile(from_input, backend="hpu_backend")(cov)
        elif case == "compiled_from_matmul":
            out = torch.compile(lambda t: from_input(t @ t.mT / 100), backend="hpu_backend")(a)
        else:  # needs --ext: the .real of a complex covariance (a stride-2 view of the pair) as the graph input
            z = torch.complex(a, a.flip(-1))
            creal = (z @ z.mH / 100).real + 1e-3 * torch.eye(22, device="hpu") * 0
            creal = creal.contiguous() if case.endswith("contiguous") else (z @ z.mH / 100).real
            out = torch.compile(from_input, backend="hpu_backend")(creal)
            cov = creal
        torch.hpu.synchronize()
        ref = (from_cov if case in ("eager", "compiled") else from_input)(a.cpu() if case in ("eager", "compiled")
                                                                          else cov.cpu())
        if case.startswith("compiled_real_of_complex"):
            zc = torch.complex(a.cpu(), a.cpu().flip(-1))
            ref = from_input((zc @ zc.mH / 100).real)
        if case == "compiled_from_matmul":
            ref = from_input(a.cpu() @ a.cpu().mT / 100)
        result = f"ok |diff| {abs(out.item() - ref.item()):.1e}"
    except Exception as e:  # noqa: BLE001
        result = f"raises {type(e).__name__}: {[ln for ln in str(e).splitlines() if 'synNode' in ln or ln][0][:140]}"
    print(f"CASE {case} {'ext' if ext else 'no-ext'} -> {result}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case")
    parser.add_argument("--ext", action="store_true")
    args = parser.parse_args()
    if args.case:
        run(args.case, args.ext)
        return
    for case in CASES:
        p = subprocess.run([sys.executable, __file__, "--case", case] + (["--ext"] if args.ext else []),
                           capture_output=True, text=True, timeout=600)
        lines = [ln for ln in p.stdout.splitlines() if ln.startswith("CASE")]
        print(lines[-1] if lines else f"CASE {case} -> process died rc={p.returncode}", flush=True)


if __name__ == "__main__":
    main()
