"""On a Gaudi card: forward+backward of the complex linalg ops of issues #4/#5, one subprocess per op.
    PT_HPU_LAZY_MODE=0|1 python3 hpu_complex/tests/linalg_bwd_bench.py [--calls 20]

Per op: status (ok / error / process died), relative error of the input gradient against a CPU float64 run, for
the card and for the CPU in float32 (float32's own error), mean ms of
``--calls`` synced fwd+bwd calls after three warm-ups (outputs kept alive), bridge ``cpu_fallback`` ops per call
and hpu_complex's own "runs on the CPU" warnings (invisible to the bridge metric). One ``RESULT {json}`` per op.
Which hpu_complex runs is the one first on PYTHONPATH (printed as ``hpu_complex``).
"""
import argparse
import json
import os
import subprocess
import sys
import time
import warnings

N, B = 22, 32
THRESHOLD = None  # --matmul-threshold: complex mm/bmm below this many multiply-adds on the CPU (None: default)


def make_cases(torch):
    g = torch.Generator().manual_seed(0)

    def c(*shape):
        return torch.complex(torch.randn(*shape, generator=g), torch.randn(*shape, generator=g))

    A = c(B, N, N)
    herm = A @ A.mH + N * torch.eye(N)
    rect = c(B, N, 8)
    sq = c(B, N, N) / N ** 0.5
    refl, tau = torch.geqrf(rect)  # real Householder reflectors: random tau makes the product ill-conditioned
    real = torch.randn(B, N, N, generator=g)
    return {
        "eigh": ((herm,), lambda t: (lambda w, V: w.sum() + V.abs().sum())(*torch.linalg.eigh(t))),
        "svd": ((rect,), lambda t: (lambda U, S, Vh: U.abs().sum() + S.sum() + Vh.abs().sum())(
            *torch.linalg.svd(t, full_matrices=False))),
        "qr": ((rect,), lambda t: (lambda Q, R: Q.abs().sum() + R.abs().sum())(*torch.linalg.qr(t))),
        "eig": ((sq,), lambda t: (lambda w, V: w.abs().sum() + V.abs().sum())(*torch.linalg.eig(t))),
        "matrix_exp": ((sq,), lambda t: torch.linalg.matrix_exp(t).abs().sum()),
        "householder_product": ((refl, tau), lambda a, t: torch.linalg.householder_product(a, t).abs().sum()),
        "eig(real f32)": ((real,), lambda t: (lambda w, V: w.abs().sum() + V.abs().sum())(*torch.linalg.eig(t))),
        "eigvals(real f32)": ((real,), lambda t: torch.linalg.eigvals(t).abs().sum()),
        "eig(real f32) fwd only": ((real,), None),
    }


def run_one(name, calls):
    import habana_frameworks.torch.core as htcore
    import habana_frameworks.torch.hpu as hthpu
    import torch
    from habana_frameworks.torch.hpu.metrics import metric_global

    import hpu_complex
    hpu_complex.enable()
    if THRESHOLD is not None:
        hpu_complex.set_matmul_cpu_threshold(THRESHOLD)
    lazy = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"
    hpu = torch.device("hpu")
    inputs, loss_fn = make_cases(torch)[name]
    fwd_only = loss_fn is None
    row = {"op": name, "lazy": lazy, "hpu_complex": hpu_complex.__file__.rsplit("/", 3)[-3],
           "matmul_threshold": THRESHOLD}

    def sync():
        if lazy:
            htcore.mark_step()
        hthpu.synchronize()

    def fb():
        return dict(dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {})

    def call(xs):
        if fwd_only:
            with torch.no_grad():
                return torch.linalg.eig(xs[0])
        xs = [x.detach().requires_grad_() for x in xs]
        loss = loss_fn(*xs)
        loss.backward()
        return xs[0].grad

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            xh = [x.to(hpu) for x in inputs]
            sync()
            if not fwd_only:
                gc, gh = call(inputs), call(xh)
                sync()
                g64 = call([x.to(torch.complex128 if x.is_complex() else torch.float64) for x in inputs])
                rel = lambda a: float(((a.cpu().to(g64.dtype) - g64).abs().max() / g64.abs().max()).item())  # noqa: E731
                # vs a CPU float64 run: the card's error next to float32's own (the CPU float32 run)
                row["grad_rel_err_vs_f64"] = {"hpu": rel(gh), "cpu_f32": rel(gc)}
            for _ in range(3):
                call(xh)
            sync()
            f0 = fb()
            keep = []
            t = time.perf_counter()
            for _ in range(calls):
                keep.append(call(xh))
            sync()
            row["ms_per_call"] = round((time.perf_counter() - t) / calls * 1e3, 3)
            f1 = fb()
        row["bridge_fallback_per_call"] = {k: (f1[k] - f0.get(k, 0)) / calls for k in f1 if f1[k] - f0.get(k, 0)}
        row["hpu_complex_cpu_ops"] = sorted({str(w.message).split(" on a complex")[0].replace("hpu_complex: ", "")
                                             for w in caught if "runs on the CPU" in str(w.message)})
        row["status"] = "ok"
    except Exception as e:  # noqa: BLE001
        row.update(status="error", error=f"{type(e).__name__}: {str(e).splitlines()[0][:200]}")
    print("RESULT " + json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--op")
    parser.add_argument("--only", help="comma-separated op names")
    parser.add_argument("--matmul-threshold", type=int)
    args = parser.parse_args()
    global THRESHOLD
    THRESHOLD = args.matmul_threshold
    if args.op:
        run_one(args.op, args.calls)
        return 0
    import torch
    names = list(make_cases(torch))
    if args.only:
        names = [n for n in names if n in args.only.split(",")]
    bad = 0
    for name in names:
        extra = [] if THRESHOLD is None else ["--matmul-threshold", str(THRESHOLD)]
        p = subprocess.run([sys.executable, __file__, "--op", name, "--calls", str(args.calls)] + extra,
                           capture_output=True, text=True, timeout=900)
        lines = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT")]
        if lines:
            print(lines[-1], flush=True)
            bad += json.loads(lines[-1][7:])["status"] != "ok"
        else:
            what = [ln.strip() for ln in p.stderr.splitlines() if "what()" in ln or "Error" in ln][:2]
            print("RESULT " + json.dumps({"op": name, "status": "process died", "rc": p.returncode,
                                          "error": what}), flush=True)
            bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
