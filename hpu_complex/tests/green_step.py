"""On a Gaudi card: train steps (forward, backward, Adam) of spd_learn's Green model (complex Gabor wavelets).
    PT_HPU_LAZY_MODE=0|1 python3 hpu_complex/tests/green_step.py [--batch 32] [--steps 20]

Issue #4's gate. Builds Green through spd_diag (``P.build``, BNCI2014_001 width, a real batch), checks one
step's loss and gradients against the same model on the CPU (relative max error, dropout off), then times ``--steps``
synced steps after five warm-ups and counts the bridge ``cpu_fallback`` ops per step plus hpu_complex's own
"runs on the CPU" warnings (the bridge metric does not see those). One ``RESULT {json}`` line; exits 1 on an
error, a NaN, or a gradient error above max(1e-3, 10x the CPU float32-vs-float64 error).
"""
import argparse
import copy
import statistics
import json
import os
import sys
import time
import traceback
import warnings

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu
import torch
from habana_frameworks.torch.hpu.metrics import metric_global

LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"
HPU = torch.device("hpu")
DATASET = "BNCI2014_001"
WARMUP = 5
GRAD_TOL = 1e-3


def sync():
    if LAZY:
        htcore.mark_step()
    hthpu.synchronize()


def fb_ops():
    return dict(dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {})


def delta(a, b):
    return {k: b[k] - a.get(k, 0) for k in b if b[k] - a.get(k, 0)}


def rel(a, b):
    return ((a.detach().cpu().double() - b.detach().double()).abs().max() / b.detach().double().abs().max().clamp_min(1e-12)).item()


def grads_vs_cpu(model_cpu, model_hpu, X, y, device=HPU):
    """Relative max error of the loss and of every parameter gradient of model_hpu (run on ``device``, input cast to
    its dtype) against model_cpu, one fwd+bwd each."""
    def loss_of(model, xb, yb):
        out = model(xb)
        out = out[0] if isinstance(out, tuple) else out
        return torch.nn.functional.cross_entropy(out, yb)

    lc = loss_of(model_cpu, X, y)
    lc.backward()
    lh = loss_of(model_hpu, X.to(device, torch.float32), y.to(device))
    lh.backward()
    if device == HPU:
        sync()
    worst = {"loss": rel(lh, lc)}
    for (name, pc), (_, ph) in zip(model_cpu.named_parameters(), model_hpu.named_parameters()):
        if pc.grad is None:
            continue
        if ph.grad is None:
            worst[name] = float("inf")
            continue
        worst[name] = rel(ph.grad, pc.grad)
    model_cpu.zero_grad(set_to_none=True)
    model_hpu.zero_grad(set_to_none=True)
    return worst


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--ab", action="store_true",
                        help="also time, in this process, every complex product on the CPU (the route before the "
                             "card GEMM) and every one on the card, interleaved with the default")
    parser.add_argument("--rounds", type=int, default=5,
                        help="--ab rounds; each round times every variant once, in a rotated order")
    parser.add_argument("--compile", action="store_true", help='time torch.compile(model, backend="hpu_backend")')
    args = parser.parse_args()

    import hpu_complex
    import hpu_ops
    from spd_diag import protocol as P
    from spd_diag.cache import open_cache
    from spd_diag.models import MODELS

    P._enable_device("hpu")  # hpu_ops.enable() and hpu_complex.enable()
    name = next(k for k in MODELS if k.lower() == "green")
    row = {"model": name, "batch": args.batch, "lazy": LAZY, "hpu_complex": hpu_complex.__file__,
           "hpu_ops": hpu_ops.__file__}
    try:
        cache = open_cache(DATASET)
        info, sd = cache.info, cache.subject(cache.subjects[0], "X")
        X = torch.as_tensor(sd.X[:args.batch], dtype=torch.float32)
        X = X / X.std()
        y = torch.as_tensor(sd.y[:args.batch]).long()
        row["n_ch"] = X.shape[1]
        torch.manual_seed(0)
        model_cpu = P.build(name, X.shape[1], info, None)
        model = copy.deepcopy(model_cpu).to(HPU)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # Dropout draws differ between devices: the comparison runs with dropout off, BatchNorm in train mode.
            drops = [m for mod in (model_cpu, model) for m in mod.modules() if isinstance(m, torch.nn.Dropout)]
            for m in drops:
                m.eval()
            errs = grads_vs_cpu(model_cpu, model, X, y)
            # float32's own error on this step: the same CPU model, float32 against float64 (complex128 wavelets)
            model_64 = copy.deepcopy(model_cpu).double()
            for m in model_64.modules():
                if getattr(m, "dtype", None) == torch.complex64:
                    m.dtype = torch.complex128
                elif getattr(m, "dtype", None) == torch.float32:
                    m.dtype = torch.float64
            errs_f32 = grads_vs_cpu(model_64, model_cpu, X.double(), y, device="cpu")
            for m in drops:
                m.train()
            row["grad_rel_err_max"] = max(errs.values())
            row["grad_rel_err_worst3"] = {k: round(errs[k], 7) for k in sorted(errs, key=errs.get)[-3:]}
            row["cpu_f32_vs_f64_rel_err_max"] = max(errs_f32.values())
            row["cpu_f32_vs_f64_worst3"] = {k: round(errs_f32[k], 7) for k in sorted(errs_f32, key=errs_f32.get)[-3:]}

            opt = torch.optim.Adam(model.parameters(), 1e-3)
            xb, yb = X.to(HPU), y.to(HPU)
            losses = []
            run = torch.compile(model, backend="hpu_backend") if args.compile else model
            row["compiled"] = args.compile

            def step():
                opt.zero_grad(set_to_none=True)
                out = run(xb)
                out = out[0] if isinstance(out, tuple) else out
                loss = torch.nn.functional.cross_entropy(out, yb)
                loss.backward()
                if LAZY:
                    htcore.mark_step()
                opt.step()
                if LAZY:
                    htcore.mark_step()
                losses.append(loss.detach())

            for _ in range(WARMUP):
                step()
            sync()
            f0 = fb_ops()
            t = time.perf_counter()
            for _ in range(args.steps):
                step()
            sync()
            row["ms_per_step"] = round((time.perf_counter() - t) / args.steps * 1e3, 2)
            row["fallback_per_step"] = {k: v / args.steps for k, v in delta(f0, fb_ops()).items()}
            if args.ab:
                # Whether the default threshold sent any of Green's products to the CPU before the A/B started.
                row["cpu_route_at_default"] = any("complex mm/bmm" in str(w.message) for w in caught)
                variants = {"all products on the CPU": 1 << 62, "default (CPU below 2^22)": 1 << 22,
                            "all products on the card": 0}
                labels = list(variants)
                row["ab_rounds"] = args.rounds
                row["ab_ms_per_step"] = {k: [] for k in variants}
                for r in range(args.rounds):
                    for label in labels[r % len(labels):] + labels[:r % len(labels)]:  # rotate: no fixed position
                        hpu_complex.set_matmul_cpu_threshold(variants[label])
                        for _ in range(2):
                            step()
                        sync()
                        t = time.perf_counter()
                        for _ in range(args.steps):
                            step()
                        sync()
                        row["ab_ms_per_step"][label].append(round((time.perf_counter() - t) / args.steps * 1e3, 2))
                hpu_complex.set_matmul_cpu_threshold(1 << 22)
                row["ab_median_min_max"] = {k: [statistics.median(v), min(v), max(v)]
                                            for k, v in row["ab_ms_per_step"].items()}
        row["cpu_warnings"] = sorted({str(w.message)[:160] for w in caught if "runs on the CPU" in str(w.message)})
        ls = torch.stack(losses).cpu()
        row["loss_first_last"] = [round(ls[0].item(), 5), round(ls[-1].item(), 5)]
        row["finite"] = bool(torch.isfinite(ls).all())
        ok = row["finite"] and row["grad_rel_err_max"] < max(GRAD_TOL, 10 * row["cpu_f32_vs_f64_rel_err_max"])
        row["status"] = "ok" if ok else "wrong"
    except Exception as e:  # noqa: BLE001  (the failure is the result)
        row.update(status="error", error=repr(e)[:400],
                   where=[f"{f.filename.split('site-packages/')[-1]}:{f.lineno}"
                          for f in traceback.extract_tb(e.__traceback__)][-5:])
    print("RESULT " + json.dumps(row, default=str), flush=True)
    return 0 if row["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
