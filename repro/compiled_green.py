"""Probe: spd_learn Green under torch.compile(backend="hpu_backend") in eager mode (PT_HPU_LAZY_MODE=0), on Gaudi2.
    python3 repro/compiled_green.py [--steps 5] [--batch 32]

Builds Green as green_step.py does, without the CPU float64 comparison (which allocates complex128 and so makes
hpu_complex run every float64-holding graph eagerly, hiding the compiled path). Compares one compiled step's loss and
gradients with the same model run eagerly on the card, then trains --steps compiled steps. One RESULT line.
"""
import argparse
import copy
import json
import time
import traceback

import habana_frameworks.torch.hpu as hthpu
import torch

HPU = torch.device("hpu")


def rel(a, b):
    a, b = a.detach().cpu().double(), b.detach().cpu().double()
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-12)).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    from spd_diag import protocol as P
    from spd_diag.cache import open_cache
    from spd_diag.models import MODELS

    P._enable_device("hpu")
    row = {}
    try:
        cache = open_cache("BNCI2014_001")
        sd = cache.subject(cache.subjects[0], "X")
        X = torch.as_tensor(sd.X[:args.batch], dtype=torch.float32)
        X = (X / X.std()).to(HPU)
        y = torch.as_tensor(sd.y[:args.batch]).long().to(HPU)
        torch.manual_seed(0)
        model = P.build(next(k for k in MODELS if k.lower() == "green"), X.shape[1], cache.info, None)
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = 0.0
        model = model.to(HPU)
        ref = copy.deepcopy(model)
        row["host_in_use_before"] = bool(torch.ops.hpu_complex.host_in_use())

        def loss_of(mod):
            out = mod(X)
            out = out[0] if isinstance(out, tuple) else out
            return torch.nn.functional.cross_entropy(out, y)

        le = loss_of(ref)
        le.backward()
        run = torch.compile(model, backend="hpu_backend")
        lc = loss_of(run)
        lc.backward()
        hthpu.synchronize()
        row["loss_eager_compiled"] = [le.item(), lc.item()]
        errs = {n: rel(p.grad, q.grad) for (n, p), (_, q) in zip(model.named_parameters(), ref.named_parameters())
                if q.grad is not None}
        row["grad_rel_err_vs_eager_max"] = max(errs.values())
        row["grad_worst3"] = {k: errs[k] for k in sorted(errs, key=errs.get)[-3:]}
        opt = torch.optim.Adam(model.parameters(), 1e-3)
        losses = []
        t = None
        for i in range(args.steps + 2):
            if i == 2:
                hthpu.synchronize()
                t = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            loss = loss_of(run)
            loss.backward()
            opt.step()
            losses.append(loss.detach())
        hthpu.synchronize()
        row["ms_per_step"] = round((time.perf_counter() - t) / args.steps * 1e3, 2)
        row["losses"] = [round(v, 5) for v in torch.stack(losses).cpu().tolist()]
        row["host_in_use_after"] = bool(torch.ops.hpu_complex.host_in_use())
        row["status"] = "ok"
    except Exception as e:  # noqa: BLE001
        row.update(status="error", error=repr(e)[:600],
                   where=[f"{f.filename.split('site-packages/')[-1]}:{f.lineno}"
                          for f in traceback.extract_tb(e.__traceback__)][-6:])
    print("RESULT " + json.dumps(row, default=str), flush=True)


if __name__ == "__main__":
    main()
