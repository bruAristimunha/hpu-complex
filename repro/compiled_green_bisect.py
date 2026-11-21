"""Bisect compiled Green's wrong wavelet gradients (eager mode, hpu_backend): Green's forward cut after each stage.
    PT_HPU_LAZY_MODE=0 python3 repro/compiled_green_bisect.py

Each stage is compiled with torch.compile(backend="hpu_backend") and its foi/fwhm gradients (Green's complex part)
are compared with the same function run eagerly on the card. One ``STAGE <name> -> <rel err>`` line per stage.
"""
import math
import os
import sys

import habana_frameworks.torch.core  # noqa: F401
import torch
import torch.nn.functional as F

import hpu_complex

hpu_complex.enable()
HPU = torch.device("hpu")


def wavelets(tt, foi, fwhm):
    foi_oct = 2 ** torch.clamp(foi, -2.0, 6.0)
    fwhm_oct = 2 ** torch.clamp(fwhm, -6.0, 1.0)
    w = torch.stack([torch.exp(2j * math.pi * f * tt) * torch.exp(-4 * math.log(2) * tt**2 / h**2)
                     for f, h in zip(foi_oct, fwhm_oct)], dim=0)
    w = w / torch.linalg.norm(w, dim=-1, keepdim=True)
    w *= math.sqrt(2.0 / 128) * torch.sqrt(math.log(2) * foi_oct).unsqueeze(1)
    return w


def wavelet_parts(stage, tt, foi, fwhm):
    f = 2 ** torch.clamp(foi, -2.0, 6.0)
    h = 2 ** torch.clamp(fwhm, -6.0, 1.0)
    if stage == "w_exp_scalar":  # one frequency, no stack
        return torch.exp(2j * math.pi * f[0] * tt)[None]
    if stage == "w_mul_scalar":
        return (2j * math.pi * f[0] * tt)[None]
    if stage == "w_mul_vector":
        return 2j * math.pi * f[:, None] * tt
    if stage == "w_exp_vector":
        return torch.exp(2j * math.pi * f[:, None] * tt)
    if stage == "w_gauss":
        return torch.exp(2j * math.pi * f[:, None] * tt) * torch.exp(-4 * math.log(2) * tt**2 / h[:, None] ** 2)
    if stage == "w_two_rows":  # two selected frequencies, stacked
        return torch.stack([torch.exp(2j * math.pi * f[0] * tt), torch.exp(2j * math.pi * f[1] * tt)])
    if stage == "w_two_rows_real":  # the same with real ops only: is it the complex part?
        return torch.complex(torch.stack([torch.cos(2 * math.pi * f[0] * tt), torch.cos(2 * math.pi * f[1] * tt)]),
                             torch.stack([torch.sin(2 * math.pi * f[0] * tt), torch.sin(2 * math.pi * f[1] * tt)]))
    if stage == "w_two_sum":  # two selected frequencies, added (no stack)
        return (torch.exp(2j * math.pi * f[0] * tt) + torch.exp(2j * math.pi * f[1] * tt))[None]
    if stage == "w_sel1_mul":  # f[1] alone: a select at a nonzero offset
        return (2j * math.pi * f[1] * tt)[None]
    if stage == "w_sel1_mul_c":  # the same, complex scalar times real 0-d first, then times tt (Green's order)
        return ((f[1] * 2j) * tt)[None]
    if stage == "w_stack_vec":  # rows of the vector form, stacked
        v = torch.exp(2j * math.pi * f[:, None] * tt)
        return torch.stack([v[i] for i in range(v.shape[0])])
    if stage.startswith("w_expn"):  # w_expn<N>: N selected frequencies, stacked
        return torch.stack([torch.exp(2j * math.pi * a * tt) for a in f[:int(stage[6:])]])
    if stage.startswith("w_muln"):  # w_muln<N>: no exp
        return torch.stack([2j * math.pi * a * tt for a in f[:int(stage[6:])]])
    if stage == "w_exp10":
        return torch.stack([torch.exp(2j * math.pi * a * tt) for a in f])
    if stage == "w_gauss10_real":  # real Gaussians only, one row per selected h
        return torch.complex(torch.stack([torch.exp(-4 * math.log(2) * tt**2 / b**2) for b in h]),
                             torch.zeros(10, tt.shape[0], device=tt.device))
    if stage == "w_gauss2":
        return torch.stack([torch.exp(2j * math.pi * a * tt) * torch.exp(-4 * math.log(2) * tt**2 / b**2)
                            for a, b in zip(f[:2], h[:2])], dim=0)
    if stage == "w_gauss10_mulc":  # Gaussian times a complex constant, one row per selected h
        return torch.stack([(1 + 1j) * torch.exp(-4 * math.log(2) * tt**2 / b**2) for b in h])
    w = torch.stack([torch.exp(2j * math.pi * a * tt) * torch.exp(-4 * math.log(2) * tt**2 / b**2)
                     for a, b in zip(f, h)], dim=0)
    if stage == "w_stack":
        return w
    w = w / torch.linalg.norm(w, dim=-1, keepdim=True)
    if stage == "w_norm":
        return w
    return w * torch.sqrt(f).unsqueeze(1)


W_STAGES = ["w_mul_scalar", "w_exp_scalar", "w_two_rows", "w_two_rows_real", "w_two_sum", "w_sel1_mul",
            "w_sel1_mul_c", "w_stack_vec", "w_exp10", "w_gauss10_real", "w_gauss2", "w_gauss10_mulc", "w_mul_vector", "w_exp_vector", "w_gauss", "w_stack", "w_norm",
            "w_scale"]


def stage_fn(stage):
    def fn(tt, foi, fwhm, X, R):
        if stage in W_STAGES or stage.startswith(("w_expn", "w_muln")):
            w = wavelet_parts(stage, tt, foi, fwhm)
            return (torch.view_as_real(w) * R[: w.numel() * 2].view(*w.shape, 2)).sum()
        w = wavelets(tt, foi, fwhm)
        if stage == "wavelet":
            return (torch.view_as_real(w) * R[: w.numel() * 2].view(*w.shape, 2)).sum()
        k = torch.cat((w.real, w.imag)).unsqueeze(1)
        if stage == "cat_real_imag":
            return (k * R[: k.numel()].view(k.shape)).sum()
        resp = F.conv1d(X.reshape(-1, 1, X.shape[-1]), k)
        if stage == "conv":
            return (resp * R[: resp.numel()].view(resp.shape)).sum()
        z = torch.complex(*resp.split(w.shape[0], dim=1))
        z = z.view(X.shape[0], X.shape[1], w.shape[0], -1).swapaxes(1, 2)
        if stage == "complex":
            return (torch.view_as_real(z).abs()).sum()
        cov = (z @ z.mT.conj() / z.size(-1))
        if stage == "cov_abs":
            return cov.abs().sum()
        c = cov.real
        if stage == "cov_real":
            return (c * R[: c.numel()].view(c.shape)).sum()
        if stage == "cov_real_contig":
            c = c.contiguous()
            return (c * R[: c.numel()].view(c.shape)).sum()
        tr = c.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True)
        if stage == "trace":
            return tr.sum()
        c = c / tr.unsqueeze(-1).clamp(min=1e-12)
        return (c * R[: c.numel()].view(c.shape)).sum()
    return fn


STAGES = W_STAGES + ["wavelet", "cat_real_imag", "conv", "complex", "cov_abs", "cov_real", "cov_real_contig", "trace",
          "trace_norm"]


def main():
    stages = sys.argv[1:] or STAGES
    torch.manual_seed(0)
    tt = torch.linspace(-0.25, 0.25, 64, device=HPU)
    foi0 = torch.linspace(1.0, 4.5, 10)
    fwhm0 = torch.linspace(-3.0, -1.0, 10)
    X = torch.randn(4, 22, 200, device=HPU)
    R = torch.randn(4 * 20 * 22 * 200, device=HPU)
    for stage in stages:
        fn = stage_fn(stage)
        grads = []
        for compiled in (False, True):
            torch._dynamo.reset()
            foi = foi0.clone().to(HPU).requires_grad_()
            fwhm = fwhm0.clone().to(HPU).requires_grad_()
            run = torch.compile(fn, backend="hpu_backend") if compiled else fn
            try:
                loss = run(tt, foi, fwhm, X, R)
                loss.backward()
                torch.hpu.synchronize()
                zero = lambda g: torch.zeros(10) if g is None else g.cpu()  # noqa: E731
                grads.append((loss.item(), zero(foi.grad), zero(fwhm.grad)))
            except Exception as e:  # noqa: BLE001
                if os.environ.get("TRACE"):
                    import traceback
                    traceback.print_exc()
                grads.append(f"{'compiled' if compiled else 'eager'} {type(e).__name__}: {str(e).splitlines()[0][:150]}")
        if any(isinstance(g, str) for g in grads):
            print(f"STAGE {stage} -> {[g for g in grads if isinstance(g, str)]}", flush=True)
            continue
        (le, fe, he), (lc, fc, hc) = grads
        rel = lambda a, b: ((a - b).abs().max() / b.abs().max().clamp_min(1e-12)).item()  # noqa: E731
        print(f"STAGE {stage} -> loss {abs(lc - le) / max(abs(le), 1e-12):.1e} foi {rel(fc, fe):.1e} "
              f"fwhm {rel(hc, he):.1e}", flush=True)


if __name__ == "__main__":
    main()
