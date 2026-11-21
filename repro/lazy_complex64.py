"""Standalone Issue #3 reproducer: complex64 & Habana lazy mode (PT_HPU_LAZY_MODE=1).

Demonstrates three modes:
  --mode habana-bug : Without hpu_complex in PT_HPU_LAZY_MODE=1:
      (a) torch.linalg.eig(float32_hpu) fails inside Habana itself
          ("Complex datatype is not supported on HPU device.").
      (b) torch.empty(0, device="hpu").set_(storage, ...) over lazy HPU
          tensors fails with synStatus 30 when a lazy op is dropped behind
          storage(), and produces silent wrong values on strided/offset set_
          (diff=8.0) and copy_() into set_ (diff=7.0), proving HbLazyTensor
          cannot safely bridge PrivateUse1 storage views.
  --mode guarded    : With hpu_complex.enable() in PT_HPU_LAZY_MODE=1:
      Real-only float32 lazy ops and the float-pair API (hpu_complex.rfft /
      hpu_complex.stft) pass cleanly, and complex64 / torch.fft.rfft /
      torch.linalg.eig(float32) compute right: each complex64 storage holds
      the Habana tensor it lives in, and the kernels read it through Habana's
      own views of that tensor (no set_, unlike (b) above). What lazy mode
      still refuses (complex64 under torch.inference_mode) raises a clean
      RuntimeError (zero process aborts).
  --mode workaround : In eager mode (PT_HPU_LAZY_MODE=0) with hpu_complex:
      complex64 ops, torch.fft.rfft fwd+bwd, and torch.compile(backend="hpu_backend")
      on simple complex+FFT modules pass cleanly. (Note: compiled spd_learn.models.Green
      fails with "strided_view synStatus 1" at its float32 aten._linalg_eigh node,
      see repro/compiled_real_eigh.py; it trains in plain eager mode.)

Usage on Gaudi2:
  python3 repro/lazy_complex64.py --mode all
  PT_HPU_LAZY_MODE=1 python3 repro/lazy_complex64.py --mode guarded
  PT_HPU_LAZY_MODE=0 python3 repro/lazy_complex64.py --mode workaround
"""
import argparse
import os
import subprocess
import sys


def _spawn(mode: str, lazy_env: str, subcase: str | None = None) -> tuple[int, str, str]:
    env = os.environ.copy()
    env["PT_HPU_LAZY_MODE"] = lazy_env
    cmd = [sys.executable, __file__, "--mode", mode, "--in-process"]
    if subcase is not None:
        cmd += ["--subcase", subcase]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout, end="", flush=True)
    if proc.stderr and proc.returncode != 0:
        print(proc.stderr, file=sys.stderr, end="", flush=True)
    return proc.returncode, proc.stdout, proc.stderr


def _run_habana_subcase(subcase: str) -> None:
    import torch
    import habana_frameworks.torch.core as htcore
    import habana_frameworks.torch.hpu as hthpu
    from habana_frameworks.torch.utils.internal import is_lazy

    assert is_lazy(), f"Expected PT_HPU_LAZY_MODE=1, got is_lazy()={is_lazy()}"
    hpu = torch.device("hpu")

    if subcase == "eig_f32_no_ext":
        x = torch.randn(4, 4, device=hpu)
        htcore.mark_step()
        try:
            torch.linalg.eig(x)
        except RuntimeError as exc:
            print(f"[habana-bug] eig_f32_no_ext raised expected RuntimeError: {exc}")
            assert "Complex datatype is not supported" in str(exc), exc
            return
        raise AssertionError("Expected Habana RuntimeError for linalg.eig(float32) in lazy mode")

    if subcase == "set_dropped_synstatus30":
        torch.manual_seed(0)
        x = torch.randn(4, 4, device=hpu)
        w = torch.randn(4, 4, device=hpu)
        out = x @ w
        htcore.mark_step()
        s = out.untyped_storage()
        del out
        g = torch.empty(0, dtype=torch.float32, device=hpu).set_(s, 0, (4, 4), (4, 1))
        try:
            y = g * 2
            htcore.mark_step()
            hthpu.synchronize()
            _ = y.cpu()
        except RuntimeError as exc:
            print(f"[habana-bug] set_dropped_synstatus30 raised expected RuntimeError: {exc}")
            assert "synStatus 30" in str(exc), exc
            return
        raise AssertionError("Expected synStatus 30 when lazy op output behind storage() is dropped")

    if subcase == "set_strided_silent_wrong":
        f = torch.arange(16, dtype=torch.float32, device=hpu)
        htcore.mark_step()
        hthpu.synchronize()
        g = torch.empty(0, dtype=torch.float32, device=hpu).set_(f.untyped_storage(), 4, (3, 2), (4, 1))
        y = g + 10
        htcore.mark_step()
        hthpu.synchronize()
        ref = f.cpu().as_strided((3, 2), (4, 1), 4) + 10
        diff = float((y.cpu() - ref).abs().max())
        print(f"[habana-bug] set_strided_silent_wrong max_diff={diff}")
        assert diff > 1.0, f"Expected silent wrong strided set_ result, got diff={diff}"
        return

    if subcase == "set_copy_silent_wrong":
        f = torch.zeros(8, dtype=torch.float32, device=hpu)
        htcore.mark_step()
        hthpu.synchronize()
        g = torch.empty(0, dtype=torch.float32, device=hpu).set_(f.untyped_storage(), 0, (8,), (1,))
        g.copy_(torch.full((8,), 7.0, device=hpu))
        htcore.mark_step()
        hthpu.synchronize()
        diff = float((f.cpu() - 7.0).abs().max())
        print(f"[habana-bug] set_copy_silent_wrong max_diff={diff}")
        assert diff > 1.0, f"Expected silent wrong copy_() into set_ view, got diff={diff}"
        return

    raise ValueError(f"Unknown habana-bug subcase: {subcase}")


def run_habana_bug() -> None:
    subcases = [
        "eig_f32_no_ext",
        "set_dropped_synstatus30",
        "set_strided_silent_wrong",
        "set_copy_silent_wrong",
    ]
    for sc in subcases:
        rc, _, _ = _spawn("habana-bug", "1", subcase=sc)
        if rc != 0:
            raise SystemExit(rc)


def run_guarded() -> None:
    import torch
    import habana_frameworks.torch.core as htcore
    import habana_frameworks.torch.hpu as hthpu
    from habana_frameworks.torch.utils.internal import is_lazy
    import hpu_complex

    assert is_lazy(), f"Expected PT_HPU_LAZY_MODE=1, got is_lazy()={is_lazy()}"
    hpu = torch.device("hpu")
    hpu_complex.enable()

    # 1. Real-only float32 lazy ops must continue working cleanly.
    torch.manual_seed(0)
    x_cpu = torch.randn(4, 4, requires_grad=True)
    x_hpu = x_cpu.detach().clone().to(hpu).requires_grad_()
    loss_cpu = (x_cpu @ x_cpu.T).sum()
    loss_hpu = (x_hpu @ x_hpu.T).sum()
    loss_cpu.backward()
    loss_hpu.backward()
    htcore.mark_step()
    hthpu.synchronize()
    assert float((loss_hpu.detach().cpu() - loss_cpu.detach()).abs()) < 1e-4
    assert float((x_hpu.grad.cpu() - x_cpu.grad).abs().max()) < 1e-4

    # 2. Device-agnostic float-pair API works in lazy mode.
    sig = torch.randn(2, 64, device=hpu, requires_grad=True)
    spec = hpu_complex.rfft(sig)
    stft_mag = hpu_complex.stft_magnitude(sig, 32, 16)
    (spec.abs().sum() + stft_mag.sum()).backward()
    htcore.mark_step()
    hthpu.synchronize()
    assert torch.isfinite(sig.grad.cpu()).all()
    print("[guarded] real-only lazy fwd+bwd and hpu_complex.rfft/stft_magnitude: PASS")

    # 3. complex64 / torch.fft / linalg.eig(float32) compute in lazy mode, against the CPU.
    torch.manual_seed(0)
    pair, re, im = torch.randn(4, 4, 2), torch.randn(4, 4), torch.randn(4, 4)
    sig, mat, c64 = torch.randn(4, 32), torch.randn(4, 4), torch.randn(4, 4, dtype=torch.complex64)
    lazy_ops = [
        ("zeros_c64", lambda d: torch.zeros(4, 4, dtype=torch.complex64, device=d)),
        ("ones_c64", lambda d: torch.ones(4, 4, dtype=torch.complex64, device=d)),
        ("to_hpu_c64", lambda d: c64.to(d)),
        ("view_as_complex", lambda d: torch.view_as_complex(pair.to(d))),
        ("torch.complex", lambda d: torch.complex(re.to(d), im.to(d))),
        ("torch.fft.rfft", lambda d: torch.fft.rfft(sig.to(d))),
        ("torch.linalg.eigvals(f32).abs", lambda d: torch.linalg.eigvals(mat.to(d)).abs().sort(-1).values),
        ("real * 2j (aborted the process before)", lambda d: re.to(d) * 2j),
    ]
    for name, fn in lazy_ops:
        got = fn(hpu)
        htcore.mark_step()
        ref = fn("cpu")
        diff = float((torch.view_as_real(got.cpu()) - torch.view_as_real(ref)).abs().max() if ref.is_complex()
                     else (got.cpu() - ref).abs().max())
        assert diff < 1e-4, (name, diff)
        print(f"[guarded] {name} computes in lazy mode: max_diff={diff:.1e}")
    try:
        with torch.inference_mode():
            torch.fft.rfft(sig.to(hpu))
    except RuntimeError as exc:
        assert "PT_HPU_LAZY_MODE=0" in str(exc), str(exc)
        print(f"[guarded] complex64 under inference_mode raised clean RuntimeError: {str(exc).splitlines()[0]}")
    else:
        raise AssertionError("Expected RuntimeError for complex64 under inference_mode in lazy mode")


def run_workaround() -> None:
    import torch
    from habana_frameworks.torch.utils.internal import is_lazy
    import hpu_complex

    assert not is_lazy(), f"Expected PT_HPU_LAZY_MODE=0, got is_lazy()={is_lazy()}"
    hpu = torch.device("hpu")
    hpu_complex.enable()

    torch.manual_seed(0)
    xc = torch.randn(4, 4, dtype=torch.complex64, requires_grad=True)
    rc = torch.randn(4, 32, requires_grad=True)
    xh = xc.detach().clone().to(hpu).requires_grad_()
    rh = rc.detach().clone().to(hpu).requires_grad_()

    def fn(z, r):
        return (z @ z).abs().sum() + torch.fft.rfft(r).abs().sum()

    loss_ref = fn(xc, rc)
    loss_ref.backward()

    loss_eager = fn(xh, rh)
    loss_eager.backward()
    torch.hpu.synchronize()
    assert float((loss_eager.detach().cpu() - loss_ref.detach()).abs()) < 1e-4
    assert float((xh.grad.cpu() - xc.grad).abs().max()) < 1e-4
    assert float((rh.grad.cpu() - rc.grad).abs().max()) < 1e-4
    print("[workaround] eager PT_HPU_LAZY_MODE=0 complex64 + rfft fwd+bwd: PASS")

    # Simple complex+FFT module also works with torch.compile(backend="hpu_backend") in eager mode.
    # Caveat: compiled spd_learn.models.Green fails with "strided_view synStatus 1" at its
    # float32 aten._linalg_eigh node (repro/compiled_real_eigh.py); it trains in plain eager mode.
    xh2 = xc.detach().clone().to(hpu).requires_grad_()
    rh2 = rc.detach().clone().to(hpu).requires_grad_()
    compiled_fn = torch.compile(fn, backend="hpu_backend")
    loss_comp = compiled_fn(xh2, rh2)
    loss_comp.backward()
    torch.hpu.synchronize()
    assert float((loss_comp.detach().cpu() - loss_ref.detach()).abs()) < 1e-4
    assert float((xh2.grad.cpu() - xc.grad).abs().max()) < 1e-4
    assert float((rh2.grad.cpu() - rc.grad).abs().max()) < 1e-4
    print("[workaround] eager + torch.compile(backend='hpu_backend') simple module: PASS")


def main() -> None:
    ap = argparse.ArgumentParser(description="Issue #3 lazy-mode complex64 reproducer")
    ap.add_argument("--mode", choices=["habana-bug", "guarded", "workaround", "all"], default="all")
    ap.add_argument("--in-process", action="store_true")
    ap.add_argument("--subcase", type=str, default=None)
    args = ap.parse_args()

    if args.in_process:
        if args.mode == "habana-bug":
            _run_habana_subcase(args.subcase)
        elif args.mode == "guarded":
            run_guarded()
        elif args.mode == "workaround":
            run_workaround()
        return

    if args.mode == "habana-bug":
        run_habana_bug()
        return

    if args.mode == "guarded":
        if os.environ.get("PT_HPU_LAZY_MODE") == "1":
            run_guarded()
        else:
            rc, _, _ = _spawn("guarded", "1")
            raise SystemExit(rc)
        return

    if args.mode == "workaround":
        if os.environ.get("PT_HPU_LAZY_MODE", "0") == "0":
            run_workaround()
        else:
            rc, _, _ = _spawn("workaround", "0")
            raise SystemExit(rc)
        return

    run_habana_bug()
    for mode, lazy_env in (("guarded", "1"), ("workaround", "0")):
        rc, _, _ = _spawn(mode, lazy_env)
        if rc != 0:
            raise SystemExit(rc)
    print("ALL_REPRO_MODES_OK")


if __name__ == "__main__":
    main()
