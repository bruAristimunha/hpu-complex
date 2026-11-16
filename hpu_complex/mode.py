"""Complex arithmetic via TorchDispatchMode (complex tensors only).

Why a mode, not C++ kernels: in Habana's fork, mul/sub/div/neg are
composite aliases of each other, so C++ fallbacks for the float cases
recurse into themselves (measured: segfault in randn). A dispatch mode
sits above the dispatcher, intercepts only when a complex HPU tensor is
involved, and computes with plain float ops -- float code paths are
never touched.
"""
import torch
from torch.utils._python_dispatch import TorchDispatchMode

aten = torch.ops.aten

# Optional fused-FFT hook, set by hpu_complex.tpc.enable_tpc_fft().
FFT_R2C_HOOK = None


def _pair(t):
    if t.is_complex():
        if t.is_conj():
            t = t.conj_physical()
        return torch.view_as_real(t)
    return torch.stack([t, torch.zeros_like(t)], dim=-1)


def _mul(a, b):
    pa, pb = _pair(a), _pair(b)
    re = pa[..., 0] * pb[..., 0] - pa[..., 1] * pb[..., 1]
    im = pa[..., 0] * pb[..., 1] + pa[..., 1] * pb[..., 0]
    return torch.view_as_complex(torch.stack([re, im], dim=-1).contiguous())


def _div(a, b):
    pa, pb = _pair(a), _pair(b)
    denom = pb[..., 0] ** 2 + pb[..., 1] ** 2
    re = (pa[..., 0] * pb[..., 0] + pa[..., 1] * pb[..., 1]) / denom
    im = (pa[..., 1] * pb[..., 0] - pa[..., 0] * pb[..., 1]) / denom
    return torch.view_as_complex(torch.stack([re, im], dim=-1).contiguous())


class ComplexMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is aten._fft_r2c.default and FFT_R2C_HOOK is not None:
            for _attempt in (1, 2):
                try:
                    out = FFT_R2C_HOOK(*args)
                    if out is not None:
                        return out
                    break
                except Exception:
                    # First graph compile after a custom-kernel graph can
                    # fail once on this stack; retry, then fall back.
                    if _attempt == 2:
                        pass
            # fall through to the GEMM path on None/failure
        from torch.utils._python_dispatch import _disable_current_modes

        with _disable_current_modes():
            return self._dispatch(func, args, kwargs)

    def _dispatch(self, func, args, kwargs):
        tensors = [t for t in args if isinstance(t, torch.Tensor)]
        if not any(t.is_complex() for t in tensors):
            try:
                return func(*args, **kwargs)
            except RuntimeError as e:
                # After a custom TPC kernel graph runs, the next
                # strided_view graph compile fails once, then heals.
                if "synNodeCreateWithId" in str(e):
                    return func(*args, **kwargs)
                raise
        if func is aten.mul.Tensor:
            return _mul(args[0], args[1])
        if func is aten.div.Tensor:
            return _div(args[0], args[1])
        if func is aten.add.Tensor:
            alpha = args[2] if len(args) > 2 else kwargs.get("alpha", 1)
            pair = _pair(args[0]) + _pair(args[1]) * alpha
            return torch.view_as_complex(pair.contiguous())
        if func is aten.sub.Tensor:
            alpha = args[2] if len(args) > 2 else kwargs.get("alpha", 1)
            pair = _pair(args[0]) - _pair(args[1]) * alpha
            return torch.view_as_complex(pair.contiguous())
        if func is aten.neg.default:
            pair = torch.view_as_real(args[0])
            return torch.view_as_complex((-pair).contiguous())
        if func is aten.view_as_real.default:
            # With the TPC plugin loaded, the stock view path breaks on
            # extension-built tensors (strided_view); ours does not.
            z = args[0]
            if z.is_conj():
                z = z.conj_physical()
            return torch.stack([z.real, z.imag], dim=-1)
        if func is aten.view_as_complex.default:
            pair = args[0]
            return torch.complex(pair[..., 0].contiguous(),
                                 pair[..., 1].contiguous())
        if func is aten.conj_physical.default:
            pair = torch.view_as_real(args[0])
            out = torch.stack([pair[..., 0], -pair[..., 1]], dim=-1)
            return torch.view_as_complex(out.contiguous())
        return func(*args, **kwargs)


MODE = ComplexMode()
