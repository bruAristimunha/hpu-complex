"""hpu_complex: real torch.fft/complex64 on Gaudi, without Intel's compiler.

User code stays unmodified. Internally a complex spectrum is carried as a
float tensor [..., 2] (interleaved re/im) -- the same representation CUDA
(cuComplex/float2) and MPS (MPSGraph complex) use. The complex dtype never
crosses into SynapseAI, whose type system has no complex entry.

Two activation paths, decided by probes.py on the target cluster:
  A) plugin  -- torch.library registrations only, no bridge rebuild
  B) fork    -- same decompositions registered inside a bridge fork

enable() is safe to call anywhere: on non-HPU machines it only registers
the (device-agnostic) decompositions for testing.
"""
from . import _core as core
from ._core import fft, irfft, magnitude, rfft, stft, stft_magnitude

_REGISTERED = False


def enable():
    """Activate: C++ kernels (Gaudi stack) + decomps + dispatch mode.

    After this, unmodified torch.fft/torch.stft/torch.complex work on HPU.
    Idempotent. On non-Gaudi machines only the decompositions register.
    """
    global _REGISTERED
    if _REGISTERED:
        return
    import torch

    if hasattr(torch, "hpu") and torch.hpu.is_available():
        from .extension import load_extension

        load_extension()
        from .mode import MODE

        MODE.__enter__()
    from .decompositions import register_all

    register_all()
    _REGISTERED = True


__all__ = ["core", "enable", "rfft", "irfft", "fft", "magnitude",
           "stft", "stft_magnitude"]
