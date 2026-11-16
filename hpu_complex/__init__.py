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
_MODE_ENTERED = False


def enable(enter_mode: bool = False):
    """Activate: C++ kernels (Gaudi stack) + decompositions.

    After this, unmodified torch.fft/torch.stft/torch.complex work on HPU.
    Idempotent. On non-Gaudi machines only the decompositions register.

    The complex-arithmetic TorchDispatchMode is NOT entered by default:
    forward complex mul/add/div already route through the C++ .out
    kernels (the fork's composite aliases land there), so a process-wide
    mode would tax every unrelated op for no benefit. Enter it only for
    features that hook the mode itself -- the fused TPC FFT
    (hpu_complex.tpc.enable_tpc_fft) -- via enter_mode=True here, or
    preferably scoped with complex_mode().
    """
    global _REGISTERED, _MODE_ENTERED
    import torch

    if hasattr(torch, "hpu") and torch.hpu.is_available():
        from .extension import load_extension

        load_extension()
    if not _REGISTERED:
        from .decompositions import register_all

        register_all()
        _REGISTERED = True
    if enter_mode and not _MODE_ENTERED:
        from .mode import MODE

        MODE.__enter__()
        _MODE_ENTERED = True


def complex_mode():
    """Scope the complex-arithmetic dispatch mode to a block.

    Inside the block, ops involving a complex tensor are computed from
    float ops by the mode; every other op passes through untouched.
    Outside it, complex arithmetic uses the C++ .out kernels. Needed
    only to activate mode hooks such as the fused TPC FFT.
    """
    from .mode import ComplexMode

    return ComplexMode()


__all__ = ["core", "enable", "complex_mode", "rfft", "irfft", "fft",
           "magnitude", "stft", "stft_magnitude"]
