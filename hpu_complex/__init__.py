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


def _touches_complex(node):
    """An FX node with a complex output or a complex input."""
    import torch

    def complex_dtypes(n):
        dtypes = list(n.meta.get("output_dtypes") or ())
        val = n.meta.get("val")
        for v in val if isinstance(val, (list, tuple)) else (val,):
            if isinstance(v, torch.Tensor):
                dtypes.append(v.dtype)
        return any(isinstance(d, torch.dtype) and d.is_complex for d in dtypes)

    return complex_dtypes(node) or any(complex_dtypes(n) for n in node.all_input_nodes)


def _complex_nodes_run_eagerly():
    """torch.compile(backend="hpu_backend"): complex nodes run eagerly, on the C++ kernels.

    Synapse has no complex dtype, yet the partitioner fuses complex nodes into its
    graphs and then fails allocating their outputs. Eager fallback keeps the float
    nodes around them in the fused graphs. The as_strided batching pass must skip
    them too: its hpu::batch_as_strided op has no complex kernel (CodeBrain).
    """
    try:
        from habana_frameworks.torch.dynamo.compile_backend import passes
        from habana_frameworks.torch.dynamo.compile_backend._passes import batch_as_strided
    except ImportError:
        return
    original = passes.is_eager_fallback_required
    if getattr(original, "hpu_complex", False):
        return

    def is_eager_fallback_required(node, is_dynamic=False):
        return _touches_complex(node) or original(node, is_dynamic=is_dynamic)

    is_eager_fallback_required.hpu_complex = True
    passes.is_eager_fallback_required = is_eager_fallback_required

    support = batch_as_strided.BatchAsStridedOperatorSupport
    batchable = support.is_node_supported
    support.is_node_supported = lambda self, submodules, node: (
        not _touches_complex(node) and batchable(self, submodules, node))


def enable(enter_mode: bool = False):
    """Activate: C++ kernels (Gaudi stack); complex nodes of compiled graphs run eagerly on them.

    After this, unmodified torch.fft/torch.stft/torch.complex work on HPU,
    eager and under torch.compile(backend="hpu_backend"). Idempotent. The
    kernels see only complex tensors (PrivateUse1 key), so float code is
    untouched. On non-Gaudi machines only the CPU decompositions register.

    The complex-arithmetic TorchDispatchMode is NOT entered by default; it
    is only needed for features that hook the mode itself -- the fused TPC
    FFT (hpu_complex.tpc.enable_tpc_fft) -- via enter_mode=True here, or
    preferably scoped with complex_mode().
    """
    global _REGISTERED, _MODE_ENTERED
    import torch

    if hasattr(torch, "hpu") and torch.hpu.is_available():
        from .extension import load_extension

        load_extension()
        _complex_nodes_run_eagerly()
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
