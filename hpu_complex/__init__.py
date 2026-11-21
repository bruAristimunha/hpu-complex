"""hpu_complex: real torch.fft/complex64 on Gaudi, without Intel's compiler.

User code stays unmodified. A complex64 HPU tensor is a complex view over
interleaved float storage (re, im) -- the layout CUDA (cuComplex/float2) and
MPS use -- tagged with the PrivateUse1 dispatch key. The extension's kernels
(cpp_ext/fft_ext.cpp) compute on float views of that storage, so the
complex dtype never crosses into SynapseAI, whose type system has no
complex entry.

enable() is safe to call anywhere: without a Gaudi it does nothing.
hpu_complex.rfft/irfft/fft/stft (the _core module) are a separate,
device-agnostic float API on [..., 2] pairs.
"""
import importlib.util
import os
import warnings

from . import _core as core
from ._core import fft, irfft, magnitude, rfft, stft, stft_magnitude

_MODE_ENTERED = False


def _touches_complex(node):
    """An FX node with a complex output or a complex input."""
    import torch
    from torch.utils._pytree import tree_leaves

    def complex_dtypes(n):
        dtypes = list(n.meta.get("output_dtypes") or ())
        dtypes += [v.dtype for v in tree_leaves(n.meta.get("val")) if isinstance(v, torch.Tensor)]
        return any(isinstance(d, torch.dtype) and d.is_complex for d in dtypes)

    return complex_dtypes(node) or any(complex_dtypes(n) for n in node.all_input_nodes)


def _complex_out(node):
    """An FX node with a complex output."""
    import torch
    from torch.utils._pytree import tree_leaves

    return any(isinstance(v, torch.Tensor) and v.is_complex() for v in tree_leaves(node.meta.get("val")))


def _is_view(node):
    return node.op == "call_function" and bool(getattr(node.target, "is_view", False))


def _views_complex_storage(node):
    """A float view of complex storage: view_as_real / real / imag / view(dtype) of a complex node, or any view of
    such a view (transitive), e.g. view_as_real(z.diagonal())[..., 0].mT."""
    while _is_view(node) and not _complex_out(node):
        inputs = node.all_input_nodes
        if not inputs:
            return False
        node = inputs[0]  # a view op's tensor argument comes first
        if _complex_out(node):
            return True
    return False


def _view_chain_reaches_complex(node):
    """node, or a view on the single-user view chain the fusion pass would follow from it, touches complex
    storage (complex dtype in or out, or a float view of complex storage): that chain must not become one
    as_strided, whose strides would come from fake tensors (eig's V.real.mT, x.view(4, 6).view(complex64))."""
    while True:
        if _touches_complex(node) or _views_complex_storage(node):
            return True
        if len(node.users) != 1:
            return False
        node = next(iter(node.users))
        if not _is_view(node):
            return False


def _runs_eagerly(node):
    """A complex node, or a reader of a float view of complex storage (see is_eager_fallback_required)."""
    return _touches_complex(node) or any(_views_complex_storage(n) for n in node.all_input_nodes)


def _complex_nodes_run_eagerly():
    """torch.compile(backend="hpu_backend"): complex nodes run eagerly, on the C++ kernels.

    Synapse has no complex dtype, yet the partitioner fuses complex nodes into its
    graphs and then fails allocating their outputs. Eager fallback keeps the float
    nodes around them in the fused graphs. The as_strided batching pass must skip
    them too: its hpu::batch_as_strided op has no complex kernel (CodeBrain).
    """
    try:
        from habana_frameworks.torch.dynamo.compile_backend import passes
    except ImportError:
        warnings.warn("hpu_complex: Habana's torch.compile passes not found; complex nodes of "
                      "hpu_backend graphs are not forced to run eagerly")
        return
    original = passes.is_eager_fallback_required
    if getattr(original, "hpu_complex", False):
        return

    def is_eager_fallback_required(node, is_dynamic=False):
        # A node that reads a float view of complex storage runs eagerly too, however long the view chain: the view
        # can be strided (view_as_real of a diagonal in the qr backward, then select), and a fused graph misreads it
        # (compiled qr gradient 0.3-0.7 off with no error, Gaudi2, bridge 1.21.4). Fresh float results of complex
        # nodes (abs, angle) and of those eager readers stay fusable.
        return _runs_eagerly(node) or original(node, is_dynamic=is_dynamic)

    is_eager_fallback_required.hpu_complex = True
    passes.is_eager_fallback_required = is_eager_fallback_required

    # pass_mark_placement settles HPU-to-HPU _to_copy nodes (casts to or from complex) before it asks
    # the predicate above, so its placement of complex nodes is overruled afterwards. Their old placement
    # is dropped first: on a second run the pass asserts that it recomputes the stored one.
    mark_placement = getattr(passes, "pass_mark_placement", None)
    if mark_placement is not None:
        def pass_mark_placement(ctx):
            complex_nodes = [n for n in ctx.graph_module.graph.nodes
                             if n.op == "call_function" and _runs_eagerly(n)]
            for node in complex_nodes:
                node.meta.pop("placement", None)
            changed = mark_placement(ctx)
            for node in complex_nodes:
                node.meta["placement"] = "eager"
            return changed

        passes.pass_mark_placement = pass_mark_placement

    # pass_fuse_view_chains replaces a chain of views by one as_strided with the strides traced by the fake
    # tensors. For complex linalg outputs those differ from the real ones (torch's meta of linalg_eig gives
    # contiguous eigenvectors, the CPU kernel column-major ones), so the baked as_strided reads the matrix
    # transposed: compiled eig/eigvals gradients were wrong with no error. No chain starts at, or runs into, a view
    # touching complex storage: complex views, float views of it (V.real.mT) and float views feeding view(complex).
    fuse_view_chains = getattr(passes, "pass_fuse_view_chains", None)
    if fuse_view_chains is not None:
        def pass_fuse_view_chains(ctx):
            for node in ctx.graph_module.graph.nodes:
                if node.op == "call_function" and _view_chain_reaches_complex(node):
                    node.meta["visited"] = True  # the pass starts no chain at a visited node
            return fuse_view_chains(ctx)

        passes.pass_fuse_view_chains = pass_fuse_view_chains

    try:  # private, and absent before 1.20: its absence must not cost the patches above
        from habana_frameworks.torch.dynamo.compile_backend._passes import batch_as_strided
    except ImportError:
        return
    support = batch_as_strided.BatchAsStridedOperatorSupport
    batchable = support.is_node_supported
    support.is_node_supported = lambda self, submodules, node: (
        not _runs_eagerly(node) and not _views_complex_storage(node) and batchable(self, submodules, node))


def _bridge_is_lazy():
    """Whether the bridge runs in lazy mode. Without its is_lazy() (older bridges, some of which default to lazy),
    PT_HPU_LAZY_MODE decides, and unset means lazy: the complex entry points then raise a clear error instead of
    aborting the process in Habana's lazy kernels (bridge 1.21.4 has is_lazy() and defaults to eager, measured)."""
    try:
        from habana_frameworks.torch.utils.internal import is_lazy

        return bool(is_lazy())
    except Exception:  # noqa: BLE001  any failure of a private API
        env = os.environ.get("PT_HPU_LAZY_MODE")
        if env is None:
            warnings.warn("hpu_complex: cannot ask the bridge whether it runs lazily and PT_HPU_LAZY_MODE is unset; "
                          "assuming lazy mode (complex64 on HPU raises). Set PT_HPU_LAZY_MODE=0 for eager mode")
            return True
        return env != "0"


def enable(enter_mode: bool = False):
    """Activate: C++ kernels (Gaudi stack); complex nodes of compiled graphs run eagerly on them.

    After this, unmodified torch.fft/torch.stft/torch.complex work on HPU,
    eager and under torch.compile(backend="hpu_backend"). Idempotent. The
    compute kernels see only complex tensors (PrivateUse1 key); float HPU
    ops reach the extension only as a dtype check in empty/empty_strided
    at BackendSelect. Without a Gaudi it does nothing, and warns when
    habana_frameworks is installed but torch.hpu is not loaded yet.

    The complex-arithmetic TorchDispatchMode is NOT entered by default; it
    is only needed for features that hook the mode itself -- the fused TPC
    FFT (hpu_complex.tpc.enable_tpc_fft) -- via enter_mode=True here, or
    preferably scoped with complex_mode().
    """
    global _MODE_ENTERED
    import torch

    if hasattr(torch, "hpu") and torch.hpu.is_available():
        from .extension import load_extension

        load_extension()
        lazy = _bridge_is_lazy()
        torch.ops.hpu_complex.set_lazy_mode(lazy)
        _complex_nodes_run_eagerly()
    elif not hasattr(torch, "hpu") and importlib.util.find_spec("habana_frameworks") is not None:
        warnings.warn("hpu_complex: habana_frameworks is installed but torch.hpu is not loaded, so the "
                      "extension is not enabled; import habana_frameworks.torch.core before "
                      "hpu_complex.enable()")
    if enter_mode and not _MODE_ENTERED:
        from .mode import MODE

        MODE.__enter__()
        _MODE_ENTERED = True


def set_matmul_cpu_threshold(macs: int):
    """Complex mm/bmm with fewer multiply-adds (batch * m * k * n) than ``macs`` run on the CPU (warned once).

    Default 2**22: below it, one copy each way and the CPU product beat the card's GEMM and its layout copies on
    Gaudi2 (bridge 1.21.4). 0 keeps every product on the card. Needs ``enable()`` on a Gaudi first.
    """
    import torch

    torch.ops.hpu_complex.set_matmul_cpu_threshold(int(macs))


def complex_mode():
    """Scope the complex-arithmetic dispatch mode to a block.

    Inside the block, ops involving a complex tensor are computed from
    float ops by the mode; every other op passes through untouched.
    Outside it, complex ops run on the extension's PrivateUse1 kernels. Needed
    only to activate mode hooks such as the fused TPC FFT.
    """
    from .mode import ComplexMode

    return ComplexMode()


__all__ = ["core", "enable", "complex_mode", "set_matmul_cpu_threshold", "rfft", "irfft", "fft",
           "magnitude", "stft", "stft_magnitude"]
