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
import math
import warnings

from . import _core as core
from ._core import fft, irfft, magnitude, rfft, stft, stft_magnitude

_MODE_ENTERED = False


def _node_dtypes(n):
    import torch
    from torch.utils._pytree import tree_leaves

    dtypes = list(n.meta.get("output_dtypes") or ())
    dtypes += [v.dtype for v in tree_leaves(n.meta.get("val")) if isinstance(v, torch.Tensor)]
    return [d for d in dtypes if isinstance(d, torch.dtype)]


def _float64_may_be_host_backed(graph):
    """Whether float64 values of this graph may live in host memory: it has a complex128 value, or the process has
    allocated a complex128 HPU tensor already (a float64 graph input may then be a view or a result of one)."""
    import torch

    cached = getattr(graph, "_hpu_complex_f64_host", None)
    if cached is None:
        cached = (torch.ops.hpu_complex.host_in_use()
                  or any(torch.complex128 in _node_dtypes(n) for n in graph.nodes))
        graph._hpu_complex_f64_host = cached
    return cached


def _touches_complex(node):
    """An FX node with a complex output or a complex input, or a float64 one when float64 may be host-backed.

    Host-backed tensors (complex128 and its float64 views and results) are host memory reporting the HPU device: a
    Synapse graph would read that pointer as device memory (a segfault). Their nodes run eagerly instead.
    """
    import torch

    def touched(n):
        dtypes = _node_dtypes(n)
        return (any(d.is_complex for d in dtypes)
                or (torch.float64 in dtypes and _float64_may_be_host_backed(node.graph)))

    return touched(node) or any(touched(n) for n in node.all_input_nodes)


def _holds_host_values(gm):
    """A Dynamo graph whose values may be host-backed: a complex128 value, or float64 once complex128 was allocated."""
    import torch

    host_in_use = torch.ops.hpu_complex.host_in_use()
    for node in gm.graph.nodes:
        dtypes = {getattr(v, "dtype", None) for v in (node.meta.get("example_value"), node.meta.get("val"))}
        if torch.complex128 in dtypes or (host_in_use and torch.float64 in dtypes):
            return True
    return False


def _complex128_graphs_run_eagerly():
    """torch.compile(backend="hpu_backend") of a graph holding complex128 values: the graph runs eagerly, whole.

    Its complex128 ops compute on the CPU either way. Measured (fix round 2, Gaudi2, bridge 1.21.4): with every node
    of such a graph placed eagerly, hpu_backend still returned a wrong gradient through eig (max error 1.6; aot_eager
    with the same nodes was exact), and before that fix float64 nodes in Synapse graphs read host memory (a segfault).
    """
    import functools

    from torch._dynamo.backends import registry

    try:
        original = registry.lookup_backend("hpu_backend")
    except Exception:  # noqa: BLE001 (InvalidBackend: Habana's backend is not registered)
        return
    if getattr(original, "hpu_complex", False):
        return

    @functools.wraps(original)
    def hpu_backend(gm, example_inputs, **kwargs):
        if _holds_host_values(gm):
            return gm.forward
        return original(gm, example_inputs, **kwargs)

    hpu_backend.hpu_complex = True
    registry._COMPILER_FNS["hpu_backend"] = hpu_backend


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
        return _touches_complex(node) or original(node, is_dynamic=is_dynamic)

    is_eager_fallback_required.hpu_complex = True
    passes.is_eager_fallback_required = is_eager_fallback_required

    # pass_mark_placement settles HPU-to-HPU _to_copy nodes (casts to or from complex) before it asks
    # the predicate above, so its placement of complex nodes is overruled afterwards. Their old placement
    # is dropped first: on a second run the pass asserts that it recomputes the stored one.
    mark_placement = getattr(passes, "pass_mark_placement", None)
    if mark_placement is not None:
        def pass_mark_placement(ctx):
            complex_nodes = [n for n in ctx.graph_module.graph.nodes
                             if n.op == "call_function" and _touches_complex(n)]
            for node in complex_nodes:
                node.meta.pop("placement", None)
            changed = mark_placement(ctx)
            for node in complex_nodes:
                node.meta["placement"] = "eager"
            return changed

        passes.pass_mark_placement = pass_mark_placement

    _complex128_graphs_run_eagerly()

    try:  # private, and absent before 1.20: its absence must not cost the patches above
        from habana_frameworks.torch.dynamo.compile_backend._passes import batch_as_strided
    except ImportError:
        return
    support = batch_as_strided.BatchAsStridedOperatorSupport
    batchable = support.is_node_supported
    support.is_node_supported = lambda self, submodules, node: (
        not _touches_complex(node) and batchable(self, submodules, node))


def _host_backed(t):
    """A complex128 HPU tensor or a float64 view of one: tagged (PrivateUse1) and not complex64, so in host memory."""
    import torch

    return (t.device.type == "hpu" and t.dtype != torch.complex64
            and torch._C._dispatch_keys(t).has(torch._C.DispatchKey.PrivateUse1))


def _pickle_complex128_through_the_cpu():
    """torch.save of a complex128 HPU tensor (or a float64 view of one): pickled as its CPU values.

    Their storage is host memory reporting the HPU device; torch.save would copy it out through Habana's own storage
    copy, which reads the pointer as device memory. They rebuild with .to(device), honoring map_location. Each tensor
    is pickled on its own, so tensors that were views of one another load as independent copies.
    """
    import torch

    original = torch.Tensor._reduce_ex_internal
    if getattr(original, "hpu_complex", False):
        return

    def _reduce_ex_internal(self, proto):
        if _host_backed(self):
            # Its bytes as a latin-1 str, not a CPU tensor: torch.load(map_location="hpu") would restore a tensor's
            # storage on the card, at float32 precision for float64; and weights_only=True accepts a str, not bytes.
            values = self.detach().cpu().resolve_conj().resolve_neg().contiguous()
            return (_rebuild_host_backed_from_cpu,
                    (values.numpy().tobytes().decode("latin-1"), self.dtype, str(self.device), self.requires_grad, tuple(values.shape)))
        return original(self, proto)

    _reduce_ex_internal.hpu_complex = True
    torch.Tensor._reduce_ex_internal = _reduce_ex_internal
    torch.serialization.add_safe_globals([_rebuild_host_backed_from_cpu])  # torch.load's default weights_only=True


def _rebuild_host_backed_from_cpu(data, dtype, device, requires_grad, size=None):
    """torch.load of a tensor _pickle_complex128_through_the_cpu saved: back into host memory on the HPU (exact).

    torch's own _rebuild_device_tensor_from_cpu_tensor would rebuild a float64 view or result of a complex128 tensor
    with .to(device): an ordinary Habana float64 tensor, at float32 precision. map_location is honored; anywhere but the
    HPU it is that .to(device).
    """
    import torch

    if isinstance(data, str):
        n = math.prod(size)
        data = (torch.frombuffer(bytearray(data.encode("latin-1")), dtype=dtype, count=n) if n else torch.empty(0, dtype=dtype)).reshape(size)
    device = torch.device(torch._utils._get_restore_location(device))
    if device.type == "hpu" and dtype in (torch.float64, torch.complex128) and hasattr(torch.ops.hpu_complex,
                                                                                     "host_from_cpu"):
        tensor = torch.ops.hpu_complex.host_from_cpu(data.to(dtype), device)
    else:
        tensor = data.to(dtype=dtype, device=device)
    tensor.requires_grad = requires_grad
    return tensor


def _deepcopy_complex128_in_host_memory():
    """copy.deepcopy of a complex128 HPU tensor (or a float64 view of one): a copy of its host memory.

    torch's own __deepcopy__ clones the storage through Habana's storage copy, which reads the host pointer as device
    memory (a segfault in eager mode). Here the storage is copied by the extension (hpu_complex::host_deepcopy), once per
    storage and deepcopy call, so tensors that alias one another still alias in the copy, as torch keeps them. The rest
    is torch's __deepcopy__: graph leaves only, conjugate/negative bits resolved, requires_grad, .grad and __dict__.
    """
    import copy

    import torch

    original = torch.Tensor.__deepcopy__
    if getattr(original, "hpu_complex", False):
        return

    def __deepcopy__(self, memo):
        if (type(self) is not torch.Tensor or not self.is_leaf or not _host_backed(self)
                or torch.overrides.has_torch_function_unary(self)):
            return original(self, memo)
        if id(self) in memo:
            return memo[id(self)]
        copies = memo.setdefault("hpu_complex", {})  # storage -> a copy over its copied storage
        key = self.untyped_storage()._cdata
        with torch.no_grad():
            new = torch.ops.hpu_complex.host_deepcopy(self, copies.get(key))
            copies.setdefault(key, new)
            if self.is_conj():
                new = new.conj_physical()
            if self.is_neg():
                new = new.neg()
        if self.requires_grad:
            new.requires_grad_()
        if self.grad is not None:
            new.grad = copy.deepcopy(self.grad, memo)
        new.__dict__ = copy.deepcopy(self.__dict__, memo)
        memo[id(self)] = new
        return new

    __deepcopy__.hpu_complex = True
    torch.Tensor.__deepcopy__ = __deepcopy__


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
        _complex_nodes_run_eagerly()
        _pickle_complex128_through_the_cpu()
        _deepcopy_complex128_in_host_memory()
    elif not hasattr(torch, "hpu") and importlib.util.find_spec("habana_frameworks") is not None:
        warnings.warn("hpu_complex: habana_frameworks is installed but torch.hpu is not loaded, so the "
                      "extension is not enabled; import habana_frameworks.torch.core before "
                      "hpu_complex.enable()")
    if enter_mode and not _MODE_ENTERED:
        from .mode import MODE

        MODE.__enter__()
        _MODE_ENTERED = True


def complex_mode():
    """Scope the complex-arithmetic dispatch mode to a block.

    Inside the block, ops involving a complex tensor are computed from
    float ops by the mode; every other op passes through untouched.
    Outside it, complex ops run on the extension's PrivateUse1 kernels. Needed
    only to activate mode hooks such as the fused TPC FFT.
    """
    from .mode import ComplexMode

    return ComplexMode()


__all__ = ["core", "enable", "complex_mode", "rfft", "irfft", "fft",
           "magnitude", "stft", "stft_magnitude"]
