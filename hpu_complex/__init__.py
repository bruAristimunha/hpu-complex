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
import os
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


def _holds_host_values(gm, example_inputs=()):
    """A Dynamo graph whose values may be host-backed: a complex128 value, float64 once complex128 was allocated, or an
    input that lives in host memory whatever its dtype (z128.view(torch.float32) is float32 host memory)."""
    import torch

    host_in_use = torch.ops.hpu_complex.host_in_use()
    for node in gm.graph.nodes:
        dtypes = {getattr(v, "dtype", None) for v in (node.meta.get("example_value"), node.meta.get("val"))}
        if torch.complex128 in dtypes or (host_in_use and torch.float64 in dtypes):
            return True
        grapharg = node.meta.get("grapharg") if node.op == "placeholder" else None
        example = getattr(grapharg, "example", None)
        if host_in_use and isinstance(example, torch.Tensor) and _host_backed(example):
            return True
    return host_in_use and any(isinstance(t, torch.Tensor) and _host_backed(t) for t in example_inputs)


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
        if _holds_host_values(gm, example_inputs):
            return gm.forward
        return original(gm, example_inputs, **kwargs)

    hpu_backend.hpu_complex = True
    registry._COMPILER_FNS["hpu_backend"] = hpu_backend


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

    _complex128_graphs_run_eagerly()

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


def _host_backed(t):
    """A tensor in host memory that reports the HPU device: complex128, its real and dtype views (z128.view(complex64)
    included), float64 results of its ops. The extension decides from the storage itself, never from the dtype."""
    import torch

    if t.device.type != "hpu" or not torch._C._dispatch_keys(t).has(torch._C.DispatchKey.PrivateUse1):
        return False
    try:
        return bool(torch.ops.hpu_complex.is_host_backed(t))
    except (AttributeError, RuntimeError):  # extension not loaded: no host-backed tensor can exist
        return False


def _dense(t):
    """t's elements fill one block of memory, in some order of its dims (a permutation of contiguous strides)."""
    dims = sorted((d for d in range(t.dim()) if t.shape[d] != 1), key=lambda d: t.stride(d))
    expected = 1
    for d in dims:
        if t.stride(d) != expected:
            return False
        expected *= t.shape[d]
    return True


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
            # A dense layout (a transpose, channels_last) is kept: the bytes in memory order, with the strides.
            values = self.detach().cpu().resolve_conj().resolve_neg()
            if not _dense(values):
                values = values.contiguous()
            flat = values.as_strided((values.numel(),), (1,), values.storage_offset())
            return (_rebuild_host_backed_from_cpu,
                    (flat.numpy().tobytes().decode("latin-1"), self.dtype, str(self.device), self.requires_grad,
                     tuple(values.shape), tuple(values.stride())))
        return original(self, proto)

    _reduce_ex_internal.hpu_complex = True
    torch.Tensor._reduce_ex_internal = _reduce_ex_internal


def _restore_device(device, data):
    """torch.load's map_location for a rebuilt tensor: a str, device or dict as torch's _get_restore_location, and a
    callable as torch calls it for a storage (storage, location) -> a storage (its device) or None (the saved one)."""
    import torch

    map_location = torch.serialization._serialization_tls.map_location
    if callable(map_location) and not isinstance(map_location, (str, torch.device, dict)):
        restored = map_location(data.untyped_storage(), device)
        return torch.device(device) if restored is None else torch.device(restored.device)
    return torch.device(torch._utils._get_restore_location(device))


def _rebuild_host_backed_from_cpu(data, dtype, device, requires_grad, size=None, stride=None):
    """torch.load of a tensor _pickle_complex128_through_the_cpu saved: back into host memory on the HPU (exact).

    torch's own _rebuild_device_tensor_from_cpu_tensor would rebuild a float64 view or result of a complex128 tensor
    with .to(device): an ordinary Habana float64 tensor, at float32 precision. map_location is honored (callables too);
    anywhere but the HPU it is that .to(device). Works in a process without a Gaudi (map_location="cpu").
    """
    import torch

    if isinstance(data, str):
        n = math.prod(size)
        flat = (torch.frombuffer(bytearray(data.encode("latin-1")), dtype=dtype, count=n) if n
                else torch.empty(0, dtype=dtype))
        data = flat.as_strided(size, stride) if stride is not None else flat.reshape(size)
    device = _restore_device(device, data)
    if device.type == "hpu" and dtype in (torch.float64, torch.complex128) and hasattr(torch.ops.hpu_complex,
                                                                                     "host_from_cpu"):
        tensor = torch.ops.hpu_complex.host_from_cpu(data.to(dtype), device)
    else:
        tensor = data.to(dtype=dtype, device=device)
    tensor.requires_grad = requires_grad
    return tensor


def _allow_rebuild_in_torch_load():
    """torch.load's default weights_only=True accepts the rebuilder of host-backed tensors in any process that imported
    hpu_complex, with or without a Gaudi (a checkpoint written on the card loads on a CPU machine)."""
    import torch

    try:
        torch.serialization.add_safe_globals([_rebuild_host_backed_from_cpu])
    except AttributeError:  # torch < 2.4: no weights_only allowlist
        pass


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


try:  # at import: no torch here is fine (enable() needs it anyway)
    _allow_rebuild_in_torch_load()
except ImportError:
    pass


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

    _allow_rebuild_in_torch_load()  # a checkpoint of host-backed tensors loads on a CPU machine too
    if hasattr(torch, "hpu") and torch.hpu.is_available():
        from .extension import load_extension

        load_extension()
        lazy = _bridge_is_lazy()
        torch.ops.hpu_complex.set_lazy_mode(lazy)
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
