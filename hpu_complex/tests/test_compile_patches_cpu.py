"""CPU checks of the hpu_backend pass patches (hpu_complex._complex_nodes_run_eagerly) and of the layout rule the
C++ extension's dense_copy relies on.

Habana's compile backend is replaced by stub modules, so these run without habana_frameworks. The gradients the
patches fix are checked on Gaudi only (pu1_check.py, "compiled eigvals/eig/qr ... fwd+bwd"). Run:
python -m pytest hpu_complex/tests/test_compile_patches_cpu.py -q
"""
import sys
import types

import pytest
import torch
from torch._prims_common import is_non_overlapping_and_dense
from torch.fx.experimental.proxy_tensor import make_fx

import hpu_complex


@pytest.fixture
def stub_passes(monkeypatch):
    """A stand-in for habana_frameworks.torch.dynamo.compile_backend.passes, its originals recording calls."""
    calls = {"fuse": [], "placement": 0}
    passes = types.ModuleType("passes")
    passes.is_eager_fallback_required = lambda node, is_dynamic=False: False

    def pass_mark_placement(ctx):
        calls["placement"] += 1
        for node in ctx.graph_module.graph.nodes:
            node.meta["placement"] = "eager" if passes.is_eager_fallback_required(node) else "hpu_cluster"
        return False

    def pass_fuse_view_chains(ctx):  # records the nodes the real pass could start a chain at
        calls["fuse"] = [n.name for n in ctx.graph_module.graph.nodes
                         if n.op == "call_function" and not n.meta.get("visited", False)]
        return False

    passes.pass_mark_placement = pass_mark_placement
    passes.pass_fuse_view_chains = pass_fuse_view_chains
    names = ["habana_frameworks", "habana_frameworks.torch", "habana_frameworks.torch.dynamo",
             "habana_frameworks.torch.dynamo.compile_backend"]
    for name in names:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    backend = sys.modules["habana_frameworks.torch.dynamo.compile_backend"]
    backend.passes = passes
    monkeypatch.setitem(sys.modules, "habana_frameworks.torch.dynamo.compile_backend.passes", passes)
    hpu_complex._complex_nodes_run_eagerly()
    return passes, calls


def _ctx(fn, *args):
    gm = make_fx(fn)(*args)
    return types.SimpleNamespace(graph_module=gm)


def _node(ctx, prefix):
    return next(n for n in ctx.graph_module.graph.nodes if n.name.startswith(prefix))


def test_float_reader_of_a_complex_node_runs_eagerly(stub_passes):
    passes, calls = stub_passes
    z = torch.randn(3, 5, 5, dtype=torch.complex64)
    # view_as_real of a complex diagonal, then float ops on it (the qr backward's pattern), then a float op that
    # reads only float nodes.
    ctx = _ctx(lambda t: (torch.view_as_real(t.diagonal(0, -2, -1))[..., 0] * 0.5).exp(), z)
    passes.pass_mark_placement(ctx)
    placement = {n.name: n.meta["placement"] for n in ctx.graph_module.graph.nodes if n.op == "call_function"}
    assert placement[_node(ctx, "diagonal").name] == "eager"       # complex in and out
    assert placement[_node(ctx, "view_as_real").name] == "eager"   # complex in
    assert placement[_node(ctx, "select").name] == "eager"         # reads view_as_real's float view
    assert placement[_node(ctx, "exp").name] == "hpu_cluster"      # reads float nodes only
    assert calls["placement"] == 1


@pytest.mark.parametrize("fn", [
    lambda t: torch.view_as_real(t.diagonal(0, -2, -1))[..., 0] * 0.5,   # select of view_as_real, then mul
    lambda t: t.diagonal(0, -2, -1).real * 2,                            # .real traces to view_as_real + select
    lambda t: torch.view_as_real(t)[..., 1].mT.unsqueeze(0) + 1,         # a longer float view chain
])
def test_every_reader_of_a_float_view_of_complex_storage_runs_eagerly(stub_passes, fn):
    """The float views of complex storage are closed transitively: a view of such a view is one too, and the first
    node that materializes (mul, add) reads one, so it runs eagerly; a reader of its fresh result is fusable."""
    passes, _ = stub_passes
    ctx = _ctx(lambda t: fn(t).exp(), torch.randn(3, 5, 5, dtype=torch.complex64))
    passes.pass_mark_placement(ctx)
    placement = {n.name: n.meta["placement"] for n in ctx.graph_module.graph.nodes if n.op == "call_function"}
    materializing = [n for n in ctx.graph_module.graph.nodes if n.op == "call_function"
                     and n.target in (torch.ops.aten.mul.Tensor, torch.ops.aten.add.Tensor)]
    assert materializing and all(placement[n.name] == "eager" for n in materializing)
    assert all(v == "eager" for k, v in placement.items() if not k.startswith("exp"))
    assert placement[_node(ctx, "exp").name] == "hpu_cluster"


def test_fuse_view_chains_skips_float_views_of_complex_storage(stub_passes):
    """eig's V.real.mT.unsqueeze(0): the float chain after view_as_real addresses complex storage, whose strides the
    fake tensors can get wrong; no chain may start on it (nor upstream of a complex view: x.view(...).view(c64))."""
    passes, calls = stub_passes
    ctx = _ctx(lambda t: t.real.mT.unsqueeze(0), torch.randn(3, 5, 5, dtype=torch.complex64))
    passes.pass_fuse_view_chains(ctx)
    assert not calls["fuse"], calls["fuse"]
    ctx = _ctx(lambda x: x.view(4, 6).view(torch.complex64), torch.randn(24))
    passes.pass_fuse_view_chains(ctx)
    assert not calls["fuse"], calls["fuse"]


def test_fresh_float_result_of_a_complex_node_stays_fusable(stub_passes):
    passes, _ = stub_passes
    ctx = _ctx(lambda t: t.abs().log1p().mean(), torch.randn(4, 6, dtype=torch.complex64))
    passes.pass_mark_placement(ctx)
    assert _node(ctx, "abs").meta["placement"] == "eager"
    assert _node(ctx, "log1p").meta["placement"] == "hpu_cluster"


def test_float_graph_is_untouched(stub_passes):
    passes, _ = stub_passes
    ctx = _ctx(lambda t: (t * 2).exp().sum(-1), torch.randn(4, 3))
    passes.pass_mark_placement(ctx)
    assert all(n.meta["placement"] == "hpu_cluster" for n in ctx.graph_module.graph.nodes if n.op == "call_function")


def test_fuse_view_chains_skips_complex_views(stub_passes):
    passes, calls = stub_passes
    z = torch.randn(3, 5, 5, dtype=torch.complex64)
    x = torch.randn(3, 5, 5)
    ctx = _ctx(lambda t, r: (t.transpose(-2, -1).unsqueeze(0), r.transpose(-2, -1).unsqueeze(0)), z, x)
    passes.pass_fuse_view_chains(ctx)
    complex_views = {n.name for n in ctx.graph_module.graph.nodes
                     if n.op == "call_function" and n.meta["val"].is_complex()}
    assert complex_views and not complex_views & set(calls["fuse"])
    assert any(name.startswith("transpose") for name in calls["fuse"])  # the float chain is still fused


def test_patches_are_idempotent(stub_passes):
    passes, _ = stub_passes
    first = (passes.is_eager_fallback_required, passes.pass_mark_placement, passes.pass_fuse_view_chains)
    hpu_complex._complex_nodes_run_eagerly()
    assert (passes.is_eager_fallback_required, passes.pass_mark_placement, passes.pass_fuse_view_chains) == first


def _permute_block(t):
    """habana_view's rank-6+ dense path in pure torch: the dims from the largest stride to the smallest as one
    contiguous block at t's offset, permuted back (cpp_ext/fft_ext.cpp habana_view; used by dense_copy)."""
    order = sorted(range(t.dim()), key=lambda d: -t.stride(d))
    block = [t.size(d) for d in order]
    dims = [order.index(d) for d in range(t.dim())]
    flat = t.as_strided((t.untyped_storage().nbytes() // t.element_size(),), (1,), 0)
    return flat.narrow(0, t.storage_offset(), t.numel()).view(block).permute(dims)


@pytest.mark.parametrize("make", [
    lambda x: x[1, 1:].permute(0, 2, 1, 3, 4, 5, 6)[..., 0] * 1,       # a permuted op result (strides kept)
    lambda x: x[1].permute(5, 4, 3, 2, 1, 0, 6),                       # offset, dense, rank 7
    lambda x: x[0, 1, 1].permute(4, 3, 2, 1, 0),                       # rank 5
])
def test_permute_block_reads_dense_permuted_layouts(make):
    x = torch.randn(2, 2, 3, 2, 2, 3, 2, 2)
    t = make(x)
    assert is_non_overlapping_and_dense(t) and not t.is_contiguous()
    got = _permute_block(t)
    assert got.stride() == t.stride()
    assert torch.equal(got, t)
    assert torch.equal(got.contiguous(), t.contiguous())


def test_a_sliced_view_is_not_dense():
    """dense_copy keeps contiguous() for such a view (habana_view's as_strided path); only dense ones take the
    permute block."""
    x = torch.randn(2, 2, 3, 2, 2, 3, 2, 2)
    assert not is_non_overlapping_and_dense(x[1, 1:].permute(0, 2, 1, 3, 4, 5, 6)[..., 0])


def test_lazy_mode_defaults_to_lazy_without_the_bridge_query(monkeypatch):
    """Without habana_frameworks' is_lazy(), an unset PT_HPU_LAZY_MODE counts as lazy (guards raise, no abort)."""
    monkeypatch.setitem(sys.modules, "habana_frameworks.torch.utils.internal", None)  # import raises ImportError
    monkeypatch.delenv("PT_HPU_LAZY_MODE", raising=False)
    with pytest.warns(UserWarning, match="assuming lazy mode"):
        assert hpu_complex._bridge_is_lazy() is True
    monkeypatch.setenv("PT_HPU_LAZY_MODE", "0")
    assert hpu_complex._bridge_is_lazy() is False
    monkeypatch.setenv("PT_HPU_LAZY_MODE", "1")
    assert hpu_complex._bridge_is_lazy() is True


# ---- complex Python scalars in AOT backward graphs (compiled Green's wavelet gradient) -------------------------


def _green_wavelet_rows(foi, tt):
    """Green's Gabor rows, reduced: three selected frequencies, each a 0-d real tensor times a Python complex."""
    f = 2 ** foi
    return torch.view_as_real(torch.stack([(2j * torch.pi * a) * tt for a in f[:3]])).sum()


def _aot_graphs(fn, *args):
    from functorch.compile import aot_function

    graphs = {}

    def capture(tag):
        def compiler(gm, example_inputs):
            graphs[tag] = (gm, list(example_inputs))
            return gm.forward  # run the traced graph as is
        return compiler

    compiled = aot_function(fn, fw_compiler=capture("fw"), bw_compiler=capture("bw"))
    return compiled, graphs


def _vals(gm):
    return [n.meta.get("val") for n in gm.graph.nodes if isinstance(n.meta.get("val"), torch.Tensor)]


def test_backward_of_a_python_complex_scalar_traces_to_complex128():
    """The captured pattern: mul(grad, _conj(<Python complex>)), a complex128 constant in the backward graph."""
    foi = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    tt = torch.linspace(-0.25, 0.25, 8)
    compiled, graphs = _aot_graphs(_green_wavelet_rows, foi, tt)
    compiled(foi, tt).backward()
    gm, _ = graphs["bw"]
    conj = [n for n in gm.graph.nodes if n.op == "call_function" and n.target == torch.ops.aten._conj.default]
    assert conj and all(isinstance(n.args[0], complex) for n in conj)
    assert any(v.dtype == torch.complex128 for v in _vals(gm))


def test_fold_conj_of_python_scalars_keeps_the_backward_complex64_and_exact():
    foi = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    tt = torch.linspace(-0.25, 0.25, 8)
    _green_wavelet_rows(foi, tt).backward()
    ref = foi.grad.clone()

    folded = {}

    def bw_compiler(gm, example_inputs):
        folded["changed"] = hpu_complex._fold_conj_of_python_scalars(gm)
        folded["dtypes"] = {v.dtype for v in _vals(gm)}
        folded["conj"] = [n for n in gm.graph.nodes if n.op == "call_function"
                          and n.target == torch.ops.aten._conj.default]
        return gm.forward

    from functorch.compile import aot_function

    run = aot_function(_green_wavelet_rows, fw_compiler=lambda gm, ex: gm.forward, bw_compiler=bw_compiler)
    foi.grad = None
    run(foi, tt).backward()
    assert folded["changed"] and not folded["conj"]
    assert torch.complex128 not in folded["dtypes"] and torch.float64 not in folded["dtypes"]
    assert torch.complex64 in folded["dtypes"]  # the metadata was recomputed: the products stay complex64
    torch.testing.assert_close(foi.grad, ref)


def test_fold_conj_of_python_scalars_leaves_other_graphs_alone():
    z = torch.randn(4, dtype=torch.complex64)
    gm = make_fx(lambda t: torch.view_as_real(t.conj() * 2j).sum())(z)
    before = str(gm.graph)
    assert not hpu_complex._fold_conj_of_python_scalars(gm)
    assert str(gm.graph) == before


@pytest.fixture
def stub_backends(monkeypatch):
    """Stand-ins for compile_backend.backends (its compilers record the partitioner they ran under) and .config."""
    import contextlib

    backends = types.ModuleType("backends")
    config = types.ModuleType("config")
    config.use_cpp_partitioner = True

    @contextlib.contextmanager
    def patch(**changes):
        saved = {k: getattr(config, k) for k in changes}
        for k, v in changes.items():
            setattr(config, k, v)
        try:
            yield
        finally:
            for k, v in saved.items():
                setattr(config, k, v)

    config.patch = patch
    seen = []

    def compiler(tag):
        def compile_graph(gm, example_inputs, **kwargs):
            seen.append((tag, config.use_cpp_partitioner))
            return gm.forward
        return compile_graph

    backends.hpu_training_compiler_fw = compiler("fw")
    backends.hpu_training_compiler_bw = compiler("bw")
    backends.hpu_inference_compiler = compiler("inf")
    names = ["habana_frameworks", "habana_frameworks.torch", "habana_frameworks.torch.dynamo",
             "habana_frameworks.torch.dynamo.compile_backend"]
    for name in names:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    package = sys.modules["habana_frameworks.torch.dynamo.compile_backend"]
    package.backends, package.config = backends, config
    monkeypatch.setitem(sys.modules, "habana_frameworks.torch.dynamo.compile_backend.backends", backends)
    monkeypatch.setitem(sys.modules, "habana_frameworks.torch.dynamo.compile_backend.config", config)
    monkeypatch.setattr(torch.ops.hpu_complex, "host_in_use", lambda: False, raising=False)
    hpu_complex._aot_graphs_keep_complex64_scalars()
    return backends, config, seen


def test_aot_compilers_are_wrapped_once(stub_backends):
    backends, _, seen = stub_backends
    fw = backends.hpu_training_compiler_fw
    hpu_complex._aot_graphs_keep_complex64_scalars()
    assert backends.hpu_training_compiler_fw is fw
    gm = make_fx(lambda t: t * 2)(torch.randn(3))
    backends.hpu_inference_compiler(gm, [torch.randn(3)], dyn_graph_module=None)
    backends.hpu_training_compiler_bw(gm, [torch.randn(3)])
    assert seen == [("inf", True), ("bw", True)]  # a float graph keeps the C++ partitioner


def test_graphs_with_eager_complex_nodes_use_the_python_partitioner(stub_backends):
    """The captured pattern: eager 0-d float results of complex nodes (the gradients of selected frequencies) read by
    one fused graph; the C++ partitioner's fused graph doubled one of them on Gaudi2."""
    backends, config, seen = stub_backends
    foi = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    tt = torch.linspace(-0.25, 0.25, 8)
    graphs = {}

    def capture(tag):
        def compile_graph(gm, example_inputs):
            graphs[tag] = gm
            return gm.forward
        return compile_graph

    from functorch.compile import aot_function

    aot_function(_green_wavelet_rows, fw_compiler=capture("fw"), bw_compiler=capture("bw"))(foi, tt).backward()
    backends.hpu_training_compiler_bw(graphs["bw"], [])
    backends.hpu_training_compiler_fw(graphs["fw"], [])
    assert seen == [("bw", False), ("fw", False)] and config.use_cpp_partitioner is True


def test_bmm_view_pass_skips_complex_bmm(monkeypatch):
    """Habana's pass_remove_unnecessary_bmm_view hands a bmm the 4-D tensors before its views; the extension's complex
    bmm takes 3-D only (compiled z @ z.mT.conj() raised). It still sees float bmm nodes."""
    passes = types.ModuleType("passes")
    passes.is_eager_fallback_required = lambda node, is_dynamic=False: False
    seen = []

    def pass_remove_unnecessary_bmm_view(ctx):
        seen.extend(n.name for n in ctx.graph_module.graph.nodes if n.target == torch.ops.aten.bmm.default)
        return False

    passes.pass_remove_unnecessary_bmm_view = pass_remove_unnecessary_bmm_view
    names = ["habana_frameworks", "habana_frameworks.torch", "habana_frameworks.torch.dynamo",
             "habana_frameworks.torch.dynamo.compile_backend"]
    for name in names:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["habana_frameworks.torch.dynamo.compile_backend"].passes = passes
    monkeypatch.setitem(sys.modules, "habana_frameworks.torch.dynamo.compile_backend.passes", passes)
    hpu_complex._complex_nodes_run_eagerly()
    z = torch.randn(2, 3, 4, 5, dtype=torch.complex64)
    x = torch.randn(2, 3, 4, 5)
    gm = make_fx(lambda a, b: ((a @ a.mT.conj()).abs() + b @ b.mT))(z, x)
    ctx = types.SimpleNamespace(graph_module=gm)
    bmms = [n for n in gm.graph.nodes if n.target == torch.ops.aten.bmm.default]
    assert len(bmms) == 2
    passes.pass_remove_unnecessary_bmm_view(ctx)
    float_bmm = [n.name for n in bmms if not n.meta["val"].is_complex()]
    assert seen == float_bmm
    assert all(n.target == torch.ops.aten.bmm.default for n in bmms)  # restored
    torch.testing.assert_close(gm(z, x), (z @ z.mT.conj()).abs() + x @ x.mT)
