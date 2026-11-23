"""The C++ kernels themselves on the CPU: the review fixes of PRs #9/#10 (CODEX_REVIEW findings), against torch.

cpu_stub/hpu_guard_stub.cpp stands in for Habana's float kernels (torch's CPU kernels on host memory that reports the
HPU device), so complex64 tensors here run the extension's real card path: its pair formulas, storages, out= and
overlap handling, the AutogradHPU complex-scalar interception and the BackendSelect factories. Lazy mode and Habana's
own numerics are not stood in: those cases are in lazy_check.py and green_ops_check.py on the card.

python -m pytest hpu_complex/tests/test_review_fixes_cpu.py -q
"""
import io
import math
import pathlib
import subprocess
import sys
import textwrap
import warnings

import pytest
import torch

from hpu_complex.tests.test_complex128_cpu import HPU, extension, tagged  # noqa: F401

C64, C128, F64 = torch.complex64, torch.complex128, torch.float64
INF, NAN = math.inf, math.nan
REPO = pathlib.Path(__file__).resolve().parents[2]


def h(values, dtype=C64):
    return torch.tensor(values, dtype=dtype).to(HPU)


def bits_equal(got, ref):
    """Equal values, NaN where torch has NaN, signed zeros included."""
    g = got.detach().cpu().resolve_conj()
    if g.is_complex():
        g, ref = torch.view_as_real(g), torch.view_as_real(ref.resolve_conj())
    assert g.dtype == ref.dtype and g.shape == ref.shape, (g.dtype, ref.dtype, g.shape, ref.shape)
    nan = ref.isnan()
    assert torch.equal(g.isnan(), nan), (g, ref)
    gi = g.view(torch.int64 if g.dtype == F64 else torch.int32)
    ri = ref.view(torch.int64 if ref.dtype == F64 else torch.int32)
    assert torch.equal(gi[~nan], ri[~nan]), (g, ref)


def div_equal(got, ref):
    """Complex division against torch's: parts finite on both sides within 2 float32 ulp (rtol 2**-23, atol
    2**-148 for the subnormal range, where an ulp is absolute), signs exact on nonzero parts (a zero part's
    sign through Smith's scaled form differs: -0 here where torch x86 gives +0), a non-finite part (inf or
    NaN) where torch has one, and at the overflow edge a finite part past float32 max / 2 where torch has inf
    (Smith scaling stays finite there; torch's x86 form overflows). torch's own CPU complex division is not
    bitwise stable across platforms: x86 torch 2.6 gives a NaN real part for x / -0j where ARM torch 2.6 gives
    inf, and 1-ulp differences elsewhere (CI vs local, measured)."""
    g = got.detach().cpu().resolve_conj()
    if g.is_complex():
        g, ref = torch.view_as_real(g), torch.view_as_real(ref.resolve_conj())
    assert g.dtype == ref.dtype and g.shape == ref.shape, (g.dtype, ref.dtype, g.shape, ref.shape)
    both_finite = torch.isfinite(g) & torch.isfinite(ref)
    overflow = (torch.isfinite(g) & (g.abs() > 1.7e38)) | (torch.isfinite(ref) & (ref.abs() > 1.7e38))
    assert bool((both_finite | ~torch.isfinite(g) & ~torch.isfinite(ref) | overflow).all()), (g, ref)
    torch.testing.assert_close(g[both_finite], ref[both_finite], rtol=2 ** -23, atol=2 ** -148)
    nonzero = both_finite & (ref != 0)
    assert torch.equal(torch.signbit(g[nonzero]), torch.signbit(ref[nonzero])), (g, ref)


# ---- #1 a failed set_ installs nothing -------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [C64, C128])
def test_failed_set_leaves_a_saved_tensor_unchanged(dtype):
    x = h([2 + 0j], dtype).requires_grad_()
    loss = (x * x).real.sum()
    b = h([7 + 0j], dtype)
    with torch.no_grad(), pytest.raises(RuntimeError, match="out of bounds"):
        x.set_(b.untyped_storage(), 0, (2,), (1,))
    assert x.shape == (1,) and x._version == 0
    loss.backward()
    assert x.grad.cpu().item() == 4  # 2 x, with x = 2: main installed b's storage first and gave 14


@pytest.mark.parametrize("dtype", [C64, C128])
def test_failed_set_keeps_numel_and_storage_consistent(dtype):
    x = h([1, 2, 3], dtype)
    b = h([7], dtype)
    with pytest.raises(RuntimeError):
        x.set_(b.untyped_storage(), 0, (3,), (1,))
    assert x.numel() == 3 and x.untyped_storage().nbytes() == 3 * x.element_size()
    assert torch.equal(x.cpu(), torch.tensor([1, 2, 3], dtype=dtype))


# ---- #3 a growing out= keeps the values outside its range -----------------------------------------------


@pytest.mark.parametrize("dtype", [C64, C128])
def test_growing_out_keeps_the_prefix(dtype):
    base = h([11, 22, 33], dtype)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.neg(h([7, 8, 9, 10], dtype), out=base[1:])
        ref = torch.tensor([11, 22, 33], dtype=dtype)
        torch.neg(torch.tensor([7, 8, 9, 10], dtype=dtype), out=ref[1:])
    assert torch.equal(base.cpu(), ref)  # main lost base[0]: [0, -7, -8]


@pytest.mark.parametrize("dtype", [C64, C128])
def test_views_taken_before_growth_see_the_result(dtype):
    out = h([0, 0], dtype)
    alias = out.view(-1)
    real = torch.view_as_real(out)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.mul(h([1, 2, 3, 4], dtype), 2, out=out)
    assert torch.equal(alias.cpu(), torch.tensor([2, 4], dtype=dtype))
    assert torch.equal(real.cpu()[:, 0], torch.tensor([2.0, 4.0], dtype=real.dtype))
    assert torch.equal(out.cpu(), torch.tensor([2, 4, 6, 8], dtype=dtype))


# ---- #4 set_() detaches the storage; set_(storage) works -----------------------------------------------


@pytest.mark.parametrize("dtype", [C64, C128])
def test_set_without_arguments_detaches(dtype):
    z = h([1, 2], dtype)
    alias = z.view(-1)
    z.set_()
    assert z.shape == (0,) and z.untyped_storage().nbytes() == 0
    z.resize_(3).fill_(99)
    assert torch.equal(alias.cpu(), torch.tensor([1, 2], dtype=dtype))  # main overwrote it with 99


@pytest.mark.parametrize("dtype", [C64, C128])
def test_set_whole_storage(dtype):
    a, b = h([1, 2], dtype), h([3, 4, 5], dtype)
    a.set_(b.untyped_storage())
    assert torch.equal(a.cpu(), torch.tensor([3, 4, 5], dtype=dtype))
    b[0] = 9
    assert a.cpu()[0] == 9  # aliases, as torch


# ---- #5 overlapping in-place writes raise -------------------------------------------------------------


@pytest.mark.parametrize("dtype", [C64, C128])
@pytest.mark.parametrize("op", ["add_", "mul_"])
def test_partially_overlapping_inplace_raises(dtype, op):
    z = h([1, 2, 3, 4, 5], dtype)
    with pytest.raises(RuntimeError, match="single memory location"):
        getattr(z[1:], op)(z[:-1])
    assert torch.equal(z.cpu(), torch.tensor([1, 2, 3, 4, 5], dtype=dtype))


def test_partially_overlapping_out_raises_complex64():
    v = h([1, 2, 3, 4, 5])
    with pytest.raises(RuntimeError, match="single memory location"):
        torch.mul(v[:-1], v[:-1], out=v[1:])


def test_full_overlap_inplace_is_fine():
    z = h([1 + 1j, 2])
    z.mul_(z)
    assert torch.equal(z.cpu(), torch.tensor([1 + 1j, 2]) ** 2)


# ---- #6 device indices --------------------------------------------------------------------------------


def test_host_copies_keep_the_device_index():
    z = h([1j], C128)
    assert z.to("hpu:1").device == torch.device("hpu", 1)
    assert torch.empty_like(z.real, device="hpu:1").device == torch.device("hpu", 1)
    assert torch.equal(z.to("hpu:1").cpu(), z.cpu())


def test_mixed_devices_raise():
    with pytest.raises(RuntimeError, match="same device"):
        h([1j], C128).to("hpu:0") + h([1j], C128).to("hpu:1")
    with pytest.raises(RuntimeError, match="same device"):
        torch.pow(h([1j, 2], C128), torch.tensor([2.0, 2.0]))
    with pytest.raises(RuntimeError, match="same device"):
        h([1j, 2]) * torch.tensor([1j, 2])  # complex64 with a CPU vector: main moved it to the card silently
    assert torch.equal((h([1j, 2], C128) * torch.tensor(2.0)).cpu(), torch.tensor([2j, 4], dtype=C128))  # 0-d CPU


# ---- #7 dtype views keep host memory host memory -------------------------------------------------------


def test_complex128_view_of_card_storage_raises():
    with pytest.raises(RuntimeError, match="view of complex64 HPU storage"):
        h([1j, 2]).view(C128)


def test_complex64_view_of_host_memory_stays_host_backed():
    import hpu_complex

    z = h([1 + 2j, 3 - 4j], C128)
    v = z.view(C64)
    assert v.dtype == C64 and hpu_complex._host_backed(v) and hpu_complex._host_backed(torch.view_as_real(v))
    ref = torch.tensor([1 + 2j, 3 - 4j], dtype=C128).view(C64)
    assert torch.equal(torch.view_as_complex(torch.view_as_real(v)).cpu(), ref)  # main raised
    assert torch.equal(v.exp().cpu(), ref.exp())
    assert not hpu_complex._host_backed(h([1j]))


def test_compile_guard_sees_host_memory_of_any_dtype():
    import hpu_complex

    gm = torch.fx.symbolic_trace(lambda x: torch.relu(x))
    f32_host = h([1 + 2j], C128).view(torch.float32)
    assert hpu_complex._holds_host_values(gm, [f32_host])  # main: eligible for a Synapse graph
    assert not hpu_complex._holds_host_values(gm, [torch.ones(2)])


# ---- #8 complex64 with float64 promotes to complex128 --------------------------------------------------


@pytest.mark.parametrize("op", [torch.add, torch.sub, torch.mul, torch.div])
@pytest.mark.parametrize("swap", [False, True])
def test_complex64_with_float64_is_complex128(op, swap):
    z, r = torch.tensor([1 + 0.5j], dtype=C64), torch.tensor([2 ** -25 + 1], dtype=F64)
    args = (r, z) if swap else (z, r)
    got = op(*(a.to(HPU) for a in args))
    ref = op(*args)
    assert got.dtype == C128 == ref.dtype and tagged(got)
    assert torch.equal(got.cpu(), ref)  # main: complex64, the float64 operand rounded to float32


def test_complex64_inplace_with_float64_stays_complex64():
    z, r = torch.tensor([1 + 0.5j], dtype=C64), torch.tensor([2 ** -25 + 1], dtype=F64)
    got = z.to(HPU)
    got.add_(r.to(HPU))
    assert torch.equal(got.cpu(), z.clone().add_(r))


def test_where_and_cat_promote_too():
    z, r = torch.tensor([1 + 0.5j], dtype=C64), torch.tensor([2 ** -25 + 1], dtype=F64)
    c = torch.tensor([True])
    assert torch.equal(torch.where(c.to(HPU), r.to(HPU), z.to(HPU)).cpu(), torch.where(c, r, z))
    assert torch.equal(torch.cat([z.to(HPU), r.to(HPU)]).cpu(), torch.cat([z, r]))


# ---- #9 checkpoints load in a fresh CPU process --------------------------------------------------------

_WRITER = """
import io, sys, torch
sys.path.insert(0, {repo!r})
exec(open({env!r}).read())
from hpu_complex import _pickle_complex128_through_the_cpu
_pickle_complex128_through_the_cpu()
z = torch.tensor([[1 + 1e-12j, 2, 3], [4, 5, 6 - 1e-300j]], dtype=torch.complex128).to(torch.device("hpu"))
torch.save({{"z": z, "t": z.mT, "re": z.real.mT}}, {path!r})
"""

_ENV = """
import pathlib, sys, types, torch
from torch.utils.cpp_extension import load
stub = types.ModuleType("torch.hpu"); stub._lazy_init = lambda: None; stub.is_available = lambda: False
stub.is_current_stream_capturing = lambda: False
sys.modules["torch.hpu"] = torch.hpu = stub
from hpu_complex.extension import load_extension
load_extension()
load(name="hpu_guard_stub", sources=[{stub!r}], is_python_module=False)
"""

_READER = """
import sys, torch
sys.path.insert(0, {repo!r})
import hpu_complex
hpu_complex.enable()  # no Gaudi here: does nothing but allow the rebuilder
ref = torch.tensor([[1 + 1e-12j, 2, 3], [4, 5, 6 - 1e-300j]], dtype=torch.complex128)
got = torch.load({path!r}, map_location="cpu")  # weights_only=True, torch 2.6's default
assert got["z"].device.type == "cpu" and torch.equal(got["z"], ref), got
assert got["t"].stride() == ref.mT.stride() and torch.equal(got["t"], ref.mT), got["t"].stride()
assert torch.equal(got["re"], ref.real.mT)  # a non-dense view: its values (torch.save keeps no shared storage here)
called = []
cb = torch.load({path!r}, map_location=lambda storage, loc: called.append(loc) or storage)
assert called and torch.equal(cb["z"], ref) and cb["z"].device.type == "cpu", called
print("READ_OK")
"""


def test_checkpoint_loads_in_a_fresh_cpu_process(tmp_path):
    path = str(tmp_path / "ckpt.pt")
    env = tmp_path / "env.py"
    env.write_text(_ENV.format(stub=str(REPO / "hpu_complex/tests/cpu_stub/hpu_guard_stub.cpp")))
    run = dict(check=False, capture_output=True, text=True, timeout=1200, cwd=tmp_path)
    w = subprocess.run([sys.executable, "-c", _WRITER.format(repo=str(REPO), env=str(env), path=path)], **run)
    assert w.returncode == 0, w.stderr[-3000:]
    r = subprocess.run([sys.executable, "-c", _READER.format(repo=str(REPO), path=path)], **run)
    assert r.returncode == 0 and "READ_OK" in r.stdout, r.stderr[-3000:]


# ---- #10 division (Smith) and addition (alpha) as torch's complex arithmetic ----------------------------

DIV_CASES = [(1.0, 1e20j), (1.0, 1e-30j), (1 + 1j, 1e20j), (1 + 1j, 1e-30 + 1e-30j), (3e38 + 3e38j, 3e38 + 1e38j),
             (1 + 1j, 0j), (0j, 0j), (1 + 1j, complex(INF, 0)), (complex(INF, 1), 2 + 0j), (1.0, 3 + 4j),
             (-0.0, 1 + 0j), (2.0, complex(0, -0.0) + 1), (complex(NAN, 1), 1j)]


@pytest.mark.parametrize("a,b", DIV_CASES)
def test_division_is_torch_complex_division(a, b):
    ta = torch.tensor([a], dtype=torch.float32 if isinstance(a, float) else C64)
    tb = torch.tensor([b], dtype=C64)
    bits_equal(ta.to(HPU) / tb.to(HPU), ta / tb)


@pytest.mark.parametrize("scalar", [1e20j, 1e-30j, 3 + 4j, -2j])
def test_real_tensor_over_complex_scalar(scalar):
    x = torch.tensor([1.0, -2.0, 0.5])
    bits_equal(x.to(HPU) / scalar, x / scalar)  # main: 0j for 1e20j, nan - inf j for 1e-30j


ADD_CASES = [complex(INF, 0), complex(-INF, 2), complex(NAN, 0), complex(0, INF), 1 + 2j, complex(-0.0, -0.0)]


@pytest.mark.parametrize("c", ADD_CASES)
@pytest.mark.parametrize("alpha", [0, 1, 2.5, -1, 1j, 2 - 1j])
def test_add_alpha_nan_placement_like_torch(c, alpha):
    """torch's a + alpha b, alpha b a complex product. One exception, kept for speed (README Limits): two complex
    tensors on the card with a real alpha are pair arithmetic, a + alpha b per part, as main computed them."""
    x = torch.tensor([1.0, -0.0])
    z = torch.tensor([1 + 1j, complex(-0.0, 0.0)], dtype=C64)
    w = torch.tensor([c, c], dtype=C64)

    def pair_sum(p, q, s):
        q = torch.view_as_real(q)
        return torch.view_as_complex(torch.view_as_real(p) + (q if s == 1 else q * s))

    for op, sign in ((torch.add, 1), (torch.sub, -1)):
        got = op(z.to(HPU), w.to(HPU), alpha=alpha)
        bits_equal(got, op(z, w, alpha=alpha) if isinstance(alpha, complex) else pair_sum(z, w, float(alpha) * sign))
    bits_equal(torch.add(x.to(HPU), w.to(HPU), alpha=alpha), torch.add(x, w, alpha=alpha))
    bits_equal(torch.add(z.to(HPU), torch.tensor(c), alpha=alpha), torch.add(z, torch.tensor(c), alpha=alpha))
    if not isinstance(alpha, complex):
        bits_equal(torch.add(w.to(HPU), x.to(HPU), alpha=alpha), torch.add(w, x, alpha=alpha))


def test_real_plus_infinite_complex_scalar():
    x = torch.tensor([1.0])
    bits_equal(x.to(HPU) + complex(INF, 0), x + complex(INF, 0))  # main: inf + 0j, torch inf + nan j


# ---- #11 vector_norm through the scaled modulus, every ord ----------------------------------------------

NORM_INPUTS = {
    "tiny": [1e-30 + 0j], "huge": [3e19 + 3e19j], "inf_nan": [complex(INF, NAN)],
    "mixed": [1e-30 + 0j, 3e19 + 0j, 1 + 1j], "zero": [0j, 0j], "small_vec": [1e-25 + 1e-25j, 2e-25j],
}


@pytest.mark.parametrize("name", list(NORM_INPUTS))
@pytest.mark.parametrize("ord", [2, 1, 3.5, 0.5, INF, -INF, 0, -1, -2])
def test_vector_norm_like_torch(name, ord):
    """The exact norm rounded to float32 (complex128 on the CPU): torch's own complex64 kernel squares unscaled for
    several elements (inf for "mixed", 0 for "small_vec"), its single-element path does not."""
    z = torch.tensor(NORM_INPUTS[name], dtype=C64)
    got = torch.linalg.vector_norm(z.to(HPU), ord).cpu()
    ref = torch.linalg.vector_norm(z.to(C128), ord).float()
    torch.testing.assert_close(got, ref, rtol=2e-6, atol=0, equal_nan=True)


def test_vector_norm_gradient_of_a_tiny_element():
    z = h([1e-30 + 0j]).requires_grad_()
    torch.linalg.vector_norm(z).backward()
    assert torch.equal(z.grad.cpu(), torch.tensor([1 + 0j]))  # main: 0


@pytest.mark.parametrize("dim,keepdim", [(None, False), (None, True), (0, False), ((0, 1), True), (-1, True), ((), False)])
@pytest.mark.parametrize("ord", [2, 1, -1, INF])
def test_vector_norm_dims(dim, keepdim, ord):
    z = torch.randn(3, 4, dtype=C64, generator=torch.Generator().manual_seed(0))
    torch.testing.assert_close(torch.linalg.vector_norm(z.to(HPU), ord, dim, keepdim).cpu(),
                               torch.linalg.vector_norm(z, ord, dim, keepdim), rtol=2e-6, atol=1e-7)


def test_vector_norm_duplicate_dims_raise_for_a_scalar():
    with pytest.raises(RuntimeError, match="appears multiple times"):
        torch.linalg.vector_norm(h(1 + 2j), 2, dim=(0, -1))


def test_vector_norm_out_dtype_rule():
    with pytest.raises(RuntimeError, match="Expected out tensor to have dtype"):
        torch.linalg.vector_norm(h([1 + 2j], C128), out=torch.empty((), dtype=C128, device=HPU))
    out = torch.empty((), dtype=torch.float32, device=HPU)
    torch.linalg.vector_norm(h([3 + 4j]), out=out)
    assert out.cpu().item() == 5


# ---- #12 cat.out --------------------------------------------------------------------------------------


def test_cat_out_computes_in_out_dtype():
    out = torch.empty(2, dtype=C128, device=HPU)
    torch.cat([h([1 + 0j]), torch.tensor([16777217], dtype=torch.int32).to(HPU)], out=out)
    assert out.cpu()[1].real.item() == 16777217  # main: 16777216 (cat in complex64 first)


def test_cat_out_rejects_what_torch_rejects():
    with pytest.raises(RuntimeError, match="can't be cast"):
        torch.cat([h([1 + 2j], C128)], out=torch.empty(1, dtype=F64, device=HPU))
    with pytest.raises(RuntimeError, match="can't be cast"):
        torch.cat([h([1 + 2j])], out=torch.empty(1, dtype=torch.float32, device=HPU))
    z = h([1 + 2j], C128)
    with pytest.raises(RuntimeError, match="single memory location|refer to any of the output"):
        torch.cat([z], out=z)


def test_cat_out_complex128_into_complex64():
    out = torch.empty(1, dtype=C64, device=HPU)
    torch.cat([h([1 + 2j], C128)], out=out)
    assert torch.equal(out.cpu(), torch.tensor([1 + 2j], dtype=C64))


# ---- #2 (CPU side) set_ of complex128 onto a complex64 storage keeps its bytes (lazy: lazy_check.py) ----


def test_complex128_set_onto_complex64_card_storage_copies_bytes():
    z = h([1 + 2j, 3 + 4j])
    t = torch.empty(0, dtype=C128, device=HPU)
    t.set_(z.untyped_storage(), 0, (1,), (1,))
    assert torch.equal(t.cpu(), torch.tensor([1 + 2j, 3 + 4j]).view(C128))
