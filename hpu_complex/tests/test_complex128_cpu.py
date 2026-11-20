"""CPU checks of complex128 HPU tensors (host memory, computed on the CPU in float64), without a Gaudi.

complex128 HPU tensors never touch Habana storage, so stock (CPU) torch runs them once two things stand in for the
Gaudi stack: the extension itself (built here with torch.utils.cpp_extension, needs ninja and a C++ compiler) and a
no-op HPU device guard (cpu_stub/hpu_guard_stub.cpp). Ops whose results Habana would hold (int, bool, float32 on the
HPU) need the real stack and are covered by c128_check.py on the card: among them det, inv, solve, cholesky (int32
pivots or info) and pinv (float64 HPU tolerance scalars). Skipped where habana_frameworks is installed
(the Gaudi image) or the extension cannot be built.

python -m pytest hpu_complex/tests/test_complex128_cpu.py -q
"""
import importlib.util
import pathlib
import shutil
import sys
import types
import warnings

import pytest
import torch

HERE = pathlib.Path(__file__).parent

if importlib.util.find_spec("habana_frameworks") is not None:
    pytest.skip("Gaudi image: c128_check.py covers complex128 on the card", allow_module_level=True)
if shutil.which("ninja") is None:
    pytest.skip("ninja not found: cannot build the extension", allow_module_level=True)


@pytest.fixture(scope="module", autouse=True)
def extension():
    from torch.utils.cpp_extension import load

    from hpu_complex.extension import load_extension

    stub = types.ModuleType("torch.hpu")  # torch's device lazy-init imports torch.hpu for device="hpu"
    stub._lazy_init = lambda: None
    stub.is_available = lambda: False
    saved = sys.modules.get("torch.hpu")
    sys.modules["torch.hpu"] = stub
    try:
        load_extension()
        load(name="hpu_guard_stub", sources=[str(HERE / "cpu_stub" / "hpu_guard_stub.cpp")], is_python_module=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        if saved is None:
            sys.modules.pop("torch.hpu", None)
        else:
            sys.modules["torch.hpu"] = saved


HPU = torch.device("hpu")


def tagged(t):
    return torch._C._dispatch_keys(t).has(torch._C.DispatchKey.PrivateUse1)


def assert_hpu_exact(got, ref, atol=0.0):
    """got: a complex128/float64 HPU tensor in host memory; ref: the CPU result. Equal to atol (default: bit-exact)."""
    assert got.device.type == "hpu" and tagged(got), (got.device, torch._C._dispatch_keys(got))
    assert got.dtype == ref.dtype and got.shape == ref.shape, (got.dtype, ref.dtype, got.shape, ref.shape)
    torch.testing.assert_close(got.cpu(), ref, atol=atol, rtol=0)


def c128(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, dtype=torch.complex128, generator=g)


def test_round_trip_keeps_float64_bits():
    x = torch.tensor([1 + 1e-12j, -3e-300 + 2j, float("inf") - 1j, complex(float("nan"), 0.5)], dtype=torch.complex128)
    z = x.to(HPU)
    assert z.device == torch.device("hpu", 0) and z.dtype == torch.complex128
    back = z.cpu()
    assert torch.equal(torch.view_as_real(back).view(torch.int64), torch.view_as_real(x).view(torch.int64))


@pytest.mark.parametrize("shape", [(), (0,), (1,), (1, 1), (3, 0, 2), (2, 3, 4)])
def test_factories_and_odd_shapes(shape):
    assert_hpu_exact(torch.zeros(shape, dtype=torch.complex128, device=HPU), torch.zeros(shape, dtype=torch.complex128))
    assert_hpu_exact(torch.full(shape, 1 - 2j, dtype=torch.complex128, device=HPU),
                     torch.full(shape, 1 - 2j, dtype=torch.complex128))
    x = c128(*shape)
    assert_hpu_exact(x.to(HPU) * 2 - 1j, x * 2 - 1j)
    assert_hpu_exact(torch.zeros_like(x.to(HPU)), torch.zeros_like(x))


def test_views_write_through():
    x = c128(3, 4, 5)
    z = x.to(HPU)
    r = torch.view_as_real(z)
    assert r.dtype == torch.float64 and tagged(r)
    torch.view_as_real(z[1, ::2])[..., 1].mul_(-3)  # a strided, offset view of a view
    x_ref = x.clone()
    torch.view_as_real(x_ref[1, ::2])[..., 1].mul_(-3)
    assert_hpu_exact(z, x_ref)
    assert_hpu_exact(torch.view_as_complex(r), x_ref)
    assert_hpu_exact(z.real, x_ref.real)
    assert_hpu_exact(z.imag, x_ref.imag)
    assert_hpu_exact(z.permute(2, 0, 1)[1:, :, 1:3].reshape(-1), x_ref.permute(2, 0, 1)[1:, :, 1:3].reshape(-1))
    assert_hpu_exact(z.view(torch.float64), x_ref.view(torch.float64))


def test_conjugate_and_negative_views():
    x = c128(4, 3)
    z = x.to(HPU)
    assert_hpu_exact(z.conj(), x.conj())
    assert_hpu_exact(z.conj().resolve_conj(), x.conj().resolve_conj())
    assert_hpu_exact(z.conj().imag, x.conj().imag)  # _neg_view
    assert_hpu_exact(z.mH @ z, x.mH @ x)
    w = z.clone()
    w.conj().copy_(z)  # a write through a conjugate view stores conj(z)
    assert_hpu_exact(w, x.conj())


def test_arithmetic_type_promotion_and_inplace():
    x, y = c128(3, 4), c128(3, 4, seed=1)
    z, v = x.to(HPU), y.to(HPU)
    assert_hpu_exact(z * v + z / v - 2.5, x * y + x / y - 2.5)
    assert_hpu_exact(z * 1j, x * 1j)
    assert_hpu_exact(torch.where(x.real > 0, z, v), torch.where(x.real > 0, x, y))
    assert_hpu_exact(z.sum(), x.sum())
    assert_hpu_exact(z.sum(0, keepdim=True), x.sum(0, keepdim=True))
    w, w_ref = z.clone(), x.clone()
    w.mul_(v)
    w.add_(v, alpha=2)
    w_ref.mul_(y)
    w_ref.add_(y, alpha=2)
    assert_hpu_exact(w, w_ref)
    w.fill_(3 + 4j)
    assert_hpu_exact(w, torch.full_like(x, 3 + 4j))
    w.zero_()
    assert_hpu_exact(w, torch.zeros_like(x))


def test_float64_results_stay_exact():
    # Real float64 results (abs, view_as_real arithmetic, structured ops such as pow) stay in host memory.
    x = c128(5, 5)
    z = x.to(HPU)
    assert_hpu_exact(z.abs(), x.abs())
    assert_hpu_exact(z.angle(), x.angle())
    assert_hpu_exact(torch.view_as_real(z).pow(2), torch.view_as_real(x).pow(2))
    assert_hpu_exact(torch.linalg.svdvals(z), torch.linalg.svdvals(x))
    assert_hpu_exact(torch.view_as_real(z).clone(), torch.view_as_real(x))
    assert_hpu_exact(torch.empty_like(z.real).copy_(z.real), x.real)
    assert_hpu_exact(z.real.clone(memory_format=torch.contiguous_format), x.real.contiguous())
    assert_hpu_exact(z.to(torch.complex128, copy=True), x)
    assert_hpu_exact(z.real.to(torch.float64, copy=True), x.real)


@pytest.mark.parametrize("n", [1, 6])
def test_linalg_matches_torch(n):
    a = c128(2, n, n)
    b = c128(2, n, 3, seed=1)
    ah, bh = a.to(HPU), b.to(HPU)
    assert_hpu_exact(ah @ bh, a @ b)
    assert_hpu_exact(torch.einsum("bij,bjk->bik", ah, bh), torch.einsum("bij,bjk->bik", a, b))
    assert_hpu_exact(torch.linalg.eigvals(ah), torch.linalg.eigvals(a))
    for got, ref in zip(torch.linalg.eig(ah), torch.linalg.eig(a)):
        assert_hpu_exact(got, ref)
    for got, ref in zip(torch.linalg.svd(ah), torch.linalg.svd(a)):
        assert_hpu_exact(got, ref)
    for got, ref in zip(torch.linalg.qr(ah), torch.linalg.qr(a)):
        assert_hpu_exact(got, ref)
    assert_hpu_exact(torch.linalg.matrix_exp(0.1 * ah), torch.linalg.matrix_exp(0.1 * a))
    assert_hpu_exact(torch.fft.fft(ah), torch.fft.fft(a))
    assert_hpu_exact(torch.fft.ifft(ah, dim=1), torch.fft.ifft(a, dim=1))


def test_gradients_match_torch():
    a = c128(3, 4, 4)

    def grad(x):
        x = x.detach().clone().requires_grad_()
        (x @ x.mH).abs().pow(2).sum().backward()
        return x.grad

    assert_hpu_exact(grad(a.to(HPU)), grad(a), atol=1e-12)



def test_deepcopy_copies_host_memory_and_keeps_aliases():
    import copy

    from hpu_complex import _deepcopy_complex128_in_host_memory

    _deepcopy_complex128_in_host_memory()  # enable() installs it; it needs torch.hpu, absent here
    x = c128(3, 4)
    z = x.to(HPU)
    d = copy.deepcopy(z)
    assert_hpu_exact(d, x)
    d.mul_(2)  # an independent copy
    assert_hpu_exact(z, x)
    # Tensors sharing a storage share the copied one, as torch's deepcopy keeps them (z.real is a float64 view).
    state = copy.deepcopy({"z": z, "real": z.real, "tail": z[1:], "f64": z.abs()})
    state["z"].zero_()
    assert_hpu_exact(state["real"], torch.zeros(3, 4, dtype=torch.float64))
    assert_hpu_exact(state["tail"], torch.zeros(2, 4, dtype=torch.complex128))
    assert_hpu_exact(state["f64"], x.abs())
    # Conjugate views resolve, as torch's deepcopy does; requires_grad and .grad come along.
    c = copy.deepcopy(z.conj())
    assert not c.is_conj()
    assert_hpu_exact(c, x.conj().resolve_conj())
    w = z.clone().requires_grad_()
    w.grad = z * 2
    dw = copy.deepcopy(w)
    assert dw.requires_grad and dw.is_leaf
    assert_hpu_exact(dw.detach(), x)
    assert_hpu_exact(dw.grad, x * 2)
    p = copy.deepcopy(torch.nn.Parameter(z.clone()))
    assert isinstance(p, torch.nn.Parameter)
    assert_hpu_exact(p.detach(), x)


@pytest.mark.parametrize("op", [
    lambda t: t.t_(),
    lambda t: t.transpose_(0, 1),
    lambda t: t.unsqueeze_(0),
    lambda t: t.unsqueeze_(1).squeeze_(1),
    lambda t: t.as_strided_((2, 2), (1, 4), 1),
])
def test_inplace_restride_keeps_storage(op):
    # torch only changes sizes and strides: other views of the tensor keep sharing its memory.
    x = c128(3, 4)
    z, ref = x.to(HPU), x.clone()
    r, rr = z.real, ref.real
    assert op(z) is z
    op(ref)
    assert z.shape == ref.shape and z.stride() == ref.stride() and z.storage_offset() == ref.storage_offset()
    assert_hpu_exact(z, ref)
    z.mul_(2)
    ref.mul_(2)
    assert_hpu_exact(r, rr)  # the old .real view sees the write
    assert_hpu_exact(z, ref)
