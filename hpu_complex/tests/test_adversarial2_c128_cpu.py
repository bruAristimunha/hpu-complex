"""Review round 2b (adversarial, second tester): an op sweep of complex128 HPU tensors on the CPU stand-in.

Each case runs on CPU complex128 and on the HPU copy (host memory), compares to 1e-12, and checks that a case where
torch raises also raises. Breaks are xfail(strict=True). Ops whose results Habana holds (bool predicates, int pivots:
isfinite, ==, allclose, cholesky, inv, det, slogdet, solve, lstsq, a boolean mask, eig's double backward) cannot run on
this stand-in; they are cases of hpu_complex/tests/c128_adversarial_check.py on the card.

python -m pytest hpu_complex/tests/test_adversarial2_c128_cpu.py -q -rxX
"""
import pytest
import torch

from hpu_complex.tests.test_complex128_cpu import HPU, c128, extension, tagged  # noqa: F401

F64, C128 = torch.float64, torch.complex128


def to_hpu(x):
    return x.to(HPU) if isinstance(x, torch.Tensor) else x


def back(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().resolve_conj().resolve_neg()
    if isinstance(x, (tuple, list)):
        return [back(e) for e in x]
    return x


def same(got, ref, atol):
    if isinstance(ref, torch.Tensor):
        assert isinstance(got, torch.Tensor), type(got)
        assert got.dtype == ref.dtype and got.shape == ref.shape, (got.dtype, ref.dtype, got.shape, ref.shape)
        if ref.is_floating_point() or ref.is_complex():
            torch.testing.assert_close(got, ref, atol=atol, rtol=0, equal_nan=True)
        else:
            assert torch.equal(got, ref)
    elif isinstance(ref, (list, tuple)):
        assert len(got) == len(ref)
        for g, r in zip(got, ref):
            same(g, r, atol)
    else:
        assert got == ref, (got, ref)


A = c128(3, 4)
B = c128(3, 4, seed=1)
M = c128(4, 4, seed=2) + 4 * torch.eye(4, dtype=C128)
H = M @ M.mH  # Hermitian positive definite
V = c128(5, seed=3)

OPS = {
    # unary
    "exp": (lambda a: a.exp(), [A]), "log": (lambda a: a.log(), [A]), "sqrt": (lambda a: a.sqrt(), [A]),
    "sin": (lambda a: a.sin(), [A]), "tanh": (lambda a: a.tanh(), [A]), "reciprocal": (lambda a: a.reciprocal(), [A]),
    "angle": (lambda a: a.angle(), [A]), "conj_physical": (lambda a: a.conj_physical(), [A]),
    "neg": (lambda a: -a, [A]), "pow_scalar": (lambda a: a ** 2.5, [A]), "pow_tensor": (lambda a, b: a ** b, [A, B]),
    "isfinite": (lambda a: a.isfinite(), [A]), "isnan": (lambda a: a.isnan(), [A]),
    "sigmoid": (lambda a: torch.sigmoid(a), [A]), "rsqrt": (lambda a: a.rsqrt(), [A]),
    "abs_square": (lambda a: (a * a.conj()).real, [A]),
    # binary / ternary
    "eq": (lambda a, b: a == b, [A, B]), "ne": (lambda a, b: a != b, [A, B]),
    "addcdiv": (lambda a, b: torch.addcdiv(a, a, b, value=0.5), [A, B]),
    "lerp": (lambda a, b: torch.lerp(a, b, 0.3), [A, B]),
    "matmul": (lambda a, b: a @ b.mT, [A, B]), "einsum": (lambda a, b: torch.einsum("ij,kj->ik", a, b), [A, B]),
    "vdot": (lambda a: torch.vdot(a, a), [V]), "outer": (lambda a: torch.outer(a, a), [V]),
    "kron": (lambda a, b: torch.kron(a, b), [A, B]),
    "mul_f64_cpu_scalar": (lambda a: a * torch.tensor(2.0, dtype=F64), [A]),
    "mul_python_complex": (lambda a: a * (1 - 2j), [A]),
    "div_int": (lambda a: a / 3, [A]), "rsub": (lambda a: 1 - a, [A]),
    "allclose": (lambda a, b: torch.allclose(a, b), [A, B]),
    "equal": (lambda a: torch.equal(a, a.clone()), [A]),
    # reductions
    "sum_dim": (lambda a: a.sum(1), [A]), "mean_dim": (lambda a: a.mean(0, keepdim=True), [A]),
    "prod": (lambda a: a.prod(), [A]), "cumprod": (lambda a: a.cumprod(1), [A]),
    "std": (lambda a: a.std(), [A]), "norm_fro": (lambda a: torch.linalg.norm(a), [A]),
    "vector_norm_inf": (lambda a: torch.linalg.vector_norm(a, float("inf")), [A]),
    "trace": (lambda a: a.trace(), [M]), "diag": (lambda a: a.diag(), [M]), "diag_embed": (lambda a: torch.diag_embed(a), [A]),
    "tril": (lambda a: a.tril(), [M]), "triu_": (lambda a: a.clone().triu_(1), [M]),
    "item": (lambda a: a[1, 2].item(), [A]), "tolist": (lambda a: a.tolist(), [A]),
    # linalg (host results)
    "eigvals": (lambda a: torch.linalg.eigvals(a), [M]), "eigh_L": (lambda a: torch.linalg.eigh(a)[0], [H]),
    "eigvalsh": (lambda a: torch.linalg.eigvalsh(a), [H]),
    "svdvals": (lambda a: torch.linalg.svdvals(a), [A]), "qr_R": (lambda a: torch.linalg.qr(a)[1], [M]),
    "matrix_exp": (lambda a: torch.linalg.matrix_exp(a), [M]),
    "cholesky": (lambda a: torch.linalg.cholesky(a), [H]),
    "inv": (lambda a: torch.linalg.inv(a), [M]), "det": (lambda a: torch.linalg.det(a), [M]),
    "slogdet": (lambda a: torch.linalg.slogdet(a), [M]), "solve": (lambda a, b: torch.linalg.solve(a, b.mT), [M, B]),
    "lstsq": (lambda a, b: torch.linalg.lstsq(a, b.mT[:4]).solution, [M, B]),
    "matrix_power3": (lambda a: torch.linalg.matrix_power(a, 3), [M]),
    "matrix_power_neg": (lambda a: torch.linalg.matrix_power(a, -2), [M]),
    "cross": (lambda a, b: torch.linalg.cross(a[:, :3], b[:, :3]), [A, B]),
    # shape / views / indexing
    "flip": (lambda a: a.flip(0), [A]), "roll": (lambda a: a.roll(1, 1), [A]), "rot90": (lambda a: a.rot90(), [A]),
    "repeat": (lambda a: a.repeat(2, 1), [A]), "tile": (lambda a: a.tile(2), [A]),
    "stack": (lambda a, b: torch.stack([a, b]), [A, B]), "cat_dim1": (lambda a, b: torch.cat([a, b], 1), [A, B]),
    "chunk": (lambda a: a.chunk(2, 1), [A]), "movedim": (lambda a: a.movedim(0, 1).contiguous(), [A]),
    "reshape_noncontig": (lambda a: a.mT.reshape(-1), [A]), "flatten_view": (lambda a: a.flatten(), [A]),
    "expand_contig": (lambda a: a[:1].expand(3, 4).contiguous(), [A]),
    "gather": (lambda a: a.gather(1, torch.tensor([[0, 1], [2, 3], [1, 1]], device=a.device)), [A]),
    "index_select": (lambda a: a.index_select(1, torch.tensor([3, 0], device=a.device)), [A]),
    "take": (lambda a: a.take(torch.tensor([0, 5, 11], device=a.device)), [A]),
    "nonzero_mask": (lambda a: a[a.real > 0], [A]),
    "scatter_add": (lambda a, b: a.clone().scatter_add_(1, torch.tensor([[0, 0, 1, 1]] * 3, device=a.device), b), [A, B]),
    "index_put_accumulate": (lambda a, b: a.clone().index_put_((torch.tensor([0, 0]),), b[:2], accumulate=True), [A, B]),
    "masked_scatter": (lambda a, b: a.masked_scatter(a.real > 0, b), [A, B]),
    "pad": (lambda a: torch.nn.functional.pad(a, (1, 2)), [A]),
    "diagonal_write": (lambda a: a.clone().diagonal().fill_(7)._base.clone(), [M]),
    "polar": (lambda a: torch.polar(a.abs(), a.angle()), [A]),
    "real_imag_assign": (lambda a, b: _assign_real(a, b), [A, B]),
    # fft (host)
    "fft": (lambda a: torch.fft.fft(a), [A]), "ifft_dim0": (lambda a: torch.fft.ifft(a, dim=0), [A]),
    "fft2": (lambda a: torch.fft.fft2(a), [A]), "fft_n_pad": (lambda a: torch.fft.fft(a, n=7), [A]),
    "irfft": (lambda a: torch.fft.irfft(a), [A]), "hfft": (lambda a: torch.fft.hfft(a), [A]),
    "fftshift": (lambda a: torch.fft.fftshift(a), [A]),
    # empty / 0-d / n=1
    "empty_sum": (lambda a: a[:0].sum(0), [A]), "zero_d_mul": (lambda a: a[0, 0] * a, [A]),
    "n1_matmul": (lambda a: a[:1, :1] @ a[:1, :1], [A]),
    "broadcast_batch_matmul": (lambda a, b: a.reshape(3, 1, 4, 1) @ b.reshape(1, 3, 1, 4), [A, B]),
    "offset_transposed_exp": (lambda a: a[1:, 1:].mT.exp(), [A]),
    "stepped_add_": (lambda a, b: _stepped(a, b), [A, B]),
}


def _assign_real(a, b):
    a = a.clone()
    a.real = b.imag
    a.imag[0] = b.real[1]
    return a


def _stepped(a, b):
    a = a.clone()
    a[:, ::2] += b[:, 1::2]
    return a


@pytest.mark.parametrize("name", list(OPS))
def test_op_matches_cpu(name):
    fn, args = OPS[name]
    try:
        ref = fn(*[a.clone() for a in args])
        ref_err = None
    except Exception as e:  # noqa: BLE001
        ref, ref_err = None, e
    if ref_err is not None:
        with pytest.raises(Exception):
            fn(*[to_hpu(a) for a in args])
        return
    got = fn(*[to_hpu(a) for a in args])
    same(back(got), back(ref), atol=1e-12)


def test_raises_on_device_mismatch():
    """torch: a non-scalar CPU tensor and an HPU tensor in one op raises; a 0-d CPU tensor is a scalar."""
    z = A.to(HPU)
    for op in (lambda: z + B, lambda: z * B, lambda: z - B, lambda: z / B, lambda: B + z,
               lambda: z.clone().add_(B), lambda: z.clone().mul_(B), lambda: B.clone().add_(z)):
        with pytest.raises(RuntimeError, match="same device"):
            op()
    same(back(z * torch.tensor(2.0 + 1j, dtype=C128)), A * (2 + 1j), 0)


def test_complex_result_into_real_raises():
    """torch: a complex result written into a real tensor raises (it does not drop the imaginary part)."""
    z = A.to(HPU)
    for op in (lambda: torch.mul(z, z, out=z.clone().real), lambda: torch.add(z, z, out=z.clone().imag),
               lambda: z.clone().real.mul_(z), lambda: z.clone().real.add_(z)):
        with pytest.raises(RuntimeError, match="can't be cast"):
            op()
    r = z.clone().real
    torch.abs(z, out=r)  # a real result into a real out is fine
    same(back(r), A.abs(), 0)


def test_grad_accumulates_over_two_backwards():
    x = A.clone().requires_grad_()
    z = A.to(HPU).requires_grad_()
    for t in (x, z):
        for k in range(2):
            (t.abs() ** (k + 2)).sum().backward()
    same(back(z.grad), x.grad, 1e-12)
    assert tagged(z.grad)


def test_grad_of_view_as_real_of_slice():
    def g(x):
        x = x.requires_grad_()
        torch.view_as_real(x[1:, ::2]).pow(2).sum().backward()
        return x.grad
    same(back(g(A.to(HPU))), g(A.clone()), 1e-12)


def test_out_variant_wrong_size_resizes():
    z = A.to(HPU)
    out = torch.empty(1, dtype=C128, device=HPU)
    ref_out = torch.empty(1, dtype=C128)
    torch.exp(z, out=out)
    torch.exp(A, out=ref_out)
    same(back(out), ref_out, 0)


def test_out_variant_view_of_bigger():
    base = torch.zeros(6, 4, dtype=C128, device=HPU)
    torch.mul(A.to(HPU), 2, out=base[::2])
    ref = torch.zeros(6, 4, dtype=C128)
    torch.mul(A, 2, out=ref[::2])
    same(back(base), ref, 0)


def test_resize_then_view_sees_writes():
    z = torch.zeros(2, 2, dtype=C128, device=HPU)
    z.resize_(4, 4)
    v = z[0]
    z.fill_(3)
    same(back(v), torch.full((4,), 3, dtype=C128), 0)


def test_view_taken_before_growth_keeps_values_and_sees_writes():
    """A view taken BEFORE the storage grows: torch keeps the old bytes and the view keeps aliasing the storage."""
    for dev in ("cpu", HPU):
        z = torch.tensor([[1, 2], [3, 4]], dtype=C128, device=dev)
        before, real = z.view(-1), z.real  # a complex and a float64 view
        z.resize_(4, 4)
        same(back(z.view(-1)[:4]), torch.tensor([1, 2, 3, 4], dtype=C128), 0)  # main: the prefix was lost
        z.fill_(3)
        same(back(before), torch.full((4,), 3, dtype=C128), 0)
        same(back(real), torch.full((2, 2), 3, dtype=F64), 0)


def test_requires_grad_leaf_inplace_raises():
    z = A.to(HPU).requires_grad_()
    with pytest.raises(RuntimeError, match="leaf"):
        z.mul_(2)


def test_version_counter_detects_modification():
    """torch raises when a tensor saved for backward is modified in place."""
    x = A.to(HPU).requires_grad_()
    y = x * 1
    w = y.exp()  # exp saves its result
    w.mul_(2)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        w.sum().abs().backward()


def test_sparse_or_strided_noncontig_clone_keeps_strides():
    z = A.to(HPU).mT
    c = z.clone()
    assert c.stride() == A.mT.clone().stride()
    c2 = z.clone(memory_format=torch.contiguous_format)
    assert c2.stride() == A.mT.clone(memory_format=torch.contiguous_format).stride()


FACTORIES = {
    "eye": lambda d: torch.eye(3, dtype=C128, device=d),
    "full": lambda d: torch.full((2, 3), 1 - 2j, dtype=C128, device=d),
    "linspace": lambda d: torch.linspace(0, 1 + 1j, 5, dtype=C128, device=d),
    "tensor": lambda d: torch.tensor([[1 + 2j, 3e-300 - 1j]], dtype=C128, device=d),
    "zeros_like_noncontig": lambda d: torch.zeros_like(A.to(d).mT),
    "ones_like": lambda d: torch.ones_like(A.to(d)),
    "full_like": lambda d: torch.full_like(A.to(d), 2j),
    "polar_f64": lambda d: torch.polar(torch.ones(3, dtype=F64).to(d) if d == "cpu" else A.to(d).abs()[0, :3] * 0 + 1,
                                       torch.zeros(3, dtype=F64).to(d) if d == "cpu" else A.to(d).abs()[0, :3] * 0),
}


@pytest.mark.parametrize("name", list(FACTORIES))
def test_factory(name):
    same(back(FACTORIES[name](HPU)), FACTORIES[name]("cpu"), 0)


@pytest.mark.parametrize("foreach", [False, True])
@pytest.mark.parametrize("opt", ["SGD", "Adam", "AdamW"])
def test_optimizer_on_complex128_parameter(opt, foreach):
    def run(dev):
        p = torch.nn.Parameter(M.clone().to(dev))
        o = getattr(torch.optim, opt)([p], lr=0.1, foreach=foreach, **({"momentum": 0.9} if opt == "SGD" else {}))
        for _ in range(3):
            o.zero_grad()
            (p @ p.mH).abs().sum().backward()
            o.step()
        return p.detach()
    same(back(run(HPU)), run("cpu"), 1e-12)


def test_clip_grad_norm_complex128():
    def run(dev):
        p = torch.nn.Parameter(M.clone().to(dev))
        (p.abs() ** 2).sum().backward()
        n = torch.nn.utils.clip_grad_norm_([p], 1.0)
        return n, p.grad
    same(back(run(HPU)), list(run("cpu")), 1e-12)


def test_overlapping_out_raises_like_torch():
    """torch raises when out= partly overlaps an input it can prove dense (a 1-D view here). The tester's 2-D case
    (x[:, :-1] into x[:, 1:]) does not raise on the CPU either: torch's check gives up on non-dense views."""
    for t in (A.clone(), A.to(HPU)):
        v = t.view(-1)
        with pytest.raises(RuntimeError, match="single memory location"):
            torch.add(v[:-1], 1, out=v[1:])
        with pytest.raises(RuntimeError, match="single memory location"):
            torch.mul(v[:-1], 2, out=v[1:])
    x = A.clone()
    torch.add(x[:, :-1], 1, out=x[:, 1:])  # no raise on the CPU: TOO_HARD
    z = A.to(HPU)
    torch.add(z[:, :-1], 1, out=z[:, 1:])  # no raise either (values: see README Limits)


def test_save_load_requires_grad_and_map_location(tmp_path):
    from hpu_complex import _pickle_complex128_through_the_cpu

    _pickle_complex128_through_the_cpu()  # enable() installs it; it needs torch.hpu, absent here
    z = A.to(HPU).requires_grad_()
    r = (z.detach() * (1 + 1e-12)).abs()  # float64 in host memory: exact float64 values Habana could not hold
    torch.save({"z": z, "r": r, "re": z.detach().real}, tmp_path / "t.pt")
    back_ = torch.load(tmp_path / "t.pt", map_location="cpu")
    assert back_["z"].requires_grad and back_["z"].device.type == "cpu"
    same(back_["z"].detach(), A, 0)
    same(back_["r"], (A * (1 + 1e-12)).abs(), 0)
    on = torch.load(tmp_path / "t.pt")  # weights_only=True (torch 2.6's default): the rebuild is a safe global
    for k in ("z", "r", "re"):
        assert on[k].device.type == "hpu" and tagged(on[k]), k  # float64 too: host memory, not Habana's
    same(back(on["z"]), A, 0)
    same(back(on["r"]), (A * (1 + 1e-12)).abs(), 0)
    same(back(on["re"]), A.real.contiguous(), 0)
    torch.save({"r": r, "empty": z.detach()[:0], "conj": z.detach().conj()}, tmp_path / "u.pt")
    mapped = torch.load(tmp_path / "u.pt", map_location="hpu")  # the values are bytes: no storage restored on the card
    assert tagged(mapped["r"]) and mapped["empty"].shape == (0, 4)
    same(back(mapped["r"]), (A * (1 + 1e-12)).abs(), 0)
    same(back(mapped["conj"]), A.conj().resolve_conj(), 0)
    buf = torch.zeros(3, 4, dtype=C128, device=HPU).real  # load_state_dict copies into a host buffer: exact
    buf.copy_(on["r"])
    same(back(buf), (A * (1 + 1e-12)).abs(), 0)


def test_module_to_complex128_and_state_dict_roundtrip():
    m = torch.nn.Linear(4, 2, dtype=C128)
    ref = m(A)
    m = m.to(HPU)
    sd = {k: v.cpu() for k, v in m.state_dict().items()}
    m2 = torch.nn.Linear(4, 2, dtype=C128).to(HPU)
    m2.load_state_dict(sd)
    same(back(m2(A.to(HPU))), ref.detach(), 1e-12)


def test_storage_offset_and_data_ptr_consistent():
    z = A.to(HPU)
    v = z[1:, 2:]
    assert v.storage_offset() == A[1:, 2:].storage_offset()
    assert v.untyped_storage().data_ptr() == z.untyped_storage().data_ptr()
    assert v.data_ptr() == z.data_ptr() + v.storage_offset() * 16
