"""complex64 linalg on the card path (hpu_complex.linalg), on the CPU stand-in, against torch.

cpu_stub/hpu_guard_stub.cpp stands in for Habana's float kernels (torch's CPU kernels on host memory reporting the HPU
device), so the kernels here run their real route: complex64 -> float pairs -> real embedding -> the real op -> back.
``FORCE_REAL_ON_CARD`` makes every size count as one the real op runs on the card (on a Gaudi that is hpu-ops' TPC
kernels; clinalg_card_check.py measures it there). Checked: values and gradients against torch (complex128 on the
CPU), torch's shapes, batch broadcast, n = 1, conjugate and strided inputs, ``info`` and errors, no CPU fallback, and
the calls that must keep torch's route (complex128, empty, traced).

python -m pytest hpu_complex/tests/test_linalg_card_cpu.py -q
"""
import warnings

import pytest
import torch

from hpu_complex.tests.test_complex128_cpu import HPU, extension, tagged  # noqa: F401

C64, C128 = torch.complex64, torch.complex128


@pytest.fixture(scope="module", autouse=True)
def card_linalg(extension):  # noqa: F811
    import hpu_complex.linalg as L

    L.FORCE_REAL_ON_CARD = True
    L.install()
    yield L
    L.uninstall()
    L.FORCE_REAL_ON_CARD = False


def rand(*shape, seed=0, dtype=C128):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, dtype=dtype, generator=g)


def hpd(*shape, seed=0):
    A = rand(*shape, seed=seed)
    n = shape[-1]
    return A @ A.mH / n + torch.eye(n, dtype=C128)


def tri(*shape, seed=0, upper=False):
    A = rand(*shape, seed=seed)
    n = shape[-1]
    return (A.triu() if upper else A.tril()) + 2 * torch.eye(n, dtype=C128)


def taken(L, name):
    return L.CALLS.get(name, 0)


def check(f, inputs, name, L, tol=2e-5, gtol=5e-5, grad=True):
    """f on the HPU (complex64) against f on the CPU (complex128) with the same complex64 values: outputs, and the
    gradient of sum |out|^2. Asserts the call was taken on the card and nothing fell back to the CPU."""
    vals = [x.to(C64) for x in inputs]
    ref_in = [v.to(C128).requires_grad_(grad) for v in vals]
    ref = f(*ref_in)
    hin = [v.to(HPU).requires_grad_(grad) for v in vals]
    before, calls = torch.ops.hpu_complex.cpu_fallback_calls(), taken(L, name)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = f(*hin)
        refs = ref if isinstance(ref, (tuple, list)) else (ref,)
        outs = out if isinstance(out, (tuple, list)) else (out,)
        if grad:
            loss = sum((o.abs() ** 2).sum() for o in outs if o.is_complex() or o.is_floating_point())
            loss.backward()
            ref_loss = sum((o.abs() ** 2).sum() for o in refs if o.is_complex() or o.is_floating_point())
            ref_loss.backward()
    assert taken(L, name) > calls, f"{name} not taken"
    assert torch.ops.hpu_complex.cpu_fallback_calls() == before, [str(w.message) for w in caught]
    assert not [w for w in caught if "runs on the CPU" in str(w.message)]
    for o, r in zip(outs, refs):
        assert o.device.type == "hpu" and o.shape == r.shape, (o.shape, r.shape)
        if r.is_complex() or (r.is_floating_point() and r.numel()):
            assert o.dtype == (C64 if r.is_complex() else torch.float32)
            assert tagged(o) or not r.is_complex()
            scale = r.abs().max().clamp_min(1).item()
            assert (o.detach().cpu().to(r.dtype) - r.detach()).abs().max().item() <= tol * scale
        else:
            assert torch.equal(o.cpu(), r.cpu().to(o.dtype)), (o, r)
    if grad:
        for h, r in zip(hin, ref_in):
            scale = r.grad.abs().max().clamp_min(1).item()
            assert (h.grad.cpu().to(C128) - r.grad).abs().max().item() <= gtol * scale
    return outs


# ---- solve / inv ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 5, 9])
def test_solve(card_linalg, n):
    check(torch.linalg.solve, [rand(3, n, n, seed=1), rand(3, n, 2, seed=2)], "solve", card_linalg)


@pytest.mark.parametrize("shapes", [((4, 4), (4, 3)), ((2, 1, 4, 4), (3, 4, 1)), ((4, 4), (4,)), ((2, 4, 4), (2, 4)),
                                    ((4, 4), (5, 2, 4, 3))])
def test_solve_shapes_and_broadcast(card_linalg, shapes):
    sa, sb = shapes
    check(torch.linalg.solve, [rand(*sa, seed=1), rand(*sb, seed=2)], "solve", card_linalg)


def test_solve_left_false_and_conjugate_strided_inputs(card_linalg):
    A = rand(2, 5, 5, seed=1)
    B = rand(2, 3, 5, seed=2)
    check(lambda a, b: torch.linalg.solve(a, b, left=False), [A, B], "solve", card_linalg)
    check(lambda a, b: torch.linalg.solve(a.mH, b[:, ::2].mT), [A, B], "solve", card_linalg)
    big = rand(2, 7, 7, seed=3)
    check(lambda a, b: torch.linalg.solve(a[:, 1:6, 2:], b.mT), [big, B], "solve", card_linalg)


def test_solve_ex_info_and_singular_error(card_linalg):
    A = rand(3, 4, 4, seed=1).to(C64)
    A[1, 2] = 0  # a zero row: singular
    X, info = torch.linalg.solve_ex(A.to(HPU), rand(3, 4, 2, seed=2).to(C64).to(HPU))
    info = info.cpu()
    assert info.dtype == torch.int32 and info.shape == (3,)
    assert info[0] == 0 and info[2] == 0 and info[1] > 0
    with pytest.raises(torch.linalg.LinAlgError, match="singular"):
        torch.linalg.solve(A.to(HPU), rand(3, 4, 2, seed=2).to(C64).to(HPU))
    with pytest.raises(RuntimeError, match="square"):
        torch.linalg.solve(rand(3, 4, seed=1).to(C64).to(HPU), rand(3, 2, seed=2).to(C64).to(HPU))


@pytest.mark.parametrize("n", [1, 3, 8])
def test_inv(card_linalg, n):
    check(torch.linalg.inv, [rand(2, n, n, seed=n)], "inv", card_linalg)
    check(torch.linalg.inv, [rand(n, n, seed=n)], "inv", card_linalg)


def test_inv_ex_info(card_linalg):
    A = rand(2, 3, 3, seed=1).to(C64)
    A[0, :, 1] = 0
    inv, info = torch.linalg.inv_ex(A.to(HPU))
    assert info.cpu().tolist()[1] == 0 and info.cpu().tolist()[0] > 0
    with pytest.raises(torch.linalg.LinAlgError):
        torch.linalg.inv(A.to(HPU))


def test_solve_double_backward(card_linalg):
    A = rand(3, 3, seed=1).to(C64).to(HPU).requires_grad_()
    B = rand(3, 2, seed=2).to(C64).to(HPU)
    (g,) = torch.autograd.grad(torch.linalg.solve(A, B).abs().sum(), A, create_graph=True)
    (gg,) = torch.autograd.grad(g.abs().sum(), A)
    Ar = A.detach().cpu().to(C128).requires_grad_()
    (gr,) = torch.autograd.grad(torch.linalg.solve(Ar, B.cpu().to(C128)).abs().sum(), Ar, create_graph=True)
    (ggr,) = torch.autograd.grad(gr.abs().sum(), Ar)
    assert (gg.cpu().to(C128) - ggr).abs().max() <= 1e-3 * ggr.abs().max()


# ---- triangular, Cholesky ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("upper", [False, True])
@pytest.mark.parametrize("left", [True, False])
@pytest.mark.parametrize("unitriangular", [False, True])
def test_solve_triangular(card_linalg, upper, left, unitriangular):
    n = 6
    A = rand(2, n, n, seed=1) + 3 * torch.eye(n, dtype=C128)  # the other triangle is not read
    B = rand(2, n, 3, seed=2) if left else rand(2, 3, n, seed=2)
    check(lambda a, b: torch.linalg.solve_triangular(a, b, upper=upper, left=left, unitriangular=unitriangular),
          [A, B], "solve_triangular", card_linalg)


@pytest.mark.parametrize("n", [1, 2, 7])
def test_solve_triangular_sizes_broadcast_conjugate(card_linalg, n):
    T = tri(n, n, seed=n, upper=True)
    check(lambda a, b: torch.linalg.solve_triangular(a, b, upper=True), [T, rand(4, n, 2, seed=2)],
          "solve_triangular", card_linalg)
    check(lambda a, b: torch.linalg.solve_triangular(a.mH, b, upper=False), [T, rand(n, 1, seed=2)],
          "solve_triangular", card_linalg)


def test_solve_triangular_zero_diagonal_is_not_finite(card_linalg):
    T = tri(3, 3, seed=1, upper=True).to(C64)
    T[1, 1] = 0
    X = torch.linalg.solve_triangular(T.to(HPU), rand(3, 1, seed=2).to(C64).to(HPU), upper=True)
    assert not torch.isfinite(torch.view_as_real(X.cpu())).all()


@pytest.mark.parametrize("n", [1, 2, 6, 11])
@pytest.mark.parametrize("upper", [False, True])
def test_cholesky(card_linalg, n, upper):
    H = hpd(3, n, n, seed=n)
    check(lambda a: torch.linalg.cholesky(a, upper=upper), [H], "cholesky", card_linalg)


def test_cholesky_reads_one_triangle(card_linalg):
    H = hpd(2, 5, 5, seed=1)
    junk = H.clone()
    junk.triu(1).add_(7 - 3j)  # no effect: triu(1) is a copy
    junk = H.tril() + torch.triu(rand(2, 5, 5, seed=9), 1)  # garbage above the diagonal
    junk.diagonal(0, -2, -1).imag.fill_(5.0)  # the imaginary part of the diagonal is ignored too
    check(torch.linalg.cholesky, [junk], "cholesky", card_linalg)
    check(lambda a: torch.linalg.cholesky(a.mH, upper=True), [junk], "cholesky", card_linalg)


def test_cholesky_info_and_error(card_linalg):
    H = hpd(3, 4, 4, seed=1).to(C64)
    H[1, 2, 2] = -5.0  # not positive definite at order 3
    L, info = torch.linalg.cholesky_ex(H.to(HPU))
    assert info.cpu().tolist() == torch.linalg.cholesky_ex(H).info.tolist() == [0, 3, 0]
    with pytest.raises(torch.linalg.LinAlgError, match="positive-definite"):
        torch.linalg.cholesky(H.to(HPU))


@pytest.mark.parametrize("upper", [False, True])
def test_cholesky_solve(card_linalg, upper):
    L = torch.linalg.cholesky(hpd(2, 5, 5, seed=1), upper=upper)
    check(lambda b, l: torch.cholesky_solve(b, l, upper=upper), [rand(2, 5, 3, seed=2), L], "cholesky_solve",
          card_linalg)
    check(lambda b, l: torch.cholesky_solve(b, l, upper=upper), [rand(3, 1, 5, 1, seed=2), L], "cholesky_solve",
          card_linalg)


# ---- matrix_exp, householder_product --------------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [1e-6, 0.3, 2.0])
def test_matrix_exp(card_linalg, scale):
    check(torch.linalg.matrix_exp, [rand(2, 6, 6, seed=1) * scale], "matrix_exp", card_linalg, tol=5e-5, gtol=2e-4)


@pytest.mark.parametrize("scale", [1e-8, 0.1, 1.0, 10.0, 300.0])
def test_matrix_exp_real_composite(scale):
    """The composite used where hpu-ops' kernel does not take the size, against float64."""
    from hpu_complex.linalg import _expm_real_composite

    M = torch.randn(4, 8, 8, generator=torch.Generator().manual_seed(1)) * scale / 8
    ref = torch.linalg.matrix_exp(M.double())
    got = _expm_real_composite(M)
    f32 = torch.linalg.matrix_exp(M)
    bar = (f32.double() - ref).abs().max() / ref.abs().max()
    assert (got.double() - ref).abs().max() / ref.abs().max() <= 10 * bar + 1e-6


def test_matrix_exp_real_composite_non_finite():
    from hpu_complex.linalg import _expm_real_composite

    M = torch.zeros(2, 3, 3)
    M[0, 1, 1] = float("nan")
    M[1, 0, 2] = float("inf")
    got = _expm_real_composite(M)
    assert got[0].isnan().any() and not torch.isfinite(got[1]).all()


@pytest.mark.parametrize("mnk", [(5, 5, 5), (6, 4, 3), (3, 1, 1), (7, 3, 2)])
def test_householder_product(card_linalg, mnk):
    m, n, k = mnk
    A = rand(2, m, n, seed=1)
    tau = rand(2, k, seed=2)
    check(torch.linalg.householder_product, [A, tau], "householder_product", card_linalg)
    a, t = torch.geqrf(rand(2, m, n, seed=3))
    check(torch.linalg.householder_product, [a, t[..., :k]], "householder_product", card_linalg)


def test_householder_product_zero_tau(card_linalg):
    a, t = torch.geqrf(rand(5, 4, seed=3))
    t[1] = 0
    check(torch.linalg.householder_product, [a, t], "householder_product", card_linalg)


# ---- eigh ---------------------------------------------------------------------------------------------------------


def hermitian(kind, b, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(b, n, n, dtype=C128, generator=g))
    if kind == "random":
        A = torch.randn(b, n, n, dtype=C128, generator=g)
        return A + A.mH
    if kind == "identity":
        return torch.eye(n, dtype=C128).expand(b, n, n).clone()
    d = torch.ones(n, dtype=torch.float64)
    if kind == "cluster":  # a cluster of n // 2 equal eigenvalues, another of the rest, one apart
        d[n // 2:] = 2
        d[-1] = 5
    elif kind == "near":  # all within 1e-4
        d = torch.linspace(1, 1 + 1e-4, n, dtype=torch.float64)
    return Q @ torch.diag_embed(d.to(C128)) @ Q.mH


def check_eigh(H, L, upper=False, tol=3e-6):
    """Gauge-free checks: eigenvalues against torch, residual |H V - V diag(w)| and orthonormality."""
    x = H.to(C64)
    hin = x.to(HPU)
    before, calls = torch.ops.hpu_complex.cpu_fallback_calls(), taken(L, "eigh")
    w, V = torch.linalg.eigh(hin, UPLO="U" if upper else "L")
    assert taken(L, "eigh") > calls and torch.ops.hpu_complex.cpu_fallback_calls() == before
    assert w.dtype == torch.float32 and V.dtype == C64 and w.shape == H.shape[:-1] and V.shape == H.shape
    w, V = w.cpu().double(), V.cpu().to(C128)
    wr = torch.linalg.eigvalsh(x.to(C128))
    scale = H.abs().amax().item()
    n = H.shape[-1]
    assert (w - wr).abs().max() <= tol * scale
    assert (x.to(C128) @ V - V * w.unsqueeze(-2)).abs().max() <= tol * scale
    assert (V.mH @ V - torch.eye(n, dtype=C128)).abs().max() <= tol


@pytest.mark.parametrize("kind", ["random", "identity", "cluster", "near"])
@pytest.mark.parametrize("n", [1, 2, 5, 8])
def test_eigh(card_linalg, kind, n):
    check_eigh(hermitian(kind, 3, n, seed=n), card_linalg)


def test_eigh_upper_and_one_triangle(card_linalg):
    H = hermitian("random", 2, 6, seed=1)
    junk = H.triu() + torch.tril(rand(2, 6, 6, seed=9), -1)
    junk.diagonal(0, -2, -1).imag.fill_(3.0)
    check_eigh(H, card_linalg)
    x = junk.to(C64).to(HPU)
    w, V = torch.linalg.eigh(x, UPLO="U")
    torch.testing.assert_close(w.cpu().double(), torch.linalg.eigvalsh(H), atol=3e-5, rtol=0)


def test_eigvalsh(card_linalg):
    H = hermitian("random", 2, 6, seed=1)
    calls = taken(card_linalg, "eigh")
    w = torch.linalg.eigvalsh(H.to(C64).to(HPU))
    assert taken(card_linalg, "eigh") > calls
    torch.testing.assert_close(w.cpu().double(), torch.linalg.eigvalsh(H), atol=3e-5, rtol=0)


def test_eigh_gradient(card_linalg):
    """A loss that does not depend on the eigenvectors' phases: torch's gradient."""
    H = hermitian("random", 2, 5, seed=3).to(C64)

    def loss(A):
        w, V = torch.linalg.eigh(A)
        return (w * torch.arange(1, 6, dtype=w.dtype, device=w.device)).sum() + (V.abs() ** 4).sum()

    a = H.to(HPU).requires_grad_()
    loss(a).backward()
    r = H.to(C128).requires_grad_()
    loss(r).backward()
    assert (a.grad.cpu().to(C128) - r.grad).abs().max() <= 1e-4 * r.grad.abs().max()


# ---- pinv, lstsq --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(4, 4), (6, 3), (3, 6), (1, 1), (2, 5, 2)])
def test_pinv(card_linalg, shape):
    check(torch.linalg.pinv, [rand(*shape, seed=1)], "pinv", card_linalg, gtol=2e-4)


def test_pinv_hermitian_and_tolerances(card_linalg):
    H = hpd(2, 4, 4, seed=1)
    check(lambda a: torch.linalg.pinv(a, hermitian=True), [H], "pinv", card_linalg, gtol=2e-4)
    A = rand(2, 5, 4, seed=2)
    A[..., -1] = A[..., 0] * 1e-4  # a tiny singular value, cut by rtol
    check(lambda a: torch.linalg.pinv(a, rtol=1e-2), [A], "pinv", card_linalg, gtol=2e-4)
    check(lambda a: torch.linalg.pinv(a, atol=0.5), [A], "pinv", card_linalg, gtol=2e-4)


@pytest.mark.parametrize("driver", [None, "gels", "gelsd", "gelss", "gelsy"])
def test_lstsq(card_linalg, driver):
    A, B = rand(2, 6, 4, seed=1), rand(2, 6, 3, seed=2)
    sol, res, rank, sv = check(lambda a, b: torch.linalg.lstsq(a, b, driver=driver), [A, B], "lstsq", card_linalg,
                               grad=False)
    ref = torch.linalg.lstsq(A.to(C64), B.to(C64), driver=driver)
    assert rank.cpu().tolist() == ref.rank.tolist()
    assert res.shape == ref.residuals.shape and sv.shape == ref.singular_values.shape
    if ref.residuals.numel():
        torch.testing.assert_close(res.cpu(), ref.residuals, rtol=1e-4, atol=1e-5)
    if ref.singular_values.numel():
        torch.testing.assert_close(sv.cpu(), ref.singular_values, rtol=1e-5, atol=1e-6)


def test_lstsq_gradient_and_vector(card_linalg):
    check(lambda a, b: torch.linalg.lstsq(a, b).solution, [rand(2, 6, 4, seed=1), rand(2, 6, 3, seed=2)], "lstsq",
          card_linalg, gtol=2e-4)
    check(lambda a, b: torch.linalg.lstsq(a, b).solution, [rand(4, 4, seed=1), rand(4, 2, seed=2)], "lstsq",
          card_linalg, gtol=2e-4)


# ---- what keeps torch's route --------------------------------------------------------------------------------------


def test_complex128_keeps_the_host_route(card_linalg):
    A, B = rand(3, 3, seed=1), rand(3, 2, seed=2)
    before = taken(card_linalg, "solve")
    X = torch.linalg.solve(A.to(HPU), B.to(HPU))
    assert taken(card_linalg, "solve") == before
    assert X.dtype == C128 and torch.equal(X.cpu(), torch.linalg.solve(A, B))


def test_empty_keeps_torch_route(card_linalg):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X = torch.linalg.solve(torch.zeros(0, 3, 3, dtype=C64, device=HPU), torch.zeros(0, 3, 2, dtype=C64, device=HPU))
    assert X.shape == (0, 3, 2)


def test_traced_calls_keep_torch_route(card_linalg, monkeypatch):
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    before = dict(card_linalg.CALLS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X = torch.linalg.solve(rand(2, 4, 4, seed=1).to(C64).to(HPU), rand(2, 4, 3, seed=2).to(C64).to(HPU))
    assert X.shape == (2, 4, 3) and card_linalg.CALLS == before


def test_uninstall_restores_the_cpu_route(card_linalg):
    card_linalg.uninstall()
    try:
        before = torch.ops.hpu_complex.cpu_fallback_calls()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.linalg.inv(rand(3, 3, seed=1).to(C64).to(HPU))
        assert torch.ops.hpu_complex.cpu_fallback_calls() > before
    finally:
        card_linalg.install()


def test_cholesky_batch_equal_to_the_real_order(card_linalg):
    """Batch 8 of order 4: the real order is 8, Habana's failing shape; taken through (8, 1, 8, 8)."""
    check(torch.linalg.cholesky, [hpd(8, 4, 4, seed=2)], "cholesky", card_linalg)
    check(torch.linalg.cholesky, [hpd(2, 8, 4, 4, seed=2)], "cholesky", card_linalg)


# ---- adversarial ---------------------------------------------------------------------------------------------------


def test_mixed_dtypes_raise_as_torch(card_linalg):
    A = rand(3, 3, seed=1).to(C64).to(HPU)
    with pytest.raises(RuntimeError) as hpu_error:
        torch.linalg.solve(A, rand(3, 2, seed=2).to(HPU))  # complex128 right-hand side
    with pytest.raises(RuntimeError) as cpu_error:
        torch.linalg.solve(A.cpu(), rand(3, 2, seed=2))
    assert str(cpu_error.value).split("(")[0] in str(hpu_error.value)


def test_offset_and_expanded_views(card_linalg):
    big = rand(4, 9, 9, seed=1)
    check(lambda a, b: torch.linalg.solve(a[1:3, 2:7, 3:8], b[:, :, 1:3]), [big, rand(2, 5, 4, seed=2)], "solve",
          card_linalg)
    check(lambda a: torch.linalg.inv(a.expand(3, 4, 4)), [rand(4, 4, seed=3)], "inv", card_linalg)
    check(lambda a, b: torch.linalg.solve_triangular(a[1:, 1:].mT, b, upper=True),
          [tri(6, 6, seed=4), rand(5, 2, seed=5)], "solve_triangular", card_linalg)


def test_non_finite_inputs_give_non_finite_outputs(card_linalg):
    A = rand(2, 3, 3, seed=1).to(C64)
    A[0, 1, 1] = complex(float("nan"), 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X = torch.linalg.solve_ex(A.to(HPU), rand(2, 3, 1, seed=2).to(C64).to(HPU)).result.cpu()
        E = torch.linalg.matrix_exp(A.to(HPU)).cpu()
    assert torch.view_as_real(X[0]).isnan().any() and torch.isfinite(torch.view_as_real(X[1])).all()
    assert torch.view_as_real(E[0]).isnan().any() and torch.isfinite(torch.view_as_real(E[1])).all()


def test_inference_mode_and_no_grad(card_linalg):
    A, B = rand(3, 3, seed=1).to(C64), rand(3, 2, seed=2).to(C64)
    ref = torch.linalg.solve(A.to(C128), B.to(C128))
    with torch.no_grad(), warnings.catch_warnings():  # solve: torch's route (its LU would be read by a traced backward)
        warnings.simplefilter("ignore")
        calls = taken(card_linalg, "solve")
        X = torch.linalg.solve(A.to(HPU), B.to(HPU))
        assert taken(card_linalg, "solve") == calls and not X.requires_grad
        calls = taken(card_linalg, "inv")
        torch.linalg.inv(A.to(HPU))
        assert taken(card_linalg, "inv") > calls
    with torch.inference_mode(), warnings.catch_warnings():  # below autograd: torch's route (the CPU fallback)
        warnings.simplefilter("ignore")
        Y = torch.linalg.solve(A.to(HPU), B.to(HPU))
    for got in (X, Y):
        assert (got.cpu().to(C128) - ref).abs().max() < 1e-5


def test_repeated_calls_and_install_idempotent(card_linalg):
    card_linalg.install()  # twice: no second registration
    A = rand(2, 4, 4, seed=1)
    for _ in range(3):
        check(torch.linalg.inv, [A], "inv", card_linalg)


@pytest.mark.parametrize("lazy", [False, True])
def test_default_routes(card_linalg, monkeypatch, lazy):
    """Without HPU_COMPLEX_LINALG=all only the measured routes are taken (ROUTES)."""
    monkeypatch.setattr(card_linalg, "FORCE_REAL_ON_CARD", False)
    monkeypatch.setattr(card_linalg, "_real_on_card", lambda kind, n2, batch=1: kind != "matrix_exp")
    monkeypatch.setattr(card_linalg, "_lazy", lambda: lazy)
    monkeypatch.delenv("HPU_COMPLEX_LINALG", raising=False)
    A, B = rand(2, 4, 4, seed=1).to(C64).to(HPU), rand(2, 4, 4, seed=2).to(C64).to(HPU)
    H = hpd(2, 4, 4, seed=3).to(C64).to(HPU)
    a, tau = (t.to(C64).to(HPU) for t in torch.geqrf(rand(2, 5, 4, seed=4)))

    def taken_by(name, f):
        before = taken(card_linalg, name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f()
        return taken(card_linalg, name) > before

    assert taken_by("solve", lambda: torch.linalg.solve(A, B))
    assert taken_by("solve", lambda: torch.linalg.solve(A, B, left=False)) == lazy
    assert taken_by("inv", lambda: torch.linalg.inv(A))
    assert taken_by("cholesky", lambda: torch.linalg.cholesky(H))
    assert taken_by("matrix_exp", lambda: torch.linalg.matrix_exp(A)) == lazy  # the composite
    assert not taken_by("householder_product", lambda: torch.linalg.householder_product(a, tau))
    assert taken_by("householder_product",
                    lambda: torch.linalg.householder_product(a.requires_grad_(), tau).abs().sum().backward())
    assert not taken_by("eigh", lambda: torch.linalg.eigh(H))
    assert not taken_by("solve_triangular", lambda: torch.linalg.solve_triangular(A, B, upper=True))
    monkeypatch.setenv("HPU_COMPLEX_LINALG", "all")
    assert taken_by("eigh", lambda: torch.linalg.eigh(H))
