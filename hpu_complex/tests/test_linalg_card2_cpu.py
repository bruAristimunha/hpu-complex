"""complex64 linalg on the card path, second round: cholesky info, HPD slogdet/logdet, eigvalsh, qr, cholesky_inverse,
cholesky_solve (inverse-based) and solve_triangular (rotation + one real substitution), on the CPU stand-in, against
torch (complex128 on the CPU). See test_linalg_card_cpu.py for the stand-in.

python -m pytest hpu_complex/tests/test_linalg_card2_cpu.py -q
"""
import warnings

import pytest
import torch

from hpu_complex.tests.test_complex128_cpu import HPU, extension  # noqa: F401
from hpu_complex.tests.test_linalg_card_cpu import C64, C128, card_linalg, check, hpd, rand, taken, tri  # noqa: F401, F811


# ---- cholesky info ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [[0], [3], [6], [2, 5]])
@pytest.mark.parametrize("upper", [False, True])
def test_cholesky_info_index_as_torch(card_linalg, bad, upper):
    """The first failing complex pivot, 1-based, as torch (computed from the factor's diagonal: Habana's info is 0)."""
    H = hpd(3, 7, 7, seed=4).to(C64)
    for k in bad:
        H[1, k, k] = -50.0
    H[2, 4, 4] = float("nan")
    ref = torch.linalg.cholesky_ex(H, upper=upper).info
    L, info = torch.linalg.cholesky_ex(H.to(HPU), upper=upper)
    assert info.dtype == torch.int32 and info.shape == (3,)
    assert info.cpu().tolist() == ref.tolist()
    with pytest.raises(torch.linalg.LinAlgError, match=f"order {ref[1].item()} is not positive-definite"):
        torch.linalg.cholesky(H.to(HPU), upper=upper)


def test_cholesky_info_matrix_and_check_errors(card_linalg):
    H = hpd(4, 4, seed=1).to(C64)
    H[0, 0] = -1.0
    _, info = torch.linalg.cholesky_ex(H.to(HPU))
    assert info.shape == () and info.item() == 1
    with pytest.raises(torch.linalg.LinAlgError):
        torch.linalg.cholesky_ex(H.to(HPU), check_errors=True)


def test_cholesky_info_from_the_diagonal_only():
    """The info rule itself on a factor like Habana's (NaN from the failing pivot on, no info of its own)."""
    from hpu_complex.linalg import _cholesky_info

    Lreal = torch.eye(8).repeat(3, 1, 1)
    Lreal[1, 4:, 4:] = float("nan")  # complex pivot 3 (real 4, 5) failed
    Lreal[2, 5, 5] = -0.0  # the second real pivot of complex 3 not positive
    assert _cholesky_info(Lreal).tolist() == [0, 3, 3]


@pytest.mark.parametrize("batch", [(8,), (2, 8), (8, 1)])
def test_cholesky_batch_equal_to_the_real_order_with_info(card_linalg, batch):
    H = hpd(*batch, 4, 4, seed=2)
    check(torch.linalg.cholesky, [H], "cholesky", card_linalg)
    assert torch.linalg.cholesky_ex(H.to(C64).to(HPU)).info.cpu().eq(0).all()


# ---- HPD slogdet / logdet ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(1, 1), (5, 5), (3, 6, 6), (2, 8, 4, 4)])
def test_hpd_slogdet(card_linalg, shape):
    from hpu_complex.linalg import hpd_logdet, hpd_slogdet

    H = hpd(*shape, seed=3)
    x = H.to(C64)
    sign, logabsdet = hpd_slogdet(x.to(HPU))
    ref = torch.linalg.slogdet(x.to(C128))
    assert sign.dtype == C64 and logabsdet.dtype == torch.float32 and logabsdet.shape == ref.logabsdet.shape
    assert torch.equal(sign.cpu(), torch.ones_like(sign.cpu()))
    torch.testing.assert_close(logabsdet.cpu().double(), ref.logabsdet, atol=2e-5 * shape[-1], rtol=1e-5)
    ld = hpd_logdet(x.to(HPU))
    assert ld.dtype == C64
    torch.testing.assert_close(ld.cpu().to(C128), torch.logdet(x.to(C128)), atol=2e-5 * shape[-1], rtol=1e-5)


def test_hpd_slogdet_gradient_is_torchs(card_linalg):
    """torch's slogdet gradient at H (g H^-1, full), also through logdet; the lower triangle is read."""
    from hpu_complex.linalg import hpd_logdet, hpd_slogdet

    H = hpd(3, 5, 5, seed=5).to(C64)
    w = torch.arange(1, 4, dtype=torch.float32)
    a = H.to(HPU).requires_grad_()
    (hpd_slogdet(a).logabsdet * w.to(HPU)).sum().backward()
    r = H.to(C128).requires_grad_()
    (torch.linalg.slogdet(r).logabsdet * w.double()).sum().backward()
    torch.testing.assert_close(a.grad.cpu().to(C128), r.grad, atol=1e-5, rtol=1e-4)
    a = H.to(HPU).requires_grad_()
    hpd_logdet(a).real.sum().backward()
    r = H.to(C128).requires_grad_()
    torch.logdet(r).real.sum().backward()
    torch.testing.assert_close(a.grad.cpu().to(C128), r.grad, atol=1e-5, rtol=1e-4)


def test_hpd_slogdet_reads_lower_and_flags_non_pd(card_linalg):
    from hpu_complex.linalg import hpd_slogdet

    H = hpd(2, 4, 4, seed=1).to(C64)
    junk = H.tril() + 9j * torch.triu(torch.ones(4, 4, dtype=C64), 1) + 0.5j * torch.eye(4, dtype=C64)
    torch.testing.assert_close(hpd_slogdet(junk.to(HPU)).logabsdet.cpu(), torch.linalg.slogdet(H).logabsdet,
                               atol=1e-5, rtol=1e-5)
    H[1, 2, 2] = -9.0
    out = hpd_slogdet(H.to(HPU)).logabsdet.cpu()
    assert torch.isfinite(out[0]) and out[1].isnan()


def test_hpd_slogdet_off_the_card_is_torchs():
    from hpu_complex.linalg import hpd_slogdet

    H = hpd(2, 4, 4, seed=1)
    sign, logabsdet = hpd_slogdet(H)
    ref = torch.linalg.slogdet(H)
    torch.testing.assert_close(logabsdet, ref.logabsdet)
    torch.testing.assert_close(sign, ref.sign)


# ---- eigvalsh -------------------------------------------------------------------------------------------------


def hermitian(b, n, seed, kind="random"):
    g = torch.Generator().manual_seed(seed)
    if kind == "random":
        A = torch.randn(b, n, n, dtype=C128, generator=g)
        return A + A.mH
    Q, _ = torch.linalg.qr(torch.randn(b, n, n, dtype=C128, generator=g))
    d = torch.ones(n, dtype=torch.float64)
    d[n // 2:] = 3.0
    return Q @ torch.diag_embed(d.to(C128)) @ Q.mH


@pytest.mark.parametrize("n", [1, 2, 5, 8])
@pytest.mark.parametrize("UPLO", ["L", "U"])
@pytest.mark.parametrize("kind", ["random", "cluster"])
def test_eigvalsh_values_and_gradient(card_linalg, n, UPLO, kind):
    H = hermitian(3, n, seed=n, kind=kind)
    weights = torch.linspace(1, 2, n, dtype=torch.float64)

    def f(a):
        return torch.linalg.eigvalsh(a, UPLO=UPLO)

    check(f, [H], "eigvalsh", card_linalg, tol=3e-6, grad=False)
    if kind == "random":  # a gradient through distinct eigenvalues only (a cluster's split is arbitrary)
        a = H.to(C64).to(HPU).requires_grad_()
        (f(a) * weights.float().to(HPU)).sum().backward()
        r = H.to(C64).to(C128).requires_grad_()
        (f(r) * weights).sum().backward()
        torch.testing.assert_close(a.grad.cpu().to(C128), r.grad, atol=2e-5, rtol=1e-4)


def test_eigvalsh_reads_one_triangle(card_linalg):
    H = hermitian(2, 6, seed=1)
    junk = H.tril() + torch.triu(rand(2, 6, 6, seed=9), 1)
    junk.diagonal(0, -2, -1).imag.fill_(3.0)
    w = torch.linalg.eigvalsh(junk.to(C64).to(HPU))
    torch.testing.assert_close(w.cpu().double(), torch.linalg.eigvalsh(H), atol=3e-5, rtol=0)
    w = torch.linalg.eigvalsh(junk.mH.to(C64).to(HPU), UPLO="U")
    torch.testing.assert_close(w.cpu().double(), torch.linalg.eigvalsh(H), atol=3e-5, rtol=0)


# ---- qr -------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(1, 1), (4, 4), (6, 3), (2, 7, 5), (3, 1, 5, 5)])
def test_qr_is_a_qr(card_linalg, shape):
    A = rand(*shape, seed=1).to(C64)
    calls = taken(card_linalg, "qr")
    Q, R = torch.linalg.qr(A.to(HPU))
    assert taken(card_linalg, "qr") > calls
    ref = torch.linalg.qr(A.to(C128))
    assert Q.shape == ref.Q.shape and R.shape == ref.R.shape and Q.dtype == R.dtype == C64
    Q, R = Q.cpu().to(C128), R.cpu().to(C128)
    n = shape[-1]
    assert (Q @ R - A.to(C128)).abs().max() < 1e-5 * A.abs().max()
    assert (Q.mH @ Q - torch.eye(n, dtype=C128)).abs().max() < 1e-5
    assert torch.equal(R, R.triu())
    d = R.diagonal(0, -2, -1)
    assert (d.imag == 0).all() and (d.real > 0).all()
    torch.testing.assert_close(d.abs(), ref.R.diagonal(0, -2, -1).abs(), atol=1e-5, rtol=1e-5)
    # torch's factors up to the real signs of diag(R): Q D, D R with D = sign(diag R_torch)
    D = torch.sgn(ref.R.diagonal(0, -2, -1).real).to(C128)
    torch.testing.assert_close(Q * D.unsqueeze(-2), ref.Q, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(R * D.unsqueeze(-1), ref.R, atol=1e-5, rtol=1e-5)


def test_qr_modes(card_linalg):
    A = rand(2, 5, 3, seed=2).to(C64)
    Q, R = torch.linalg.qr(A.to(HPU), mode="r")
    assert Q.numel() == 0 and R.shape == (2, 3, 3)
    torch.testing.assert_close(R.cpu().abs(), torch.linalg.qr(A, mode="r").R.abs(), atol=1e-5, rtol=1e-5)
    S = rand(2, 4, 4, seed=3).to(C64)
    calls = taken(card_linalg, "qr")
    Q, R = torch.linalg.qr(S.to(HPU), mode="complete")
    assert taken(card_linalg, "qr") > calls and Q.shape == (2, 4, 4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calls = taken(card_linalg, "qr")
        Q, R = torch.linalg.qr(A.to(HPU), mode="complete")  # m > n: torch's route
        assert taken(card_linalg, "qr") == calls and Q.shape == (2, 5, 5)
        Q, R = torch.linalg.qr(A.mT.to(HPU))  # m < n: torch's route
        assert taken(card_linalg, "qr") == calls and R.shape == (2, 3, 5)
    with pytest.raises(RuntimeError, match="mode='r'"):
        torch.linalg.qr(A.to(HPU).requires_grad_(), mode="r")


@pytest.mark.parametrize("shape", [(4, 4), (6, 3), (2, 5, 2)])
def test_qr_gradient_is_torchs(card_linalg, shape):
    """A loss that does not depend on the signs of diag(R) (they differ from torch's): torch's gradient."""
    A = rand(*shape, seed=4).to(C64)
    W = rand(*shape, seed=5).to(C64)

    def loss(a, w):
        Q, R = torch.linalg.qr(a)
        P = Q @ Q.mH  # the projector: sign-free
        return ((P @ w).abs() ** 2).sum() + (R.abs() ** 3).sum() + (R.mH @ R).real.sum()

    a = A.to(HPU).requires_grad_()
    loss(a, W.to(HPU)).backward()
    r = A.to(C128).requires_grad_()
    loss(r, W.to(C128)).backward()
    torch.testing.assert_close(a.grad.cpu().to(C128), r.grad, atol=5e-5, rtol=2e-4)


# ---- cholesky_inverse / cholesky_solve ------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 3, 6])
@pytest.mark.parametrize("upper", [False, True])
def test_cholesky_inverse(card_linalg, n, upper):
    L = torch.linalg.cholesky(hpd(3, n, n, seed=n), upper=upper)
    check(lambda f: torch.cholesky_inverse(f, upper=upper), [L], "cholesky_inverse", card_linalg, gtol=1e-4)
    check(lambda f: torch.cholesky_inverse(f, upper=upper), [L[0]], "cholesky_inverse", card_linalg, gtol=1e-4)


def test_cholesky_inverse_reads_one_triangle_and_batch_equal_to_order(card_linalg):
    L = torch.linalg.cholesky(hpd(8, 4, 4, seed=1))  # batch 8 = the real order
    junk = L + torch.triu(rand(8, 4, 4, seed=2), 1) + 0.3j * torch.eye(4, dtype=C128)  # diagonal's imaginary part
    got = torch.cholesky_inverse(junk.to(C64).to(HPU)).cpu().to(C128)
    torch.testing.assert_close(got, torch.cholesky_inverse(L), atol=2e-5, rtol=1e-5)


@pytest.mark.parametrize("shapes", [((2, 5, 3), (2, 5, 5)), ((3, 1, 5, 1), (2, 5, 5)), ((5, 2), (5, 5)),
                                    ((1, 1), (1, 1))])
@pytest.mark.parametrize("upper", [False, True])
def test_cholesky_solve_broadcast(card_linalg, shapes, upper):
    sb, sl = shapes
    L = torch.linalg.cholesky(hpd(*sl, seed=1), upper=upper)
    check(lambda b, f: torch.cholesky_solve(b, f, upper=upper), [rand(*sb, seed=2), L], "cholesky_solve",
          card_linalg, gtol=1e-4)


def test_cholesky_solve_double_backward(card_linalg):
    L = torch.linalg.cholesky(hpd(3, 3, seed=1)).to(C64)
    B = rand(3, 2, seed=2).to(C64)
    lh = L.to(HPU).requires_grad_()
    (g,) = torch.autograd.grad(torch.cholesky_solve(B.to(HPU), lh).abs().sum(), lh, create_graph=True)
    (gg,) = torch.autograd.grad(g.abs().sum(), lh)
    lr = L.to(C128).requires_grad_()
    (gr,) = torch.autograd.grad(torch.cholesky_solve(B.to(C128), lr).abs().sum(), lr, create_graph=True)
    (ggr,) = torch.autograd.grad(gr.abs().sum(), lr)
    assert (gg.cpu().to(C128) - ggr).abs().max() <= 1e-3 * ggr.abs().max()


# ---- solve_triangular ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("upper", [False, True])
@pytest.mark.parametrize("left", [True, False])
@pytest.mark.parametrize("unitriangular", [False, True])
@pytest.mark.parametrize("n", [1, 4])
def test_solve_triangular_flags_and_gradient(card_linalg, upper, left, unitriangular, n):
    A = rand(2, n, n, seed=1) + 3 * torch.eye(n, dtype=C128) * (1 + 1j)  # a complex diagonal; other triangle junk
    B = rand(2, n, 3, seed=2) if left else rand(2, 3, n, seed=2)
    check(lambda a, b: torch.linalg.solve_triangular(a, b, upper=upper, left=left, unitriangular=unitriangular),
          [A, B], "solve_triangular", card_linalg)


@pytest.mark.parametrize("shapes", [((3, 3), (4, 3, 2)), ((2, 1, 3, 3), (3, 3, 1)), ((3, 3), (3, 5))])
def test_solve_triangular_broadcast(card_linalg, shapes):
    sa, sb = shapes
    check(lambda a, b: torch.linalg.solve_triangular(a, b, upper=False), [tri(*sa, seed=3), rand(*sb, seed=4)],
          "solve_triangular", card_linalg)


def test_solve_triangular_imaginary_diagonal_and_zero_rows(card_linalg):
    T = tri(2, 5, 5, seed=1, upper=True)
    T.diagonal(0, -2, -1).copy_(torch.tensor([2j, -3j, 1 + 1e-3j, -1, 0.5j], dtype=C128))
    B = rand(2, 5, 3, seed=2)
    B[:, 2] = 0
    check(lambda a, b: torch.linalg.solve_triangular(a, b, upper=True), [T, B], "solve_triangular", card_linalg)


def test_solve_triangular_ill_conditioned_as_cpu_float32(card_linalg):
    """A Cholesky factor of cond 1e6: the forward error stays within 10x of CPU complex64's."""
    g = torch.Generator().manual_seed(7)
    Q, _ = torch.linalg.qr(torch.randn(16, 16, dtype=C128, generator=g))
    H = Q @ torch.diag(torch.logspace(0, -6, 16, dtype=torch.float64).to(C128)) @ Q.mH
    L = torch.linalg.cholesky(H).to(C64)
    B = torch.randn(16, 4, dtype=C64, generator=g)
    ref = torch.linalg.solve_triangular(L.to(C128), B.to(C128), upper=False)
    cpu = torch.linalg.solve_triangular(L, B, upper=False).to(C128)
    got = torch.linalg.solve_triangular(L.to(HPU), B.to(HPU), upper=False).cpu().to(C128)
    bar = ((cpu - ref).abs().max() / ref.abs().max()).item()
    assert ((got - ref).abs().max() / ref.abs().max()).item() <= 10 * bar + 1e-6


def test_solve_triangular_double_backward(card_linalg):
    T = tri(3, 3, seed=1).to(C64)
    B = rand(3, 2, seed=2).to(C64)
    t = T.to(HPU).requires_grad_()
    (g,) = torch.autograd.grad(torch.linalg.solve_triangular(t, B.to(HPU), upper=False).abs().sum(), t,
                               create_graph=True)
    (gg,) = torch.autograd.grad(g.abs().sum(), t)
    tr = T.to(C128).requires_grad_()
    (gr,) = torch.autograd.grad(torch.linalg.solve_triangular(tr, B.to(C128), upper=False).abs().sum(), tr,
                                create_graph=True)
    (ggr,) = torch.autograd.grad(gr.abs().sum(), tr)
    assert (gg.cpu().to(C128) - ggr).abs().max() <= 1e-3 * ggr.abs().max()


def test_qr_of_structured_inputs(card_linalg):
    """Inputs whose Householder columns have a zero tail (Habana's qr gives NaN there): through W E, the exact factors."""
    eye = torch.eye(4, dtype=C64).expand(2, 4, 4)
    T = tri(2, 4, 4, seed=1, upper=True).to(C64)
    T.diagonal(0, -2, -1).imag.zero_()
    for A in (eye, T, torch.diag_embed(torch.tensor([[1.0, 2j, -3.0]], dtype=C64))):
        Q, R = torch.linalg.qr(A.to(HPU))
        Q, R = Q.cpu().to(C128), R.cpu().to(C128)
        assert (Q @ R - A.to(C128)).abs().max() < 1e-5 * A.abs().max()
        assert (R.diagonal(0, -2, -1).real > 0).all()


@pytest.mark.parametrize("scale", [1e-18, 1e18])
def test_solve_triangular_extreme_diagonal_scale(card_linalg, scale):
    """The phase of each diagonal entry comes from hypot: no overflow or underflow of |d|^2."""
    T = tri(2, 4, 4, seed=3)
    T = T * scale
    B = rand(2, 4, 2, seed=4)
    x = T.to(C64)
    got = torch.linalg.solve_triangular(x.to(HPU), B.to(C64).to(HPU), upper=False).cpu().to(C128)
    ref = torch.linalg.solve_triangular(x.to(C128), B.to(C64).to(C128), upper=False)
    assert ((got - ref).abs().max() / ref.abs().max()).item() < 1e-5


def test_qr_gradient_through_q_only_and_r_only(card_linalg):
    """Only Q (sign-free through Q Q^H) or only R (through |R|) reaches the loss: torch's gradient."""
    A = rand(2, 5, 3, seed=6).to(C64)
    W = rand(2, 5, 5, seed=7).to(C64)
    for which in ("q", "r"):
        a = A.to(HPU).requires_grad_()
        r = A.to(C128).requires_grad_()
        for x, w in ((a, W.to(HPU)), (r, W.to(C128))):
            Q, R = torch.linalg.qr(x)
            val = ((Q @ Q.mH) * w).abs().sum() if which == "q" else (R.abs() ** 2).sum()
            val.backward()
        torch.testing.assert_close(a.grad.cpu().to(C128), r.grad, atol=5e-5, rtol=2e-4)


def test_eigvalsh_non_finite_input_as_the_real_kernel(card_linalg):
    """A NaN entry: torch's CPU kernel raises (both complex and the stand-in's real one, LAPACK does not converge). On
    the card hpu-ops' real Jacobi kernel returns finite values for that matrix and no error (measured,
    clinalg_adversarial2_check.py): a limit of the real kernel, the same for float32 input."""
    H = hermitian(2, 4, seed=1).to(C64)
    H[0, 2, 1] = complex(float("nan"), 0)
    with pytest.raises(torch.linalg.LinAlgError):
        torch.linalg.eigvalsh(H)
    with pytest.raises(torch.linalg.LinAlgError):
        torch.linalg.eigvalsh(H.to(HPU))


def test_eigvalsh_declined_keeps_torchs_route(card_linalg, monkeypatch):
    """A declined eigvalsh (linalg_eigvalsh is CompositeImplicitAutograd: no Autograd kernel to decline to) runs
    torch's decomposition, not this kernel again (a recursion: the card segfaulted at n = 32 before)."""
    monkeypatch.setattr(card_linalg, "FORCE_REAL_ON_CARD", False)
    monkeypatch.setattr(card_linalg, "_real_on_card", lambda kind, n2, batch=1: False)
    H = hermitian(2, 4, seed=1)
    calls = taken(card_linalg, "eigvalsh")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = torch.linalg.eigvalsh(H.to(C64).to(HPU))
        wc = torch.linalg.eigvalsh(H.to(HPU))  # complex128: the host route
    assert taken(card_linalg, "eigvalsh") == calls
    torch.testing.assert_close(w.cpu().double(), torch.linalg.eigvalsh(H), atol=3e-5, rtol=0)
    torch.testing.assert_close(wc.cpu(), torch.linalg.eigvalsh(H))
