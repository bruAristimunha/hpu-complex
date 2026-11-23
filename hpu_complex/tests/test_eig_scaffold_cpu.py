"""Fake-backend CPU tests for the pre-staged ``complex_linalg.eig`` / ``eigvals`` wiring and Wirtinger autograd.

python -m pytest hpu_complex/tests/test_eig_scaffold_cpu.py -q
"""
import warnings

import pytest
import torch

from hpu_complex.tests.test_complex128_cpu import HPU, extension, tagged  # noqa: F401

C64, C128 = torch.complex64, torch.complex128


class _EigReplica:
    COMPLEX_MAXN = 32

    @staticmethod
    def _pairs(t, dev):
        return torch.view_as_real(t.resolve_conj()).to(dev).contiguous()

    def eig(self, A, layout="pairs"):
        L, V = torch.linalg.eig(A.detach().cpu())
        return self._pairs(L, A.device), self._pairs(V, A.device)

    def eigvals(self, A, layout="pairs"):
        L = torch.linalg.eigvals(A.detach().cpu())
        return self._pairs(L, A.device)

    def _d_s(self, A, batch):
        # (D, S) of cgj_*: D is discarded by every caller, S[..., 2] == 0 reads as info 0 (regular).
        n = A.shape[-1]
        return (A.new_zeros((*batch, 3, n), dtype=torch.float32),
                A.new_zeros((*batch, 4), dtype=torch.float32))

    def cgj_solve(self, A, B, layout="pairs"):
        X = torch.linalg.solve(A.detach().cpu(), B.detach().cpu())
        return self._pairs(X, A.device), *self._d_s(A, X.shape[:-2])

    def cgj_inv(self, A, layout="pairs"):
        X = torch.linalg.inv(A.detach().cpu())
        return self._pairs(X, A.device), *self._d_s(A, X.shape[:-2])


@pytest.fixture
def eig_linalg(extension, monkeypatch):  # noqa: F811
    import hpu_complex.linalg as L

    monkeypatch.setattr(L, "_CL", _EigReplica())
    L.FORCE_REAL_ON_CARD = True
    L.install()
    try:
        yield L
    finally:
        L.uninstall()
        L.FORCE_REAL_ON_CARD = False


def _rand(*shape, seed=0, dtype=C64):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, dtype=dtype, generator=g)


def _sort_eigs(L):
    key = L.real * 1e6 + L.imag
    idx = key.argsort(dim=-1)
    return L.gather(-1, idx)


@pytest.mark.parametrize("shape", [(1, 1), (2, 2), (4, 4), (3, 5, 5), (2, 3, 8, 8)])
def test_eig_forward_residual_norms_and_multiset(eig_linalg, shape):
    A_c64 = _rand(*shape, seed=shape[-1] + len(shape))
    A_hpu = A_c64.to(HPU)
    before_fb = torch.ops.hpu_complex.cpu_fallback_calls()
    before_calls = eig_linalg.CALLS.get("eig", 0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        L, V = torch.linalg.eig(A_hpu)
    assert eig_linalg.CALLS.get("eig", 0) > before_calls
    assert torch.ops.hpu_complex.cpu_fallback_calls() == before_fb
    assert not [w for w in caught if "runs on the CPU" in str(w.message)]
    assert tagged(L) and tagged(V) and L.dtype == C64 and V.dtype == C64

    L_cpu, V_cpu = L.cpu().to(C128), V.cpu().to(C128)
    A_ref = A_c64.to(C128)
    resid = (A_ref @ V_cpu - V_cpu * L_cpu.unsqueeze(-2)).abs().max().item()
    scale = A_ref.abs().max().clamp_min(1.0).item()
    assert resid <= 5e-5 * scale
    col_norms = V_cpu.norm(dim=-2)
    torch.testing.assert_close(col_norms, torch.ones_like(col_norms), rtol=1e-5, atol=1e-5)

    ref_eigs = torch.linalg.eigvals(A_ref)
    torch.testing.assert_close(_sort_eigs(L_cpu), _sort_eigs(ref_eigs), rtol=5e-5, atol=5e-5)


def test_eig_strided_and_conjugate_inputs(eig_linalg):
    base = _rand(2, 7, 7, seed=11)
    for A_c64, make in ((base, lambda t: t.mH), (base, lambda t: t[:, 1:6, 2:])):
        L, V = torch.linalg.eig(make(A_c64.to(HPU)))
        A_ref = make(A_c64).to(C128)
        L_cpu, V_cpu = L.cpu().to(C128), V.cpu().to(C128)
        assert (A_ref @ V_cpu - V_cpu * L_cpu.unsqueeze(-2)).abs().max().item() <= 5e-5 * A_ref.abs().max().item()


def test_eigvals_forward(eig_linalg):
    A_c64 = _rand(3, 5, 5, seed=13)
    before_fb = torch.ops.hpu_complex.cpu_fallback_calls()
    before_calls = eig_linalg.CALLS.get("eigvals", 0)
    L = torch.linalg.eigvals(A_c64.to(HPU))
    assert eig_linalg.CALLS.get("eigvals", 0) > before_calls
    assert torch.ops.hpu_complex.cpu_fallback_calls() == before_fb
    assert tagged(L) and L.dtype == C64
    torch.testing.assert_close(_sort_eigs(L.cpu().to(C128)), _sort_eigs(torch.linalg.eigvals(A_c64.to(C128))),
                               rtol=5e-5, atol=5e-5)


@pytest.mark.parametrize("mode", ["both", "L_only", "V_only"])
def test_eig_wirtinger_backward(eig_linalg, mode):
    A_c64 = _rand(2, 4, 4, seed=21) + 2.0 * torch.eye(4, dtype=C64)
    A_ref = A_c64.to(C128).requires_grad_()
    A_hpu = A_c64.to(HPU).requires_grad_()

    def loss_fn(L, V):
        loss_L = (L.abs() ** 2).sum()
        loss_V = (V @ V.mH).abs().sum() + (V.abs() ** 4).sum()
        if mode == "both":
            return loss_L + loss_V
        if mode == "L_only":
            return loss_L
        return loss_V

    loss_fn(*torch.linalg.eig(A_ref)).backward()
    before_fb = torch.ops.hpu_complex.cpu_fallback_calls()
    loss_fn(*torch.linalg.eig(A_hpu)).backward()
    assert torch.ops.hpu_complex.cpu_fallback_calls() == before_fb
    scale = A_ref.grad.abs().max().clamp_min(1.0).item()
    assert (A_hpu.grad.cpu().to(C128) - A_ref.grad).abs().max().item() <= 2e-4 * scale


def test_eigvals_backward_including_repeated_eigenvalues(eig_linalg):
    A_c64 = _rand(3, 5, 5, seed=31)
    A_ref = A_c64.to(C128).requires_grad_()
    A_hpu = A_c64.to(HPU).requires_grad_()

    def eigvals_loss(t):
        ev = torch.linalg.eigvals(t)
        return (ev.abs() ** 2).sum() + 0.5 * ev.real.sum()

    eigvals_loss(A_ref).backward()
    eigvals_loss(A_hpu).backward()
    scale = A_ref.grad.abs().max().clamp_min(1.0).item()
    assert (A_hpu.grad.cpu().to(C128) - A_ref.grad).abs().max().item() <= 2e-4 * scale

    # Repeated eigenvalues: eigvals backward uses the gV is None shortcut so it avoids 0/0 in E_ij.
    Q = _rand(4, 4, seed=33) + 3.0 * torch.eye(4, dtype=C64)
    D = torch.diag(torch.tensor([1 + 0j, 1 + 0j, 2 + 1j, 3 - 1j], dtype=C64))
    rep_c64 = (Q @ D @ torch.linalg.inv(Q)).contiguous()
    rep_ref = rep_c64.to(C128).requires_grad_()
    rep_hpu = rep_c64.to(HPU).requires_grad_()
    eigvals_loss(rep_ref).backward()
    eigvals_loss(rep_hpu).backward()
    assert torch.isfinite(rep_hpu.grad.cpu()).all()
    scale_rep = rep_ref.grad.abs().max().clamp_min(1.0).item()
    assert (rep_hpu.grad.cpu().to(C128) - rep_ref.grad).abs().max().item() <= 2e-4 * scale_rep


def test_eig_autograd_gradcheck_complex128(monkeypatch):
    import hpu_complex.linalg as L

    monkeypatch.setattr(L, "_CL", _EigReplica())
    A = (_rand(3, 3, seed=41, dtype=C128) + 2.5 * torch.eye(3, dtype=C128)).requires_grad_()

    def gauge_invariant(x):
        lam, V = L._Eig.apply(x, True)
        return (lam.abs() ** 2).sum() + (V @ torch.diag_embed(lam) @ V.mH).abs().sum() + (V.abs() ** 4).sum()

    assert torch.autograd.gradcheck(gauge_invariant, (A,))
    assert torch.autograd.gradcheck(lambda x: (L._Eig.apply(x, False)[0].abs() ** 2).sum(), (A,))


def test_eig_and_eigvals_graceful_decline(eig_linalg, monkeypatch):
    A = _rand(4, 4, seed=51).to(HPU)

    def check_declines(fn, op_name):
        c0 = eig_linalg.CALLS.get(op_name, 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = fn()
        assert eig_linalg.CALLS.get(op_name, 0) == c0
        return out

    # Missing _CL or missing cl.eig
    with monkeypatch.context() as m:
        m.setattr(eig_linalg, "_CL", None)
        check_declines(lambda: torch.linalg.eig(A), "eig")
        check_declines(lambda: torch.linalg.eigvals(A), "eigvals")

    with monkeypatch.context() as m:
        m.setattr(eig_linalg._CL, "eig", None, raising=False)
        check_declines(lambda: torch.linalg.eig(A), "eig")
        # Non-grad eigvals still runs if cl.eigvals is present, while grad eigvals declines cleanly without cl.eig.
        A_grad = A.detach().requires_grad_()
        check_declines(lambda: torch.linalg.eigvals(A_grad), "eigvals")

    # n > COMPLEX_MAXN (33), empty (0, 3, 3), and complex128
    big = _rand(33, 33, seed=52).to(HPU)
    empty = torch.empty(0, 3, 3, dtype=C64, device=HPU)
    c128 = _rand(4, 4, seed=53, dtype=C128).to(HPU)
    for tensor in (big, empty, c128):
        check_declines(lambda t=tensor: torch.linalg.eig(t), "eig")
        check_declines(lambda t=tensor: torch.linalg.eigvals(t), "eigvals")
