"""Fake-backend CPU tests for M6D-complex: TPC qr wiring (default on) and eig/eigvals past n=32 (ceig_big cap).

python -m pytest hpu_complex/tests/test_m6d_qr_eig_cpu.py -q
"""
import warnings

import pytest
import torch

from hpu_complex.tests.test_complex128_cpu import HPU, extension, tagged  # noqa: F401

C64, C128 = torch.complex64, torch.complex128


class _M6dReplica:
    COMPLEX_MAXN = 32
    CEIG_MAXN = 64

    @staticmethod
    def _pairs(t, dev):
        return torch.view_as_real(t.resolve_conj()).to(dev).contiguous()

    def eig(self, A, layout="pairs"):
        L, V = torch.linalg.eig(A.detach().cpu())
        return self._pairs(L, A.device), self._pairs(V, A.device)

    def eigvals(self, A, layout="pairs"):
        return self._pairs(torch.linalg.eigvals(A.detach().cpu()), A.device)

    def qr(self, A, mode="reduced", layout="pairs"):
        Q, R = torch.linalg.qr(A.detach().cpu(), mode)
        if mode == "r":
            return A.new_zeros(0, dtype=torch.float32), self._pairs(R, A.device)
        return self._pairs(Q, A.device), self._pairs(R, A.device)

    def _d_s(self, A, batch):
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
def m6d_linalg(extension, monkeypatch):  # noqa: F811
    import hpu_complex.linalg as L

    monkeypatch.setattr(L, "_CL", _M6dReplica())
    L.FORCE_REAL_ON_CARD = True
    L.install()
    try:
        yield L
    finally:
        L.uninstall()
        L.FORCE_REAL_ON_CARD = False


def _rand(*shape, seed=0):
    return torch.randn(*shape, dtype=C64, generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("n,takes", [(32, True), (48, True), (64, True), (65, False)])
def test_eig_takes_past_32_to_ceig_maxn(m6d_linalg, n, takes):
    A = _rand(n, n).to(HPU)
    assert m6d_linalg._takes_eig(A) is takes
    assert m6d_linalg._takes_eigvals(A) is takes


def test_eig_48_residual(m6d_linalg):
    A_c = _rand(48, 48, seed=3)
    A_h = A_c.to(HPU)
    before = m6d_linalg.CALLS.get("eig", 0)
    L, V = torch.linalg.eig(A_h)
    assert m6d_linalg.CALLS.get("eig", 0) > before
    assert tagged(L) and tagged(V)
    resid = (A_c.to(C128) @ V.cpu().to(C128) - V.cpu().to(C128) * L.cpu().to(C128).unsqueeze(-2)).abs().max().item()
    assert resid <= 5e-4 * A_c.abs().max().item()


@pytest.mark.parametrize("mode", ["reduced", "complete", "r"])
@pytest.mark.parametrize("shape", [(6, 4), (8, 8), (2, 32, 16)])
def test_qr_tpc_route(m6d_linalg, mode, shape):
    A_c = _rand(*shape, seed=shape[-1])
    A_h = A_c.to(HPU)
    before = m6d_linalg.CALLS.get("qr", 0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Q, R = torch.linalg.qr(A_h, mode)
    assert m6d_linalg.CALLS.get("qr", 0) > before
    assert not [w for w in caught if "runs on the CPU" in str(w.message)]
    Qr, Rr = torch.linalg.qr(A_c, mode)
    assert R.shape == Rr.shape and Q.shape == Qr.shape
    if mode != "r":
        resid = (Q.cpu().to(C128) @ R.cpu().to(C128) - A_c.to(C128)).abs().max().item()
        assert resid <= 5e-5 * A_c.abs().max().clamp_min(1.0).item()
        k = Q.shape[-1]
        eye = torch.eye(k, dtype=C128)
        orth = (Q.cpu().to(C128).mH @ Q.cpu().to(C128) - eye).abs().max().item()
        assert orth <= 5e-5
        assert tagged(Q)


def test_qr_rank_deficient_finite(m6d_linalg):
    A_c = _rand(8, 6, seed=1)
    A_c[:, 3] = 0
    Q, R = torch.linalg.qr(A_c.to(HPU))
    assert torch.isfinite(Q.cpu()).all() and torch.isfinite(R.cpu()).all()


def test_qr_grad(m6d_linalg):
    A_c = _rand(6, 4, seed=2)
    A_h = A_c.to(HPU).detach().requires_grad_()
    Q, R = torch.linalg.qr(A_h)
    (Q.abs().sum() + R.abs().sum()).backward()
    A_r = A_c.clone().requires_grad_()
    Qr, Rr = torch.linalg.qr(A_r)
    (Qr.abs().sum() + Rr.abs().sum()).backward()
    torch.testing.assert_close(A_h.grad.cpu(), A_r.grad, rtol=1e-4, atol=1e-5)


def test_pinv_lstsq_takes_past_32(m6d_linalg):
    A48 = _rand(48, 40).to(HPU)
    B48 = _rand(48, 3).to(HPU)
    assert m6d_linalg._takes_pinv(A48)
    assert m6d_linalg._takes_lstsq(A48, B48)
    # The real SVD route itself is hpu-ops' (unavailable on the CPU stand-in): only the gate is checked here;
    # values are covered on the card (clinalg_m6d_card_check.py).
