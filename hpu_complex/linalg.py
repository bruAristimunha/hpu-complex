"""complex64 linear algebra on the card, from real ops on the float pairs (no new TPC kernel).

Without this module every complex64 linalg op on the HPU runs through the extension's CPU fallback. Here the ops
whose complex problem is exactly a real one run as that real problem on float32 HPU tensors, where Habana's kernels
or hpu-ops' TPC kernels compute it:

- the block embedding ``E(A) = [[Ar, -Ai], [Ai, Ar]]`` maps a complex n x n matrix to a real 2n x 2n one, and
  ``E(A B) = E(A) E(B)``, ``E(A^H) = E(A)^T``, ``E(A^-1) = E(A)^-1``, ``E(exp A) = exp(E(A))``,
  ``E(pinv A) = pinv(E(A))``; a complex vector ``x`` maps to ``[xr; xi]`` and ``|x| = |[xr; xi]|``. So ``solve``,
  ``inv``, ``matrix_exp``, ``pinv`` and ``lstsq`` are the real op on ``E(A)``, and the complex result is read back
  from the first block column.
- the interleaved embedding (rows and columns ordered re, im, re, im, ...) maps a triangular matrix with a real
  diagonal to a triangular real matrix: ``solve_triangular`` (after scaling the rows by the diagonal, which makes it
  unit) and ``cholesky_solve`` are real substitutions; for a Hermitian positive-definite ``H`` the real Cholesky
  factor of the interleaved ``E(H)`` is the interleaved ``E(L)`` of the complex factor (``E(L)`` is lower triangular
  with a positive diagonal and ``E(L) E(L)^T = E(H)``; the Cholesky factor is unique), so ``cholesky`` is one real
  Cholesky.
- ``householder_product`` is the compact WY form ``I - V T V^H`` with ``T^-1 = diag(1/tau) + striu(V^H V)``: complex
  products as real GEMMs and one triangular solve.

Gradients are autograd's through these real ops (the real kernels carry torch's formulas), which is the derivative
of the complex op: every map above is R-linear in the pairs, and the real op is evaluated on the structured matrix.

Each kernel sits at ``AutogradPrivateUse1`` (complex HPU tensors) and takes a call only when the real op it needs
runs on the card at that size (hpu-ops' TPC kernels, see ``_real_on_card``), the inputs are complex64 in Habana
storage (not complex128 host memory), and no tracer or functorch transform is active. Every other call goes to
torch's own kernel, hence to the extension's CPU fallback as before. ``install()`` returns an undo.
"""
import functools
import math
import os
import warnings

import torch
from torch._C import DispatchKey

aten = torch.ops.aten
C64 = torch.complex64

# Real ops whose float32 HPU kernel runs on the card, and the largest real order they take there. None: unknown
# (hpu-ops not enabled), the op is declined. Tests on the CPU stand-in set ``FORCE_REAL_ON_CARD``.
FORCE_REAL_ON_CARD = False
CALLS = {}  # op name -> calls taken on the card (card checks read it)


def _count(name):
    CALLS[name] = CALLS.get(name, 0) + 1


@functools.lru_cache(maxsize=None)
def _hpu_ops_tpc():
    try:
        import hpu_ops.tpc as tpc
    except Exception:  # noqa: BLE001  (no hpu-ops, or an old one without the kernels)
        return None
    return tpc


def _hpu_ops_has(op_name):
    """hpu-ops installed its kernel for this real op (at AutogradHPU)."""
    return torch._C._dispatch_has_kernel_for_dispatch_key(op_name, "AutogradHPU")


def _lazy():
    tpc = _hpu_ops_tpc()
    if tpc is not None:
        return tpc.lazy_mode()
    return os.environ.get("PT_HPU_LAZY_MODE", "0") != "0"


def _real_on_card(kind, n2, batch=1):
    """Whether the float32 op ``kind`` on real matrices of order ``n2`` runs on the card (not a CPU fallback)."""
    if FORCE_REAL_ON_CARD:
        return True
    if kind == "cholesky":  # Habana's own kernel (bridge 1.21.4: no fallback, measured)
        return True
    tpc = _hpu_ops_tpc()
    if tpc is None or not tpc.ready():
        return False
    if kind == "solve":  # Gauss-Jordan kernel (gj_solve; past MAXN the in-place kernel, eager only)
        if not _hpu_ops_has("aten::_linalg_solve_ex"):
            return False
        return n2 <= tpc.MAXN or (n2 <= tpc.LARGE_MAXN and tpc.large_pays(n2, batch))
    if kind == "tri":
        return _hpu_ops_has("aten::linalg_solve_triangular") and n2 <= tpc.MAXN
    if kind == "matrix_exp":
        return _hpu_ops_has("aten::linalg_matrix_exp") and n2 <= tpc.SMALL_MAXN
    if kind == "svd":
        return _hpu_ops_has("aten::_linalg_svd") and n2 <= tpc.SMALL_MAXN
    if kind == "eigh":
        return _hpu_ops_has("aten::_linalg_eigh") and n2 <= tpc.EIGH_MAXN
    return False


# ---- which calls are taken ------------------------------------------------------------------------------------------


def _tracing(tensors):
    from torch._subclasses.fake_tensor import is_fake

    if torch.compiler.is_compiling() or torch._guards.detect_fake_mode() is not None:
        return True
    if torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.PROXY) is not None:
        return True
    return any(is_fake(t) for t in tensors)


def _transformed():
    forward_ad = getattr(torch.autograd.forward_ad, "_current_level", -1) >= 0
    return torch._C._functorch.peek_interpreter_stack() is not None or forward_ad


def _host_backed(t):
    try:
        return bool(torch.ops.hpu_complex.is_host_backed(t))
    except (AttributeError, RuntimeError):
        return False


def _complex_on_card(*tensors):
    """complex64 HPU tensors in Habana storage, static non-empty shapes, one device; no tracer, no transform."""
    if _transformed() or _tracing(tensors):
        return False
    device = tensors[0].device
    for t in tensors:
        if t.dtype != C64 or t.device != device or t.device.type != "hpu" or t.numel() == 0 or _host_backed(t):
            return False
        if any(not isinstance(s, int) for s in t.shape):
            return False
    return True


def _meta(t):
    return torch.empty(t.shape, dtype=t.dtype, device="meta") if isinstance(t, torch.Tensor) else t


def _meta_call(op, *args, **kwargs):
    """``op`` on meta tensors: torch's own shape checks and errors, and the output shapes."""
    return op(*[_meta(a) for a in args], **{k: _meta(v) for k, v in kwargs.items()})


# ---- pairs and embeddings --------------------------------------------------------------------------------------------


def _pair(z):
    """(re, im) float32 views of a complex tensor (conjugate and negative bits resolved first)."""
    r = torch.view_as_real(z.resolve_conj().resolve_neg())
    return r[..., 0], r[..., 1]


def _as_complex(pairs):
    """view_as_complex of a (..., 2) float tensor, which needs even strides, also of its size-1 dims (a stack may give
    those an odd stride)."""
    if pairs.stride(-1) != 1 or any(st % 2 for st in pairs.stride()[:-1]):
        pairs = pairs.clone(memory_format=torch.contiguous_format).as_strided(
            pairs.shape, torch.empty(pairs.shape, device="meta").stride())
    return torch.view_as_complex(pairs)


def _complex(re, im):
    """A complex64 tensor from float32 parts: one stack, viewed as complex (two ops on the card)."""
    return _as_complex(torch.stack([re, im], -1))


def _embed(re, im, interleaved):
    """The real embedding of a complex (..., n, m) matrix, (..., 2n, 2m): block ``[[re, -im], [im, re]]``, or
    ``interleaved`` (rows and columns ordered re, im per index: ``E[2i + a, 2j + c]``). One stack and one copy."""
    E = torch.stack([re, -im, im, re], -1).unflatten(-1, (2, 2))  # [..., i, j, a, c]
    n, m = re.shape[-2], re.shape[-1]
    E = E.permute(*range(re.dim() - 2), -4, -2, -3, -1) if interleaved else \
        E.permute(*range(re.dim() - 2), -2, -4, -1, -3)
    return E.reshape(*re.shape[:-2], 2 * n, 2 * m)


def _block(re, im):
    """E(A) = [[re, -im], [im, re]] (..., 2n, 2m)."""
    return _embed(re, im, False)


def _block_rhs(re, im):
    """[re; im] (..., 2n, k)."""
    return torch.cat([re, im], -2)


def _from_block_rhs(X, n):
    return _complex(X[..., :n, :], X[..., n:, :])


def _interleaved(re, im):
    """E(A) with rows and columns ordered re, im per index: E[2i + a, 2j + c] (..., 2n, 2m)."""
    return _embed(re, im, True)


def _interleaved_rhs(re, im):
    """v(B) (..., 2n, k): row 2i is re B[i], row 2i + 1 is im B[i]."""
    return torch.stack([re, im], -2).reshape(*re.shape[:-2], 2 * re.shape[-2], re.shape[-1])


def _from_interleaved_rhs(X):
    """The complex (..., n, k) of v(X) (..., 2n, k): one copy, viewed as complex."""
    n2, k = X.shape[-2], X.shape[-1]
    X = X.reshape(*X.shape[:-2], n2 // 2, 2, k).transpose(-1, -2)
    return _as_complex(X.contiguous())


def _cmm(ar, ai, br, bi):
    """(ar + i ai) @ (br + i bi) as one real GEMM: [ar | ai] @ [[br, bi], [-bi, br]]. Real GEMMs on the card at
    any size (hpu_complex's complex mm sends small products to the CPU)."""
    n = br.shape[-1]
    left = torch.cat([ar, ai], -1)
    right = torch.cat([torch.cat([br, bi], -1), torch.cat([-bi, br], -1)], -2)
    out = left @ right
    return out[..., :n], out[..., n:]


def _info_from_real(info):
    """The complex info from the interleaved real Cholesky's: 0 stays 0; it fails first at an odd real index 2j - 1
    (the real pivot there is the complex one), hence j = (info + 1) // 2."""
    return torch.where(info > 0, (info + 1) // 2, info)


def _info_from_block(info, n):
    """The complex info from the real one on the block embedding, whose columns j and n + j are complex column j:
    real index k (1-based) is complex column k, or k - n past n. Exact for torch's index when the real elimination
    meets the zero pivot in the complex order (a zero column or diagonal); for other singular matrices the index
    follows the kernel's elimination order, as for float32 on the card (it is nonzero exactly when torch's is)."""
    return torch.where(info > n, info - n, info)


def _is_vector_rhs(A, B):
    return B.dim() == 1 or (B.dim() == A.dim() - 1 and B.shape == A.shape[:-1])


def _placeholder_lu(A_shape, batch, n, device):
    """LU and pivots that ``_linalg_solve_ex`` / ``_linalg_det`` return: never read (the kernels here replace the
    derivatives that read them). LU zeros, pivots the identity permutation."""
    LU = torch.zeros((*batch, n, n), dtype=C64, device=device)
    pivots = torch.arange(1, n + 1, dtype=torch.int32, device=device).expand(*batch, n).contiguous()
    return LU, pivots


# ---- solve, inv ----------------------------------------------------------------------------------------------------


def solve_ex(A, B, left=True, check_errors=False):
    """``_linalg_solve_ex`` from the real solve of ``E(A) [Xr; Xi] = [Br; Bi]``. Returns (X, LU, pivots, info)."""
    vector = _is_vector_rhs(A, B)
    if not left:  # X A = B  <=>  A^T X^T = B^T
        X, LU, pivots, info = solve_ex(A.mT, B.mT, True, check_errors)
        return X.mT, LU, pivots, info
    rhs = B.unsqueeze(-1) if vector else B
    batch = torch.broadcast_shapes(A.shape[:-2], rhs.shape[:-2])
    n = A.shape[-1]
    ar, ai = _pair(A)
    br, bi = _pair(rhs)
    X, info = torch.linalg.solve_ex(_block(ar, ai), _block_rhs(br, bi))
    X = _from_block_rhs(X, n)
    info = _info_from_block(info, n)
    if check_errors:
        torch.ops.aten._linalg_check_errors(info, "torch.linalg.solve_ex", is_matrix=A.dim() == 2)
    LU, pivots = _placeholder_lu(A.shape, batch, n, A.device)
    return (X.squeeze(-1) if vector else X), LU, pivots, info


def inv_ex(A, check_errors=False):
    n = A.shape[-1]
    ar, ai = _pair(A)
    Minv, info = torch.linalg.inv_ex(_block(ar, ai))
    info = _info_from_block(info, n)
    if check_errors:
        torch.ops.aten._linalg_check_errors(info, "linalg.inv_ex", is_matrix=A.dim() == 2)
    return _complex(Minv[..., :n, :n], Minv[..., n:, :n]), info


# ---- triangular -----------------------------------------------------------------------------------------------------


def _triangle(t, upper, diagonal=0):
    return t.triu(diagonal) if upper else t.tril(-diagonal)


def _reciprocal(re, im):
    """1 / (re + i im) in pairs, scaled by max(|re|, |im|) (no overflow or underflow of |d|^2)."""
    s = torch.maximum(re.abs(), im.abs())
    r, i = re / s, im / s
    q = s * (r * r + i * i)
    return r / q, -i / q


def solve_triangular(A, B, upper, left=True, unitriangular=False):
    """``linalg.solve_triangular``: the rows of ``T`` (and ``B``) scaled by ``1 / diag(T)`` make ``T`` unit triangular
    (its strict triangle scaled, its diagonal exactly 1); its interleaved embedding is then a real unit triangle,
    solved by one real substitution. The scaling is one pass over ``[T | B]``."""
    if not left:  # X T = B  <=>  T^T X^T = B^T
        return solve_triangular(A.mT, B.mT, not upper, True, unitriangular).mT
    n = A.shape[-1]
    ar, ai = _pair(A)
    br, bi = _pair(B)
    batch = torch.broadcast_shapes(A.shape[:-2], B.shape[:-2])
    tr, ti = _triangle(ar, upper, 1), _triangle(ai, upper, 1)
    cr = torch.cat([tr.expand(*batch, n, n), br.expand(*batch, *B.shape[-2:])], -1)
    ci = torch.cat([ti.expand(*batch, n, n), bi.expand(*batch, *B.shape[-2:])], -1)
    if not unitriangular:
        xr, xi = _reciprocal(torch.diagonal(ar, 0, -2, -1).unsqueeze(-1), torch.diagonal(ai, 0, -2, -1).unsqueeze(-1))
        cr, ci = cr * xr - ci * xi, cr * xi + ci * xr
    E = _interleaved(cr, ci)  # (..., 2n, 2(n + k)): the triangle, then v(B') in the even columns
    X = torch.linalg.solve_triangular(E[..., :2 * n], E[..., 2 * n::2].contiguous(), upper=upper, unitriangular=True)
    return _from_interleaved_rhs(X)


def _cmatmul(a, b):
    ar, ai = _pair(a)
    br, bi = _pair(b)
    return _complex(*_cmm(ar, ai, br, bi))


def _ctril(z, diagonal=0):
    re, im = _pair(z)
    return _complex(re.tril(diagonal), im.tril(diagonal))


def _cholesky_solve_forward(B, L, upper):
    if upper:
        Y = solve_triangular(L.mH, B, upper=False)
        return solve_triangular(L, Y, upper=True)
    Y = solve_triangular(L, B, upper=False)
    return solve_triangular(L.mH, Y, upper=True)


class _CholeskySolve(torch.autograd.Function):
    """torch's ``cholesky_solve_backward``: ``gB = cholesky_solve(g, L)``, ``gL = -(C L)`` (``-(L C)`` if upper) with
    ``C = gB X^H + X gB^H`` (not masked to the triangle, as torch)."""

    @staticmethod
    def forward(ctx, B, L, upper):
        X = _cholesky_solve_forward(B, L, upper)
        ctx.save_for_backward(L, X)
        ctx.upper = upper
        ctx.shapes = (B.shape, L.shape)
        return X

    @staticmethod
    def backward(ctx, g):
        L, X = ctx.saved_tensors
        gB = _CholeskySolve.apply(g, L, ctx.upper)
        gL = None
        if ctx.needs_input_grad[1]:
            common = _cmatmul(gB, X.mH)
            common = common + common.mH
            gL = -_cmatmul(L, common) if ctx.upper else -_cmatmul(common, L)
            gL = gL.sum_to_size(ctx.shapes[1])
        return gB.sum_to_size(ctx.shapes[0]), gL, None


def cholesky_solve(B, L, upper=False):
    """``cholesky_solve``: ``L L^H X = B`` (``U^H U X = B`` if ``upper``) as two triangular solves."""
    if torch.is_grad_enabled() and (B.requires_grad or L.requires_grad):
        return _CholeskySolve.apply(B, L, upper)
    return _cholesky_solve_forward(B, L, upper)


class _Cholesky(torch.autograd.Function):
    """torch's ``cholesky_backward``: ``gA = L^-H Phi(L^H gL) L^-1``, ``Phi(X) = (tril(X) + tril(X, -1)^H) / 2``. The
    two triangular solves are real ones with the interleaved factor ``E(L)`` the forward computed: ``L^H Y = Phi`` is
    ``E(L)^T v(Y) = v(Phi)``, and ``X L = Y`` is ``X~ E(L) = Y~`` on the rows ``[xr_0, -xi_0, xr_1, -xi_1, ...]``."""

    @staticmethod
    def forward(ctx, A, upper):
        L, info, Lreal = _cholesky_ex(A, upper, keep_real=True)
        ctx.save_for_backward(L, Lreal)
        ctx.upper = upper
        ctx.mark_non_differentiable(info)
        return L, info

    @staticmethod
    def backward(ctx, gL, _):
        L, E = ctx.saved_tensors
        if ctx.upper:
            L, gL = L.mH, gL.mH
        if torch.is_grad_enabled():  # create_graph: the same formula through differentiable ops on L (no saved E)
            gA = _ctril(_cmatmul(L.mH, gL))
            gA = 0.5 * (gA + _ctril(gA, -1).mH)
            gA = solve_triangular(L.mH, gA, upper=True)
            return solve_triangular(L, gA, upper=False, left=False), None
        lr, li = _pair(L)
        gr, gi = _pair(gL)
        xr, xi = _cmm(lr.mT, -li.mT, gr, gi)  # L^H gL
        xr, xi = xr.tril(), xi.tril()
        pr, pi = 0.5 * (xr + xr.tril(-1).mT), 0.5 * (xi - xi.tril(-1).mT)  # Phi
        Y = torch.linalg.solve_triangular(E.mT, _interleaved_rhs(pr, pi), upper=True)  # v(L^-H Phi)
        n = L.shape[-1]
        Y = Y.unflatten(-2, (n, 2))  # [..., i, a, j]
        Yt = torch.stack([Y[..., 0, :], -Y[..., 1, :]], -1).flatten(-2)  # rows [yr_0, -yi_0, ...]
        X = torch.linalg.solve_triangular(E, Yt, upper=False, left=False).unflatten(-1, (n, 2))
        gA = _complex(X[..., 0], -X[..., 1])
        return gA, None


def cholesky_ex(A, upper=False, check_errors=False):
    if torch.is_grad_enabled() and A.requires_grad:
        L, info = _Cholesky.apply(A, upper)
    else:
        L, info = _cholesky_ex(A, upper)
    if check_errors:
        torch.ops.aten._linalg_check_errors(info, "linalg.cholesky_ex", is_matrix=A.dim() == 2)
    return L, info


def _cholesky_ex(A, upper=False, keep_real=False):
    """``linalg_cholesky_ex``: the real Cholesky factor of the interleaved ``E(H)`` is ``E(L)``."""
    ar, ai = _pair(A)
    if upper:  # the upper triangle holds H: H = U^H U, the lower triangle of H is conj(U)^T
        ar, ai = ar.mT, -ai.mT
    lr, li = ar.tril(), ai.tril(-1)
    hr = lr + lr.tril(-1).mT
    hi = li - li.mT  # Hermitian: the imaginary part is antisymmetric; diag(H) is real (its imaginary part ignored)
    E = _interleaved(hr, hi)
    n2 = E.shape[-1]
    if E.dim() >= 3 and E.shape[-3] == n2:
        # Habana's float32 cholesky returns NaN (info 0) when the dim before the matrix equals its order (bridge
        # 1.21.4, repro/habana_cholesky_nan.py); the same batch as (..., b, 1, n, n) is right.
        Lreal, info = torch.linalg.cholesky_ex(E.unsqueeze(-3))
        Lreal, info = Lreal.squeeze(-3), info.squeeze(-1)
    else:
        Lreal, info = torch.linalg.cholesky_ex(E)
    blocks = Lreal.reshape(*Lreal.shape[:-2], n2 // 2, 2, n2 // 2, 2)
    re, im = blocks[..., 0, :, 0], blocks[..., 1, :, 0].tril(-1)  # the diagonal is real (its rounding residue: 0)
    info = _info_from_real(info)
    L = _complex(re.mT, -im.mT) if upper else _complex(re, im)
    return (L, info, Lreal) if keep_real else (L, info)


# ---- matrix_exp ------------------------------------------------------------------------------------------------------

_TAYLOR = [1.0 / math.factorial(k) for k in range(13)]


def _expm_real_composite(M):
    """exp(M) for real M (..., n, n): scaled by 2^-s so that the 1-norm is at most 1, a degree-12 Taylor polynomial
    (remainder below 1.7e-10, under float32's rounding) by Paterson-Stockmeyer (5 GEMMs; the three degree-3 blocks in
    one contraction of [I, M, M^2, M^3]), then s squarings, each matrix squared only while k < its own s. One host
    read: the largest s of the batch."""
    norm = M.abs().sum(-2).amax(-1)
    s = torch.nan_to_num(torch.ceil(torch.log2(norm)).clamp(0, 130), nan=0.0, posinf=0.0)
    s_max = int(s.max().item()) if s.numel() else 0
    Ms = M * torch.exp2(-s)[..., None, None]
    n = M.shape[-1]
    eye = torch.eye(n, dtype=M.dtype, device=M.device).expand_as(Ms)
    M2 = Ms @ Ms
    M3 = M2 @ Ms
    M4 = M2 @ M2
    coeffs = torch.tensor([_TAYLOR[0:4], _TAYLOR[4:8], _TAYLOR[8:12]], dtype=M.dtype, device=M.device)
    B = torch.einsum("jp,p...->j...", coeffs, torch.stack([eye, Ms, M2, M3]))  # B[j] = sum_p c[4j + p] M^p
    Y = torch.add(B[2], M4, alpha=_TAYLOR[12])
    Y = torch.baddbmm(B[1].reshape(-1, n, n), Y.reshape(-1, n, n), M4.reshape(-1, n, n)).reshape(M.shape)
    Y = torch.baddbmm(B[0].reshape(-1, n, n), Y.reshape(-1, n, n), M4.reshape(-1, n, n)).reshape(M.shape)
    for k in range(s_max):
        Y = torch.where((s > k)[..., None, None], Y @ Y, Y)
    return Y


def matrix_exp(A):
    n = A.shape[-1]
    ar, ai = _pair(A)
    M = _block(ar, ai)
    if _real_on_card("matrix_exp", 2 * n):
        Y = torch.linalg.matrix_exp(M)
    else:
        Y = _expm_real_composite(M)
    return _complex(Y[..., :n, :n], Y[..., n:, :n])


# ---- householder_product -----------------------------------------------------------------------------------------------


def householder_product(A, tau):
    """``H_1 ... H_k`` (first n columns), ``H_i = I - tau_i v_i v_i^H``, as ``I - V T V^H`` (compact WY). With
    ``D = diag(tau)`` and ``S = striu(V^H V)``, ``T^-1 = D^-1 + S``, so ``T = (I + D S)^-1 D``: one unit triangular
    solve, no division (a zero ``tau_i``, LAPACK's identity reflector, needs no special case, and its derivative is
    torch's)."""
    m, n = A.shape[-2], A.shape[-1]
    k = tau.shape[-1]
    ar, ai = _pair(A[..., :, :k])
    vr = ar.tril(-1) + torch.eye(m, k, dtype=ar.dtype, device=ar.device)
    vi = ai.tril(-1)
    gr, gi = _cmm(vr.mT, -vi.mT, vr, vi)  # V^H V (k x k)
    tr, ti = _pair(tau)
    tr, ti = tr.unsqueeze(-1), ti.unsqueeze(-1)  # D S: row i scaled by tau_i
    sr, si = gr.triu(1), gi.triu(1)
    U = _complex(tr * sr - ti * si, tr * si + ti * sr)
    T = solve_triangular(U, torch.diag_embed(tau), upper=True, unitriangular=True)
    t_r, t_i = _pair(T)
    wr, wi = _cmm(t_r, t_i, vr[..., :n, :].mT, -vi[..., :n, :].mT)  # T V[:n]^H (k x n)
    qr, qi = _cmm(vr, vi, wr, wi)  # V T V[:n]^H (m x n)
    return _complex(torch.eye(m, n, dtype=ar.dtype, device=ar.device) - qr, -qi)


# ---- eigh -------------------------------------------------------------------------------------------------------------

_GOLDEN = math.pi * (3 - math.sqrt(5))


def _hermitian_pairs(A, upper):
    """(re, im) of the Hermitian matrix held in one triangle of ``A`` (the imaginary part of the diagonal ignored)."""
    ar, ai = _pair(A)
    if upper:
        ar, ai = ar.mT, -ai.mT
    lr, li = ar.tril(), ai.tril(-1)
    return lr + lr.tril(-1).mT, li - li.mT


def _orthonormalize(wr, wi, passes=2):
    """Cholesky QR, ``passes`` times: ``W <- W L^-H`` with ``L L^H = W^H W``. Columns that are already orthogonal
    stay (up to their norm); the others are Gram-Schmidt'ed in order."""
    for _ in range(passes):
        gr, gi = _cmm(wr.mT, -wi.mT, wr, wi)
        L, _info = _cholesky_ex(_complex(gr, gi))
        W = solve_triangular(L.mH, _complex(wr, wi), upper=True, left=False)
        wr, wi = _pair(W)
    return wr, wi


def _eigh_forward(A, upper, compute_v):
    """``E(H) = [[Hr, -Hi], [Hi, Hr]]`` has every eigenvalue of ``H`` twice: for ``H v = l v``, ``v = x + i y``, both
    ``[x; y]`` and ``[-y; x]`` are eigenvectors, and any real eigenvector ``[x; y]`` of ``E(H)`` gives the complex
    eigenvector ``x + i y``. The sorted real eigenvalues pair up (the complex one is the mean of a pair). From pair k
    (real eigenvectors ``u_2k``, ``u_2k+1``, complex images ``a_k``, ``b_k``), ``w_k = a_k + omega_k b_k`` with
    ``|omega_k| = 1/2`` is never zero (``b_k`` is ``a_k`` times a phase when the eigenvalue is simple). Inside a cluster
    of equal eigenvalues the real solver returns any basis of the doubled space, so the ``a_k`` alone can repeat a
    vector (``u`` and ``[-y; x]`` from two pairs: ``H = I``); distinct ``omega_k`` (golden-angle phases) make the
    ``w_k`` independent, and two Cholesky QR passes orthonormalize them (Gram-Schmidt inside clusters, nothing
    across them, where the ``w_k`` are orthogonal already)."""
    n = A.shape[-1]
    hr, hi = _hermitian_pairs(A, upper)
    w, U = torch.linalg.eigh(_block(hr, hi))
    L = 0.5 * (w[..., 0::2] + w[..., 1::2])
    if not compute_v:
        return L, None
    ar, ai = U[..., :n, 0::2], U[..., n:, 0::2]
    br, bi = U[..., :n, 1::2], U[..., n:, 1::2]
    phase = torch.arange(n, dtype=w.dtype, device=w.device) * _GOLDEN
    om_r, om_i = 0.5 * torch.cos(phase), 0.5 * torch.sin(phase)
    wr = ar + om_r * br - om_i * bi
    wi = ai + om_r * bi + om_i * br
    wr, wi = _orthonormalize(wr, wi)
    return L, _complex(wr, wi)


class _Eigh(torch.autograd.Function):
    """torch's ``linalg_eig_backward`` (``is_hermitian``): ``gA = V (diag(gL) + skew(V^H gV) / E) V^H`` with
    ``skew(X) = (X - X^H) / 2`` and ``E_ij = L_j - L_i`` (1 on the diagonal). torch's check that the loss does not
    depend on the eigenvectors' phases (a host read of ``imag(diag(V^H gV))``) is not made."""

    @staticmethod
    def forward(ctx, A, upper):
        L, V = _eigh_forward(A, upper, True)
        ctx.save_for_backward(L, V)
        return L, V

    @staticmethod
    def backward(ctx, gL, gV):
        L, V = ctx.saved_tensors
        vr, vi = _pair(V)
        if gV is not None:
            gr, gi = _pair(gV)
            xr, xi = _cmm(vr.mT, -vi.mT, gr, gi)  # V^H gV
            xr, xi = 0.5 * (xr - xr.mT), 0.5 * (xi + xi.mT)
            E = L.unsqueeze(-2) - L.unsqueeze(-1)
            E = E + torch.eye(L.shape[-1], dtype=L.dtype, device=L.device)
            xr, xi = xr / E, xi / E
            if gL is not None:
                eye = torch.eye(L.shape[-1], dtype=L.dtype, device=L.device)
                xr = xr * (1 - eye) + torch.diag_embed(gL)
                xi = xi * (1 - eye)
        else:
            xr, xi = torch.diag_embed(gL), torch.zeros_like(vr)
        tr, ti = _cmm(xr, xi, vr.mT, -vi.mT)  # (...) V^H
        gr, gi = _cmm(vr, vi, tr, ti)
        return _complex(gr, gi), None


def eigh(A, UPLO="L", compute_v=True):
    """``_linalg_eigh`` -> (eigenvalues, eigenvectors); without ``compute_v`` the eigenvectors are empty."""
    upper = UPLO.upper() == "U"
    if not compute_v:
        L, _ = _eigh_forward(A, upper, False)
        return L, torch.empty(0, dtype=C64, device=A.device)
    if torch.is_grad_enabled() and A.requires_grad:
        return _Eigh.apply(A, upper)
    return _eigh_forward(A, upper, True)


# ---- pinv, lstsq ------------------------------------------------------------------------------------------------------


class _HermitianFromLower(torch.autograd.Function):
    """The Hermitian matrix of ``A``'s lower triangle (real diagonal), with the incoming gradient passed through
    unchanged: torch's ``pinv(hermitian=True)`` gradient is that of the general ``pinv`` at the Hermitian matrix
    (measured on the CPU, complex128), neither masked to the triangle nor made Hermitian."""

    @staticmethod
    def forward(ctx, A):
        return _complex(*_hermitian_pairs(A, False))

    @staticmethod
    def backward(ctx, g):
        return g


def _default_rtol(A):
    """torch's default tolerance, eps * max(m, n) of the complex matrix (the embedding's would be twice that)."""
    return torch.finfo(torch.float32).eps * max(A.shape[-2], A.shape[-1])


def pinv(A, atol=None, rtol=None, hermitian=False):
    """``linalg.pinv``: ``pinv(E(A)) = E(pinv A)``. The singular values of ``E(A)`` are those of ``A``, each twice, so
    ``atol`` and ``rtol`` keep their meaning; torch's default ``rtol`` (both unset) is passed explicitly, from the
    complex shape. ``hermitian``: the Hermitian matrix of the lower triangle (real diagonal), as torch's eigh reads it."""
    m, n = A.shape[-2], A.shape[-1]
    if hermitian:
        ar, ai = _pair(_HermitianFromLower.apply(A))
    else:
        ar, ai = _pair(A)
    if atol is None and rtol is None:
        rtol = _default_rtol(A)
    P = torch.linalg.pinv(_block(ar, ai), atol=atol, rtol=rtol, hermitian=hermitian)
    return _complex(P[..., :n, :m], P[..., n:, :m])


def lstsq(A, B, rcond=None, driver=None):
    """``linalg.lstsq``: ``min |E(A) x - [br; bi]|`` is ``min |A x - b|`` (the norms agree), so the real solution is
    ``[xr; xi]``, each residual is the complex one, the real rank is twice the complex one and the real singular
    values are the complex ones, each twice."""
    vector = _is_vector_rhs(A, B)
    rhs = B.unsqueeze(-1) if vector else B
    n = A.shape[-1]
    if rcond is None:  # torch's default from the complex shape (the embedding's would be twice it)
        rcond = _default_rtol(A)
    ar, ai = _pair(A)
    br, bi = _pair(rhs)
    X, residuals, rank, sv = torch.linalg.lstsq(_block(ar, ai), _block_rhs(br, bi), rcond=rcond, driver=driver)
    X = _from_block_rhs(X, n)
    if sv.numel():
        sv = sv[..., 0::2].contiguous()
        if rank.numel():  # from the complex singular values, one per pair: a pair split at the cutoff counts once
            rank = (sv > rcond * sv[..., :1]).sum(-1)
    elif rank.numel():  # gelsy: no singular values; a pair split at the cutoff counts once
        rank = (rank + 1) // 2
    return (X.squeeze(-1) if vector else X), residuals, rank, sv


# ---- registration ------------------------------------------------------------------------------------------------------


def _decline(op, args, kwargs):
    return op._op_dk(DispatchKey.Autograd, *args, **kwargs)


# Where each op is taken by default: the modes where it measured not slower than the CPU route, forward and forward +
# backward, batch 32, n = 16..32 (clinalg_card_check.py, Gaudi2, bridge 1.21.4; tables in TEAM_REPORT.md).
# HPU_COMPLEX_LINALG=all takes every op in both modes (nothing on the CPU, at that speed); =0 none.
ROUTES = {
    "solve": ("eager", "lazy"),
    "solve_left_false": ("lazy",),  # eager: forward + backward 0.79-1.0x
    "inv": ("eager", "lazy"),
    "cholesky": ("eager", "lazy"),
    "matrix_exp": ("eager", "lazy"),  # n <= 16: hpu-ops' kernel on E(A)
    "matrix_exp_composite": ("lazy",),  # n > 16: eager 0.2-0.4x, lazy 1.0-4x
    "pinv": ("eager", "lazy"),
    "householder_product": ("eager:grad", "lazy:grad"),  # forward alone 0.2-0.5x, with backward 2-28x
    "lstsq": ("eager:grad", "lazy:grad"),  # forward alone 0.7-0.9x, with backward 1.6-3.2x
    "solve_triangular": (),  # 0.3-0.96x forward in both modes
    "cholesky_solve": (),  # 0.25-0.4x forward
    "eigh": (),  # eager 0.2x, lazy 0.3x forward (n <= 16, the real kernel's limit)
}


def _needs_grad(args, kwargs):
    return torch.is_grad_enabled() and any(isinstance(t, torch.Tensor) and t.requires_grad
                                           for t in (*args, *kwargs.values()))


def _routed(name, args, kwargs):
    """Whether ``name`` is taken in this mode: ROUTES lists "eager" / "lazy", or "eager:grad" / "lazy:grad" for an op
    taken only when a gradient will be computed (its forward alone measured slower than the CPU route, forward +
    backward faster)."""
    if FORCE_REAL_ON_CARD or os.environ.get("HPU_COMPLEX_LINALG", "1") == "all":
        return True
    mode = "lazy" if _lazy() else "eager"
    routes = ROUTES.get(name, ())
    return mode in routes or (f"{mode}:grad" in routes and _needs_grad(args, kwargs))


def _kernel(op, name, takes, compute, meta=True):
    def kernel(*args, **kwargs):
        if _routed(name, args, kwargs) and takes(*args, **kwargs):
            if meta:  # torch's shape checks and errors (lstsq has no meta kernel: the real lstsq raises its own)
                _meta_call(op, *args, **kwargs)
            _count(name)
            return compute(*args, **kwargs)
        return _decline(op, args, kwargs)

    return kernel


def _square(A):
    return A.dim() >= 2 and A.shape[-1] == A.shape[-2]


def _batch(*tensors):
    return math.prod(torch.broadcast_shapes(*[t.shape[:-2] for t in tensors]))


def _takes_solve(A, B, left=True, check_errors=False):
    # Only with grad mode on: the LU and pivots returned here are placeholders, which only torch's own derivative
    # reads. With grad on, the derivative recorded is this module's (autograd through the real solve); a compiled
    # graph's forward runs with grad off and its traced backward reads the LU (NaN gradients under
    # torch.compile(backend="hpu_backend"), measured), so such calls, and torch.no_grad() ones, keep torch's route.
    if not torch.is_grad_enabled():
        return False
    if not (isinstance(B, torch.Tensor) and _complex_on_card(A, B) and _square(A)):
        return False
    if _is_vector_rhs(A, B) and not left:
        return False
    if not left and not _routed("solve_left_false", (A, B), {}):
        return False
    rhs = B.unsqueeze(-1) if _is_vector_rhs(A, B) else B
    if rhs.dim() < 2 or rhs.shape[-2 if left else -1] != A.shape[-1]:
        return False
    return _real_on_card("solve", 2 * A.shape[-1], _batch(A, rhs))


def _takes_inv(A, check_errors=False):
    return _complex_on_card(A) and _square(A) and _real_on_card("solve", 2 * A.shape[-1], _batch(A))


def _takes_tri(A, B, *, upper, left=True, unitriangular=False):
    if not (isinstance(B, torch.Tensor) and _complex_on_card(A, B) and _square(A) and B.dim() >= 2):
        return False
    if B.shape[-2 if left else -1] != A.shape[-1]:
        return False
    return _real_on_card("tri", 2 * A.shape[-1])


def _takes_cholesky_solve(B, L, upper=False):
    if not (isinstance(L, torch.Tensor) and _complex_on_card(B, L) and _square(L) and B.dim() >= 2):
        return False
    return B.shape[-2] == L.shape[-1] and _real_on_card("tri", 2 * L.shape[-1])


def _takes_cholesky(A, upper=False, check_errors=False):
    return _complex_on_card(A) and _square(A) and _real_on_card("cholesky", 2 * A.shape[-1])


def _takes_matrix_exp(A):
    if not (_complex_on_card(A) and _square(A)):
        return False
    return _real_on_card("matrix_exp", 2 * A.shape[-1]) or _routed("matrix_exp_composite", (A,), {})


def _takes_householder(A, tau):
    if not (isinstance(tau, torch.Tensor) and _complex_on_card(A, tau) and A.dim() >= 2):
        return False
    m, n, k = A.shape[-2], A.shape[-1], tau.shape[-1]
    if not (m >= n >= k >= 1) or tau.shape[:-1] != A.shape[:-2]:
        return False
    return _real_on_card("tri", 2 * k)


def _takes_pinv(A, atol=None, rtol=None, hermitian=False):
    tols = [t for t in (atol, rtol) if isinstance(t, torch.Tensor)]
    if not (_complex_on_card(A) and A.dim() >= 2) or any(t.device != A.device or t.is_complex() for t in tols):
        return False
    return _real_on_card("svd", 2 * max(A.shape[-2:]))


def _takes_lstsq(A, B, rcond=None, *, driver=None):
    if not (isinstance(B, torch.Tensor) and _complex_on_card(A, B) and A.dim() >= 2 and B.dim() >= 1):
        return False
    if not _is_vector_rhs(A, B) and B.dim() < A.dim():
        # torch's lstsq treats such a B (e.g. A (1, 3, 2), B (3, 4) -> solution (3, 2, 4)) by its own broadcast
        # rule: torch's route
        return False
    if B.shape[-1 if _is_vector_rhs(A, B) else -2] != A.shape[-2]:
        return False
    return _real_on_card("svd", 2 * max(A.shape[-2:]))


def _takes_eigh(A, UPLO="L", compute_v=True):
    return _complex_on_card(A) and _square(A) and _real_on_card("eigh", 2 * A.shape[-1])


def _entries():
    return [
        (aten._linalg_solve_ex.default, "solve", _takes_solve, solve_ex),
        (aten.linalg_inv_ex.default, "inv", _takes_inv, inv_ex),
        (aten.linalg_solve_triangular.default, "solve_triangular", _takes_tri,
         lambda A, B, *, upper, left=True, unitriangular=False: solve_triangular(A, B, upper, left, unitriangular)),
        (aten.cholesky_solve.default, "cholesky_solve", _takes_cholesky_solve, cholesky_solve),
        (aten.linalg_cholesky_ex.default, "cholesky", _takes_cholesky, cholesky_ex),
        (aten.linalg_matrix_exp.default, "matrix_exp", _takes_matrix_exp, matrix_exp),
        (aten.linalg_householder_product.default, "householder_product", _takes_householder, householder_product),
        (aten.linalg_pinv.atol_rtol_tensor, "pinv", _takes_pinv, pinv),
        (aten._linalg_eigh.default, "eigh", _takes_eigh, eigh),
        (aten.linalg_lstsq.default, "lstsq", _takes_lstsq,
         lambda A, B, rcond=None, *, driver=None: lstsq(A, B, rcond, driver), False),
    ]


_UNDO = None


def install(key="AutogradPrivateUse1"):
    """Register the kernels at ``key`` (complex HPU tensors). Idempotent; ``HPU_COMPLEX_LINALG=0`` keeps them off.
    Returns an undo (also ``uninstall()``)."""
    global _UNDO
    if _UNDO is not None or os.environ.get("HPU_COMPLEX_LINALG", "1") == "0":
        return uninstall
    library = torch.library.Library("aten", "IMPL")
    ops = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for op, name, takes, compute, *meta in _entries():
            library.impl(op, _kernel(op, name, takes, compute, *meta), key)
            ops.append(op)
    for op in ops:
        op._dispatch_cache.clear()

    def undo():
        library._destroy()
        for op in ops:
            op._dispatch_cache.clear()

    _UNDO = undo
    return uninstall


def uninstall():
    global _UNDO
    if _UNDO is not None:
        _UNDO()
        _UNDO = None


__all__ = ["install", "uninstall", "CALLS", "solve_ex", "inv_ex", "solve_triangular", "cholesky_solve", "cholesky_ex",
           "matrix_exp", "householder_product"]
