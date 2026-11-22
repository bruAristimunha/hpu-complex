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
  diagonal to a triangular real matrix: ``solve_triangular`` (after rotating each row by the phase of its diagonal
  entry) is one real substitution (hpu-ops' TPC kernel); for a Hermitian positive-definite ``H`` the real Cholesky
  factor of the interleaved ``E(H)`` is the interleaved ``E(L)`` of the complex factor (``E(L)`` is lower triangular
  with a positive diagonal and ``E(L) E(L)^T = E(H)``; the Cholesky factor is unique), so ``cholesky`` is one real
  Cholesky (its ``info`` read from the factor's diagonal), ``cholesky_inverse`` is Habana's real one on ``E(L)``
  (``E(H^-1)``), ``cholesky_solve`` that times ``v(B)``, and ``hpd_slogdet`` / ``hpd_logdet`` (explicit functions for
  HPD input) the log of the factor's diagonal. ``qr`` is Habana's real QR of the interleaved ``E(A)`` re-signed to a
  positive diagonal (``E(Q) E(R)`` by uniqueness), and ``eigvalsh`` the means of the pairs of the embedding's sorted
  eigenvalues.
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
_CL = None  # test hook: stand-in for hpu_ops.tpc.complex_linalg on CPU tests
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


def _complex_linalg():
    if _CL is not None:
        return _CL
    tpc = _hpu_ops_tpc()
    if tpc is not None and tpc.ready():
        return getattr(tpc, "complex_linalg", None)
    return None


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
    if kind in ("cholesky", "cholesky_inverse", "qr"):  # Habana's own kernels (bridge 1.21.4: no fallback, measured)
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


def _embed(re, im, interleaved, conj=False):
    """The real embedding of a complex (..., n, m) matrix, (..., 2n, 2m): block ``[[re, -im], [im, re]]``, or
    ``interleaved`` (rows and columns ordered re, im per index: ``E[2i + a, 2j + c]``). One stack and one copy.
    ``conj``: the embedding of ``re - i im`` (no negation op of its own)."""
    E = torch.stack([re, im, -im, re] if conj else [re, -im, im, re], -1).unflatten(-1, (2, 2))  # [..., i, j, a, c]
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


# ---- solve, inv, det, slogdet --------------------------------------------------------------------------------------


def _gj_info(S, batch_shape):
    """int32 info (1-based zero pivot, 0 if regular or non-finite) from complex_linalg's S (..., 4)."""
    return torch.where(S[..., 2] > 0, S[..., 2], 0.0).to(torch.int32).reshape(batch_shape)


class _SolveEx(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, rhs):
        cl = _complex_linalg()
        batch = torch.broadcast_shapes(A.shape[:-2], rhs.shape[:-2])
        A_c, rhs_c = A.contiguous(), rhs.contiguous()
        X_p, _D, S = cl.cgj_solve(A_c, rhs_c)
        X = _as_complex(X_p)
        info = _gj_info(S, batch)
        ctx.save_for_backward(A_c, X)
        ctx.shapes = (A.shape, rhs.shape)
        ctx.mark_non_differentiable(info)
        return X, info

    @staticmethod
    def backward(ctx, gX, _ginfo):
        if gX is None:
            return None, None
        A, X = ctx.saved_tensors
        gB = solve_ex(A.mH, gX, left=True, check_errors=False)[0]
        gA = -_cmatmul(gB, X.mH) if ctx.needs_input_grad[0] else None
        return (gA.sum_to_size(ctx.shapes[0]) if gA is not None else None), gB.sum_to_size(ctx.shapes[1])


def solve_ex(A, B, left=True, check_errors=False):
    """``_linalg_solve_ex`` via ``complex_linalg.cgj_solve`` (n <= 32) or real ``E(A)`` (n > 32). Returns (X, LU, pivots, info)."""
    vector = _is_vector_rhs(A, B)
    if not left:  # X A = B  <=>  A^T X^T = B^T
        X, LU, pivots, info = solve_ex(A.mT, B.mT, True, check_errors)
        return X.mT.contiguous(), LU, pivots, info
    rhs = B.unsqueeze(-1) if vector else B
    batch = torch.broadcast_shapes(A.shape[:-2], rhs.shape[:-2])
    n = A.shape[-1]
    cl = _complex_linalg()
    if cl is not None and n <= cl.COMPLEX_MAXN:
        if torch.is_grad_enabled() and (A.requires_grad or rhs.requires_grad):
            X, info = _SolveEx.apply(A, rhs)
        else:
            X_p, _D, S = cl.cgj_solve(A.contiguous(), rhs.contiguous())
            X = _as_complex(X_p)
            info = _gj_info(S, batch)
    else:
        ar, ai = _pair(A)
        br, bi = _pair(rhs)
        X, info = torch.linalg.solve_ex(_block(ar, ai), _block_rhs(br, bi))
        X = _from_block_rhs(X, n)
        info = _info_from_block(info, n)
    if check_errors:
        torch.ops.aten._linalg_check_errors(info, "torch.linalg.solve_ex", is_matrix=A.dim() == 2)
    LU, pivots = _placeholder_lu(A.shape, batch, n, A.device)
    return (X.squeeze(-1) if vector else X), LU, pivots, info


class _InvEx(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        cl = _complex_linalg()
        X_p, _D, S = cl.cgj_inv(A)
        Minv = _as_complex(X_p)
        info = _gj_info(S, A.shape[:-2])
        ctx.save_for_backward(Minv)
        ctx.mark_non_differentiable(info)
        return Minv, info

    @staticmethod
    def backward(ctx, g, _ginfo):
        if g is None:
            return None
        (Minv,) = ctx.saved_tensors
        return -_cmatmul(Minv.mH, _cmatmul(g, Minv.mH))


def inv_ex(A, check_errors=False):
    n = A.shape[-1]
    cl = _complex_linalg()
    if cl is not None and n <= cl.COMPLEX_MAXN:
        if torch.is_grad_enabled() and A.requires_grad:
            Minv, info = _InvEx.apply(A)
        else:
            X_p, _D, S = cl.cgj_inv(A)
            Minv = _as_complex(X_p)
            info = _gj_info(S, A.shape[:-2])
    else:
        ar, ai = _pair(A)
        Minv, info = torch.linalg.inv_ex(_block(ar, ai))
        Minv = _complex(Minv[..., :n, :n], Minv[..., n:, :n])
        info = _info_from_block(info, n)
    if check_errors:
        torch.ops.aten._linalg_check_errors(info, "linalg.inv_ex", is_matrix=A.dim() == 2)
    return Minv, info


def _det_svd_backward(A, g_det):
    U, S, Vh = svd(A, full_matrices=False, compute_uv=True)
    alpha = (det(U)[0] * det(Vh)[0]).conj()
    before = torch.cat([torch.ones_like(S[..., :1]), S[..., :-1].cumprod(-1)], -1)
    after = torch.cat([S[..., 1:].flip(-1).cumprod(-1).flip(-1), torch.ones_like(S[..., :1])], -1)
    D = (g_det * alpha)[..., None] * (before * after)
    return _cmatmul(U * D.unsqueeze(-2), Vh)


class _Det(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        ctx.set_materialize_grads(False)
        d = _as_complex(_complex_linalg().det(A))
        ctx.save_for_backward(A, d)
        return d

    @staticmethod
    def backward(ctx, g_det):
        if g_det is None:
            return None
        A, d = ctx.saved_tensors
        singular = d.abs() == 0
        if torch.is_grad_enabled() and singular.any():
            return _det_svd_backward(A, g_det)
        # Call inv_ex before building (g_det * d.conj()) so TPC's lazy mark_step does not split the cofactor graph.
        inv_ah = inv_ex(A, check_errors=False)[0].mH
        cofactor = (g_det * d.conj())[..., None, None] * inv_ah
        return torch.where(singular[..., None, None], 0.0, cofactor)


def det(A):
    """``_linalg_det`` via ``complex_linalg.det`` -> (result, LU, pivots)."""
    d = _Det.apply(A) if torch.is_grad_enabled() and A.requires_grad else _as_complex(_complex_linalg().det(A))
    LU, pivots = _placeholder_lu(A.shape, A.shape[:-2], A.shape[-1], A.device)
    return d, LU, pivots


class _Slogdet(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        ctx.set_materialize_grads(False)
        sign_p, lad = _complex_linalg().slogdet(A)
        sign = _as_complex(sign_p)
        ctx.save_for_backward(A, sign)
        return sign, lad

    @staticmethod
    def backward(ctx, g_sign, g_lad):
        if g_sign is None and g_lad is None:
            return None
        A, sign = ctx.saved_tensors
        inv_ah = inv_ex(A, check_errors=False)[0].mH
        re = g_lad if g_lad is not None else torch.zeros(sign.shape, dtype=torch.float32, device=sign.device)
        im = (torch.view_as_real((sign.conj() * g_sign).resolve_conj())[..., 1]
              if g_sign is not None else torch.zeros_like(re))
        scale = _complex(re, im)
        return scale[..., None, None] * inv_ah


def slogdet(A):
    """``_linalg_slogdet`` via ``complex_linalg.slogdet`` -> (sign, logabsdet, LU, pivots)."""
    if torch.is_grad_enabled() and A.requires_grad:
        sign, lad = _Slogdet.apply(A)
    else:
        sign_p, lad = _complex_linalg().slogdet(A)
        sign = _as_complex(sign_p)
    LU, pivots = _placeholder_lu(A.shape, A.shape[:-2], A.shape[-1], A.device)
    return sign, lad, LU, pivots


# ---- triangular -----------------------------------------------------------------------------------------------------


def _triangle(t, upper, diagonal=0):
    return t.triu(diagonal) if upper else t.tril(-diagonal)


def _real_tri(E, R, upper, unitriangular):
    """The real triangular solve ``E X = R`` (float32, one triangle of ``E`` read): hpu-ops' TPC substitution kernel
    called directly (one launch, no dispatcher route or autograd of its own), or torch's op where hpu-ops' kernels are
    not loaded (the CPU stand-in)."""
    tpc = _hpu_ops_tpc()
    if tpc is not None and E.device.type == "hpu" and E.shape[-1] <= tpc.MAXN and tpc.ready():
        if E.dim() == 3 and R.dim() == 3 and E.shape[0] == R.shape[0] and R.shape[-1] <= tpc.MAXN and \
                _fresh(E) and _fresh(R):
            # the kernel itself (through hpu-ops' lazy-mode guard): ``tri_solve`` copies every input that is a view,
            # which its own batch reshape always makes one (two copies per call, measured as launches)
            return tpc._call(tpc._tri_op(upper, unitriangular), E, R)
        return tpc.tri_solve(E, R, upper=upper, unitriangular=unitriangular)
    return torch.linalg.solve_triangular(E, R, upper=upper, unitriangular=unitriangular)


def _fresh(t):
    """A contiguous float32 tensor that owns its storage from offset 0 (no view: views break hpu-ops' kernels in lazy
    mode)."""
    return t.dtype == torch.float32 and t._base is None and t.storage_offset() == 0 and t.is_contiguous()


# Row i of the interleaved embedding of ``conj(phi_i) T``, ``phi_i = d_i / |d_i| = ur + i ui`` (``d = diag T``): entry
# ``[2i + a, 2j + c] = sum_e N[i, a, c, e] * (re, im)[T_ij][e]``, the rotation ``[[ur, ui], [-ui, ur]]`` (multiplication
# by conj(phi)) followed by row a = 0 ``(re, -im)``, a = 1 ``(im, re)``: N[i] = [[[ur, ui], [ui, -ur]], [[-ui, ur],
# [ur, ui]]]. Its c = 0 half is the rotation itself, which gives v(conj(phi) B).


def _tri_left_forward(A, B, upper, unitriangular):
    """``A X = B`` for triangular complex ``A``: the interleaved embedding of a triangle with a real diagonal is a real
    triangle. Non-unit: the rows of ``A`` and ``B`` are rotated by ``conj(phi_i)`` (``phi = diag / |diag|``), which
    makes the diagonal ``|d_i|`` (a pure rotation: no reciprocal, no scaling of the solution), and the real
    substitution divides by it. Unit: the diagonal is not read (its imaginary part, which the real kernel would read
    at ``[2i + 1, 2i]``, is masked). The embedding and the rotation are one multiply and one sum per operand."""
    n, k = A.shape[-1], B.shape[-1]
    # contiguous first: on the card a transposed view (A^H in the backward) through the broadcast products below gave
    # wrong values (relative error 0.45, measured; the same matrix copied first is right). Free when contiguous.
    va = torch.view_as_real(A.resolve_conj().resolve_neg().contiguous())  # (..., n, n, 2)
    vb = torch.view_as_real(B.resolve_conj().resolve_neg().contiguous())  # (..., n, k, 2)
    if unitriangular:
        ar, ai = va[..., 0], va[..., 1]
        E = _interleaved(ar, ai.triu(1) if upper else ai.tril(-1))
        R = vb.transpose(-1, -2).reshape(*vb.shape[:-3], 2 * n, k)  # v(B): rows (re, im) per index
    else:
        d = torch.diagonal(va, 0, -3, -2)  # (..., 2, n): (re, im) of the diagonal
        p = (d / torch.hypot(d[..., :1, :], d[..., 1:, :])).mT  # (..., n, 2): phi_i (a zero diagonal: NaN, as torch's inf)
        m = -p
        N = torch.stack([p[..., 0], p[..., 1], p[..., 1], m[..., 0], m[..., 1], p[..., 0], p[..., 0], p[..., 1]],
                        -1).unflatten(-1, (2, 2, 2))  # (..., n, a, c, e)
        # reshaped before the sum, so E and R are fresh tensors, not views (hpu-ops copies a view before its kernel)
        E = (N[..., :, :, None, :, :] * va[..., :, None, :, None, :]).reshape(*va.shape[:-3], 2 * n, 2 * n, 2).sum(-1)
        R = N[..., :, :, None, 0, :] * vb[..., :, None, :, :]
        R = R.reshape(*R.shape[:-4], 2 * n, k, 2).sum(-1)
    return _from_interleaved_rhs(_real_tri(E, R, upper, unitriangular))


class _SolveTriangular(torch.autograd.Function):
    """torch's ``linalg_solve_triangular_backward`` (left): ``gB = A^-H g`` (the other triangle), ``gA = -gB X^H``
    restricted to the triangle (strict when unit)."""

    @staticmethod
    def forward(ctx, A, B, upper, unitriangular):
        X = _tri_left_forward(A, B, upper, unitriangular)
        ctx.save_for_backward(A, X)
        ctx.flags = (upper, unitriangular)
        ctx.shapes = (A.shape, B.shape)
        return X

    @staticmethod
    def backward(ctx, g):
        A, X = ctx.saved_tensors
        upper, unit = ctx.flags
        gB = _SolveTriangular.apply(A.mH, g, not upper, unit)
        gA = None
        if ctx.needs_input_grad[0]:
            gr, gi = _pair(gB)
            xr, xi = _pair(X)
            pr, pi = _cmm(gr, gi, xr.mT, -xi.mT)  # gB X^H
            gA = _complex(-_triangle(pr, upper, int(unit)), -_triangle(pi, upper, int(unit))).sum_to_size(ctx.shapes[0])
        return gA, gB.sum_to_size(ctx.shapes[1]), None, None


def solve_triangular(A, B, upper, left=True, unitriangular=False):
    """``linalg.solve_triangular`` (``left=False``: ``X A = B`` as ``A^T X^T = B^T``, the other triangle)."""
    if not left:
        return solve_triangular(A.mT, B.mT, not upper, True, unitriangular).mT
    if torch.is_grad_enabled() and (A.requires_grad or B.requires_grad):
        return _SolveTriangular.apply(A, B, upper, unitriangular)
    return _tri_left_forward(A, B, upper, unitriangular)


def _cmatmul(a, b):
    ar, ai = _pair(a)
    br, bi = _pair(b)
    return _complex(*_cmm(ar, ai, br, bi))


def _ctril(z, diagonal=0):
    re, im = _pair(z)
    return _complex(re.tril(diagonal), im.tril(diagonal))


def _factor_embedding(L, upper):
    """The interleaved embedding of a triangular factor with the same ``L L^H`` (``U^H U``) and a real positive
    diagonal, for Habana's real kernels (which read one triangle only): ``L' = L diag(conj(phi))`` (``U' =
    diag(conj(phi)) U``), ``phi = diag / |diag|``, so ``L' L'^H = L L^H`` and ``diag(L') = |diag|``, the rotation
    ``solve_triangular`` does by rows. A complex diagonal is thus right (torch's ``cholesky_solve`` uses it); the
    rotation is ``hypot``, one division and one complex scaling, no host read."""
    v = torch.view_as_real(L.resolve_conj().resolve_neg().contiguous())
    lr, li = v[..., 0], v[..., 1]
    d = torch.diagonal(v, 0, -3, -2)  # (..., 2, n)
    p = d / torch.hypot(d[..., :1, :], d[..., 1:, :])  # (re, im) of phi (a zero diagonal: NaN, singular anyway)
    ur, ui = (p[..., 0, :, None], p[..., 1, :, None]) if upper else (p[..., 0, None, :], p[..., 1, None, :])
    rr, ri = lr * ur + li * ui, li * ur - lr * ui  # times conj(phi): rows (upper) or columns (lower)
    return _interleaved(rr, ri.triu(1) if upper else ri.tril(-1))  # the diagonal's residue of im: 0


def _from_interleaved(X):
    """The complex (..., n, m) matrix of its interleaved embedding: column 2j holds (re, im) of column j."""
    return _complex(X[..., 0::2, 0::2], X[..., 1::2, 0::2])


def _cholesky_inverse_real(L, upper):
    """``E(H^-1)`` for ``H = L L^H`` (``U^H U``): Habana's ``cholesky_inverse`` on the interleaved factor, which is the
    real Cholesky factor of ``E(H)``."""
    return torch.cholesky_inverse(_factor_embedding(L, upper), upper=upper)


class _CholeskySolve(torch.autograd.Function):
    """torch's ``cholesky_solve_backward``: ``gB = cholesky_solve(g, L)``, ``gL = -(C L)`` (``-(L C)`` if upper) with
    ``C = gB X^H + X gB^H`` (not masked to the triangle, as torch). ``gB`` reuses the forward's ``E(H^-1)``."""

    @staticmethod
    def forward(ctx, B, L, upper):
        Einv = _cholesky_inverse_real(L, upper)
        X = _apply_interleaved(Einv, B)
        ctx.save_for_backward(L, X, Einv)
        ctx.upper = upper
        ctx.shapes = (B.shape, L.shape)
        return X

    @staticmethod
    def backward(ctx, g):
        L, X, Einv = ctx.saved_tensors
        if torch.is_grad_enabled():  # create_graph: differentiable in L
            gB = _CholeskySolve.apply(g, L, ctx.upper)
        else:
            gB = _apply_interleaved(Einv, g)
        gL = None
        if ctx.needs_input_grad[1]:
            common = _cmatmul(gB, X.mH)
            common = common + common.mH
            gL = -_cmatmul(L, common) if ctx.upper else -_cmatmul(common, L)
            gL = gL.sum_to_size(ctx.shapes[1])
        return gB.sum_to_size(ctx.shapes[0]), gL, None


def _apply_interleaved(E, B):
    """``M B`` for complex B (..., n, k) and the interleaved embedding ``E = E(M)``: one real GEMM on ``v(B)``."""
    br, bi = _pair(B)
    return _from_interleaved_rhs(E @ _interleaved_rhs(br, bi))


def cholesky_solve(B, L, upper=False):
    """``cholesky_solve``: ``X = H^-1 B`` with ``E(H^-1)`` from Habana's ``cholesky_inverse`` of the interleaved
    factor, then one real GEMM (an inverse-based solve: forward error of order cond(H) eps, as two substitutions)."""
    if torch.is_grad_enabled() and (B.requires_grad or L.requires_grad):
        return _CholeskySolve.apply(B, L, upper)
    return _apply_interleaved(_cholesky_inverse_real(L, upper), B)


class _CholeskyInverse(torch.autograd.Function):
    """torch's ``cholesky_inverse_backward``: ``C = Inv (g + g^H) Inv``, ``gL = -(C L)`` (``-(L C)`` if upper)."""

    @staticmethod
    def forward(ctx, L, upper):
        inv = _from_interleaved(_cholesky_inverse_real(L, upper))
        ctx.save_for_backward(L, inv)
        ctx.upper = upper
        return inv

    @staticmethod
    def backward(ctx, g):
        L, inv = ctx.saved_tensors
        common = _cmatmul(inv, _cmatmul(g + g.mH, inv))
        return (-_cmatmul(L, common) if ctx.upper else -_cmatmul(common, L)), None


def cholesky_inverse(L, upper=False):
    """``cholesky_inverse``: ``(L L^H)^-1`` (``(U^H U)^-1``) from Habana's real ``cholesky_inverse`` of the interleaved
    factor (rotated to a real positive diagonal, see ``_factor_embedding``). Equal to torch's for a real diagonal of
    either sign. For a diagonal that is not real this is ``(L L^H)^-1``; torch's LAPACK result there (``potri``, whose
    ``lauum`` reads only the real part of the inverted diagonal) is neither that nor Hermitian (measured, 0.045 off),
    so the two differ there."""
    if torch.is_grad_enabled() and L.requires_grad:
        return _CholeskyInverse.apply(L, upper)
    return _from_interleaved(_cholesky_inverse_real(L, upper))


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
        Y = _real_tri(E.mT, _interleaved_rhs(pr, pi), True, False)  # v(L^-H Phi)
        n = L.shape[-1]
        Y = Y.unflatten(-2, (n, 2))  # [..., i, a, j]
        Yt = torch.stack([Y[..., 0, :], -Y[..., 1, :]], -1).flatten(-2)  # rows [yr_0, -yi_0, ...]
        X = _real_tri(E.mT, Yt.mT, True, False).mT.unflatten(-1, (n, 2))  # X~ E(L) = Y~  <=>  E(L)^T X~^T = Y~^T
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


def _hermitian_lower_embedding(A, upper):
    """The interleaved embedding of the Hermitian matrix held in one triangle of ``A``, right in its lower triangle
    (the only one Habana's cholesky reads; measured): the imaginary part of the diagonal is masked, and for ``upper``
    the lower triangle of ``H`` is ``conj(U)^T``."""
    ar, ai = _pair(A)
    if upper:
        return _embed(ar.mT, ai.mT.tril(-1), True, conj=True)
    return _interleaved(ar, ai.tril(-1))


def _real_cholesky(E, with_info=False):
    """Habana's real Cholesky factor of ``E`` (..., n2, n2). It returns NaN (info 0) when the dim before the matrix
    equals its order (bridge 1.21.4: every (..., n2, n2, n2) of the sweep in repro/habana_linalg_shape_sweep.py, eager
    and lazy; any other batch is right); the same batch as (..., n2, 1, n2, n2) is right in every swept shape (a view,
    no op). Its own ``info`` is a constant 0 and is not read."""
    n2 = E.shape[-1]
    if E.dim() >= 3 and E.shape[-3] == n2:
        Lreal, info = torch.linalg.cholesky_ex(E.unsqueeze(-3))
        Lreal, info = Lreal.squeeze(-3), info.squeeze(-1)
    else:
        Lreal, info = torch.linalg.cholesky_ex(E)
    return (Lreal, info) if with_info else Lreal


def _cholesky_info(Lreal, real_info):
    """torch's ``info`` for the interleaved factor. Habana's kernel reports ``info = 0`` always and gives NaN from the
    failing pivot on (measured: a non-positive pivot, a Schur-complement failure and an off-diagonal NaN alike), so
    the first real diagonal entry r (0-based) that is not positive (NaN included, LAPACK's test) is the failing real
    pivot, of complex pivot r // 2: info = r // 2 + 1. Each real position holds its complex info (1, 1, 2, 2, ...,
    cached), the good ones n + 1; the least, modulo n + 1, is that info. A LAPACK factor (the CPU stand-in) is not
    read this way: Accelerate leaves the original diagonal from the failing pivot on, so the diagonal can look good;
    its own real info is exact, mapped as (info + 1) // 2 and taken where nonzero. Float math, cast at the end.
    (Measured on the card: ``amin`` over a (NaN, x) pair skips the NaN; strided views of the diagonal and int64
    arithmetic on an ``argmax`` gave wrong indices on a lazy graph's first run; an ``argmax``-based rule took 2-6 ms
    per call in lazy mode.)"""
    n = Lreal.shape[-1] // 2
    first = _info_positions(n, Lreal.device).expand(Lreal.shape[:-1]).masked_fill(
        torch.diagonal(Lreal, 0, -2, -1) > 0, n + 1).amin(-1)
    real = real_info.float()
    info = torch.where(real > 0, torch.floor((real + 1) * 0.5), torch.remainder(first, n + 1))
    return info.to(torch.int32)


@functools.lru_cache(maxsize=None)
def _info_positions(n, device):
    """[1, 1, 2, 2, ..., n, n] (float32) on ``device``: the complex pivot (1-based) of each real diagonal position."""
    return torch.arange(1, n + 1, dtype=torch.float32).repeat_interleave(2).to(device)


def _cholesky_ex(A, upper=False, keep_real=False):
    """``linalg_cholesky_ex``: the real Cholesky factor of the interleaved ``E(H)`` is ``E(L)``."""
    n = A.shape[-1]
    Lreal, real_info = _real_cholesky(_hermitian_lower_embedding(A, upper), with_info=True)
    blocks = Lreal.reshape(*Lreal.shape[:-2], n, 2, n, 2)
    re, im = blocks[..., 0, :, 0], blocks[..., 1, :, 0].tril(-1)  # the diagonal is real (its rounding residue: 0)
    info = _cholesky_info(Lreal, real_info)
    L = _complex(re.mT, -im.mT) if upper else _complex(re, im)
    return (L, info, Lreal) if keep_real else (L, info)


# ---- Hermitian positive-definite slogdet / logdet -------------------------------------------------------------------


def _hpd_logabsdet_forward(A):
    Lreal, real_info = _real_cholesky(_hermitian_lower_embedding(A, False), with_info=True)
    # each complex pivot appears twice on the real diagonal: sum log = 2 sum log l_jj = log det H; NaN unless every
    # pivot is positive and the kernel reported no failure (an exactly singular matrix too, not -inf)
    d = torch.diagonal(Lreal, 0, -2, -1)
    ok = (d > 0).all(-1) & (real_info == 0)
    return torch.where(ok, torch.log(d).sum(-1), float("nan")), Lreal


class _HpdLogabsdet(torch.autograd.Function):
    """``log det H`` of the HPD matrix in the lower triangle of ``A``, with torch's ``slogdet`` gradient at ``H``:
    ``g A^-H = g H^-1`` (full, not masked to the triangle), ``H^-1`` from Habana's ``cholesky_inverse`` of the factor
    the forward computed. With ``create_graph`` the same ``g A^-H`` through differentiable ops on ``A``
    (``linalg.inv`` of H: the block-embedding route), so second derivatives are torch's slogdet's at H."""

    @staticmethod
    def forward(ctx, A):
        logabsdet, Lreal = _hpd_logabsdet_forward(A)
        ctx.save_for_backward(A, Lreal)
        return logabsdet

    @staticmethod
    def backward(ctx, g):
        A, Lreal = ctx.saved_tensors
        if torch.is_grad_enabled():  # H from the lower triangle, its gradient passed through as for the full matrix
            return g[..., None, None] * torch.linalg.inv(_HermitianFromLower.apply(A)).mH
        inv = torch.cholesky_inverse(Lreal)
        return _from_interleaved(g[..., None, None] * inv)


def hpd_slogdet(A):
    """``torch.linalg.slogdet`` of a Hermitian positive-definite complex ``A`` (its lower triangle read, as
    ``linalg.cholesky``): sign 1, ``logabsdet = 2 sum log diag(L)`` from one real Cholesky of the interleaved
    embedding, on the card. An explicit API, not a route of ``torch.linalg.slogdet``: torch has no Hermitian flag, and
    testing for one would read the matrix on the host. A matrix that is not positive definite, an exactly singular
    one included, gives NaN ``logabsdet`` (not -inf: ``torch.linalg.cholesky_ex`` reports which), on and off the card.
    The general complex determinant is not here. Off the card (CPU, complex128) the same math runs through torch's
    ops."""
    if not _complex_on_card(A):
        L, info = torch.linalg.cholesky_ex(A)
        d = torch.diagonal(L, 0, -2, -1).real
        logabsdet = torch.where((info == 0) & (d > 0).all(-1), 2 * torch.log(d).sum(-1), float("nan"))
        return torch.return_types.linalg_slogdet((torch.ones_like(logabsdet, dtype=A.dtype), logabsdet))
    _count("hpd_slogdet")
    if torch.is_grad_enabled() and A.requires_grad:
        logabsdet = _HpdLogabsdet.apply(A)
    else:
        logabsdet = _hpd_logabsdet_forward(A)[0]
    sign = _complex(torch.ones_like(logabsdet), torch.zeros_like(logabsdet))
    return torch.return_types.linalg_slogdet((sign, logabsdet))


def hpd_logdet(A):
    """``torch.logdet`` of a Hermitian positive-definite complex ``A`` (complex, imaginary part 0), see
    ``hpd_slogdet``."""
    logabsdet = hpd_slogdet(A).logabsdet
    return _complex(logabsdet, torch.zeros_like(logabsdet)) if logabsdet.device.type == "hpu" else \
        logabsdet.to(A.dtype)


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


# ---- eigh, svd -------------------------------------------------------------------------------------------------------


def _upper(UPLO):
    if not isinstance(UPLO, str) or UPLO.upper() not in ("L", "U"):
        raise RuntimeError(f"Expected UPLO argument to be 'L' or 'U', but got {UPLO}")
    return UPLO.upper() == "U"


def _hermitian_pairs(A, upper):
    """(re, im) of the Hermitian matrix held in one triangle of ``A`` (the imaginary part of the diagonal ignored)."""
    ar, ai = _pair(A)
    if upper:
        ar, ai = ar.mT, -ai.mT
    lr, li = ar.tril(), ai.tril(-1)
    return lr + lr.tril(-1).mT, li - li.mT


class _Eigh(torch.autograd.Function):
    """torch's ``linalg_eig_backward`` (``is_hermitian``): ``gA = V (diag(gL) + skew(V^H gV) / E) V^H`` with
    ``skew(X) = (X - X^H) / 2`` and ``E_ij = L_j - L_i`` (1 on the diagonal)."""

    @staticmethod
    def forward(ctx, A, UPLO, compute_v):
        ctx.set_materialize_grads(False)
        L, V_p = _complex_linalg().eigh(A, UPLO=UPLO)
        V = _as_complex(V_p)
        ctx.save_for_backward(L, V)
        if not compute_v:
            return L, torch.empty(0, dtype=C64, device=A.device)
        return L, V

    @staticmethod
    def backward(ctx, gL, gV):
        if gL is None and gV is None:
            return None, None, None
        L, V = ctx.saved_tensors
        if gV is None:
            return _cmatmul(V * gL.unsqueeze(-2), V.mH), None, None
        vr, vi = _pair(V)
        gr, gi = _pair(gV)
        xr, xi = _cmm(vr.mT, -vi.mT, gr, gi)  # V^H gV
        xr, xi = 0.5 * (xr - xr.mT), 0.5 * (xi + xi.mT)
        eye = torch.eye(L.shape[-1], dtype=L.dtype, device=L.device)
        E = (L.unsqueeze(-2) - L.unsqueeze(-1)) + eye
        xr, xi = xr / E, xi / E
        if gL is not None:
            xr = xr * (1 - eye) + torch.diag_embed(gL)
        else:
            xr = xr * (1 - eye)
        xi = xi * (1 - eye)
        tr, ti = _cmm(xr, xi, vr.mT, -vi.mT)  # (...) V^H
        gr, gi = _cmm(vr, vi, tr, ti)
        return _complex(gr, gi), None, None


def eigh(A, UPLO="L", compute_v=True):
    """``_linalg_eigh`` via ``complex_linalg.eigh`` -> (eigenvalues, eigenvectors)."""
    _upper(UPLO)
    if not compute_v:
        return eigvalsh(A, UPLO), torch.empty(0, dtype=C64, device=A.device)
    if torch.is_grad_enabled() and A.requires_grad:
        return _Eigh.apply(A, UPLO, True)
    L, V_p = _complex_linalg().eigh(A, UPLO=UPLO)
    return L, _as_complex(V_p)


class _HermitianEmbedding(torch.autograd.Function):
    """Fallback interleaved embedding of ``eigvalsh`` when ``complex_linalg`` is unavailable."""

    @staticmethod
    def forward(ctx, A, upper):
        ar, ai = _pair(A)
        return _interleaved(ar, ai.triu(1) if upper else ai.tril(-1))

    @staticmethod
    def backward(ctx, g):
        return _complex(g[..., 0::2, 0::2] + g[..., 1::2, 1::2], g[..., 1::2, 0::2] - g[..., 0::2, 1::2]), None


def eigvalsh(A, UPLO="L"):
    """``linalg.eigvalsh`` via ``complex_linalg.eigvalsh`` (or ``_HermitianEmbedding`` fallback if unavailable)."""
    upper = _upper(UPLO)
    cl = _complex_linalg()
    if cl is not None and A.shape[-1] <= cl.COMPLEX_MAXN:
        if torch.is_grad_enabled() and A.requires_grad:
            return _Eigh.apply(A, UPLO, False)[0]
        return cl.eigvalsh(A, UPLO=UPLO)
    E = _HermitianEmbedding.apply(A, upper) if torch.is_grad_enabled() and A.requires_grad else \
        _HermitianEmbedding.forward(None, A, upper)
    w = torch.linalg.eigvalsh(E, UPLO="U" if upper else "L")
    return 0.5 * (w[..., 0::2] + w[..., 1::2])


def _complete_orthogonal(Q, rows=False):
    """Complete orthonormal columns of Q (..., d, k) (or rows (..., k, d) if rows=True) to (..., d, d)."""
    cl = _complex_linalg()
    if rows:
        k, d = Q.shape[-2], Q.shape[-1]
        pr, pi = _pair(-_cmatmul(Q.mH, Q))
        P = _complex(pr + torch.eye(d, dtype=torch.float32, device=Q.device), pi)
        Up, _, _ = cl.svd(P, full_matrices=False)
        V_null = _as_complex(Up)[..., :, : d - k]
        return torch.cat([Q, V_null.mH], -2)
    d, k = Q.shape[-2], Q.shape[-1]
    pr, pi = _pair(-_cmatmul(Q, Q.mH))
    P = _complex(pr + torch.eye(d, dtype=torch.float32, device=Q.device), pi)
    Up, _, _ = cl.svd(P, full_matrices=False)
    U_null = _as_complex(Up)[..., :, : d - k]
    return torch.cat([Q, U_null], -1)


class _Svd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, full_matrices, compute_uv):
        ctx.set_materialize_grads(False)
        cl = _complex_linalg()
        m, n = A.shape[-2], A.shape[-1]
        if not compute_uv and not ctx.needs_input_grad[0]:
            S = cl.svdvals(A)
            empty = torch.empty(0, dtype=C64, device=A.device)
            return empty, S, empty
        U_p, S, Vh_p = cl.svd(A, full_matrices=False)
        U, Vh = _as_complex(U_p), _as_complex(Vh_p)
        ctx.save_for_backward(U, S, Vh)
        if not compute_uv:
            empty = torch.empty(0, dtype=C64, device=A.device)
            return empty, S, empty
        if full_matrices and m != n:
            if m > n:
                U = _complete_orthogonal(U, rows=False)
            else:
                Vh = _complete_orthogonal(Vh, rows=True)
        return U, S, Vh

    @staticmethod
    def backward(ctx, gU, gS, gVh):
        if gU is None and gS is None and gVh is None:
            return None, None, None
        U, S, Vh = ctx.saved_tensors
        m, k = U.shape[-2], U.shape[-1]
        n = Vh.shape[-1]
        gUk = gU[..., :k] if gU is not None else None
        gVhk = gVh[..., :k, :] if gVh is not None else None
        if gUk is None and gVhk is None:
            return _cmatmul(U * gS.unsqueeze(-2), Vh), None, None
        S_inv = torch.where(S > 0, S.reciprocal(), 0.0)
        total = S.unsqueeze(-2) + S.unsqueeze(-1)
        safe_total = torch.where(total != 0, total, 1.0)
        diff1 = S.unsqueeze(-2) - S.unsqueeze(-1)
        distinct = diff1 != 0
        safe_diff1 = torch.where(distinct, diff1, 1.0)
        G1 = torch.where(distinct, (S.unsqueeze(-2) / safe_total) / safe_diff1, 0.0)
        G2 = torch.where(distinct, (S.unsqueeze(-1) / safe_total) / safe_diff1, 0.0)
        diag_sum = torch.zeros_like(S)
        core = None
        extra = None
        if gUk is not None:
            UhgU = _cmatmul(U.mH, gUk)
            diag_sum = diag_sum + torch.view_as_real(UhgU.diagonal(0, -2, -1).resolve_conj())[..., 1]
            core = (UhgU - UhgU.mH) * G1
            if m > k:
                extra = _cmatmul((gUk - _cmatmul(U, UhgU)) * S_inv.unsqueeze(-2), Vh)
        if gVhk is not None:
            gVhV = _cmatmul(gVhk, Vh.mH)
            diag_sum = diag_sum + torch.view_as_real(gVhV.diagonal(0, -2, -1).resolve_conj())[..., 1]
            t2 = (gVhV - gVhV.mH) * G2
            core = -t2 if core is None else (core - t2)
            if n > k:
                t_extra = _cmatmul(U * S_inv.unsqueeze(-2), gVhk - _cmatmul(gVhV, Vh))
                extra = t_extra if extra is None else (extra + t_extra)
        diag_c = _complex(gS if gS is not None else torch.zeros_like(S), 0.5 * diag_sum * S_inv)
        gA = _cmatmul(U * diag_c.unsqueeze(-2), Vh) + _cmatmul(U, _cmatmul(core, Vh))
        if extra is not None:
            gA = gA + extra
        return gA, None, None


def svd(A, full_matrices=False, compute_uv=True, *, driver=None):
    """``_linalg_svd`` via ``complex_linalg.svd`` / ``svdvals``."""
    if torch.is_grad_enabled() and A.requires_grad:
        return _Svd.apply(A, full_matrices, compute_uv)
    cl = _complex_linalg()
    if not compute_uv:
        empty = torch.empty(0, dtype=C64, device=A.device)
        return empty, cl.svdvals(A), empty
    U_p, S, Vh_p = cl.svd(A, full_matrices=False)
    U, Vh = _as_complex(U_p), _as_complex(Vh_p)
    m, n = A.shape[-2], A.shape[-1]
    if full_matrices and m != n:
        if m > n:
            U = _complete_orthogonal(U, rows=False)
        else:
            Vh = _complete_orthogonal(Vh, rows=True)
    return U, S, Vh


def svd_out(A, full_matrices=False, compute_uv=True, *, driver=None, U, S, Vh):
    u, s, vh = svd(A, full_matrices, compute_uv, driver=driver)
    if compute_uv:
        torch.ops.aten._copy_from_and_resize(u, U)
        torch.ops.aten._copy_from_and_resize(vh, Vh)
    S.resize_(s.shape).copy_(s)
    return U, S, Vh


def svdvals(A, *, driver=None):
    """``linalg_svdvals`` via ``complex_linalg.svdvals`` (or ``_Svd`` when grad is needed)."""
    if torch.is_grad_enabled() and A.requires_grad:
        return _Svd.apply(A, False, False)[1]
    return _complex_linalg().svdvals(A)


# ---- qr ---------------------------------------------------------------------------------------------------------------


def _qr_forward(A, want_q):
    """Habana's real QR of the interleaved ``E(A)`` (2m x 2n, m >= n). For ``A = Q R`` with ``diag(R) > 0``,
    ``E(A) = E(Q) E(R)``, ``E(R)`` upper triangular with the diagonal pairs ``(r_kk, r_kk)`` and ``E(Q)`` with orthonormal
    columns; the real QR with a positive diagonal is unique (full column rank), so the real factors re-signed to a
    positive diagonal are ``E(Q)``, ``E(R)``, and column 2j holds (re, im) of column j. R has a real positive
    diagonal: LAPACK's (and torch's) is real too, with the sign ``-sign(Re)`` of each Householder head; both are QR
    decompositions (torch documents the signs as not unique), and ``Q`` differs from torch's by those column signs.
    A rank-deficient ``A`` (a zero column, dependent columns) gives NaN: Habana's qr returns NaN for it, and ``W``
    does not change the rank (torch returns a QR with a zero on R's diagonal)."""
    ar, ai = _pair(A)
    E = _interleaved(ar, ai)
    W = _mixing(E.shape[-2], E.device)
    # Habana's qr returns NaN when a Householder column has a zero tail (an identity, a diagonal or triangular
    # matrix, any A whose first column is a real multiple of e_0; repro/habana_qr_nan.py). The QR of W E, W a fixed
    # dense orthogonal matrix, is (W Q) R: the same R, and Q = W^T (W Q), with columns of W E generic.
    Qe, Re = torch.linalg.qr(W @ E)
    s = torch.where(torch.diagonal(Re, 0, -2, -1) < 0, -1.0, 1.0)  # (..., 2n)
    R = _from_interleaved(Re * s.unsqueeze(-1))
    Q = _from_interleaved(W.mT @ (Qe * s.unsqueeze(-2))) if want_q else None
    return Q, R


@functools.lru_cache(maxsize=None)
def _mixing(m, device):
    """A fixed dense m x m orthogonal float32 matrix on ``device`` (QR of a seeded Gaussian, float64 on the host,
    once per size)."""
    g = torch.Generator().manual_seed(m)
    W, _ = torch.linalg.qr(torch.randn(m, m, dtype=torch.float64, generator=g))
    return W.to(torch.float32).to(device)


class _QR(torch.autograd.Function):
    """torch's ``linalg_qr_backward`` for m >= n: ``M = gR R^H - Q^H gQ``, ``gA = (Q syminvadj(triu(M)) + gQ) R^-H`` with
    ``syminvadj(X) = X + X^H`` and its diagonal's real part halved; the solve with R^H is this module's triangular
    solve."""

    @staticmethod
    def forward(ctx, A):
        Q, R = _qr_forward(A, True)
        ctx.save_for_backward(Q, R)
        return Q, R

    @staticmethod
    def backward(ctx, gQ, gR):
        Q, R = ctx.saved_tensors
        if gR is not None:
            M = _cmatmul(gR, R.mH)
            M = M - _cmatmul(Q.mH, gQ) if gQ is not None else M
        else:
            M = -_cmatmul(Q.mH, gQ)
        mr, mi = _pair(M)
        mr, mi = mr.triu(), mi.triu()
        sr, si = mr + mr.mT, mi - mi.mT  # X + X^H; its diagonal: 2 Re (imaginary part 0)
        eye = torch.eye(sr.shape[-1], dtype=sr.dtype, device=sr.device)
        sr = sr - 0.5 * sr * eye
        gA = _cmatmul(Q, _complex(sr, si))
        if gQ is not None:
            gA = gA + gQ
        return solve_triangular(R.mH, gA, upper=False, left=False)


def qr(A, mode="reduced"):
    """``linalg.qr`` for m >= n (modes ``reduced``, ``r``, and ``complete`` when m == n)."""
    if mode == "r":
        if torch.is_grad_enabled() and A.requires_grad:
            raise RuntimeError("The derivative of linalg.qr depends on Q, which is not computed when mode='r'. "
                               "Please use linalg.qr(A, mode='reduced') if you are going to differentiate through "
                               "linalg.qr.")
        return torch.empty(0, dtype=A.dtype, device=A.device), _qr_forward(A, False)[1]
    if torch.is_grad_enabled() and A.requires_grad:
        return _QR.apply(A)
    return _qr_forward(A, True)


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


def _decline(op, args, kwargs, key="AutogradPrivateUse1"):
    """torch's own kernel: CompositeImplicitAutograd for math ops, Autograd when at AutogradPrivateUse1,
    or PrivateUse2 (Extension cpu_fallback / structured_pu1) when at PrivateUse1."""
    if torch._C._dispatch_has_kernel_for_dispatch_key(op.name(), "CompositeImplicitAutograd"):
        return op._op_dk(DispatchKey.CompositeImplicitAutograd, *args, **kwargs)
    if key == "PrivateUse1":
        return op._op_dk(DispatchKey.PrivateUse2, *args, **kwargs)
    return op._op_dk(DispatchKey.Autograd, *args, **kwargs)


ROUTES = {
    "solve": ("eager", "lazy"),
    "solve_left_false": ("lazy",),  # eager: forward + backward 0.79-1.0x for real block; n<=32 uses cgj_solve
    "inv": ("eager", "lazy"),
    "det": ("eager", "lazy"),
    "slogdet": ("eager", "lazy"),
    "cholesky": ("eager", "lazy"),
    "matrix_exp": ("eager", "lazy"),
    "matrix_exp_composite": ("lazy",),
    "pinv": ("eager", "lazy"),
    "householder_product": ("eager:grad", "lazy:grad"),
    "lstsq": ("eager:grad", "lazy:grad"),
    "solve_triangular": ("lazy:grad",),
    "cholesky_solve": ("eager", "lazy"),
    "cholesky_inverse": ("eager", "lazy"),
    "qr": (),
    "eigvalsh": ("eager", "lazy"),
    "eigh": ("eager", "lazy"),
    "svd": ("eager", "lazy"),
    "svdvals": ("eager", "lazy"),
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


def _kernel(op, name, takes, compute, key="AutogradPrivateUse1", meta=True):
    def kernel(*args, **kwargs):
        if _routed(name, args, kwargs) and takes(*args, **kwargs):
            if meta:  # torch's shape checks and errors (lstsq has no meta kernel: the real lstsq raises its own)
                _meta_call(op, *args, **kwargs)
            _count(name)
            return compute(*args, **kwargs)
        return _decline(op, args, kwargs, key)

    return kernel


def _square(A):
    return A.dim() >= 2 and A.shape[-1] == A.shape[-2]


def _batch(*tensors):
    return math.prod(torch.broadcast_shapes(*[t.shape[:-2] for t in tensors]))


def _takes_solve(A, B, left=True, check_errors=False):
    # Only with grad or inference_mode on: the LU and pivots returned here are placeholders, which only torch's own
    # derivative reads. A compiled graph's forward runs with grad off (and inference_mode off) and its traced backward
    # reads LU, so such calls keep torch's route.
    if not (torch.is_grad_enabled() or torch.is_inference_mode_enabled()):
        return False
    if not (isinstance(B, torch.Tensor) and _complex_on_card(A, B) and _square(A)):
        return False
    if _is_vector_rhs(A, B) and not left:
        return False
    rhs = B.unsqueeze(-1) if _is_vector_rhs(A, B) else B
    if rhs.dim() < 2 or rhs.shape[-2 if left else -1] != A.shape[-1]:
        return False
    n = A.shape[-1]
    cl = _complex_linalg()
    if cl is not None and 1 <= n <= cl.COMPLEX_MAXN:
        return True
    if not left and not _routed("solve_left_false", (A, B), {}):
        return False
    return _real_on_card("solve", 2 * n, _batch(A, rhs))


def _takes_inv(A, check_errors=False):
    if not (_complex_on_card(A) and _square(A)):
        return False
    n = A.shape[-1]
    cl = _complex_linalg()
    if cl is not None and 1 <= n <= cl.COMPLEX_MAXN:
        return True
    return _real_on_card("solve", 2 * n, _batch(A))


def _takes_det(A):
    if not (torch.is_grad_enabled() or torch.is_inference_mode_enabled()):
        return False
    cl = _complex_linalg()
    return cl is not None and _complex_on_card(A) and _square(A) and 1 <= A.shape[-1] <= cl.COMPLEX_MAXN


def _takes_tri(A, B, *, upper, left=True, unitriangular=False):
    if not (isinstance(B, torch.Tensor) and _complex_on_card(A, B) and _square(A) and B.dim() >= 2):
        return False
    if B.shape[-2 if left else -1] != A.shape[-1]:
        return False
    return _real_on_card("tri", 2 * A.shape[-1])


def _takes_cholesky_solve(B, L, upper=False):
    if not (isinstance(L, torch.Tensor) and _complex_on_card(B, L) and _square(L) and B.dim() >= 2):
        return False
    return B.shape[-2] == L.shape[-1] and _real_on_card("cholesky_inverse", 2 * L.shape[-1])


def _takes_cholesky_inverse(L, upper=False):
    return _complex_on_card(L) and _square(L) and _real_on_card("cholesky_inverse", 2 * L.shape[-1])


def _takes_qr(A, mode="reduced"):
    if not (_complex_on_card(A) and A.dim() >= 2 and A.shape[-2] >= A.shape[-1]):
        return False
    if mode not in ("reduced", "r") and not (mode == "complete" and A.shape[-2] == A.shape[-1]):
        return False  # complete with m > n: the extra columns of Q need not keep the pair structure; torch's route
    return _real_on_card("qr", 2 * A.shape[-2])


def _takes_eigvalsh(A, UPLO="L"):
    if not (_complex_on_card(A) and _square(A)):
        return False
    cl = _complex_linalg()
    if cl is not None and 1 <= A.shape[-1] <= cl.COMPLEX_MAXN:
        return True
    return cl is None and _real_on_card("eigh", 2 * A.shape[-1])


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
    if not (_complex_on_card(A) and _square(A)):
        return False
    cl = _complex_linalg()
    if cl is not None and 1 <= A.shape[-1] <= cl.COMPLEX_MAXN:
        return True
    if not compute_v:
        return _takes_eigvalsh(A, UPLO)
    return False


def _takes_svd(A, full_matrices=False, compute_uv=True, *, driver=None):
    cl = _complex_linalg()
    if cl is None or driver is not None or not _complex_on_card(A) or A.dim() < 2:
        return False
    m, n = A.shape[-2], A.shape[-1]
    return 1 <= m <= cl.COMPLEX_MAXN and 1 <= n <= cl.COMPLEX_MAXN


def _takes_svd_out(A, full_matrices=False, compute_uv=True, *, driver=None, U, S, Vh):
    if _needs_grad((A, U, S, Vh), {}):
        return False
    if not _takes_svd(A, full_matrices, compute_uv, driver=driver):
        return False
    for out, expected_dtype in ((U, C64), (S, torch.float32), (Vh, C64)):
        if not isinstance(out, torch.Tensor) or out.device != A.device or out.dtype != expected_dtype or _host_backed(out):
            return False
    return True


def _takes_svdvals(A, *, driver=None):
    return _takes_svd(A, False, False, driver=driver)


def _entries():
    return [
        (aten._linalg_solve_ex.default, "solve", _takes_solve, solve_ex),
        (aten.linalg_inv_ex.default, "inv", _takes_inv, inv_ex),
        (aten._linalg_det.default, "det", _takes_det, det),
        (aten._linalg_slogdet.default, "slogdet", _takes_det, slogdet),
        (aten.linalg_solve_triangular.default, "solve_triangular", _takes_tri,
         lambda A, B, *, upper, left=True, unitriangular=False: solve_triangular(A, B, upper, left, unitriangular)),
        (aten.cholesky_solve.default, "cholesky_solve", _takes_cholesky_solve, cholesky_solve),
        (aten.linalg_cholesky_ex.default, "cholesky", _takes_cholesky, cholesky_ex),
        (aten.cholesky_inverse.default, "cholesky_inverse", _takes_cholesky_inverse, cholesky_inverse),
        (aten.linalg_qr.default, "qr", _takes_qr, lambda A, mode="reduced": qr(A, mode)),
        (aten.linalg_eigvalsh.default, "eigvalsh", _takes_eigvalsh, eigvalsh),
        (aten.linalg_matrix_exp.default, "matrix_exp", _takes_matrix_exp, matrix_exp),
        (aten.linalg_householder_product.default, "householder_product", _takes_householder, householder_product),
        (aten.linalg_pinv.atol_rtol_tensor, "pinv", _takes_pinv, pinv),
        (aten._linalg_eigh.default, "eigh", _takes_eigh, eigh),
        (aten._linalg_svd.default, "svd", _takes_svd, svd),
        (aten._linalg_svd.U, "svd", _takes_svd_out, svd_out),
        (aten.linalg_svdvals.default, "svdvals", _takes_svdvals, svdvals),
        (aten.linalg_lstsq.default, "lstsq", _takes_lstsq,
         lambda A, B, rcond=None, *, driver=None: lstsq(A, B, rcond, driver), False),
    ]


_UNDO = None


def install(keys=("AutogradPrivateUse1", "PrivateUse1")):
    """Register the kernels at ``keys`` (complex HPU tensors, grad & inference_mode). Idempotent; ``HPU_COMPLEX_LINALG=0`` keeps them off.
    Returns an undo (also ``uninstall()``)."""
    global _UNDO
    if _UNDO is not None or os.environ.get("HPU_COMPLEX_LINALG", "1") == "0":
        return uninstall
    if isinstance(keys, str):
        keys = (keys,)
    library = torch.library.Library("aten", "IMPL")
    ops = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for op, name, takes, compute, *meta in _entries():
            for key in keys:
                library.impl(op, _kernel(op, name, takes, compute, key, *meta), key)
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


__all__ = ["install", "uninstall", "CALLS", "solve_ex", "inv_ex", "det", "slogdet", "solve_triangular",
           "cholesky_solve", "cholesky_ex", "matrix_exp", "householder_product", "cholesky_inverse", "qr",
           "eigh", "eigvalsh", "svd", "svdvals", "hpd_slogdet", "hpd_logdet"]
