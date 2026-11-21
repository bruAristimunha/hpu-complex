"""The C++ kernels on the CPU stand-in (cpu_stub/hpu_guard_stub.cpp): the second review of PR #11 (CODEX_REVIEW.txt,
Part B), each case with the review's own inputs and stock torch (or the exact complex128 value) as the reference.

Lazy-mode cases set the extension's lazy flag: the CPU stand-in runs the lazy storages (a storage whose context holds
the Habana tensor) on fake HPU tensors, so their metadata and guards are checked here; Habana's lazy graphs are not
(lazy_check.py on the card). Single-element tensors keep torch's scalar CPU kernels as the reference on every
platform (its vectorized complex kernels place NaNs differently on x86 and ARM).

python -m pytest hpu_complex/tests/test_review2_cpu.py -q
"""
import contextlib
import io
import math
import pathlib
import subprocess
import sys
import warnings

import pytest
import torch

from hpu_complex.tests.test_complex128_cpu import HPU, extension, tagged  # noqa: F401
from hpu_complex.tests.test_review_fixes_cpu import bits_equal, h

C64, C128, F64 = torch.complex64, torch.complex128, torch.float64
INF, NAN = math.inf, math.nan
REPO = pathlib.Path(__file__).resolve().parents[2]


@contextlib.contextmanager
def lazy_flag():
    torch.ops.hpu_complex.set_lazy_mode(True)
    try:
        yield
    finally:
        torch.ops.hpu_complex.set_lazy_mode(False)


def outcome(fn):
    """('ok', value) or ('error', first line of the message)."""
    try:
        return "ok", fn()
    except RuntimeError as e:
        return "error", str(e).splitlines()[0]


# ---- B1 (Critical) a failed set_ installs nothing; a singleton negative stride is accepted as torch does ----------


@pytest.mark.parametrize("dtype", [C64, C128])
def test_set_with_a_singleton_negative_stride_is_torchs(dtype):
    """Stock accepts the stride of a size-1 dim, whatever its sign, and bumps the version (the saved x is stale)."""
    x = h([2], dtype).requires_grad_()
    loss = (x * x).real.sum()
    with torch.no_grad():
        x.set_(h([7], dtype).untyped_storage(), 0, (1,), (-1,))  # main: raised after installing the storage
    assert x.cpu().item() == 7 and x._version == 1
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        loss.backward()  # main: gradient 14 from the new storage, no error


@pytest.mark.parametrize("dtype", [C64, C128])
@pytest.mark.parametrize("offset,stride,match", [(0, (-1,), "Negative strides"), (-1, (1,), "invalid storage offset")])
def test_failed_set_leaves_self_and_a_saved_tensor_unchanged(dtype, offset, stride, match):
    x = h([2, 3], dtype).requires_grad_()
    loss = (x * x).real.sum()
    storage = h([7, 8, 9], dtype).untyped_storage()
    with torch.no_grad(), pytest.raises(RuntimeError, match=match):
        x.set_(storage, offset, (2,), stride)
    assert x._version == 0 and x.untyped_storage().nbytes() == 2 * x.element_size()
    assert torch.equal(x.detach().cpu(), torch.tensor([2, 3], dtype=dtype))
    loss.backward()
    assert torch.equal(x.grad.cpu(), torch.tensor([4, 6], dtype=dtype))


# ---- B2 (Critical) a storage slice does not hide the lazy owner -------------------------------------------------


@pytest.mark.parametrize("piece", [slice(0, 0), slice(None), slice(8, 16)])
@pytest.mark.parametrize("dtype", [C64, C128])
def test_set_onto_a_slice_of_a_lazy_storage_raises(piece, dtype):
    with lazy_flag():
        s = h([1j, 2j]).untyped_storage()
        assert s[piece].nbytes() == len(range(16)[piece])
        with pytest.raises(RuntimeError, match="slice of a complex64 HPU storage made in Habana lazy mode"):
            h([], dtype).set_(s[piece])  # main: accepted (an owner-context address read as device bytes)


def test_set_onto_a_slice_of_host_memory_raises():
    s = h([1j, 2j], C128).untyped_storage()
    for dtype in (C64, C128):
        with pytest.raises(RuntimeError, match="slice of a complex128 HPU storage"):
            h([], dtype).set_(s[16:32])  # main: host memory taken for Habana storage


def test_set_onto_a_whole_storage_still_aliases():
    z = h([1j, 2j])
    t = torch.empty(0, dtype=C64, device=HPU).set_(z.untyped_storage())
    z[1] = 5
    assert t.cpu()[1] == 5


# ---- B3 (High) lazy growth counts the owner's bytes; siblings share the storage ---------------------------------


def test_lazy_growth_through_a_half_owner_raises_before_changing_anything():
    with lazy_flag():
        r = h([0] * 8, torch.float16)
        v = r.view(C64)
        with warnings.catch_warnings(), pytest.raises(RuntimeError, match="no float view of a .*Half storage"):
            warnings.simplefilter("ignore")
            torch.neg(h([1j] * 8), out=v)
        assert v.shape == (2,) and v._version == 0  # main: shape (8,), 64 bytes over a 32-byte owner
        assert v.untyped_storage().nbytes() == r.untyped_storage().nbytes() == 16


def test_lazy_growth_is_seen_by_a_sibling_over_the_same_owner():
    with lazy_flag():
        r = h([0.0] * 4, torch.float32)
        a, b = torch.view_as_complex(r.view(2, 2)), torch.view_as_complex(r.view(2, 2))
        assert a.untyped_storage().nbytes() == b.untyped_storage().nbytes() == 16
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.neg(h([1j, 2j, 3j, 4j]), out=a)
        assert a.untyped_storage().nbytes() == b.untyped_storage().nbytes() == 32  # main: b kept 16
        assert torch.equal(a.cpu(), torch.tensor([-1j, -2j, -3j, -4j]))


# ---- B4 (High) a real self of an in-place op keeps torch's cast check -------------------------------------------


@pytest.mark.parametrize("op", ["add_", "mul_"])
def test_float64_self_with_a_complex_operand_raises(op):
    x = h([2], F64)
    with pytest.raises(RuntimeError, match="ComplexDouble can't be cast to the desired output type Double"):
        getattr(x, op)(h([1 + 1j]))  # main: float64 [3], the imaginary part dropped
    assert x.cpu().item() == 2 and x._version == 0


def test_complex32_result_raises_instead_of_complex64():
    assert torch.result_type(torch.tensor(1j), torch.ones(1, dtype=torch.float16)) == torch.complex32
    with pytest.raises(RuntimeError, match="ComplexHalf"):
        h(1j) + h([1], torch.float16)  # main: complex64


# ---- B5 (High) vector_norm keeps every representable contribution; NaN placement -------------------------------


@pytest.mark.parametrize("values,ord", [([1e20, 1e-30], 0.01), ([1e30, 1e-20], -0.01), ([1e20, 1e-30], 0.5),
                                        ([1e30, 1e-20], -0.5), ([3e38, 1e-45 + 1e-45j], 0.3), ([1, 2, 3], 0.7),
                                        ([1e-30, 2e-30], 0.25), ([0j, 1 + 1j], -0.5), ([1, 2 + 1j], -0.99)])
def test_vector_norm_small_orders_keep_small_terms(values, ord):
    """|ord| < 1: a term (|z| / M)^ord that is representable survives even when |z| / M under- or overflows. The final
    ^(1 / ord) multiplies relative errors by 1 / |ord|, in torch's float32 kernel as here."""
    z = torch.tensor(values, dtype=C64)
    got = torch.linalg.vector_norm(z.to(HPU), ord).cpu()
    ref = torch.linalg.vector_norm(z.to(C128), ord).float()
    torch.testing.assert_close(got, ref, rtol=4e-6 / abs(ord), atol=0)  # main: 1e20 for the first (8.57e31)


@pytest.mark.parametrize("values", [[INF, NAN], [complex(INF, NAN), NAN], [complex(INF, NAN), 1.0],
                                    [complex(NAN, 1), complex(1, INF)], [NAN, 1.0], [INF, 1.0]])
@pytest.mark.parametrize("ord", [2, 1, 3.0, 0.5, -1, INF, -INF, 0])
def test_vector_norm_nan_placement(values, ord):
    """Per element |z| = hypot (inf beats NaN within an element); across elements a NaN modulus makes the norm NaN."""
    z = torch.tensor(values, dtype=C64)
    got = torch.linalg.vector_norm(z.to(HPU), ord).cpu()
    ref = torch.linalg.vector_norm(z.abs().double(), ord).float()  # torch's rule on the exact moduli
    assert torch.equal(got.isnan(), ref.isnan()) and (ref.isnan() | (got == ref)).all(), (got, ref)


def test_vector_norm_nan_placement_over_a_dim():
    z = torch.tensor([[INF, NAN], [complex(INF, NAN), 1.0]], dtype=C64)
    got = torch.linalg.vector_norm(z.to(HPU), 2, dim=1).cpu()
    assert got[0].isnan() and got[1] == INF, got  # main: [inf, inf]


# ---- B6 (High) cat.out casts each input to out's dtype; an aliased real view raises ----------------------------


def test_cat_out_casts_each_input_before_concatenating():
    out = h([0, 0])
    torch.cat([h([4097], torch.int32), h([1], torch.float16)], out=out)
    assert torch.equal(out.cpu(), torch.tensor([4097, 1], dtype=C64))  # main: 4096 (concatenated in float16)


@pytest.mark.parametrize("view", [lambda z: z.view(torch.float32), lambda z: torch.view_as_real(z).flatten(),
                                  lambda z: z.view(torch.float32)[1:3]])
def test_cat_out_over_a_real_view_of_itself_raises(view):
    z = h([1 + 2j, 3 + 4j])
    with pytest.raises(RuntimeError, match="single memory location|refer to any of the output"):
        torch.cat([view(z)], out=z)  # main: grew z and overwrote it
    assert torch.equal(z.cpu(), torch.tensor([1 + 2j, 3 + 4j]))


def test_elementwise_out_over_a_real_view_of_itself_raises():
    z = h([1 + 2j, 3 + 4j, 5j])
    with pytest.raises(RuntimeError, match="single memory location"):
        torch.mul(z.view(torch.float32)[2:4], h([1j, 1j]), out=z[:2])  # main: the untagged real view unchecked


# ---- B7 (High) a compiled card graph never receives host memory --------------------------------------------------


def test_compiled_card_graph_is_not_reused_for_a_host_backed_input():
    import hpu_complex
    from torch._dynamo.backends import registry

    calls = []

    def recording_backend(gm, example_inputs, **kwargs):
        def compiled(*args):
            calls.append([hpu_complex._host_backed(a) for a in args if isinstance(a, torch.Tensor)])
            return gm.forward(*args)
        return compiled

    saved = registry._COMPILER_FNS.get("hpu_backend"), registry._BACKENDS.get("hpu_backend")
    registry._COMPILER_FNS["hpu_backend"] = recording_backend
    registry._BACKENDS.setdefault("hpu_backend", None)
    try:
        hpu_complex._complex128_graphs_run_eagerly()
        torch._dynamo.reset()
        f = torch.compile(lambda z: z.view(torch.float32) + 1, backend="hpu_backend")
        card, host = h([1, 2]), h([7], C128).view(C64)
        assert card.shape == host.shape and card.stride() == host.stride()
        assert torch.equal(f(card).cpu(), card.cpu().view(torch.float32) + 1)
        assert calls == [[False]], calls  # the card call reached the compiled graph
        assert torch.equal(f(host).cpu(), host.cpu().view(torch.float32) + 1)
        assert calls == [[False]], calls  # main: [[False], [True]], the host input in the graph
        assert torch.equal(f(card).cpu(), card.cpu().view(torch.float32) + 1)
        assert calls == [[False], [False]], calls  # and card calls still do afterwards
    finally:
        torch._dynamo.reset()
        for table, entry in zip((registry._COMPILER_FNS, registry._BACKENDS), saved):
            if entry is None:
                table.pop("hpu_backend", None)
            else:
                table["hpu_backend"] = entry


# ---- B8 (High) device checks cover written tensors and masks -----------------------------------------------------


@pytest.mark.parametrize("call", [
    lambda z: h([1], C128).add_(h([2], C128).to("hpu:1")),
    lambda z: h([1], C128).mul_(h([2], C128).to("hpu:1")),
    lambda z: h([1], C128).to("hpu:1").add_(h([2])),
    lambda z: torch.add(z, z, out=z.to("hpu:1")),
    lambda z: torch.mul(z, 2, out=z.to("hpu:1")),
    lambda z: torch.cat([z], out=z.to("hpu:1")),
    lambda z: torch.cat([z, z.to("hpu:1")]),
    lambda z: torch.where(torch.tensor([True]).to("hpu:1"), z, z),
    lambda z: torch.where(torch.tensor([True]).to("hpu:1"), z.to(C128), z.to(C128)),
])
def test_mixed_devices_raise_for_written_tensors_and_masks(call):
    with pytest.raises(RuntimeError, match="same device"):
        call(h([1 + 1j]))


def test_cpu_scalar_masks_and_operands_still_pass():
    """A CPU mask of any shape stays accepted (copied to the card; README Limits): pu1_check's rank-6 rows use one."""
    z = h([1 + 1j, 2])
    assert torch.equal(torch.where(torch.tensor(True), z, z).cpu(), z.cpu())
    assert torch.equal(torch.where(torch.tensor([True, False]), z, 0).cpu(), torch.tensor([1 + 1j, 0]))
    assert torch.equal(torch.where(z.abs() > 1.5, z, 0).cpu(), torch.tensor([0, 2], dtype=C64))
    assert torch.equal(torch.mul(z, torch.tensor(2j), out=h([0, 0])).cpu(), z.cpu() * 2j)


# ---- B9 (High) division by a real tensor and a complex alpha follow torch's NaN placement -----------------------

# NaN placement and values, not zero signs or the last ulp (README Limits: the per-part quotient plus torch's cross
# terms, kept for speed). Not a zero divisor next to an infinite part (NaN here, inf in torch), nor a subnormal
# divisor whose 1 / c overflows in torch (its 0 * inf cross term is NaN: (1 - 0j) / 1e-45).
REAL_DIV = [(complex(INF, 1), 2.0), (complex(1, INF), -3.0), (complex(NAN, 1), 2.0), (1 + 1j, 0.0),
            (complex(INF, 1), -2.0), (complex(-INF, INF), 3.0), (complex(-0.0, 1), -2.0), (3 + 4j, INF),
            (3 + 4j, -INF), (1 + 1j, NAN), (1e30 + 1e-30j, 1e-10), (1 + 2j, 3.0)]


@pytest.mark.parametrize("a,b", REAL_DIV)
def test_complex_over_real_tensor_like_torch(a, b):
    ta, tb = torch.tensor([a], dtype=C64), torch.tensor([b])
    got = torch.view_as_real((ta.to(HPU) / tb.to(HPU)).cpu())
    ref = torch.view_as_real(ta / tb)
    assert torch.equal(got.isnan(), ref.isnan()), (got, ref)  # main: inf + 0.5j for the first, torch inf + nan j
    torch.testing.assert_close(got[~ref.isnan()], ref[~ref.isnan()], rtol=1.2e-7, atol=0)


@pytest.mark.parametrize("c", [complex(INF, 0), complex(NAN, 0), complex(-0.0, -0.0), 1 + 2j])
@pytest.mark.parametrize("alpha", [1 + 0j, -1 + 0j, 2.5 + 0j])
def test_complex_alpha_with_zero_imaginary_part_is_a_complex_product(c, alpha):
    z, w = torch.tensor([1 + 1j], dtype=C64), torch.tensor([c], dtype=C64)
    bits_equal(torch.add(z.to(HPU), w.to(HPU), alpha=alpha), torch.add(z, w, alpha=alpha))  # main: inf + 1j
    bits_equal(torch.sub(z.to(HPU), w.to(HPU), alpha=alpha), torch.sub(z, w, alpha=alpha))
    x = torch.tensor([INF])
    bits_equal(torch.add(z.to(HPU), x.to(HPU), alpha=alpha), torch.add(z, x, alpha=alpha))


def test_real_alpha_pair_arithmetic_is_the_documented_exception():
    """README Limits: two complex64 card tensors with a real alpha add per part (torch's alpha b is a complex product,
    inf + 0i in b gives inf + nan i there). A regression to the exact form would be slower, not wrong: pinned here."""
    z, w = h([1 + 1j]), h([complex(INF, 0)])
    got = torch.view_as_real(torch.add(z, w, alpha=1).cpu())
    assert got[0, 0] == INF and got[0, 1] == 1


# ---- B10 (High) reductions into out= compute in out's dtype; neg's out dtype is exact ------------------------------


def test_sum_into_a_real_out_writes_the_real_part():
    out, ref = h(0, torch.float32), torch.tensor(0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.sum(h([1 + 3j, 2 - 1j]), 0, out=out)  # main raised
        torch.sum(torch.tensor([1 + 3j, 2 - 1j]), 0, out=ref)
    assert out.cpu().item() == ref.item() == 3


def test_sum_into_a_complex128_out_sums_in_complex128():
    out = h(0, C128)
    torch.sum(h([2 ** 24, 1, -2 ** 24]), 0, out=out)
    assert out.cpu().item() == 1  # main: 0 (summed in complex64 first)


def test_sum_dtype_and_out_must_agree():
    with pytest.raises(RuntimeError, match="Expected out tensor to have dtype"):
        torch.sum(h([1 + 3j]), 0, dtype=C128, out=h(0j))


def test_neg_out_dtype_must_match_exactly():
    with pytest.raises(RuntimeError, match="Found dtype ComplexDouble but expected ComplexFloat"):
        torch.neg(h([1j]), out=h([0], C128))  # main: accepted
    with pytest.raises(RuntimeError, match="Found dtype ComplexFloat but expected ComplexDouble"):
        torch.neg(h([1j], C128), out=h([0]))


# ---- B11 (High) checkpoints: the map_location callback's storage, the legacy format, bfloat16 views ------------

_WRITER = """
import sys, torch
sys.path.insert(0, {repo!r})
exec(open({env!r}).read())
from hpu_complex import _pickle_complex128_through_the_cpu
_pickle_complex128_through_the_cpu()
z = torch.tensor([[1 + 1e-12j, 2, 3], [4, 5, 6 - 1e-300j]], dtype=torch.complex128).to(torch.device("hpu"))
obj = {{"z": z, "t": z.mT, "bf": torch.tensor([1j], dtype=torch.complex128).to("hpu").view(torch.bfloat16)}}
torch.save(obj, {path!r})
torch.save(obj, {legacy!r}, _use_new_zipfile_serialization=False)
"""

_READER = """
import sys, torch
sys.path.insert(0, {repo!r})
import hpu_complex  # import alone (no enable()): the rebuilder is allowed and the legacy format sees map_location
ref = torch.tensor([[1 + 1e-12j, 2, 3], [4, 5, 6 - 1e-300j]], dtype=torch.complex128)
bf = torch.tensor([1j], dtype=torch.complex128).view(torch.bfloat16)
for path in ({path!r}, {legacy!r}):
    got = torch.load(path, map_location="cpu")  # weights_only=True, torch 2.6's default, both formats
    assert got["z"].device.type == "cpu" and torch.equal(got["z"], ref), got
    assert got["t"].stride() == ref.mT.stride() and torch.equal(got["t"], ref.mT)
    assert torch.equal(got["bf"].view(torch.int16), bf.view(torch.int16)), got["bf"]
    seen = []
    def zeros(storage, location):
        seen.append(location)
        return torch.UntypedStorage(storage.nbytes()).fill_(0)
    cb = torch.load(path, map_location=zeros)
    assert seen and "hpu" in seen[0], seen
    assert torch.equal(cb["z"], torch.zeros_like(ref)) and torch.equal(cb["t"], torch.zeros_like(ref.mT)), cb
    assert cb["t"].stride() == ref.mT.stride()
print("READ_OK")
"""


def test_checkpoint_restore_honors_the_callback_and_the_legacy_format(tmp_path):
    from hpu_complex.tests.test_review_fixes_cpu import _ENV

    path, legacy = str(tmp_path / "zip.pt"), str(tmp_path / "legacy.pt")
    env = tmp_path / "env.py"
    env.write_text(_ENV.format(stub=str(REPO / "hpu_complex/tests/cpu_stub/hpu_guard_stub.cpp")))
    run = dict(check=False, capture_output=True, text=True, timeout=1200, cwd=tmp_path)
    w = subprocess.run([sys.executable, "-c", _WRITER.format(repo=str(REPO), env=str(env), path=path, legacy=legacy)],
                       **run)
    assert w.returncode == 0, w.stderr[-3000:]  # main: bfloat16 .numpy() raised
    r = subprocess.run([sys.executable, "-c", _READER.format(repo=str(REPO), path=path, legacy=legacy)], **run)
    assert r.returncode == 0 and "READ_OK" in r.stdout, r.stderr[-3000:]  # main: the callback's zeros were ignored


def test_checkpoint_callback_storage_in_process():
    import hpu_complex

    hpu_complex._pickle_complex128_through_the_cpu()
    z = h([1 + 2j, 3], C128)
    buf = io.BytesIO()
    torch.save(z, buf)
    buf.seek(0)
    got = torch.load(buf, map_location=lambda s, loc: torch.UntypedStorage(s.nbytes()).fill_(0))
    assert got.device.type == "cpu" and torch.equal(got, torch.zeros(2, dtype=C128))
    buf.seek(0)
    back = torch.load(buf, map_location=lambda s, loc: None)
    assert hpu_complex._host_backed(back) and torch.equal(back.cpu(), z.cpu())


# ---- B12 (Medium) alpha validation of the complex pair kernel ------------------------------------------------------


@pytest.mark.parametrize("call", [lambda z: z.add_(z, alpha=True), lambda z: torch.add(z, z, alpha=1e39),
                                  lambda z: torch.sub(z, z, alpha=True), lambda z: z.sub_(z, alpha=-1e39),
                                  lambda z: torch.add(z, torch.tensor([1.0, 2.0]).to(z.device), alpha=True)])
def test_alpha_checks_like_torch(call):
    ref, got = h([2, 3]).cpu(), h([2, 3])
    expected = outcome(lambda: call(ref))
    assert expected[0] == "error"
    assert outcome(lambda: call(got))[1] == expected[1]  # main: [4, 6] written for alpha=True, inf for 1e39
    assert torch.equal(got.cpu(), torch.tensor([2, 3], dtype=C64)) and got._version == 0


# ---- B14 (Medium) the fake backend keeps a resized out='s strides ------------------------------------------------


def test_stub_out_resize_keeps_the_kernels_strides():
    x = torch.tensor([[1, 0, 2], [0, 3, 0]], dtype=torch.float32)
    got = torch.nonzero(x.to(HPU), out=torch.zeros(6, dtype=torch.int64).to(HPU))
    ref = torch.nonzero(x, out=torch.zeros(6, dtype=torch.int64))
    assert torch.equal(got.cpu(), ref), got.cpu()  # main: [[0, 0], [1, 0], [2, 1]]


# ---- B16 (Low) / B17 (Low) mutation-sensitive variants of earlier tests ------------------------------------------


def test_complex64_inplace_with_float64_rounds_once():
    """From -1: the float64 2**-25 + 1 rounded to float32 first (r.float()) differs from torch's complex128 sum by
    2**-25, where the earlier test's input could not tell the two apart."""
    z, r = torch.tensor([-1 + 0.5j], dtype=C64), torch.tensor([2 ** -25 + 1], dtype=F64)
    got = z.to(HPU)
    got.add_(r.to(HPU))
    ref = z.clone().add_(r)
    assert torch.equal(got.cpu(), ref) and not torch.equal(ref, z.clone().add_(r.float()))


# ---- adversarial pass: a real HPU tensor and a complex scalar through the out= overloads ------------------------


@pytest.mark.parametrize("call", [
    lambda x, o: torch.ops.aten.mul.Scalar_out(x, 2j, out=o), lambda x, o: torch.ops.aten.add.Scalar_out(x, 2j, alpha=3, out=o),
    lambda x, o: torch.ops.aten.sub.Scalar_out(x, 2j, alpha=1, out=o), lambda x, o: torch.ops.aten.div.Scalar_out(x, 2j, out=o),
    lambda x, o: torch.mul(x, torch.tensor(2j), out=o)])
def test_complex_scalar_out_forms_compute_on_the_hpu_side(call):
    """Habana's mul.out of x_hpu * tensor(2j) reaches the pair kernels with a CPU 0-d complex operand: the value is
    computed on x's device (main computed it on the CPU and copied it in; the out= device check now raises for that)."""
    x = torch.linspace(-1, 1, 6).reshape(2, 3)
    got, ref = torch.empty(2, 3, dtype=C64, device=HPU), torch.empty(2, 3, dtype=C64)
    call(x.to(HPU), got)
    call(x, ref)
    bits_equal(got, ref)


# ---- final round of PR #15's review --------------------------------------------------------------------------


def test_lazy_storage_follows_the_owners_growth():
    """A complex tensor made over a Habana storage that grew since an earlier one (r.resize_) gets the cached lazy
    storage at the new size: main reported 16 bytes over a 32-byte owner (set_ raised, storage().cpu() truncated)."""
    with lazy_flag():
        r = h([0.0] * 4, torch.float32)
        a = torch.view_as_complex(r.view(2, 2))
        r.resize_(8)
        c = torch.view_as_complex(r.view(4, 2))
        assert c.untyped_storage().nbytes() == a.untyped_storage().nbytes() == 32
        t = torch.empty(0, dtype=C64, device=HPU).set_(c.untyped_storage(), 0, (4,))
        assert t.shape == (4,)


def test_lazy_cat_out_over_an_untagged_real_view_raises():
    """Lazy mode: z.view(float32) is Habana's own view of the storage's Habana tensor, untagged and over the Habana
    storage, not z's lazy storage: check_lazy_overlap's untagged-source branch."""
    with lazy_flag():
        z = h([1 + 2j, 3 + 4j])
        v = z.view(torch.float32)
        assert not tagged(v) and not v.untyped_storage()._cdata == z.untyped_storage()._cdata
        with pytest.raises(RuntimeError, match="single memory location|refer to any of the output"):
            torch.cat([v], out=z)
        assert torch.equal(z.cpu(), torch.tensor([1 + 2j, 3 + 4j]))


@pytest.mark.parametrize("call", [lambda a, b: torch.add(a, b), lambda a, b: torch.mul(a, b),
                                  lambda a, b: torch.sub(a, b), lambda a, b: torch.div(a, b),
                                  lambda a, b: torch.where(b > 0, a, b)])
def test_complex32_result_raises_on_the_host_path_too(call):
    """A 0-d complex128 HPU tensor with a float16 vector is complex32 (torch.result_type): raises like the card path
    (main returned a complex32 HPU tensor from the host path)."""
    with pytest.raises(RuntimeError, match="ComplexHalf"):
        call(h(1j, C128), h([1, 2], torch.float16))


@pytest.mark.parametrize("values,ord", [([1e-30] * 3, 0.01), ([1e30] * 3, -0.01), ([1e-30] * 10000, 0.1),
                                        ([1e-20] * 3, 0.01), ([1.0] * 3, 0.01)])
def test_vector_norm_small_orders_rescale_before_overflow(values, ord):
    """sum^(1 / ord) overflowed float32 before the rescale (inf for 3 x 1e-30 at ord 0.01, torch's exact 5.15e17)."""
    z = torch.tensor(values, dtype=C64)
    got = torch.linalg.vector_norm(z.to(HPU), ord).cpu()
    ref = torch.linalg.vector_norm(z.to(C128), ord).float()
    torch.testing.assert_close(got, ref, rtol=4e-6 / abs(ord), atol=0)


@pytest.mark.parametrize("ord", [0.5, 0.25, -0.5, 0.7])
def test_vector_norm_small_orders_keep_float32_accuracy(ord):
    """The direct power wherever it fits: the log form alone was 2.7e-6 off at ord 0.5 (pu1_check's 1e-6 row)."""
    z = torch.tensor([1e-30, 3e19 + 3e19j, 1e-25j], dtype=C64)
    got = torch.linalg.vector_norm(z.to(HPU), ord).cpu()
    ref = torch.linalg.vector_norm(z.to(C128), ord).float()
    torch.testing.assert_close(got, ref, rtol=1e-6, atol=0)


def test_vector_norm_small_order_gradient_at_zero_stays_finite():
    for o in (0.5, 0.01, -0.5):
        b = h([0j, 0j]).requires_grad_()
        torch.linalg.vector_norm(b, o).backward()
        assert torch.equal(b.grad.cpu(), torch.zeros(2, dtype=C64)), (o, b.grad)
