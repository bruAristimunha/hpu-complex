import warnings, copy, os
import habana_frameworks.torch.core as htcore, habana_frameworks.torch.hpu as hthpu, torch
from habana_frameworks.torch.hpu.metrics import metric_global
import hpu_ops, hpu_complex
hpu_ops.enable(); hpu_complex.enable()
H = torch.device("hpu"); LAZY = os.environ.get("PT_HPU_LAZY_MODE") == "1"
def sync():
    if LAZY: htcore.mark_step()
    torch.hpu.synchronize()
def fb(): return dict(metric_global("cpu_fallback").stats()).get("TotalNumber", 0)
print("lazy", LAZY)
torch.manual_seed(0)
x = torch.randn(32, 22, 22, dtype=torch.float64).to(H)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    f0 = fb(); ev = torch.linalg.eigvals(x); sync(); f1 = fb()
print("eigvals", ev.dtype, ev.device, ev.shape, "fallback +%d" % (f1 - f0), "maxdiff", (ev.cpu() - torch.linalg.eigvals(x.cpu())).abs().max().item())
print("warnings", sorted({str(m.message)[:90] for m in w}))
z = torch.randn(3, 4, dtype=torch.complex128).to(H); zc = z.cpu()
r = z.real; z.t_(); z.mul_(2); sync()
ref = zc.clone(); rr = ref.real; ref.t_(); ref.mul_(2)
print("t_: values", torch.equal(z.cpu(), ref), "stride", z.stride(), "torch", ref.stride(), "alias kept", torch.equal(r.cpu(), rr))
z = torch.randn(3, 4, dtype=torch.complex128).to(H); r = z.real; z.unsqueeze_(0); z.zero_(); sync()
print("unsqueeze_ then zero_: old .real view zeroed (torch: True)", bool((r.cpu() == 0).all()))
try:
    z = torch.randn(3, 4, dtype=torch.complex128).to(H); d = copy.deepcopy(z); sync()
    print("deepcopy", d.dtype, d.device, torch.equal(d.cpu(), z.cpu()))
except Exception as e:
    print("deepcopy FAIL", type(e).__name__, str(e)[:200])
