"""CPU only (no Gaudi): does the FFT-convolution reference of pu1_check raise MKL's DFTI error by itself?"""
import torch

torch.manual_seed(0)
x = torch.randn(6, 30, 200, requires_grad=True)
kernel = torch.randn(30, 200)
for name, f in (("forward", lambda: torch.fft.irfft(torch.fft.rfft(x, n=400) * torch.fft.rfft(kernel, n=400), n=400)),):
    try:
        y = f()
        print("CPU", name, "OK", tuple(y.shape), flush=True)
        y.sum().backward()
        print("CPU backward OK", float(x.grad.abs().sum()), flush=True)
    except Exception as e:
        print("CPU", name, "ERROR", type(e).__name__, str(e).splitlines()[0][:200], flush=True)
x2 = torch.randn(6, 30, 200, requires_grad=True)
y = torch.fft.irfft(torch.fft.rfft(x2, n=400) * torch.fft.rfft(kernel, n=400), n=400)
try:
    y.backward(torch.ones_like(y))
    print("CPU backward with ones_like grad OK", flush=True)
except Exception as e:
    print("CPU backward ones_like ERROR", str(e).splitlines()[0][:200], flush=True)
