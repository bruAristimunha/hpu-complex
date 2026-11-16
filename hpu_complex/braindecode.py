"""Braindecode-on-Gaudi helpers.

patch_braindecode() flips Braindecode's two HPU gates
(functional.spectral_input and functional._real_dft.needs_real_dft)
when this plugin provides on-device FFT, so spectral models
(BIOT, BrainBERT, Brant, CBraMod, LUNA, MAPA, NeuroRVQ, ...)
stop copying every forward to the CPU. Everything here is composed
from the plugin FFT - no new kernels.
"""
import torch

_PATCHED = False


def fft_available():
    """True when on-device complex FFT is live (plugin enabled + HPU)."""
    from . import _REGISTERED

    return bool(_REGISTERED and hasattr(torch, "hpu")
                and torch.hpu.is_available())


def patch_braindecode():
    """Make Braindecode keep spectral work on the HPU. Idempotent.

    Returns True when patched, False when the plugin is not active
    (Braindecode then keeps its CPU fallback, unchanged behavior).
    """
    global _PATCHED
    if _PATCHED:
        return True
    if not fft_available():
        return False
    try:
        from braindecode.functional import _real_dft, functions
    except Exception:
        return False

    def spectral_input_ondevice(x):
        return x.to(torch.promote_types(x.dtype, torch.float32))

    functions.spectral_input = spectral_input_ondevice
    _real_dft.needs_real_dft = lambda device: False
    _PATCHED = True
    return True


def analytic_signal(x, dim=-1):
    """Hilbert analytic signal via FFT (complex64 on HPU).

    Forward-only in practice: gradients through .real/.imag of any
    HPU complex tensor do not flow in this stack (fork autograd view
    tracking); .abs() backward is verified. Feature extraction use
    (Braindecode's Hilbert/PLV front ends) is unaffected.
    """
    n = x.shape[dim]
    f = torch.fft.fft(x.to(torch.float32), dim=dim)
    h = torch.zeros(n, device=x.device, dtype=torch.float32)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1.0
        h[1:n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1:(n + 1) // 2] = 2.0
    shape = [1] * x.dim()
    shape[dim] = n
    return torch.fft.ifft(f * h.reshape(shape), dim=dim)


def ft_surrogate(x, phase, phase_noise_magnitude):
    """Braindecode FTSurrogate math on-device, complex64.

    `phase` is a real phase tensor (radians), broadcastable to the
    rfft spectrum's shape without the last (time) handling of the
    original: this mirrors augmentation.functional.ft_surrogate,
    which multiplies the full FFT by exp(i * magnitude * phase).
    The original computes in float64; this is the complex64 path
    (accuracy decision: fine for augmentation noise).
    """
    f = torch.fft.fft(x.to(torch.float32), dim=-1)
    shifted = torch.fft.ifft(
        f * torch.exp(1j * phase_noise_magnitude * phase), dim=-1)
    return shifted.real.float()


def aggregate_trial_predictions(window_preds, i_window_in_trial, n_trials):
    """On-device replacement for trial_preds_from_window_preds (mean).

    window_preds: [n_windows, n_classes]; i_window_in_trial: long
    trial index per window. Returns [n_trials, n_classes] means;
    only this small tensor ever needs to leave the device.
    """
    out = window_preds.new_zeros((n_trials, window_preds.shape[1]))
    out.index_add_(0, i_window_in_trial, window_preds)
    counts = window_preds.new_zeros((n_trials, 1))
    counts.index_add_(
        0, i_window_in_trial,
        window_preds.new_ones((window_preds.shape[0], 1)))
    return out / counts.clamp_min(1)


def renorm(x, maxnorm, dim=0):
    """torch.renorm(p=2) equivalent, composed (renorm is broken on HPU).

    Scales slices along `dim` whose L2 norm exceeds maxnorm, matching
    the semantics Braindecode's MaxNormParametrize workaround uses.
    """
    dims = tuple(d for d in range(x.dim()) if d != dim)
    norms = torch.linalg.vector_norm(x.float(), dim=dims, keepdim=True)
    scale = (maxnorm / norms.clamp_min(torch.finfo(torch.float32).tiny)
             ).clamp(max=1.0)
    return (x.float() * scale).to(x.dtype)


def spectrogram(x, n_fft, hop_length=None, window=None):
    """torchaudio.transforms.Spectrogram(power=None) equivalent.

    torchaudio keeps its window on CPU on this stack, which makes the
    transform fail with a device mismatch on HPU tensors. This routes
    through torch.stft with the window on x's device instead.
    """
    if window is None:
        window = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
    return torch.stft(
        x, n_fft, hop_length=hop_length, window=window,
        return_complex=True)


def loader_kwargs(num_workers=4):
    """DataLoader kwargs that stop starving the HPU (audit A4).

    pin_memory intentionally omitted: Habana pinned-transfer benefit
    is unverified; workers + persistence + prefetch are the win.
    """
    return {
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
        "prefetch_factor": 2 if num_workers > 0 else None,
    }
