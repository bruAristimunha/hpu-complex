"""Build/load the C++ extension (first call compiles, then cached).

Registers kernels for c10::DispatchKey::HPU (entry points: _fft_r2c,
view_as_complex, view.dtype, complex, _neg_view), AutogradHPU (_fft_r2c and
complex-scalar arithmetic), BackendSelect (complex factories and
complex-scalar fallback), and PrivateUse1 (complex tensors). Builds against
either the Gaudi stack or stock CPU PyTorch (used with a stub guard in the CPU
test suite). The build goes to torch's extension directory
(TORCH_EXTENSIONS_DIR, else ~/.cache/torch_extensions).
"""
import pathlib

_LOADED = False


def load_extension():
    global _LOADED
    if _LOADED:
        return
    from torch.utils.cpp_extension import load

    src = pathlib.Path(__file__).parent / "cpp_ext" / "fft_ext.cpp"
    load(name="hpu_complex_pu1_ext", sources=[str(src)], verbose=False,
         is_python_module=False)
    _note_storage_slice()
    _LOADED = True


def _note_storage_slice():
    """Show the extension the deleter of a Python storage slice (s[a:b]), so set_ follows a slice of a lazy-mode or
    host-memory storage to the storage it was taken of instead of reading it as Habana bytes."""
    import torch

    sliced = torch.UntypedStorage(2)[0:1]
    torch.ops.hpu_complex.note_storage_slice(torch.empty(0, dtype=torch.uint8).set_(sliced))
