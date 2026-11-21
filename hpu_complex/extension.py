"""Build/load the C++ extension (first call compiles, then cached).

Must run inside the Gaudi stack (habana torch fork): the extension
registers kernels for the fork's c10::DispatchKey::HPU (entry points),
AutogradHPU (the rfft backward), BackendSelect (complex factories) and
PrivateUse1 (complex tensors). The build goes to torch's default
extension directory (TORCH_EXTENSIONS_DIR, else ~/.cache/torch_extensions).
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
