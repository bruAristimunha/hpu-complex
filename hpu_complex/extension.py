"""Build/load the C++ extension (first call compiles, then cached).

Must run inside the Gaudi stack (habana torch fork): the extension
registers kernels for the fork's c10::DispatchKey::HPU, which stock
torch headers do not have.
"""
import pathlib

_LOADED = False


def load_extension():
    global _LOADED
    if _LOADED:
        return
    import torch
    from torch.utils.cpp_extension import load

    src = pathlib.Path(__file__).parent / "cpp_ext" / "fft_ext.cpp"
    build_dir = pathlib.Path("/tmp/hpu_complex_ext_build")
    build_dir.mkdir(exist_ok=True)
    load(name="hpu_complex_ext", sources=[str(src)],
         build_directory=str(build_dir), verbose=False,
         is_python_module=False)
    _LOADED = True
