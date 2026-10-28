
"""Register NVIDIA CUDA DLL directories for this project's Windows venv."""

import os
import sysconfig

_DLL_HANDLES = []

if os.name == "nt":
    site_packages = sysconfig.get_paths()["purelib"]

    dll_directories = [
        os.path.join(site_packages, "nvidia", "cu13", "bin", "x86_64"),
        os.path.join(site_packages, "nvidia", "cu13", "bin"),
        os.path.join(site_packages, "nvidia", "cudnn", "bin"),
    ]

    for directory in dll_directories:
        if os.path.isdir(directory):
            _DLL_HANDLES.append(os.add_dll_directory(directory))
