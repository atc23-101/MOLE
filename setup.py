import setuptools
from setuptools import setup, Extension
from torch.utils import cpp_extension
import os

extensions = []

extensions.append(
        cpp_extension.CUDAExtension(
            name = "MOLE.native_atn_wrapper",
            sources=[
                "MOLE/cpp/native_atn_wrapper.cpp",
            ],
            extra_compile_args=[]
        )
)

        
extensions.append(
        cpp_extension.CUDAExtension(
            name = "MOLE.core",
            sources=[
                "MOLE/cpp/core.cpp",
                "MOLE/cpp/memory.cpp",
                "MOLE/cpp/parameter.cpp",
                "MOLE/cpp/data_transfer.cpp",
                "MOLE/cpp/transfer.cu",
                "MOLE/cpp/stream.cpp",
                "MOLE/cpp/communication.cpp",
                "MOLE/cpp/gelu.cu",
                "MOLE/cpp/gelubwd.cpp"

            ],
            extra_compile_args=["-O2"]
        )
)

cmdclass = {}
cmdclass["build_ext"] = cpp_extension.BuildExtension 

setup(
    name = "MOLE",
    packages = setuptools.find_packages(),
    version = "0.1.0",
    author="NONE",
    author_email="NONE",
    description="Mixture of expert OffLoadEr",
    ext_modules=extensions,
    cmdclass=cmdclass,
)

