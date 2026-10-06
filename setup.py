"""
Setup configuration for compiling Tiny-Seq2Seq C++ extension module.
"""

import sys
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

extra_compile_args = []
if sys.platform == "win32":
    extra_compile_args = ["/O2", "/std:c++17", "/EHsc", "/arch:AVX2"]
else:
    extra_compile_args = ["-O3", "-std=c++17", "-march=native"]

ext_modules = [
    Pybind11Extension(
        name="seq2seq_c_batcher",
        sources=["cpp/batch_builder.cpp"],
        extra_compile_args=extra_compile_args,
        cxx_std=17,
    ),
]

setup(
    name="tiny_seq2seq",
    version="0.1.0",
    author="Tiny-Seq2Seq Team",
    description="Scaled-down reproduction of Sutskever et al. (2014) Seq2Seq",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.10",
)
