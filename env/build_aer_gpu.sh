#!/bin/bash
# Build a CUDA-accelerated qiskit-aer wheel from source.
# ------------------------------------------------------------------
# Why: PyPI's `qiskit-aer-gpu` stops at 0.15.1 (qiskit 1.x only); for qiskit 2.x the
# GPU backend must be built from source. Takes only a few minutes.
#
# The prebuilt wheel in this folder (qiskit_aer-0.17.2-cp311-cp311-linux_x86_64.whl)
# targets Python 3.11 / linux x86_64 / sm_86 (RTX 3090) / CUDA 12.x. If that matches
# your machine:  pip install env/qiskit_aer-0.17.2-*.whl  and skip this script.
#
# Known build pitfalls (encoded below):
#   1) pyproject requires conan<2.0.0
#   2) pip's default cmake 4.x FATALs on conan's old deps
#      ("cmake_minimum_required(VERSION 2.8)") -> need cmake<4
#   3) official docs say g++<=8 — outdated; CUDA 12.8 supports gcc 13.x
# ------------------------------------------------------------------
set -e
AER_VERSION=${AER_VERSION:-0.17.2}
CUDA_ARCH=${CUDA_ARCH:-8.6}      # 8.6=RTX30xx, 8.9=RTX40xx, 9.0=H100 (nvidia-smi to check)

pip install "conan<2.0.0" "cmake<4" scikit-build pybind11 ninja
[ -d aer-src ] || git clone --depth 1 -b ${AER_VERSION} \
    https://github.com/Qiskit/qiskit-aer.git aer-src
cd aer-src
python setup.py bdist_wheel -- -DAER_THRUST_BACKEND=CUDA -DAER_CUDA_ARCH=${CUDA_ARCH} --
echo "wheel ready:"
ls dist/*.whl
echo "install with:  pip install dist/qiskit_aer-*.whl"
