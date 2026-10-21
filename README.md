# Needle — Autograd and Accelerated NDArray

A lightweight tensor and neural-network framework I built to explore automatic differentiation, neural-network training, and efficient array computation on CPUs and GPUs. The project includes a Python tensor API, a reverse-mode autograd engine, neural-network building blocks, optimizers, and C++/CUDA array kernels.

## Features

- Computational graph construction and reverse-mode automatic differentiation
- Tensor arithmetic, broadcasting, reshaping, reductions, and matrix multiplication
- Neural-network layers including linear layers, activations, normalization, dropout, and residual blocks
- Softmax cross-entropy loss, SGD with momentum and gradient clipping, and Adam
- Parameter initialization, datasets, batching, and image transforms
- Multi-dimensional arrays backed by NumPy, native C++, or CUDA
- Optimized elementwise operations, reductions, and tiled matrix multiplication

## Project layout

- `python/needle/` — Tensor API, autograd, operations, layers, optimizers, and data utilities
- `src/` — C++ and CUDA array backends
- `tests/` — Functional smoke tests

## Running the tests

The NumPy tensor backend does not require compiled extensions:

```bash
PYTHONPATH=python NEEDLE_BACKEND=np python -m pytest tests
```

## Native backend setup

To build the native array backends, install CMake, a C++ compiler, and pybind11. CUDA support also requires a CUDA toolkit and compatible GPU/toolchain. Then run:

```bash
make
```

The default `NEEDLE_BACKEND=nd` uses the NDArray implementation, with device selection via `cpu()`, `cuda()`, and `cpu_numpy()`.

## Status

The NumPy-backed functional smoke tests have passed. Native C++ and CUDA backend execution has not been fully validated in the packaged environment.
