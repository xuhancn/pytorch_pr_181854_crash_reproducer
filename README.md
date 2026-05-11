# CUTLASS EVT Multi-.so DEVICE_LOST Reproducer

Reproducer for a Level Zero / SYCL runtime bug that causes `UR_RESULT_ERROR_DEVICE_LOST`
on Intel BMG/Xe2 GPUs (e.g., Arc Pro B60, Arc B580) when two CUTLASS SYCL kernels
are loaded as separate `.so` files via `dlopen`.

**Context**: This bug was discovered while developing
[PyTorch PR #181854](https://github.com/pytorch/pytorch/pull/181854) — CUTLASS EVT
(Epilogue Visitor Tree) SiLU epilogue fusion for XPU inductor. The Llama MLP pattern
`silu(x @ gate_w.T) * (x @ up_w.T)` generates two CUTLASS GEMMs compiled as separate
shared libraries. Loading both into the same process corrupts the first kernel's
SPIR-V device code, causing a GPU crash.

## Root Cause

When two CUTLASS SYCL `.so` files are loaded via `ctypes.CDLL` (`dlopen`), the
Level Zero runtime **invalidates the first `.so`'s un-JIT'd SPIR-V device code**
when loading the second. Calling the first kernel then executes corrupted code
→ `DEVICE_LOST`.

The bug is **load-order dependent** and **Intel Level Zero-specific** (confirmed by
CUDA apple-to-apple testing — see [CUDA comparison](#cuda-comparison) below):

| Test | Sequence | XPU (BMG) | CUDA (RTX 5090 D) |
|------|----------|-----------|-------------------|
| A | plain kernel only | ✅ PASS | ✅ PASS |
| B | EVT kernel only | ✅ PASS | ✅ PASS |
| **C** | **load plain → load EVT → call plain** | **❌ DEVICE_LOST** | **✅ PASS** |
| D | load EVT → load plain → call plain | ✅ PASS | ✅ PASS |
| E | load plain → call plain → load EVT → call plain | ✅ PASS | ✅ PASS |
| F | load plain → load EVT → call EVT | ✅ PASS | ✅ PASS |

Test C exactly matches PyTorch inductor's compilation pattern.

## Prerequisites

- **GPU**: Intel BMG/Xe2 (Arc Pro B60, Arc B580, etc.)
- **Intel oneAPI**: 2025.3+ (`icpx` compiler)
- **Python**: 3.11+
- **PyTorch**: With XPU support (nightly or source build)

## Setup

```bash
git clone --recursive https://github.com/xuhancn/pytorch_pr_181854_crash_reproducer.git
cd pytorch_pr_181854_crash_reproducer

# If you forgot --recursive:
git submodule update --init

# Set up oneAPI environment
source /opt/intel/oneapi/setvars.sh  # or ~/intel/oneapi/setvars.sh

# Install PyTorch with XPU support (nightly)
pip install --pre torch pytorch-triton-xpu --index-url https://download.pytorch.org/whl/nightly/xpu
```

## Usage

### Option 1: Pure C++ Reproducer (no PyTorch)

Zero Python/PyTorch dependency — only requires `icpx` (oneAPI) and the `sycl-tla`
submodule. Builds two CUTLASS kernel `.so` files and a C++ test harness, then runs
the 6-test load-order matrix.

```bash
source /opt/intel/oneapi/setvars.sh  # or ~/intel/oneapi/setvars.sh
./build_and_run.sh
```

Options:
- `--build-only` — Just compile, don't run tests

### Option 2: Python Reproducer (requires PyTorch XPU)

Uses PyTorch for GPU memory allocation and SYCL queue access. Compiles kernels
via `icpx` and tests load order via `ctypes.CDLL`.

```bash
python standalone_repro_evt.py
```

Options:
- `--no-xs` — Disable IGC backend optimization flags
- `--build-only` — Just compile, don't run tests

### Option 3: CUDA Comparison (Apple-to-Apple)

Proves the same dlopen pattern works correctly on NVIDIA CUDA. Requires an NVIDIA GPU
and `nvcc`.

```bash
python test_cuda_multi_so_dlopen.py
```

**Expected output** on any NVIDIA GPU:
```
CUDA Test Matrix: .so Load Order vs Kernel Call
  A: plain only                                           [PASS]
  B: evt only                                             [PASS]
  C: load plain -> load evt -> call plain  [XPU BUG]      [PASS]
  D: load evt -> load plain -> call plain                 [PASS]
  E: load plain -> call plain -> load evt -> call plain   [PASS]
  F: load plain -> load evt -> call evt                   [PASS]

  ✅ ALL 6 TESTS PASS on CUDA
```

### Expected Output (XPU)

On affected Intel hardware, both XPU reproducers produce:

```
Test Matrix: .so Load Order vs Kernel Call
  A: plain only                                           [PASS]
  B: EVT only                                             [PASS]
  C: load plain -> load EVT -> call plain  [BUG]          [FAIL]
  D: load EVT -> load plain -> call plain                 [PASS]
  E: load+call plain -> load EVT -> call plain            [PASS]
  F: load plain -> load EVT -> call EVT                   [PASS]
```

## Files

| File | Description |
|------|-------------|
| `repro_no_pytorch.cpp` | Pure C++ test harness (no PyTorch dependency) |
| `build_and_run.sh` | Build script for C++ reproducer |
| `standalone_repro_evt.py` | Python reproducer with PyTorch XPU |
| `test_cuda_multi_so_dlopen.py` | **CUDA apple-to-apple comparison** (proves CUDA handles multi-.so correctly) |
| `_inductor_kernel1_plain.sycl` | Exact inductor-generated plain GEMM SYCL source |
| `_inductor_kernel2_evt.sycl` | Exact inductor-generated EVT GEMM SYCL source |
| `evt_device_lost_analysis.md` | Detailed root cause analysis and debugging history |
| `cuda_test_results.md` | Full CUDA validation test results (RTX 5090 D, SM120) |
| `third_party/sycl-tla/` | CUTLASS for Intel GPUs (git submodule) |

## Workarounds

1. **JIT-on-load**: Call each kernel immediately after `dlopen` (before loading the next `.so`)
2. **Reverse load order**: Load the EVT kernel `.so` before the plain kernel `.so`
3. **Single `.so`**: Combine both kernels into one shared library

## CUDA Comparison

The same 6-test load-order matrix was run on NVIDIA CUDA (RTX 5090 D, SM120) as an
apple-to-apple comparison. **All 6 tests pass on CUDA**, confirming the DEVICE_LOST
crash is specific to Intel Level Zero / SYCL runtime, not a general GPU driver issue.

See [cuda_test_results.md](cuda_test_results.md) for full details including environment
setup, EVT unit test results, and SM120 architecture analysis.

## Environment Tested

| Component | Version |
|-----------|---------|
| GPU (XPU) | Intel Arc Pro B60 (BMG/Xe2, 0xE20B) |
| GPU (CUDA) | NVIDIA GeForce RTX 5090 D (SM120, Blackwell) |
| Level Zero Loader | 1.28.0 |
| GPU Driver (Intel) | 1.14.37435+12 (NEO 26.09.37435.12) |
| CUDA Toolkit | 13.0 |
| sycl-tla | v0.9 (latest main) |
| PyTorch (XPU) | 2.13.0a0+git8f75890 / 2.13.0.dev20260506+xpu |
| PyTorch (CUDA) | 2.12.0.dev20260407+cu128 |
| icpx | oneAPI 2025.3 |

## Detailed Analysis

See [evt_device_lost_analysis.md](evt_device_lost_analysis.md) for the full root cause
analysis, inductor compilation model details, debugging history, and `TORCH_LOGS`
instructions.

## Related Links

- [PyTorch PR #181854](https://github.com/pytorch/pytorch/pull/181854) — CUTLASS EVT SiLU epilogue fusion for XPU
- [sycl-tla](https://github.com/intel/sycl-tla) — CUTLASS for Intel GPUs
- [PyTorch PR #177612](https://github.com/pytorch/pytorch/pull/177612) — CUTLASS GEMM support for XPU inductor