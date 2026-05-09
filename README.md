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

The bug is **load-order dependent**:

| Test | Sequence | Result |
|------|----------|--------|
| A | plain kernel only | ✅ PASS |
| B | EVT kernel only | ✅ PASS |
| **C** | **load plain → load EVT → call plain** | **❌ DEVICE_LOST** |
| D | load EVT → load plain → call plain | ✅ PASS |
| E | load plain → call plain → load EVT → call plain | ✅ PASS |
| F | load plain → load EVT → call EVT | ✅ PASS |

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

### Standalone Reproducer (recommended)

This compiles the exact inductor-generated SYCL kernels as two separate `.so` files
and runs a 6-test load-order matrix. **Works with any PyTorch version** that has XPU
support.

```bash
python standalone_repro_evt.py
```

Options:
- `--no-xs` — Disable IGC backend optimization flags
- `--build-only` — Just compile, don't run tests

**Expected output** on affected hardware:
```
Test Matrix: .so Load Order vs Kernel Call
  A: plain only                                           [PASS]
  B: evt only                                             [PASS]
  C: load plain -> load evt -> call plain  [BUG]          [FAIL]
  D: load evt -> load plain -> call plain                 [PASS]
  E: load plain -> call plain -> load evt -> call plain   [PASS]
  F: load plain -> load evt -> call evt                   [PASS]
```

### Inductor-Level Reproducer

This uses `torch.compile` with CUTLASS EVT epilogue fusion enabled.
**Requires PyTorch with [PR #181854](https://github.com/pytorch/pytorch/pull/181854) merged.**

```bash
python repro_inductor.py
```

**Expected output** on affected hardware:
```
❌ DEVICE_LOST — Bug reproduced!
  Error: level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)
```

## Files

| File | Description |
|------|-------------|
| `standalone_repro_evt.py` | Self-contained reproducer with 6-test load-order matrix |
| `repro_inductor.py` | Inductor-level reproducer via `torch.compile` |
| `_inductor_kernel1_plain.sycl` | Exact inductor-generated plain GEMM SYCL source |
| `_inductor_kernel2_evt.sycl` | Exact inductor-generated EVT GEMM SYCL source |
| `evt_device_lost_analysis.md` | Detailed root cause analysis and debugging history |
| `third_party/sycl-tla/` | CUTLASS for Intel GPUs (git submodule) |

## Workarounds

1. **JIT-on-load**: Call each kernel immediately after `dlopen` (before loading the next `.so`)
2. **Reverse load order**: Load the EVT kernel `.so` before the plain kernel `.so`
3. **Single `.so`**: Combine both kernels into one shared library

## Environment Tested

| Component | Version |
|-----------|---------|
| GPU | Intel Arc Pro B60 (BMG/Xe2, 0xE20B) |
| Level Zero Loader | 1.28.0 |
| GPU Driver | 1.14.37435+12 (NEO 26.09.37435.12) |
| sycl-tla | v0.8 (latest main) |
| PyTorch | 2.13.0a0+git8f75890 / 2.13.0.dev20260506+xpu |
| icpx | oneAPI 2025.3 |

## Related Links

- [PyTorch PR #181854](https://github.com/pytorch/pytorch/pull/181854) — CUTLASS EVT SiLU epilogue fusion for XPU
- [sycl-tla](https://github.com/intel/sycl-tla) — CUTLASS for Intel GPUs
- [PyTorch PR #177612](https://github.com/pytorch/pytorch/pull/177612) — CUTLASS GEMM support for XPU inductor