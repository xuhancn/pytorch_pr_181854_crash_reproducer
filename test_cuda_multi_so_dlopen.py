#!/usr/bin/env python3
"""
CUDA apple-to-apple comparison for the multi-.so dlopen DEVICE_LOST bug.

This script mirrors the XPU standalone reproducer (standalone_repro_evt.py) but
uses CUDA kernels compiled with nvcc. It proves that the CUDA driver correctly
handles multiple .so files loaded via dlopen — the same pattern that causes
UR_RESULT_ERROR_DEVICE_LOST on Intel Level Zero / SYCL runtime.

The test uses two CUDA .so files containing __global__ kernels of different
complexity (mimicking the plain GEMM vs EVT GEMM pattern), and runs the same
6-test load-order matrix used in the XPU reproducer.

Prerequisites:
    - NVIDIA GPU (SM70+)
    - CUDA toolkit (nvcc)
    - PyTorch with CUDA support

Usage:
    python test_cuda_multi_so_dlopen.py            # Run full test matrix
    python test_cuda_multi_so_dlopen.py --build-only  # Just compile
"""

import os
import sys
import subprocess
import ctypes
import shutil
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(SCRIPT_DIR, "_build_cuda")

# ─── CUDA Kernel Sources ─────────────────────────────────────────────────
# Two separate kernel sources that will be compiled into separate .so files.
# We use template-heavy code to generate large .so files (~100KB+), similar
# to the CUTLASS kernels on XPU that trigger the Level Zero bug.

KERNEL1_SOURCE = r'''
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Template-heavy GEMM-like kernel to produce non-trivial device code
template <int TILE_M, int TILE_N, int TILE_K>
__global__ void gemm_kernel_plain(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K)
{
    int row = blockIdx.y * TILE_M + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x;

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; k += TILE_K) {
            #pragma unroll
            for (int kk = 0; kk < TILE_K && (k + kk) < K; ++kk) {
                acc += __half2float(A[row * K + k + kk]) *
                       __half2float(B[(k + kk) * N + col]);
            }
        }
        C[row * N + col] = __float2half(acc);
    }
}

// Instantiate multiple template variants to increase .so size
template __global__ void gemm_kernel_plain<16, 16, 8>(const half*, const half*, half*, int, int, int);
template __global__ void gemm_kernel_plain<32, 32, 8>(const half*, const half*, half*, int, int, int);
template __global__ void gemm_kernel_plain<16, 16, 16>(const half*, const half*, half*, int, int, int);
template __global__ void gemm_kernel_plain<32, 32, 16>(const half*, const half*, half*, int, int, int);

extern "C" {
int cuda_kernel_plain(
    const half* A, const half* B, half* C,
    int M, int N, int K,
    cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    gemm_kernel_plain<16, 16, 8><<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    return cudaGetLastError();
}
}
'''

KERNEL2_SOURCE = r'''
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// "EVT-like" kernel: GEMM + fused SiLU activation + elementwise multiply
// More complex than kernel1, mimicking the CUTLASS EVT epilogue kernel

__device__ __forceinline__ float silu_activation(float x) {
    return x / (1.0f + expf(-x));
}

template <int TILE_M, int TILE_N, int TILE_K>
__global__ void gemm_silu_mul_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    const half* __restrict__ aux,   // auxiliary input (like up_proj output)
    half* __restrict__ C,
    int M, int N, int K)
{
    int row = blockIdx.y * TILE_M + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x;

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; k += TILE_K) {
            #pragma unroll
            for (int kk = 0; kk < TILE_K && (k + kk) < K; ++kk) {
                acc += __half2float(A[row * K + k + kk]) *
                       __half2float(B[(k + kk) * N + col]);
            }
        }
        // Fused epilogue: silu(acc) * aux
        float silu_val = silu_activation(acc);
        float aux_val = __half2float(aux[row * N + col]);
        C[row * N + col] = __float2half(silu_val * aux_val);
    }
}

// Instantiate multiple template variants
template __global__ void gemm_silu_mul_kernel<16, 16, 8>(const half*, const half*, const half*, half*, int, int, int);
template __global__ void gemm_silu_mul_kernel<32, 32, 8>(const half*, const half*, const half*, half*, int, int, int);
template __global__ void gemm_silu_mul_kernel<16, 16, 16>(const half*, const half*, const half*, half*, int, int, int);
template __global__ void gemm_silu_mul_kernel<32, 32, 16>(const half*, const half*, const half*, half*, int, int, int);

extern "C" {
int cuda_kernel_evt(
    const half* A, const half* B, const half* aux, half* C,
    int M, int N, int K,
    cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    gemm_silu_mul_kernel<16, 16, 8><<<grid, block, 0, stream>>>(A, B, aux, C, M, N, K);
    return cudaGetLastError();
}
}
'''


def get_cuda_arch():
    """Auto-detect the CUDA compute capability."""
    try:
        import torch
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            return f"{cap[0]}{cap[1]}"
    except Exception:
        pass
    # Fallback: try nvidia-smi
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True, timeout=10
        ).strip().split("\n")[0]
        return out.replace(".", "")
    except Exception:
        return "70"  # safe default


def compile_kernel(src_code, so_name, arch):
    """Compile a CUDA kernel source string into a .so file."""
    os.makedirs(BUILD_DIR, exist_ok=True)
    so_path = os.path.join(BUILD_DIR, so_name)
    cu_path = os.path.join(BUILD_DIR, so_name.replace(".so", ".cu"))

    # Write source
    with open(cu_path, "w") as f:
        f.write(src_code)

    if os.path.exists(so_path) and os.path.getmtime(so_path) > os.path.getmtime(cu_path):
        return so_path

    print(f"  Compiling {so_name}...")

    # Compile: nvcc → .o → .so (two-step, matching inductor pattern)
    obj_path = cu_path.replace(".cu", ".o")

    # Step 1: compile to .o
    cmd = [
        "nvcc",
        f"-gencode=arch=compute_{arch},code=[sm_{arch},compute_{arch}]",
        "-std=c++17", "-O3", "--compiler-options", "-fPIC",
        "-c", "-o", obj_path, cu_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"    Compile error: {result.stderr[-300:]}")
        raise RuntimeError(f"nvcc compilation failed for {so_name}")

    # Step 2: link to .so
    cmd = [
        "nvcc",
        f"-gencode=arch=compute_{arch},code=[sm_{arch},compute_{arch}]",
        "-shared", "-o", so_path, obj_path,
        "-lcudart",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        print(f"    Link error: {result.stderr[-300:]}")
        raise RuntimeError(f"nvcc linking failed for {so_name}")

    so_size = os.path.getsize(so_path) / 1024
    print(f"  Built {so_name} ({so_size:.0f} KB)")
    return so_path


def run_test_in_subprocess(test_label, test_code, so1, so2):
    """Run a test in a subprocess for clean GPU state."""
    code = f"""
import torch, ctypes, sys

M = N = K = 256

x = torch.randn(M, K, device='cuda', dtype=torch.float16)
w = torch.randn(K, N, device='cuda', dtype=torch.float16)
aux = torch.randn(M, N, device='cuda', dtype=torch.float16)
y = torch.empty(M, N, device='cuda', dtype=torch.float16)
stream = torch.cuda.current_stream().cuda_stream

SO1 = {so1!r}
SO2 = {so2!r}

def load_plain():
    lib = ctypes.CDLL(SO1)
    return lib.cuda_kernel_plain

def load_evt():
    lib = ctypes.CDLL(SO2)
    return lib.cuda_kernel_evt

def call_plain(fn):
    ret = fn(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(w.data_ptr()),
        ctypes.c_void_p(y.data_ptr()),
        M, N, K,
        ctypes.c_void_p(stream))
    torch.cuda.synchronize()
    if ret != 0:
        raise RuntimeError(f"CUDA kernel returned error {{ret}}")
    return ret

def call_evt(fn):
    ret = fn(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(w.data_ptr()),
        ctypes.c_void_p(aux.data_ptr()),
        ctypes.c_void_p(y.data_ptr()),
        M, N, K,
        ctypes.c_void_p(stream))
    torch.cuda.synchronize()
    if ret != 0:
        raise RuntimeError(f"CUDA kernel returned error {{ret}}")
    return ret

try:
{test_code}
    print("PASS")
except Exception as e:
    print(f"FAIL: {{e}}")
"""
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=60, env=env,
    )
    stdout = result.stdout.strip().split("\n")[-1] if result.stdout.strip() else ""
    passed = "PASS" in stdout
    status = "PASS" if passed else "FAIL"
    print(f"  {test_label:55s} [{status}]")
    if not passed:
        if result.stdout.strip():
            last_line = result.stdout.strip().split("\n")[-1]
            if "FAIL:" in last_line:
                print(f"    -> {last_line}")
        if result.stderr.strip():
            for line in result.stderr.strip().split("\n")[-3:]:
                print(f"    stderr: {line}")
    return passed


def main():
    import torch

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    dev = torch.cuda.get_device_properties(0)
    arch = get_cuda_arch()
    print(f"Device:  {dev.name}")
    print(f"SM:      {dev.major}.{dev.minor} (compute_{arch})")
    print(f"Driver:  {torch.version.cuda}")
    print()

    # Check for nvcc
    if not shutil.which("nvcc"):
        print("ERROR: nvcc not found. Install CUDA toolkit or add to PATH.")
        sys.exit(1)

    print("Building CUDA kernels (separate .so per kernel, like inductor):")
    so1 = compile_kernel(KERNEL1_SOURCE, "kernel1_plain.so", arch)
    so2 = compile_kernel(KERNEL2_SOURCE, "kernel2_evt.so", arch)
    print("Build complete.\n")

    if "--build-only" in sys.argv:
        print("Build-only mode, skipping tests.")
        return

    print("=" * 70)
    print("CUDA Test Matrix: .so Load Order vs Kernel Call")
    print("  Apple-to-apple comparison with XPU dlopen reproducer.")
    print("  Each test runs in a separate subprocess for clean GPU state.")
    print("=" * 70)
    results = {}

    tests = [
        ("A: plain only",
         "    fn = load_plain(); call_plain(fn)"),
        ("B: evt only",
         "    fn = load_evt(); call_evt(fn)"),
        ("C: load plain -> load evt -> call plain  [XPU BUG]",
         "    fn = load_plain(); load_evt(); call_plain(fn)"),
        ("D: load evt -> load plain -> call plain",
         "    load_evt(); fn = load_plain(); call_plain(fn)"),
        ("E: load plain -> call plain -> load evt -> call plain",
         "    fn = load_plain(); call_plain(fn); load_evt(); call_plain(fn)"),
        ("F: load plain -> load evt -> call evt",
         "    load_plain(); fn = load_evt(); call_evt(fn)"),
    ]

    for label, code in tests:
        results[label] = run_test_in_subprocess(label, code, so1, so2)

    print()
    print("=" * 70)
    print("Results")
    print("=" * 70)

    all_pass = all(results.values())
    if all_pass:
        print("  ✅ ALL 6 TESTS PASS on CUDA")
        print()
        print("  The CUDA driver correctly maintains separate device code modules")
        print("  loaded via dlopen. Multiple .so files coexist without interference.")
        print()
        print("  Compare with XPU (Intel Level Zero):")
        print("  Test C (load plain → load evt → call plain) causes DEVICE_LOST")
        print("  on Intel BMG/Xe2 GPUs due to a Level Zero runtime bug that")
        print("  invalidates the first .so's SPIR-V when loading the second .so.")
    else:
        print("  ⚠️  Some tests FAILED — unexpected on CUDA")
        for label, passed in results.items():
            if not passed:
                print(f"    FAILED: {label}")


if __name__ == "__main__":
    main()
