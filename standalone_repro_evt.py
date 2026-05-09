#!/usr/bin/env python3
"""
Standalone reproducer for CUTLASS EVT DEVICE_LOST on BMG/Xe2.

ROOT CAUSE: When two CUTLASS SYCL kernel .so files are loaded into the same
process via ctypes.CDLL, loading the second .so invalidates the first .so's
un-JIT'd device code in the Level Zero runtime. This causes DEVICE_LOST when
the first kernel is subsequently called.

The bug is ORDER-DEPENDENT:
  - Load plain.so → Load evt.so → call plain → CRASH  ← inductor's pattern
  - Load evt.so → Load plain.so → call plain → OK     (reverse order)
  - Load plain.so → call plain → Load evt.so → OK     (JIT before 2nd load)
  - Load plain.so → Load evt.so → call evt → OK       (call 2nd, not 1st)

This exactly matches the inductor crash: inductor compiles the plain GEMM
kernel first (loading its .so), then compiles the EVT kernel (loading its
.so), then calls the plain GEMM → DEVICE_LOST.

Prerequisites:
    source /opt/intel/oneapi/setvars.sh   # or wherever oneAPI is installed
    pip install torch (with XPU support)
    git submodule update --init           # to get third_party/sycl-tla

Usage:
    python standalone_repro_evt.py              # Run full test matrix
    python standalone_repro_evt.py --no-xs      # Without IGC backend flags
    python standalone_repro_evt.py --build-only # Just compile, don't test
"""

import os
import sys
import subprocess
import ctypes
import shlex
import shutil

SYCL_TLA_DIR = os.environ.get(
    "TORCHINDUCTOR_CUTLASS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party", "sycl-tla"),
)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(SCRIPT_DIR, "_build_standalone")

USE_XS_FLAGS = "--no-xs" not in sys.argv


def get_gpu_target():
    """Auto-detect the SYCL GPU target from the installed XPU device."""
    try:
        import torch
        dev_name = torch.xpu.get_device_properties(0).name.lower()
        if "b60" in dev_name or "b580" in dev_name or "b570" in dev_name or "bmg" in dev_name:
            return "intel_gpu_bmg_g21"
        elif "pvc" in dev_name or "max" in dev_name or "1550" in dev_name or "1100" in dev_name:
            return "intel_gpu_pvc"
        elif "lnl" in dev_name:
            return "intel_gpu_lnl_m"
        else:
            # Fallback: try to compile for spir64_gen (generic)
            return "intel_gpu_bmg_g21"
    except Exception:
        return "intel_gpu_bmg_g21"


def get_compile_flags():
    """Return the exact compile and link flags inductor uses for XPU CUTLASS."""
    gpu_target = get_gpu_target()
    include_dirs = [
        os.path.join(SYCL_TLA_DIR, "include"),
        os.path.join(SYCL_TLA_DIR, "tools/library/include"),
        os.path.join(SYCL_TLA_DIR, "tools/library/src"),
        os.path.join(SYCL_TLA_DIR, "tools/util/include"),
    ]

    xs_arg = (
        '-options "-igc_opts '
        "'VISAOptions=-perfmodel,"
        "VectorAliasBBThreshold=100000000000,"
        "ExtraOCLOptions=-cl-intel-256-GRF-per-thread'\" "
        "-options -ze-opt-large-register-file"
    )

    compiler_lib = os.path.join(os.path.dirname(os.path.dirname(
        subprocess.check_output(["which", "icpx"], text=True).strip()
    )), "lib")

    # Flags passed to BOTH compile and link (matching inductor exactly)
    flags = [
        "-DCUTLASS_ENABLE_SYCL",
        "-DSYCL_INTEL_TARGET",
        "-DCUTLASS_VERSIONS_GENERATED",
        "-O3", "-DNDEBUG",
        "-std=c++20", "-fPIC",
        "-fsycl",
        "-fsycl-targets=" + gpu_target,
        "-Xspirv-translator",
        "-spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,"
        "+SPV_INTEL_subgroup_matrix_multiply_accumulate",
        "-fno-sycl-instrument-device-code",
        "-DMKL_ILP64", "-MD",
    ]
    if USE_XS_FLAGS:
        flags.extend(["-Xs", xs_arg])
    flags.extend([
        f"-L{compiler_lib}",
        "-Xlinker", f"-rpath={compiler_lib}",
        "-lsycl",
    ])

    return include_dirs, flags


def compile_kernel(src_path, so_name, include_dirs, flags):
    """Compile a .sycl source into a .so, matching inductor's exact pipeline.

    Uses subprocess.list2cmdline + shlex.split round-trip for correct -Xs
    quoting, then subprocess.check_output(list) — same as inductor does.
    """
    os.makedirs(BUILD_DIR, exist_ok=True)
    obj_path = os.path.join(BUILD_DIR, so_name.replace(".so", ".o"))
    so_path = os.path.join(BUILD_DIR, so_name)

    # Copy to .cpp (inductor uses .cpp; icpx doesn't recognize .sycl)
    cpp_path = os.path.join(BUILD_DIR, os.path.basename(src_path).replace(".sycl", ".cpp"))
    shutil.copy2(src_path, cpp_path)

    if os.path.exists(so_path) and os.path.getmtime(so_path) > os.path.getmtime(src_path):
        return so_path

    base_flags = ["-I" + d for d in include_dirs] + ["-isystem", "/include"] + flags

    # Step 1: Compile
    cmd_list = ["icpx"] + base_flags + ["-c", "-o", obj_path, cpp_path]
    cmd_parts = shlex.split(subprocess.list2cmdline(cmd_list))
    print(f"  Compiling {os.path.basename(src_path)}...")
    result = subprocess.run(cmd_parts, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        for line in result.stderr.split("\n"):
            if "error:" in line:
                print(f"    {line}")
        raise RuntimeError(f"Compilation failed for {src_path}")

    # Step 2: Link
    cmd_list = ["icpx"] + base_flags + ["-shared", "-o", so_path, obj_path]
    cmd_parts = shlex.split(subprocess.list2cmdline(cmd_list))
    print(f"  Linking {so_name}...")
    result = subprocess.run(cmd_parts, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"    Link error: {result.stderr[-200:]}")
        raise RuntimeError(f"Linking failed for {so_name}")

    return so_path


def run_test_in_subprocess(test_label, test_code, so1, so2):
    """Run a test in a subprocess for clean GPU state."""
    code = f"""
import torch, ctypes, sys
c_void_p = ctypes.c_void_p
M = N = K = 256
from torch._C import _xpu_getCurrentRawStream as get_raw_stream

x = torch.randn(M, K, device='xpu', dtype=torch.float16)
w = torch.randn(N, K, device='xpu', dtype=torch.float16)
buf0 = torch.randn(M, N, device='xpu', dtype=torch.float16)
y = torch.empty(M, N, device='xpu', dtype=torch.float16)
stream0 = get_raw_stream(0)

SO1 = {so1!r}
SO2 = {so2!r}

def load_plain():
    lib = ctypes.CDLL(SO1)
    return lib.cutlass_fused_mm_t_ccb68346

def load_evt():
    lib = ctypes.CDLL(SO2)
    return lib.cutlass_fused_mm_mul_silu_t_02f33700

def call_plain(fn):
    ret = fn(c_void_p(x.data_ptr()), c_void_p(w.data_ptr()), c_void_p(y.data_ptr()),
             256, 256, 256, 1, 256, 256, 0, 256, 0, 0, 0, 2, None, None, c_void_p(stream0))
    torch.xpu.synchronize()
    return ret

def call_evt(fn):
    ret = fn(c_void_p(x.data_ptr()), c_void_p(w.data_ptr()), c_void_p(buf0.data_ptr()),
             c_void_p(y.data_ptr()), c_void_p(y.data_ptr()),
             256, 256, 256, 1, 256, 256, 0, 256, 0, 0, 0, 0, 0, 2, None, None, c_void_p(stream0))
    torch.xpu.synchronize()
    return ret

try:
{test_code}
    print("PASS")
except Exception as e:
    print(f"FAIL: {{e}}")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=60,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    stdout = result.stdout.strip().split("\n")[-1] if result.stdout.strip() else ""
    passed = "PASS" in stdout
    status = "PASS" if passed else "FAIL"
    print(f"  {test_label:55s} [{status}]")
    if not passed and result.stdout.strip():
        last_line = result.stdout.strip().split("\n")[-1]
        if "FAIL:" in last_line:
            print(f"    -> {last_line}")
    return passed


def main():
    import torch

    d = torch.xpu.get_device_properties(0)
    print(f"Device:     {d.name}")
    print(f"Driver:     {d.driver_version}")
    print(f"sycl-tla:   {SYCL_TLA_DIR}")
    print(f"-Xs flags:  {'enabled' if USE_XS_FLAGS else 'disabled'}")
    print()

    include_dirs, flags = get_compile_flags()

    kernel1_src = os.path.join(SCRIPT_DIR, "_inductor_kernel1_plain.sycl")
    kernel2_src = os.path.join(SCRIPT_DIR, "_inductor_kernel2_evt.sycl")

    if not os.path.exists(kernel1_src) or not os.path.exists(kernel2_src):
        print("ERROR: Missing kernel sources")
        print("  Need: _inductor_kernel1_plain.sycl, _inductor_kernel2_evt.sycl")
        sys.exit(1)

    print("Building kernels (separate .so per kernel, like inductor):")
    so1 = compile_kernel(kernel1_src, "kernel1_plain.so", include_dirs, flags)
    so2 = compile_kernel(kernel2_src, "kernel2_evt.so", include_dirs, flags)
    print("Build complete.\n")

    if "--build-only" in sys.argv:
        print("Build-only mode, skipping tests.")
        return

    print("=" * 70)
    print("Test Matrix: .so Load Order vs Kernel Call")
    print("  Each test runs in a separate subprocess for clean GPU state.")
    print("=" * 70)
    results = {}

    tests = [
        ("A: plain only",
         "    fn = load_plain(); call_plain(fn)"),
        ("B: evt only",
         "    fn = load_evt(); call_evt(fn)"),
        ("C: load plain -> load evt -> call plain  [BUG]",
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
    print("Analysis")
    print("=" * 70)
    all_expected = True
    for label, passed in results.items():
        is_bug_test = "[BUG]" in label
        expected = not passed if is_bug_test else passed
        if not expected:
            all_expected = False
            print(f"  UNEXPECTED: {label} -> {'PASS' if passed else 'FAIL'}")

    if all_expected:
        print("  All tests match expected behavior.")
        print()
        print("  Root cause: loading plain.so THEN evt.so invalidates the plain")
        print("  kernel's un-JIT'd SPIR-V device code registration in Level Zero.")
        print("  This is a SYCL runtime / Level Zero driver bug where dlopen of a")
        print("  second SYCL .so corrupts the first .so's registered device code.")
        print()
        print("  Workarounds:")
        print("  1. JIT kernels immediately after loading (call once before loading next .so)")
        print("  2. Reverse the load order (load EVT .so first)")
        print("  3. Combine both kernels into a single .so")
    else:
        print()
        print("  Some tests had unexpected results. The bug pattern may differ on")
        print("  this hardware/driver combination.")


if __name__ == "__main__":
    main()
