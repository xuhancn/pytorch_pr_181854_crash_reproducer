#!/usr/bin/env python3
"""
Inductor-level reproducer for CUTLASS EVT DEVICE_LOST on BMG/Xe2.

This uses torch.compile with CUTLASS EVT epilogue fusion enabled to trigger
the bug through PyTorch's normal compilation pipeline.

The Llama MLP pattern: silu(x @ gate_w.T) * (x @ up_w.T)
  - Inductor compiles this as two CUTLASS GEMMs (plain + EVT-fused)
  - Each GEMM is compiled into a separate .so file
  - Loading the second .so invalidates the first .so's SPIR-V device code
  - Calling the first kernel → DEVICE_LOST

Prerequisites:
    source /opt/intel/oneapi/setvars.sh
    pip install torch (with XPU support, built from PR #181854 branch)
    git submodule update --init

NOTE: Requires PyTorch with PR #181854 merged (XPU CUTLASS EVT epilogue fusion).
Until that PR lands in nightly, use a source build from the PR branch.
The standalone_repro_evt.py works with any PyTorch version.

Usage:
    python repro_inductor.py
"""

import os
import sys

# Ensure sycl-tla is findable
script_dir = os.path.dirname(os.path.abspath(__file__))
default_cutlass_dir = os.path.join(script_dir, "third_party", "sycl-tla")
if "TORCHINDUCTOR_CUTLASS_DIR" not in os.environ:
    if os.path.isdir(default_cutlass_dir):
        os.environ["TORCHINDUCTOR_CUTLASS_DIR"] = default_cutlass_dir
        print(f"Using sycl-tla from: {default_cutlass_dir}")
    else:
        print("ERROR: sycl-tla not found. Either:")
        print("  1. Run: git submodule update --init")
        print("  2. Set: export TORCHINDUCTOR_CUTLASS_DIR=/path/to/sycl-tla")
        sys.exit(1)
else:
    print(f"Using sycl-tla from: {os.environ['TORCHINDUCTOR_CUTLASS_DIR']}")

import torch
import torch._inductor.config as config

# Print environment info
d = torch.xpu.get_device_properties(0)
print(f"Device:  {d.name}")
print(f"Driver:  {d.driver_version}")
print(f"PyTorch: {torch.__version__}")
print()

# Configure inductor for CUTLASS EVT fusion
config.max_autotune = True
config.max_autotune_gemm_backends = "CUTLASS"
config.cutlass.cutlass_epilogue_fusion_enabled = True
# Skip benchmarking — force CUTLASS to be selected
config.benchmark_epilogue_fusion = False


def llama_mlp(x, gate_w, up_w):
    """Llama MLP pattern: silu(x @ gate_w.T) * (x @ up_w.T)"""
    gate = x @ gate_w.t()
    up = x @ up_w.t()
    return torch.nn.functional.silu(gate) * up


# Use small dimensions for quick compilation
M, K, N = 256, 256, 256

x = torch.randn(M, K, device="xpu", dtype=torch.float16)
gate_w = torch.randn(N, K, device="xpu", dtype=torch.float16)
up_w = torch.randn(N, K, device="xpu", dtype=torch.float16)

# Compute reference (eager mode)
ref = llama_mlp(x, gate_w, up_w)
print(f"Eager result: shape={ref.shape}, dtype={ref.dtype}")

# Compile with CUTLASS EVT fusion
print("\nCompiling with torch.compile (CUTLASS EVT fusion enabled)...")
print("  This will generate two separate .so files (plain GEMM + EVT GEMM)")
print("  If the bug is present, the first kernel call will crash with DEVICE_LOST")
print()

compiled_fn = torch.compile(llama_mlp)

try:
    result = compiled_fn(x, gate_w, up_w)
    torch.xpu.synchronize()

    # Verify correctness
    max_diff = (result - ref).abs().max().item()
    print(f"✅ PASS — no crash!")
    print(f"  Max difference from eager: {max_diff:.6f}")
    if max_diff < 0.1:
        print(f"  Numerical accuracy: OK")
    else:
        print(f"  ⚠️  Large numerical difference (may be expected with fp16)")

except RuntimeError as e:
    if "DEVICE_LOST" in str(e) or "UR_RESULT_ERROR" in str(e):
        print(f"❌ DEVICE_LOST — Bug reproduced!")
        print(f"  Error: {e}")
        print()
        print("  Root cause: loading two CUTLASS SYCL .so files via dlopen")
        print("  invalidates the first .so's SPIR-V device code in Level Zero.")
        print("  See evt_device_lost_analysis.md for full analysis.")
        sys.exit(1)
    else:
        print(f"❌ Unexpected error: {e}")
        sys.exit(2)
