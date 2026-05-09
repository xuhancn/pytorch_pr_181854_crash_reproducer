# EVT SiLU Fusion — BMG/B60 DEVICE_LOST Root Cause Analysis

## Problem Statement

When PyTorch inductor generates two CUTLASS GEMM kernels for the Llama MLP pattern
(`silu(x @ gate_w.T) * (x @ up_w.T)`), and the second kernel uses CUTLASS EVT
(Epilogue Visitor Tree) to fuse `silu(buf0) * accumulator` into its epilogue,
the GPU crashes with `UR_RESULT_ERROR_DEVICE_LOST` on Intel Arc Pro B60 (BMG/Xe2).

This bug **does not affect PVC** (Intel Data Center GPU Max series), which is used
in the PyTorch CI. It is specific to BMG/Xe2 consumer/workstation GPUs.

## Root Cause (CONFIRMED)

**The bug is in the Level Zero / SYCL runtime, NOT in the kernel code or EVT logic.**

When two separate CUTLASS SYCL kernel `.so` files are loaded into the same process
via `ctypes.CDLL` (or `dlopen`), loading the second `.so` **invalidates the first
`.so`'s un-JIT'd SPIR-V device code registration** in the Level Zero runtime. When
the first kernel is subsequently called, it executes corrupted/invalid device code,
causing `DEVICE_LOST`.

### Key Evidence: Load Order Test Matrix

Each test runs in a **separate subprocess** with clean GPU state:

| Test | Sequence | Expected | Result |
|------|----------|----------|--------|
| A | plain kernel only | PASS | ✅ PASS |
| B | EVT kernel only | PASS | ✅ PASS |
| **C** | **load plain → load EVT → call plain** | **FAIL** | **❌ DEVICE_LOST** |
| D | load EVT → load plain → call plain | PASS | ✅ PASS |
| E | load plain → call plain → load EVT → call plain | PASS | ✅ PASS |
| F | load plain → load EVT → call EVT | PASS | ✅ PASS |

### Why Test C Fails

1. `ctypes.CDLL("plain.so")` — SYCL runtime registers plain kernel's SPIR-V
2. `ctypes.CDLL("evt.so")` — SYCL runtime registers EVT kernel's SPIR-V,
   **corrupts/invalidates** the previously registered plain kernel's SPIR-V
3. Call plain kernel — executes corrupted device code → **DEVICE_LOST**

### Why Tests D and E Pass

- **Test D** (reverse order): EVT loaded first, then plain. The plain kernel's
  SPIR-V registration is the most recent and not corrupted.
- **Test E** (JIT before 2nd load): The plain kernel is **called** (JIT-compiled to
  native code) before EVT is loaded. Once JIT'd, the kernel is cached in native
  form and survives the second `.so` load.

### Inductor Compilation Model (confirmed via `TORCH_LOGS="output_code"`)

Inductor compiles each CUTLASS kernel as a **separate `.so` module**. The generated
output code contains exactly **two `async_compile.xpu()` calls**, each producing an
independent shared library:

```python
# From inductor-generated output (TORCH_LOGS="output_code"):
cutlass_fused_mm_t_ccb68346 = async_compile.xpu(r'''...kernel1 source...''', "so")
cutlass_fused_mm_mul_silu_t_02f33700 = async_compile.xpu(r'''...kernel2 source...''', "so")
async_compile.wait(globals())   # ← both .so are dlopen'd by this point
del async_compile

# Later, in the call() function:
cutlass_fused_mm_t_ccb68346.cutlass_fused_mm_t_ccb68346(...)      # ← CRASH here
cutlass_fused_mm_mul_silu_t_02f33700.cutlass_fused_mm_mul_silu_t_02f33700(...)
```

**Code path for each `async_compile.xpu()` call:**
```
async_compile.xpu(source)
  → self.submit(task)                    # submit to ThreadPoolExecutor (20 threads)
    → XPUCodeCache.load(source, "so")
      → XPUCodeCache.compile(source, "so")
        → compile to .o: icpx -c -o kernel.o kernel.cpp
        → link to .so:   icpx -shared -o kernel.so kernel.o
      → DLLWrapper(kernel.so)
        → cdll.LoadLibrary(kernel.so)    # ← dlopen! Registers SPIR-V with L0 runtime
```

**Observed in inductor cache** (7 `.so` files total):
- `rn/*.so` — winning plain GEMM kernel (`cutlass_fused_mm_t_ccb68346`)
- `zv/*.so` — winning EVT GEMM kernel (`cutlass_fused_mm_mul_silu_t_02f33700`)
- 5 other `.so` — autotuning candidates (128×256, 256×256, 128×128, etc.), loaded during
  benchmarking but only the 2 winners appear in the final output code

**Threading does NOT affect the bug**: tested with `TORCHINDUCTOR_COMPILE_THREADS=1`
(forcing sequential, deterministic `dlopen` order) — still crashes. The bug is purely
in the sequential `dlopen(plain.so) → dlopen(evt.so)` order, not a race condition.

### Why This Matches the Inductor Crash

The inductor crash is exactly Test C from the load-order matrix:
1. `dlopen(plain.so)` — SYCL runtime registers plain kernel's SPIR-V
2. `dlopen(evt.so)` — SYCL runtime registers EVT kernel's SPIR-V,
   **invalidates** the plain kernel's registration
3. `async_compile.wait()` completes — both `DLLWrapper` objects ready
4. Call plain GEMM → executes corrupted device code → **DEVICE_LOST**

### Nature of the Bug

This is a **Level Zero / SYCL runtime bug** (not a compiler or kernel code bug):
- The SPIR-V binary in both `.so` files is valid (each works fine individually)
- The EVT kernel itself is correct (test F: calling EVT after both loads works)
- The issue is in how the SYCL runtime manages device code images from multiple
  dynamically-loaded shared libraries
- Not a threading issue (reproduces with single-threaded compilation)

## EVT Type Tree (for reference)

```cpp
using Accum = cutlass::epilogue::fusion::XeAccFetch;
using Buf0 = cutlass::epilogue::fusion::XeAuxLoad<half_t, Stride<int64_t, Int<1>, Int<0>>>;
using SiLu = cutlass::epilogue::fusion::XeCompute<cutlass::epilogue::thread::SiLu, float, float>;
using EVT_SiLu = cutlass::epilogue::fusion::XeEVT<SiLu, Buf0>;
using Mul = cutlass::epilogue::fusion::XeCompute<cutlass::multiplies, half_t, float>;
using EVT_SiLuMul = cutlass::epilogue::fusion::XeEVT<Mul, EVT_SiLu, Accum>;
```

## Reproducer

### Standalone Test (recommended)

```bash
source ~/intel/oneapi/setvars.sh
conda activate xu_pytorch
cd 08_fusing_pr_reuse_inductor_XPU_cutlass
python standalone_repro_evt.py
```

This compiles the exact inductor-generated SYCL source as two separate `.so` files,
then runs the 6-test load-order matrix above. Test C is expected to FAIL with
DEVICE_LOST; all others should PASS.

### Via Inductor

Use `TORCH_LOGS="output_code"` to inspect the generated compilation plan:

```bash
TORCHINDUCTOR_CUTLASS_DIR=/path/to/sycl-tla \
TORCH_LOGS="output_code" \
python -c "
import torch
import torch._inductor.config as config
config.max_autotune = True
config.max_autotune_gemm_backends = 'CUTLASS'
config.cutlass.cutlass_epilogue_fusion_enabled = True
config.benchmark_epilogue_fusion = False

def llama_mlp(x, gate_w, up_w):
    gate = x @ gate_w.t()
    up = x @ up_w.t()
    return torch.nn.functional.silu(gate) * up

M, K, N = 256, 256, 256
x = torch.randn(M, K, device='xpu', dtype=torch.float16)
gate_w = torch.randn(N, K, device='xpu', dtype=torch.float16)
up_w = torch.randn(N, K, device='xpu', dtype=torch.float16)
compiled = torch.compile(llama_mlp)
result = compiled(x, gate_w, up_w)  # crashes here
" 2>&1 | grep -E 'async_compile|cutlass_fused'
```

The log output confirms two separate `async_compile.xpu()` calls (two `.so` files):
```
cutlass_fused_mm_t_ccb68346 = async_compile.xpu(r'''...''', "so")
cutlass_fused_mm_mul_silu_t_02f33700 = async_compile.xpu(r'''...''', "so")
async_compile.wait(globals())
```

Additional useful torch log options (see [torch logs tutorial](https://docs.pytorch.org/tutorials/recipes/torch_logs.html)):
- `TORCH_LOGS="output_code"` — shows the full generated Python wrapper + SYCL source
- `TORCH_LOGS="+inductor"` — verbose inductor debug logging
- `TORCH_LOGS="schedule"` — shows kernel scheduling decisions

## Workarounds

### For PyTorch Inductor (immediate)

1. **JIT-on-load**: After each `DLLWrapper.open()` / `cdll.LoadLibrary()`, immediately
   call the kernel with a workspace-size query (no GPU work, just triggers JIT).
   This pre-compiles the SPIR-V before the next `.so` is loaded.

2. **Reverse load order**: Load the EVT kernel `.so` before the plain kernel `.so`.
   The most recently loaded kernel's device code is not corrupted.

3. **Single `.so`**: Combine both kernels into one `.so` file (avoids the
   multi-dlopen race condition entirely).

### For Users (existing defaults are safe)

- `cutlass_epilogue_fusion_enabled=False` (default): EVT fusion never attempted
- `benchmark_epilogue_fusion=True`: Triton typically wins on BMG, so EVT not selected

## Debugging History

1. Initially observed crash in inductor EVT test on B60
2. Suspected kernel code bug → built standalone reproducer with identical EVT types
3. Standalone reproducer passed all tests → suspected inductor-specific issue
4. Extracted exact inductor-generated SYCL source → still crashed through inductor
5. Compiled extracted source with manual `icpx` → standalone worked, inductor crashed
6. **Key breakthrough**: loaded both `.so` files simultaneously via ctypes → crashed!
7. Discovered load-order dependency through systematic 6-test matrix
8. Confirmed: inductor's `.so` files have the same behavior (not a compilation issue)
9. Root cause confirmed: Level Zero / SYCL runtime multi-`.so` device code corruption

## Environment

| Component | Version |
|-----------|---------|
| GPU | Intel Arc Pro B60 (BMG/Xe2, 0xE20B, 20 Xe-cores, 22.7GB) |
| Level Zero Loader | 1.28.0 |
| GPU Driver | 1.14.37435+12 (NEO 26.09.37435.12) |
| sycl-tla | v0.8 (commit 2fc09973) |
| PyTorch | 2.13.0a0+git8f75890 (xu_cutlass_evt_silu_fusion branch) |
| icpx | oneAPI 2025.3 |
| SYCL target | intel_gpu_bmg_g21 |

## Next Steps

1. **File Level Zero / compute-runtime bug** with the standalone reproducer
   (standalone_repro_evt.py + kernel source files)
2. **Implement JIT-on-load workaround** in `DLLWrapper` or `XPUCodeCache.load()` to
   pre-compile SPIR-V before loading the next `.so`
3. **Test on other BMG devices** (Arc B580, B570, etc.) to confirm scope
4. **Test on newer Level Zero drivers** once available
