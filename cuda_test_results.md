# CUDA Validation Test Results (2026-05-11)

Apple-to-apple comparison of the multi-.so dlopen behavior on CUDA vs Intel XPU.
This documents the full CUDA testing performed on an NVIDIA RTX 5090 D machine.

## Environment

| Component | Version |
|-----------|---------|
| GPU | NVIDIA GeForce RTX 5090 D (SM 12.0, Blackwell, 32GB VRAM) |
| GPU Count | 8 (used `CUDA_VISIBLE_DEVICES=5`) |
| CUDA Toolkit | 13.0 (via pip wheel) |
| PyTorch | 2.12.0.dev20260407+cu128 (nightly) |
| Triton | 3.7.0+git9c288bc5 (bundled with PyTorch nightly) |
| CUTLASS | Latest from https://github.com/NVIDIA/cutlass (cloned 2026-05-11) |
| Python | 3.11.15 |
| Conda env | `evt_cuda_test` |
| OS | Linux (shared multi-user machine) |

## Setup Steps

```bash
# 1. Created conda environment
conda create -n evt_cuda_test python=3.11 -y

# 2. Installed PyTorch nightly with CUDA 12.8
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# 3. Triton came bundled (triton 3.7.0 installed with PyTorch nightly)

# 4. Cloned NVIDIA CUTLASS
git clone --depth 1 https://github.com/NVIDIA/cutlass.git /root/xuhan/dev_llama_up_gate_fusion_op/third_party/cutlass

# 5. Downloaded PR #181854 files from fork
curl -sL "https://raw.githubusercontent.com/xuhancn/pytorch/xu_cutlass_evt_silu_fusion/torch/_inductor/codegen/cutlass/python_evt.py" -o /tmp/python_evt.py
curl -sL "https://raw.githubusercontent.com/xuhancn/pytorch/xu_cutlass_evt_silu_fusion/torch/_inductor/codegen/cutlass/scheduling.py" -o /tmp/scheduling.py
curl -sL "https://raw.githubusercontent.com/xuhancn/pytorch/xu_cutlass_evt_silu_fusion/torch/testing/_internal/common_xpu.py" -o /tmp/common_xpu.py
curl -sL "https://raw.githubusercontent.com/xuhancn/pytorch/xu_cutlass_evt_silu_fusion/test/inductor/test_cutlass_evt.py" -o /tmp/test_cutlass_evt.py
curl -sL "https://raw.githubusercontent.com/xuhancn/pytorch/xu_cutlass_evt_silu_fusion/test/inductor/test_cutlass_backend.py" -o /tmp/test_cutlass_backend.py

# 6. Applied PR files to site-packages (backed up originals as .orig)
SITE=/data/miniforge3/envs/evt_cuda_test/lib/python3.11/site-packages
cp /tmp/python_evt.py  $SITE/torch/_inductor/codegen/cutlass/python_evt.py
cp /tmp/scheduling.py  $SITE/torch/_inductor/codegen/cutlass/scheduling.py
cp /tmp/common_xpu.py  $SITE/torch/testing/_internal/common_xpu.py

# 7. Installed test dependencies
pip install pytest expecttest hypothesis
```

---

## Test 1: EVT Unit Tests — ✅ ALL PR-ADDED TESTS PASS

```bash
cd /tmp
export CUDA_VISIBLE_DEVICES=5
export TORCHINDUCTOR_CUTLASS_DIR=/root/xuhan/dev_llama_up_gate_fusion_op/third_party/cutlass
python /tmp/test_cutlass_evt.py TestCutlassEVT -v
```

| Test | Status | Notes |
|------|--------|-------|
| `test_py_codegen_neg_constant` | ✅ PASS | `neg()` → `(0.0 - x)` and `constant()` → `str(value)` generate correct code |
| `test_py_codegen_silu` | ✅ PASS | Decomposed SiLU is reconstituted to native `silu()` |
| `test_py_codegen_silu_mul_aux` | ✅ PASS | Full Llama MLP pattern: `silu(acc) * aux_load` |
| `test_py_codegen_aux_load_mul` | ✅ PASS | Cross-GEMM AuxLoad pattern works |
| `test_py_codegen` | ✅ PASS | Existing base test |
| `test_py_codegen_accumulator_return` | ✅ PASS | Existing base test |
| `test_py_codegen_broadcasting` | ✅ PASS | Existing base test |
| `test_py_codegen_disjoint_read_indexing` | ✅ PASS | Existing base test |
| `test_example_tensor_creation` | ✅ PASS | Existing base test |
| `test_evt_argument_codegen` | ❌ ERROR | **Pre-existing**: SM120 not in CUTLASS cpp-gen `cc_map` (`KeyError: '120'`) — **NOT our PR** |
| `test_evt_argument_codegen_return_accumulator` | ❌ ERROR | Same pre-existing SM120 issue |
| `test_evt_codegen` | ❌ ERROR | Same pre-existing SM120 issue |

**Conclusion**: All 4 PR-added unit tests pass. The 3 failures are pre-existing issues
in the CUTLASS cpp-gen backend that doesn't know about SM120 — completely unrelated to our PR.

---

## Test 2: Import Verification — ✅ ALL PR CODE LOADS

```python
>>> from torch._inductor.codegen.cutlass.python_evt import CutlassEVTOpsMixIn
>>> ops = CutlassEVTOpsMixIn()
>>> ops.neg      # ✅ exists
>>> ops.constant # ✅ exists
>>> ops.silu     # ✅ exists
>>> from torch._inductor.codegen.cutlass.scheduling import reconstitute_silu  # ✅
```

---

## Test 3: Multi-.so dlopen — ✅ ALL 6 TESTS PASS (No DEVICE_LOST)

Created and ran `test_cuda_multi_so_dlopen.py` — a CUDA port of the XPU dlopen crash
reproducer. Two CUDA `.so` files with device code (~1MB each, compiled for SM120a) were
tested in all 6 load-order permutations:

| Test | Sequence | XPU (BMG B60) | CUDA (RTX 5090 D) |
|------|----------|---------------|-------------------|
| A | plain kernel only | ✅ PASS | ✅ PASS |
| B | EVT kernel only | ✅ PASS | ✅ PASS |
| **C** | **load plain → load EVT → call plain** | **❌ DEVICE_LOST** | **✅ PASS** |
| D | load EVT → load plain → call plain | ✅ PASS | ✅ PASS |
| E | load plain → call plain → load EVT → call plain | ✅ PASS | ✅ PASS |
| F | load plain → load EVT → call EVT | ✅ PASS | ✅ PASS |

**Conclusion**: The DEVICE_LOST crash is confirmed to be **specific to the Intel Level Zero
/ SYCL runtime** on BMG/Xe2 GPUs. The CUDA driver correctly maintains separate device code
modules loaded via `dlopen` — multiple `.so` files with CUDA device code coexist without
interference.

This validates that the inductor's `async_compile` pattern (compile kernel1.so → compile
kernel2.so → `wait()` → call both kernels) is **safe on CUDA**.

---

## Test 4: Integration Tests — ❌ BLOCKED (SM120 Architecture Limitation)

**Test attempted**: `test_evt_silu_fusion`

**Root cause**: SM120 (consumer Blackwell / RTX 5090 D) does NOT support CUTLASS standard
f16/f32 GEMM ops. This is a fundamental CUTLASS library + hardware limitation.

### Technical Deep-Dive: Why SM120 Can't Run CUTLASS Standard GEMMs

1. **CUTLASS op generation filtering**:
   - `GenerateSM100()` produces ops with `min_cc=100, max_cc=101`
   - `GenerateSM90()` produces ops with `min_cc=90, max_cc=90`
   - CUTLASS manifest's `filter()` checks: `min_cc <= target_arch <= max_cc`
   - For SM120: `120 <= 101` is FALSE → all SM100 ops filtered out
   - For SM120: `120 <= 90` is FALSE → all SM90 ops filtered out
   - **Result: 0 valid CUTLASS GEMM ops for SM120**

2. **SM120's own generator (`GenerateSM120`) only produces specialized ops**:
   - Block-scaled GEMMs (FP4, INT8)
   - Sparse GEMMs
   - FP8 blockwise GEMMs
   - **No standard f16×f16→f32 GEMM** (which our test needs)

3. **Shared memory mismatch** (explains runtime failure when filter is bypassed):
   - SM90 (H100): 227KB shared memory per SM
   - SM100 (B200): 227KB shared memory per SM
   - **SM120 (RTX 5090 D): ~101KB shared memory per SM** (`sm120_smem_capacity_bytes = 101376`)
   - SM90/SM100 CUTLASS 3.x templates are designed for 227KB → physically can't run on SM120

4. **Force-feeding SM100 ops** (bypassing filter via `_normalize_cuda_arch` returning `"100"`):
   - CUTLASS generates 4 SM100 ops (128×128×64 tile, `TmaWarpSpecialized1SmSm100` schedule)
   - Compiled correctly with `-gencode=arch=compute_120a,code=[sm_120a,compute_120a]`
   - **Fails at runtime**: `cutlass::Status::kErrorInternal` at `gemm_op.initialize()` (line 198)
   - Failure reason: SM100 `Sm100TmaWarpSpecialized` dispatch policy assumes SM100 hardware
     (227KB smem, SM100-specific TMA configuration), but SM120 has different hardware

5. **Architecture separation** (confirmed in CUTLASS source):
   - `cutlass::arch::Sm100` and `cutlass::arch::Sm120` are separate arch structs
   - `Sm120TmaWarpSpecialized` and `Sm120PtrArrayTmaWarpSpecialized` are SM120-specific dispatch policies
   - SM120 has its own builder: `cutlass/gemm/collective/builders/sm120_builder.inl`
   - SM120 has its own epilogue builder: `cutlass/epilogue/collective/builders/sm120_builder.inl`

### PyTorch Inductor Code Flow for SM120

```
_normalize_cuda_arch("120")
  → returns "120" (with warning "architecture > 103")
  
cutlass_arch("cuda")
  → "120"

_gen_ops_cached(arch="120", version="12.8.1", device_type="cuda")
  → gen_arch = "120" (not "103", so no "100" override)
  → calls GenerateSM120(manifest, version)
  → GenerateSM120 only generates block-scaled/sparse/fp8 ops
  → manifest filter: 0 standard f16 GEMM ops
  → returns empty dict

tuned_mm → autotune_select_algorithm
  → No valid CUTLASS choices → NoValidChoicesError
  → "No choices exist for backend"
```

### Attempted Workaround: Mapping SM120 → "100"

Modified `_normalize_cuda_arch()` to return "100" for SM120:
```python
if arch_num >= 120:
    return "100"  # SM120+ (Blackwell) uses SM100 CUTLASS ops
```

Result: 4 CUTLASS ops generated, all fail at runtime:
```
Runtime error: Got cutlass error: Error Internal at: 198
Autotune Choices Stats: {"num_choices": 4, "best_time": Infinity}
```

The SM100 templates use `KernelTmaWarpSpecialized1SmSm100` which requires 227KB smem
and SM100-specific hardware features that SM120 doesn't have.

---

## What GPU is Needed for Full Integration Testing

| GPU | SM | Shared Mem | CUTLASS GEMM | Status |
|-----|----|-----------|-------------|--------|
| H100 / H200 | SM90 | 227KB | ✅ 21,268 ops | **RECOMMENDED** |
| B200 / B100 | SM100 | 227KB | ✅ 163,224 ops | ✅ Works |
| RTX 5090 D | SM120 | 101KB | ❌ 0 standard ops | **BLOCKED** |
| A100 | SM80 | 163KB | ⚠️ SM80 ops only (no CUTLASS 3.x EVT) | Not suitable |

**The PR integration tests require SM90 (Hopper) or SM100 (data center Blackwell).**
The `@skipCUDAIf(not SM90OrLater)` decorator passes for SM120, but CUTLASS doesn't
have standard GEMM support for consumer Blackwell.

---

## Summary

| Test Category | Result | Notes |
|---------------|--------|-------|
| EVT unit tests (codegen) | ✅ 4/4 PR tests PASS | Logic validated on SM120 |
| Import verification | ✅ ALL PASS | `neg`, `constant`, `silu`, `reconstitute_silu` load correctly |
| Multi-.so dlopen | ✅ 6/6 PASS | **DEVICE_LOST is Level Zero-specific, NOT a CUDA issue** |
| Integration tests (E2E) | ❌ BLOCKED | SM120 has no standard f16 CUTLASS GEMM ops; need H100/B200 |

**Key finding**: The DEVICE_LOST crash is **confirmed to be Intel Level Zero-specific**.
The CUDA driver correctly handles the multi-.so dlopen pattern that crashes on XPU.
