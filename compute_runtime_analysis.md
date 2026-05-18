# Intel compute-runtime Root Cause Analysis: BMG/Xe2 DEVICE_LOST on Multi-.so CUTLASS Kernels

## Executive Summary

The DEVICE_LOST crash on BMG/Xe2 GPUs when loading multiple SYCL kernel `.so` files
via `dlopen` is caused by a **SYCL kernel name collision** in the CUTLASS SYCL library
design. Both the "plain" and "EVT" CUTLASS kernel `.so` files embed device binaries
(zebin) registered under the **same SYCL kernel class name**:

```
cutlass3x_xe20_tensorop_gemm_f16_f16_f32_void_f16_df16_64x128x32_1x1x1_0_tnt_align8
```

When both `.so` files are loaded via `dlopen`, the SYCL `ProgramManager` has two
different device binaries (44KB plain vs 66KB EVT) registered under the same kernel
name. On kernel dispatch, the runtime selects the **wrong binary** (EVT's 66KB image
instead of plain's 44KB image) → GPU executes incompatible code → **DEVICE_LOST**.

**This is NOT a GPU driver bug.** It reproduces across all tested driver versions
(26.09, 26.14, 26.18) including the absolute latest release. The previously suspected
NEO-12287 (2MB ISA alignment) fix is a separate, unrelated issue.

### Key Evidence

| Test | Behavior | Binary Size at `urProgramCreateWithBinary` | Result |
|------|----------|-------------------------------------------|--------|
| A: load plain → call plain | Correct binary selected | **44,456 bytes** (plain) | ✅ PASS |
| C: load plain → load evt → call plain | **Wrong binary selected** | **66,120 bytes** (evt!) | ❌ DEVICE_LOST |
| D: load evt → load plain → call plain | Correct binary selected | 44,456 bytes (plain) | ✅ PASS |
| E: load plain → call → load evt → call | Cached from first call | 44,456 bytes (plain) | ✅ PASS |

---

## True Root Cause: SYCL Kernel Name Collision

### The Problem

CUTLASS SYCL uses the **GEMM core type** as the SYCL kernel class name. The epilogue
functor (plain `LinearCombination` vs complex EVT `Sm90TreeVisitor`) is passed as a
**kernel argument** rather than as a template parameter that affects the kernel type name.

This means different `.so` files with different CUTLASS epilogues produce device images
with **identical kernel names but different compiled code**.

```
# Both .so files export this SAME kernel type name:
_ZTS83cutlass3x_xe20_tensorop_gemm_f16_f16_f32_void_f16_df16_64x128x32_1x1x1_0_tnt_align8

# kernel1_plain.so epilogue: ...FusionCallbacks<LinearCombination<f16,f,...>>...
# kernel2_evt.so epilogue:   ...Sm90TreeVisitor<Sm90Compute<multiplies>...<SiLu>...<XeAuxLoad>...>...
```

### How the SYCL Runtime Fails

1. `dlopen(kernel1_plain.so)` → `__sycl_register_lib` registers image A (44KB zebin)
   with kernel name `cutlass3x_...align8`
2. `dlopen(kernel2_evt.so)` → `__sycl_register_lib` registers image B (66KB zebin)
   with the **same** kernel name `cutlass3x_...align8`
3. User calls the plain kernel → SYCL runtime looks up `cutlass3x_...align8`
4. `ProgramManager` finds **two** entries in `m_KernelIDs2BinImage` (multimap)
5. `urDeviceSelectBinary` selects the **wrong** image (B instead of A)
6. `urProgramCreateWithBinary` loads the 66KB EVT binary
7. `urKernelCreate` creates a kernel with EVT code
8. Kernel is dispatched with plain kernel's argument layout (416 bytes) but EVT's code
   expects different arguments → **GPU HANG / DEVICE_LOST**

### Why Load Order Matters

- **Test C (plain first, evt second) → FAIL**: The multimap/lookup picks the last-
  registered image (evt's binary) for the shared kernel name
- **Test D (evt first, plain second) → PASS**: The lookup picks the last-registered
  image (plain's binary), which is correct for calling plain
- **Test E (call before loading evt) → PASS**: The kernel is compiled and cached
  before the collision exists; subsequent calls use the cached kernel

---

## Previously Suspected: NEO-12287 (2MB ISA Alignment) — RULED OUT

The NEO-12287 fix was previously suspected as the root cause but has been **definitively
ruled out** through testing:

---

## Bug Tracker References

### True Root Cause (Kernel Name Collision) — **FIXED** ✅

- **Layer**: PyTorch Inductor CUTLASS codegen (`gemm_template.py`)
- **Impact**: Any two CUTLASS `.so` files sharing GEMM core parameters but different
  epilogues will collide
- **Status**: **FIXED** — unique SYCL kernel class names via `KERNEL_NAME` hash suffix
- **Fix location**: `torch/_inductor/codegen/cutlass/gemm_template.py` →
  `CUTLASS3xGemmTemplate._define_gemm_instance()`

### Previously Suspected: NEO-12287 (Separate Issue, Fixed)

- **Internal**: NEO-12287, HSD-18042276431
- **Primary fix commit**: [`4078022318bca0dfb466c0aeba5a392f47abf7a0`](https://github.com/intel/compute-runtime/commit/4078022318bca0dfb466c0aeba5a392f47abf7a0) — *2025-11-19* — "fix: configure ISA Pool params based on productHelper"
- **Secondary fix commit**: [`e2228201ce6da6e7ae657d7fa0d309d9a38d3912`](https://github.com/intel/compute-runtime/commit/e2228201ce6da6e7ae657d7fa0d309d9a38d3912) — *2025-05-30* — "fix: Avoid redundant padding in ISA allocations"
- **Reverted earlier attempt**: [`bf20ae7ae82a468801a921f4d2e01b9d4b92eb0b`](https://github.com/intel/compute-runtime/commit/bf20ae7ae82a468801a921f4d2e01b9d4b92eb0b) (2025-02-19) → reverted by `f5e37e725cc47cae` (2025-03-10), then correctly re-landed as `4078022318bca0`
- **Status**: Fixed in driver 26.14+, but **does NOT resolve this crash**

---

## The Three Inter-Related Bugs (Fixed by `4078022318bca0`)

### Bug 1: Wrong `isaAllocationPageSize` Threshold for BMG

The `isaAllocationPageSize` determines whether a module's compiled ISA goes through the
**ISA pool** (shared allocation) or gets a **dedicated GPU allocation**. Before the fix,
BMG used `64KB` — the same as non-BMG devices.

**Before fix** (`module_imp.cpp`):
```cpp
ModuleImp::ModuleImp(...) {
    auto &gfxCoreHelper = device->getGfxCoreHelper();
    auto &hwInfo = device->getHwInfo();
    this->isaAllocationPageSize = gfxCoreHelper.useSystemMemoryPlacementForISA(hwInfo)
        ? MemoryConstants::pageSize     // 4KB (system memory path)
        : MemoryConstants::pageSize64k; // 64KB ← WRONG for BMG!
}
```

**After fix** (new `getIsaAllocationPageSize()` method):
```cpp
size_t ModuleImp::getIsaAllocationPageSize() const {
    auto &gfxCoreHelper = device->getGfxCoreHelper();
    auto &hwInfo = device->getHwInfo();
    if (gfxCoreHelper.useSystemMemoryPlacementForISA(hwInfo)) {
        return MemoryConstants::pageSize;     // 4KB
    }
    if (device->getProductHelper().is2MBLocalMemAlignmentEnabled()) {
        return MemoryConstants::pageSize2M;   // 2MB ← CORRECT for BMG
    } else {
        return MemoryConstants::pageSize64k;  // 64KB for other local-mem devices
    }
}
```

**Effect**: With `isaAllocationPageSize = 64KB`, any module with compiled ISA > 64KB
takes the dedicated allocation path (`allocateKernelsIsaMemory`). Large CUTLASS kernels
(185KB+ SPIR-V → well above 64KB compiled ISA) always hit this path.

### Bug 2: No 2MB Alignment for ISA in DRM Heap Allocator

When the dedicated allocation path is taken, `drm_memory_manager.cpp` allocates the GPU
virtual address with **zero explicit alignment**, defaulting to 64KB.

**Before fix** (`drm_memory_manager.cpp`):
```cpp
case AllocationType::kernelIsa:
case AllocationType::kernelIsaInternal:
case AllocationType::internalHeap:
case AllocationType::debugModuleArea:
    gpuAddress = getCanonizedHeapAllocationAddress(
        heapAssigner.get32BitHeapIndex(...),
        gmmHelper, gfxPartition, sizeAllocated,
        /* alignment = */ 0,   // ← NO explicit alignment! Defaults to 64KB
        false);
    break;
```

**After fix**:
```cpp
case AllocationType::kernelIsa:
case AllocationType::kernelIsaInternal:
case AllocationType::internalHeap:
case AllocationType::debugModuleArea: {
    size_t alignment = 0;
    if (gmmHelper->getRootDeviceEnvironment().getHelper<ProductHelper>()
                .is2MBLocalMemAlignmentEnabled()) {
        alignment = MemoryConstants::pageSize2M;  // ← 2MB alignment for BMG
    }
    gpuAddress = getCanonizedHeapAllocationAddress(
        heapAssigner.get32BitHeapIndex(...),
        gmmHelper, gfxPartition, sizeAllocated, alignment, false);
    break;
}
```

**Effect**: ISA allocations on BMG get GPU virtual addresses that are only 64KB-aligned.
BMG hardware requires 2MB alignment for ISA in local memory. GPU instruction fetch from
a non-2MB-aligned local memory address causes a page table fault → DEVICE_LOST.

### Bug 3: ISA Pool `poolAlignment` Not Configured for BMG

Even for the ISA pool path (small kernels), the pool's internal alignment was wrong.

**Before fix** (`isa_pool_allocator.h` default member values):
```cpp
size_t userAllocationSize = MemoryConstants::pageSize2M * 2; // 4MB
size_t builtinAllocationSize = MemoryConstants::pageSize64k; // 64KB (WRONG for BMG builtins)
size_t poolAlignment = 1u;  // ← WRONG for BMG (should be 2MB)
```

**After fix** (`isa_pool_allocator.cpp`, called from constructor):
```cpp
void ISAPoolAllocator::initAllocParams() {
    if (device->getProductHelper().is2MBLocalMemAlignmentEnabled()) {
        userAllocationSize = MemoryConstants::pageSize2M * 2;  // 4MB
        builtinAllocationSize = MemoryConstants::pageSize2M;   // 2MB ← FIXED
        poolAlignment = MemoryConstants::pageSize2M;           // 2MB ← FIXED
    } else {
        userAllocationSize = MemoryConstants::pageSize2M * 2;
        builtinAllocationSize = MemoryConstants::pageSize64k;
    }
}
```

Additionally, a new `alignToPoolSize()` is used when creating ISA pool backing
allocations, and a `DEBUG_BREAK_IF` assertion guards against future violations:
```cpp
ISAPool::ISAPool(Device *device, bool isBuiltin, size_t storageSize) : ... {
    DEBUG_BREAK_IF(device->getProductHelper().is2MBLocalMemAlignmentEnabled() &&
                   !isAligned(storageSize, MemoryConstants::pageSize2M));
    // ...
}
```

---

## Fix Summary Matrix

| Component | Before Fix | After Fix (`4078022318bca0`) |
|-----------|-----------|-------------------------------|
| `isaAllocationPageSize` for BMG | `64KB` (hardcoded) | `2MB` (`is2MBLocalMemAlignmentEnabled()`) |
| DRM heap alignment for `kernelIsa` (BMG) | `0` → defaults to 64KB | `pageSize2M = 2MB` |
| ISA Pool `poolAlignment` for BMG | `1` (no alignment) | `pageSize2M = 2MB` |
| ISA Pool `builtinAllocationSize` for BMG | `64KB` (default) | `pageSize2M = 2MB` |
| Pool backing allocation size alignment | Not aligned | `alignToPoolSize()` → 2MB-aligned |
| `qualifiesFor2MBPages` for ISA on BMG | `false` (address/size not aligned) | `true` (both 2MB-aligned) |

---

## The Secondary Bug: Double ISA Padding (Commit `e2228201`)

This is a **memory waste bug**, not a corruption bug. Before the fix, ISA pool backing
allocations received double-padding from `getPaddingForISAAllocation()` (3584 bytes on
BMG/Xe2). The `HeapAllocator` inside the pool tracked only the requested size, not the
actual allocation size including padding, wasting 3584 bytes per pool.

The fix sets `allocProperties.isaPaddingIncluded = true` and initializes the HeapAllocator
with the actual `GraphicsAllocation` size.

---

## Why the Multi-.so CUTLASS Scenario Triggers This Bug

### The ISA Allocation Decision Tree

```
zeModuleCreate(SPIR-V) → IGC compile → native ISA binary
  │
  ├─ ISA size ≤ isaAllocationPageSize?
  │   ├─ YES → ISA Pool path (shared GraphicsAllocation)
  │   │         Pool backing: 4MB, poolAlignment=1 (bug!) or 2MB (fixed)
  │   │         Sub-allocation offset from pool HeapAllocator
  │   │
  │   └─ NO → Dedicated allocation path (allocateKernelsIsaMemory)
  │            DRM heap: alignment=0/64KB (bug!) or 2MB (fixed)
  │
  └─ Debugger active? → Per-kernel allocation (not relevant here)
```

### Small Kernels (34KB SPIR-V → ~30KB compiled ISA)

- 30KB < 64KB (`isaAllocationPageSize`) → **ISA pool path**
- Pool backing allocation is 4MB. The OS/DRM allocator may happen to place it at a
  2MB-aligned address **by luck** (4MB allocations are often naturally 2MB-aligned)
- → Works by coincidence, not by design

### Large CUTLASS Kernels (185KB+ SPIR-V → >64KB compiled ISA)

- ISA > 64KB (`isaAllocationPageSize`) → **dedicated allocation path**
- `drm_memory_manager::getCanonizedHeapAllocationAddress(alignment=0)`
- GPU VA assigned at next available 64KB-aligned slot in the internal heap
- As more modules are loaded, the heap pointer advances → eventually lands at a
  non-2MB-aligned address
- → **Guaranteed failure** once address alignment becomes unfavorable

### Why the Load Order Matters

The DRM internal heap allocator hands out addresses sequentially. The exact GPU VA
depends on all prior allocations in the process:

| Test | Sequence | ISA GPU VA Alignment | Result |
|------|----------|---------------------|--------|
| A | plain only | First allocation → likely 2MB-aligned | ✅ PASS |
| B | EVT only | First allocation → likely 2MB-aligned | ✅ PASS |
| **C** | load plain → load EVT → call plain | Two `zeModuleCreate` calls; second shifts heap → plain's ISA at non-2MB VA | **❌ DEVICE_LOST** |
| D | load EVT → load plain → call plain | Different allocation order → plain lands at 2MB-aligned VA | ✅ PASS |
| E | load plain → call plain → load EVT → call plain | Plain compiled first (good VA), cached → survives | ✅ PASS |
| F | load plain → load EVT → call EVT | EVT is the last compiled → happens to get aligned VA | ✅ PASS |

> **Note**: The exact pass/fail for each test depends on the GPU VA heap state, which is
> deterministic but depends on all prior allocations. The pattern observed in the test
> matrix is consistent with alignment-sensitive failures in a sequential heap allocator.

### The `qualifiesFor2MBPages` Gate

After allocating any `GraphicsAllocation`, compute-runtime checks:
```cpp
allocation->setQualifiesFor2MBPages(
    allocation->isAllocatedInLocalMemoryPool() &&
    productHelper.is2MBLocalMemAlignmentEnabled() &&
    isAligned(allocation->getGpuAddress(), MemoryConstants::pageSize2M) &&
    isAligned(allocation->getUnderlyingBufferSize(), MemoryConstants::pageSize2M));
```

Before the fix, ISA allocations on BMG fail both alignment checks:
- GPU VA is 64KB-aligned (not 2MB) → first check fails
- ISA size rounded to 64KB, not 2MB → second check fails
- → `qualifiesFor2MBPages = false` → GPU uses 4KB/64KB pages for ISA in local memory

On BMG with `is2MBLocalMemAlignmentEnabled()=true`, the hardware **requires** 2MB pages
for local memory ISA access. Using smaller pages causes incorrect GPU behavior during
instruction fetch → page fault → **DEVICE_LOST**.

---

## What Was Investigated and Ruled Out

### GPU Driver ISA Alignment (NEO-12287)

The NEO-12287 fix (2MB ISA alignment for BMG) was tested with driver versions 26.14
and 26.18 — the crash still reproduces. The alignment fix is correct but addresses a
**different** failure mode (random ISA placement issues with large dedicated allocations).

### Cross-SO Handler/Allocator Hypothesis

The original hypothesis was that handler objects allocated in one `.so`'s allocator
context are being passed to another `.so`, causing heap corruption. This was ruled out:

- Each `zeModuleCreate` call is independent — no handles are shared between modules
- The ISA Pool Allocator does share a backing `GraphicsAllocation` across modules, but
  the sub-allocation tracking (HeapAllocator) is correct
- The `ur_program_handle_t_` objects are per-program with independent lifecycle

The actual "cross-module" interaction is purely through the **GPU virtual address heap**:
the DRM heap allocator hands out sequential addresses, and loading module2 consumes
address space that shifts module1's (or subsequent modules') ISA to a non-2MB-aligned
address.

---

## Other Bugs Found During Investigation (Not the Root Cause)

### 1. Data Race on `m_EliminatedKernelArgMasks` (intel/llvm)

**File**: `sycl/source/detail/program_manager/program_manager.cpp`

`addImage()` writes to `m_EliminatedKernelArgMasks` **before** acquiring any mutex,
while `getEliminatedKernelArgMask()` iterates the same map without locking. Concurrent
`dlopen` + kernel launch = undefined behavior. Not relevant to the sequential reproducer.

### 2. `removeAllRelatedEntries` Only Removes First Match (intel/llvm)

**File**: `sycl/source/detail/kernel_program_cache.hpp`

Despite its name, `removeAllRelatedEntries(ImageId)` uses `std::find_if` which stops at
the first match. If a program has been cached for multiple device/spec-const
configurations, stale cache entries survive `.so` unload.

### 3. New `libsycl` Implementation Drops Duplicate Kernel Names (intel/llvm)

**File**: `libsycl/src/detail/program_manager.cpp`

`MKernelIDToDevImageJIT` is `std::unordered_map` (not multimap). When two `.so` files
share a kernel name, the second registration is silently dropped. The production code
uses `unordered_multimap` + `urDeviceSelectBinary` and handles this correctly.

---

## Environment and Driver Version

| Component | Version | Resolves Crash? |
|-----------|---------|-----------------|
| GPU | Intel Arc Pro B60 (BMG/Xe2, 0xE20B) | N/A (hardware) |
| Level Zero Loader | 1.28.0 | N/A (loader only) |
| GPU Driver (NEO) | 26.09.37435.12 → **26.18.38308.1** (tested) | **❌ No** — this is NOT a driver bug |
| IGC Compiler | 2.34.4 (tested) | **❌ No** — rebuilt .so still crashes |
| SYCL Runtime | oneAPI 2025.3 + conda nightly (tested) | **❌ No** — crashes with both |

**Tested driver versions**: 26.09.37435.12, 26.14.37833.4, 26.18.38308.1 — ALL reproduce the crash.

**Package source**: `ppa:kobuk-team/intel-graphics` (Ubuntu 25.10 / Questing)

---

## Fix: Unique SYCL Kernel Names via Inductor Codegen (Verified ✅)

### The Fix (PyTorch Inductor — `gemm_template.py`)

The fix is in `CUTLASS3xGemmTemplate._define_gemm_instance()` in
`torch/_inductor/codegen/cutlass/gemm_template.py`. For XPU/SYCL targets, the CUTLASS
kernel struct name is made unique by appending the `KERNEL_NAME` placeholder, which
`scheduling.py` replaces with a per-`.so` SHA-256 hash.

**Before** (both `.so` files get the same struct → same SYCL kernel class name):
```python
op_type = match.groups()[0]  # "cutlass3x_..._align8"
op_def += f"\n  using {op_type}_device_type = GemmUniversalAdapter<{op_type}>;\n"
```

**After** (each `.so` gets a unique struct name → unique SYCL kernel class name):
```python
op_type = match.groups()[0]  # "cutlass3x_..._align8"
if self.device_type == "xpu":
    unique_op_type = f"{op_type}_{Placeholder.KERNEL_NAME}"
    op_def = op_def.replace(f"struct {op_type} :", f"struct {unique_op_type} :")
    op_def += f"\n  using {unique_op_type}_device_type = GemmUniversalAdapter<{unique_op_type}>;\n"
```

After `scheduling.py` replaces `KERNEL_NAME` → `cutlass_fused_mm_t_ccb68346`:
```
# kernel1_plain.so struct name:
cutlass3x_..._align8_cutlass_fused_mm_t_ccb68346

# kernel2_evt.so struct name:
cutlass3x_..._align8_cutlass_fused_mm_mul_silu_t_02f33700
```

The SYCL `ProgramManager` now sees **different** kernel class names → selects the
correct device binary for each kernel → **no more DEVICE_LOST**.

### Verification Results

Compiled fixed kernel sources with unique struct names and ran the full test matrix:

| Test | Original (Bug) | Fixed |
|------|:---:|:---:|
| A: plain only | ✅ PASS | ✅ PASS |
| B: evt only | ✅ PASS | ✅ PASS |
| **C: load plain → load evt → call plain** | **❌ DEVICE_LOST** | **✅ PASS** |
| D: load evt → load plain → call plain | ✅ PASS | ✅ PASS |
| E: load plain → call → load evt → call | ✅ PASS | ✅ PASS |
| F: load plain → load evt → call evt | ✅ PASS | ✅ PASS |

**All 6 tests PASS with the fix.** Test C is the critical one — it was the only
failing case and it now passes.

### End-to-End Validation (PyTorch Build from Source)

Built PyTorch from source (`torch-2.13.0a0+gitfa73e24`, branch `xu_cutlass_evt_silu_fusion`)
with `USE_XPU=1` in a clean conda environment and ran the full CUTLASS backend test suite:

| Test | Result |
|------|:---:|
| `test_evt_silu_fusion` | ✅ PASS |
| `test_evt_llama_mlp_pattern` | ✅ PASS |
| `test_evt_aux_load_mul` | ✅ PASS |
| `test_py_codegen_neg_constant` | ✅ PASS |
| `test_py_codegen_silu` | ✅ PASS |
| `test_py_codegen_aux_load_mul` | ✅ PASS |
| `test_py_codegen_silu_mul_aux` | ✅ PASS |
| `test_max_autotune_cutlass_backend_regular_mm` (8 variants) | ✅ PASS |

**All 15 tests pass.** The CUTLASS kernels correctly benchmark and execute with unique
kernel names. Multiple `.so` files with different epilogues no longer conflict.

### Why This Fix Works

1. The `KERNEL_NAME` placeholder already exists in PyTorch Inductor for the C function
   name (`extern "C" int KERNEL_NAME(...)`)
2. `scheduling.py:107` does a global text replace: `src_code.replace("KERNEL_NAME", kernel_name)`
   where `kernel_name` is a unique per-`.so` SHA-256 hash (e.g., `cutlass_fused_mm_t_ccb68346`)
3. By embedding `KERNEL_NAME` in the struct name, the same replacement automatically
   makes the struct unique → the SYCL kernel class name becomes unique
4. The fix is scoped to `device_type == "xpu"` only, with zero impact on CUDA

### Workarounds (If Fix Cannot Be Applied)

1. **JIT-Before-Load** (Proven by Test E): Force kernel compilation before loading the
   second `.so`. Once cached, the SYCL runtime uses the cached kernel.
2. **Single `.so`**: Combine all CUTLASS kernels into one shared object.
3. **Disable EVT Fusion**: `cutlass_epilogue_fusion_enabled=False`

### GPU Driver Upgrade (For NEO-12287 Separately)

The 2MB ISA alignment issue (NEO-12287) is a separate real bug that may affect other
scenarios. If you want the latest driver for other stability improvements:

```bash
# Latest release from GitHub (26.18.38308.1):
mkdir -p /tmp/neo && cd /tmp/neo
wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.34.4/intel-igc-core-2_2.34.4+21428_amd64.deb
wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.34.4/intel-igc-opencl-2_2.34.4+21428_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/26.18.38308.1/libze-intel-gpu1_26.18.38308.1-0_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/26.18.38308.1/intel-opencl-icd_26.18.38308.1-0_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/26.18.38308.1/libigdgmm12_22.10.0_amd64.deb
sudo dpkg -i *.deb
```

---

## Appendix: Key Source Files Analyzed

### compute-runtime (`intel/compute-runtime`)

| File | Purpose |
|------|---------|
| `level_zero/core/source/module/module_imp.cpp` (118KB) | `ModuleImp::initialize`, `setIsaGraphicsAllocations`, `transferIsaSegmentsToAllocation`, `allocateKernelsIsaMemory` |
| `level_zero/core/source/module/module_imp.h` | `ModuleImp` class, `isaAllocationPageSize`, `sharedIsaAllocation`, `kernelsIsaParentRegion` |
| `shared/source/utilities/isa_pool_allocator.h` | `ISAPoolAllocator`, `ISAPool` — pool-based ISA memory management |
| `shared/source/os_interface/linux/drm_memory_manager.cpp` | GPU VA allocation for `AllocationType::kernelIsa` |
| `shared/source/memory_manager/memory_manager.cpp` | `setQualifiesFor2MBPages`, `allocInUse` |
| `shared/source/helpers/gfx_core_helper_pvc_and_later.inl` | `getPaddingForISAAllocation()` = 0xE00 (3584 bytes) for Xe2 |
| `shared/source/dll/buffer_pool_size.cpp` | ISA pool default params: `startingOffset=64KB`, `chunkAlignment=64KB` |

### SYCL Runtime (`intel/llvm`)

| File | Purpose |
|------|---------|
| `sycl/source/detail/program_manager/program_manager.cpp` (146KB) | `__sycl_register_lib` → `addImages` → `addImage`; kernel lookup and build paths |
| `sycl/source/detail/program_manager/program_manager.hpp` | `ProgramManager` member variables: `m_KernelIDs2BinImage` (multimap), `m_DeviceImages`, `m_DeviceKernelInfoMap` |
| `sycl/source/detail/kernel_program_cache.hpp` | Per-context kernel/program cache with `emhash8` fast path |
| `sycl/source/detail/device_kernel_info.hpp` | `DeviceKernelInfo`, `FastKernelSubcacheT` |
| `libsycl/src/detail/program_manager.cpp` | New `ProgramAndKernelManager::registerFatBin` |
| `libsycl/src/detail/device_image_wrapper.hpp` | `DeviceImageWrapper` — thin wrapper around `__sycl_tgt_device_image` |
| `unified-runtime/source/adapters/level_zero/program.cpp` | `urProgramBuildExp` → `zeModuleCreate` call site |
| `unified-runtime/source/adapters/level_zero/program.hpp` | `ur_program_handle_t_` state machine (`IL → Exe`), `ILCode` SPIR-V copy |
