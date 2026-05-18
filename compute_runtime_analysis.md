# Intel compute-runtime Root Cause Analysis: BMG/Xe2 DEVICE_LOST on Multi-.so CUTLASS Kernels

## Executive Summary

The DEVICE_LOST crash on BMG/Xe2 GPUs when loading multiple SYCL kernel `.so` files
via `dlopen` has been traced to a **GPU ISA memory alignment bug** in Intel's
compute-runtime (NEO driver). BMG/Xe2 hardware requires **2MB-aligned GPU virtual
addresses** for ISA (Instruction Set Architecture) allocations in local memory, but the
driver was allocating ISA with only 64KB alignment (or no alignment at all in the ISA
pool path). This causes GPU page faults during instruction fetch → `DEVICE_LOST`.

The bug has been fixed in two commits tracked under **NEO-12287 / HSD-18042276431**.
The user's driver version (`NEO 26.09.37435.12`) predates the fix.

---

## Bug Tracker References

- **Internal**: NEO-12287, HSD-18042276431
- **Primary fix commit**: [`4078022318bca0dfb466c0aeba5a392f47abf7a0`](https://github.com/intel/compute-runtime/commit/4078022318bca0dfb466c0aeba5a392f47abf7a0) — *2025-11-19* — "fix: configure ISA Pool params based on productHelper"
- **Secondary fix commit**: [`e2228201ce6da6e7ae657d7fa0d309d9a38d3912`](https://github.com/intel/compute-runtime/commit/e2228201ce6da6e7ae657d7fa0d309d9a38d3912) — *2025-05-30* — "fix: Avoid redundant padding in ISA allocations"
- **Reverted earlier attempt**: [`bf20ae7ae82a468801a921f4d2e01b9d4b92eb0b`](https://github.com/intel/compute-runtime/commit/bf20ae7ae82a468801a921f4d2e01b9d4b92eb0b) (2025-02-19) → reverted by `f5e37e725cc47cae` (2025-03-10), then correctly re-landed as `4078022318bca0`

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

### SYCL Runtime Layer (`intel/llvm`)

The SYCL runtime's `ProgramManager` and device image registration were thoroughly
analyzed across both the production (`sycl/source/detail/`) and new (`libsycl/src/detail/`)
implementations:

| Hypothesis | Status | Reason |
|------------|--------|--------|
| SPIR-V pointer invalidation across .so | ❌ Ruled out | UR adapter copies SPIR-V into `ILCode` at `urProgramCreateWithIL` |
| `unordered_map` iterator/pointer invalidation | ❌ Ruled out | `m_DeviceKernelInfoMap` uses `std::string` keys; element addresses stable after rehash |
| Kernel name collision between .so files | ❌ Ruled out | CUTLASS kernels have distinct SYCL names (different template params); `m_KernelIDs2BinImage` is a multimap |
| `zeModuleCreate` called at `dlopen` time | ❌ Ruled out | Compilation is lazy — only at first kernel dispatch |
| Data race on `m_EliminatedKernelArgMasks` | ⚠️ Real bug, not this issue | Write-without-lock in `addImage()`, but reproduces in single-threaded mode |

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

| Component | Version | Has Fix? |
|-----------|---------|----------|
| GPU | Intel Arc Pro B60 (BMG/Xe2, 0xE20B) | N/A (hardware) |
| Level Zero Loader | 1.28.0 | N/A (loader only) |
| GPU Driver (NEO) | **26.09.37435.12** (installed) | **❌ No** — fix landed 2025-11-19 |
| GPU Driver (NEO) | **26.14.37833.4** (available in PPA) | **✅ Yes** — post-fix build |
| Fix commit | `4078022318bca0` | In `master` since 2025-11-19 |
| Earlier attempt | `bf20ae7a` (2025-02-19) | Reverted 2025-03-10 |

**Package source**: `ppa:kobuk-team/intel-graphics` (Ubuntu 25.10 / Questing)

---

## Recommended Actions

### Immediate: Update GPU Driver

The fix is already available in the PPA. Version `26.14.37833.4` (build 37833) postdates
the fix commit (2025-11-19) and should contain the 2MB ISA alignment fix for BMG.

#### Upgrade Steps (Ubuntu 25.10 with `ppa:kobuk-team/intel-graphics`)

```bash
# 1. Update package index
sudo apt-get update

# 2. Upgrade GPU driver packages
sudo apt-get upgrade -y libze-intel-gpu1 intel-opencl-icd

# Or upgrade all PPA packages at once:
# sudo apt-get dist-upgrade -y

# 3. Reboot to load the new GPU driver
sudo reboot
```

#### If PPA is Not Yet Configured

```bash
# Add the Intel graphics PPA (Ubuntu 25.10)
sudo add-apt-repository ppa:kobuk-team/intel-graphics
sudo apt-get update
sudo apt-get install -y libze-intel-gpu1 intel-opencl-icd level-zero
```

#### Alternative: Intel Official Repository

For non-Ubuntu or enterprise setups, Intel provides packages at
https://dgpu-docs.intel.com/driver/client/overview.html:

```bash
# Add Intel GPU repository key and source (example for Ubuntu 24.04+)
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
  sudo gpg --dearmor -o /usr/share/keyrings/intel-graphics.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] \
  https://repositories.intel.com/gpu/ubuntu noble unified" | \
  sudo tee /etc/apt/sources.list.d/intel-gpu.list
sudo apt-get update
sudo apt-get install -y intel-opencl-icd intel-level-zero-gpu
```

### Verification

```bash
# Check installed version after upgrade:
dpkg -l libze-intel-gpu1 | grep intel
# Expected: 26.14.37833.4-1~25.10~ppa1 or newer

# Verify GPU is functional:
clinfo | grep "Device Name"
# Expected: Intel(R) Arc(TM) Pro B60 Graphics

# Quick Level Zero sanity check:
ze_info 2>/dev/null || echo "ze_info not installed (optional)"

# Run the reproducer to confirm the fix:
cd /path/to/pytorch_pr_181854_crash_reproducer
python standalone_repro_evt.py
# Expected: No DEVICE_LOST, all tests pass
```

### PyTorch Inductor Workaround (Until Driver Update)

The existing workarounds documented in `evt_device_lost_analysis.md` remain effective:

1. **JIT-on-load**: Force SPIR-V → ISA compilation immediately after each `dlopen`,
   before loading the next `.so`. This reduces the chance of the heap allocator placing
   ISA at non-2MB-aligned addresses (though it's not a guaranteed fix).

2. **Single `.so`**: Combine both CUTLASS kernels into one `.so` file. With one
   `zeModuleCreate` call, both kernels' ISA is allocated as a single contiguous block,
   more likely to be 2MB-aligned.

3. **Disable EVT fusion**: `cutlass_epilogue_fusion_enabled=False` avoids the second
   large CUTLASS `.so` entirely.

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
