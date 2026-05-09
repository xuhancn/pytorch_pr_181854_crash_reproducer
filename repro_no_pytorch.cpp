/**
 * Pure C++ reproducer for CUTLASS EVT multi-.so DEVICE_LOST on BMG/Xe2.
 *
 * No PyTorch dependency — only requires icpx (oneAPI) and sycl-tla headers.
 *
 * ROOT CAUSE: When two CUTLASS SYCL kernel .so files are loaded via dlopen,
 * loading the second .so invalidates the first .so's un-JIT'd SPIR-V device
 * code in the Level Zero runtime. Calling the first kernel then executes
 * corrupted code → DEVICE_LOST.
 *
 * Build & Run:
 *   source /opt/intel/oneapi/setvars.sh
 *   ./build_and_run.sh
 *
 * Or manually:
 *   # Build kernels and harness (see build_and_run.sh)
 *   # Run single test:
 *   ./repro_no_pytorch <kernel1.so> <kernel2.so> <test_name>
 *   # test_name: A, B, C, D, E, F, or "info"
 */

#include <sycl/sycl.hpp>
#include <dlfcn.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Function pointer types matching the CUTLASS kernel signatures
using PlainKernelFn = int (*)(
    const uint16_t* X, const uint16_t* W, uint16_t* Y,
    int M, int N, int K, int B,
    int lda, int ldb, int ldc, int ldd,
    int X_offset, int W_offset, int Y_offset,
    uint8_t swizzle,
    size_t* workspace_size, uint8_t* workspace,
    sycl::queue* stream);

using EvtKernelFn = int (*)(
    const uint16_t* X, const uint16_t* W, const uint16_t* ptr_0,
    const uint16_t* Y, uint16_t* buf2,
    int M, int N, int K, int B,
    int lda, int ldb, int ldc, int ldd,
    int X_offset, int W_offset, int ptr_0_offset, int Y_offset, int buf2_offset,
    uint8_t swizzle,
    size_t* workspace_size, uint8_t* workspace,
    sycl::queue* stream);

void* load_so(const char* path) {
    void* h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        fprintf(stderr, "dlopen(%s) failed: %s\n", path, dlerror());
        exit(2);
    }
    return h;
}

static constexpr int M = 256, N = 256, K = 256;

bool call_plain(PlainKernelFn fn, sycl::queue& q,
                uint16_t* X, uint16_t* W, uint16_t* Y) {
    try {
        fn(X, W, Y, M, N, K, 1, K, K, 0, N, 0, 0, 0, 2, nullptr, nullptr, &q);
        q.wait_and_throw();
        return true;
    } catch (std::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        return false;
    }
}

bool call_evt(EvtKernelFn fn, sycl::queue& q,
              uint16_t* X, uint16_t* W, uint16_t* buf0, uint16_t* Y) {
    try {
        fn(X, W, buf0, Y, Y, M, N, K, 1, K, K, 0, N, 0, 0, 0, 0, 0, 2,
           nullptr, nullptr, &q);
        q.wait_and_throw();
        return true;
    } catch (std::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        return false;
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr,
            "Usage: %s <kernel1_plain.so> <kernel2_evt.so> <test>\n"
            "  test: info, A, B, C, D, E, F\n", argv[0]);
        return 1;
    }
    const char* so1_path = argv[1];
    const char* so2_path = argv[2];
    const char* test = argv[3];

    // "info" mode: just print device info and exit
    if (strcmp(test, "info") == 0) {
        sycl::queue q{sycl::gpu_selector_v};
        auto dev = q.get_device();
        printf("Device:  %s\n", dev.get_info<sycl::info::device::name>().c_str());
        printf("Driver:  %s\n",
               dev.get_info<sycl::info::device::driver_version>().c_str());
        return 0;
    }

    // Create queue and allocate memory
    sycl::queue q{sycl::gpu_selector_v};
    auto* X    = sycl::malloc_device<uint16_t>(M * K, q);
    auto* W    = sycl::malloc_device<uint16_t>(N * K, q);
    auto* buf0 = sycl::malloc_device<uint16_t>(M * N, q);
    auto* Y    = sycl::malloc_device<uint16_t>(M * N, q);
    q.memset(X, 0x3C, M * K * sizeof(uint16_t));   // ~1.0 in fp16
    q.memset(W, 0x3C, N * K * sizeof(uint16_t));
    q.memset(buf0, 0x3C, M * N * sizeof(uint16_t));
    q.memset(Y, 0, M * N * sizeof(uint16_t));
    q.wait();

    bool ok = false;

    if (strcmp(test, "A") == 0) {
        // A: plain only
        auto so = load_so(so1_path);
        auto fn = (PlainKernelFn)dlsym(so, "cutlass_fused_mm_t_ccb68346");
        ok = call_plain(fn, q, X, W, Y);

    } else if (strcmp(test, "B") == 0) {
        // B: EVT only
        auto so = load_so(so2_path);
        auto fn = (EvtKernelFn)dlsym(so, "cutlass_fused_mm_mul_silu_t_02f33700");
        ok = call_evt(fn, q, X, W, buf0, Y);

    } else if (strcmp(test, "C") == 0) {
        // C: load plain → load EVT → call plain [BUG]
        auto so1 = load_so(so1_path);
        auto fn = (PlainKernelFn)dlsym(so1, "cutlass_fused_mm_t_ccb68346");
        load_so(so2_path);
        ok = call_plain(fn, q, X, W, Y);

    } else if (strcmp(test, "D") == 0) {
        // D: load EVT → load plain → call plain
        load_so(so2_path);
        auto so1 = load_so(so1_path);
        auto fn = (PlainKernelFn)dlsym(so1, "cutlass_fused_mm_t_ccb68346");
        ok = call_plain(fn, q, X, W, Y);

    } else if (strcmp(test, "E") == 0) {
        // E: load plain → call plain → load EVT → call plain
        auto so1 = load_so(so1_path);
        auto fn = (PlainKernelFn)dlsym(so1, "cutlass_fused_mm_t_ccb68346");
        ok = call_plain(fn, q, X, W, Y);
        if (ok) {
            load_so(so2_path);
            ok = call_plain(fn, q, X, W, Y);
        }

    } else if (strcmp(test, "F") == 0) {
        // F: load plain → load EVT → call EVT
        load_so(so1_path);
        auto so2 = load_so(so2_path);
        auto fn = (EvtKernelFn)dlsym(so2, "cutlass_fused_mm_mul_silu_t_02f33700");
        ok = call_evt(fn, q, X, W, buf0, Y);

    } else {
        fprintf(stderr, "Unknown test: %s\n", test);
        return 1;
    }

    printf("%s\n", ok ? "PASS" : "FAIL");

    sycl::free(X, q);
    sycl::free(W, q);
    sycl::free(buf0, q);
    sycl::free(Y, q);

    return ok ? 0 : 1;
}
