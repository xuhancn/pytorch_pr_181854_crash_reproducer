#!/bin/bash
# Build and run the pure C++ reproducer (no PyTorch dependency).
#
# Prerequisites:
#   source /opt/intel/oneapi/setvars.sh   (or ~/intel/oneapi/setvars.sh)
#   git submodule update --init           (to get third_party/sycl-tla)
#
# Usage:
#   ./build_and_run.sh              # Build everything and run test matrix
#   ./build_and_run.sh --build-only # Just build, don't run

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SYCL_TLA_DIR="${TORCHINDUCTOR_CUTLASS_DIR:-$SCRIPT_DIR/third_party/sycl-tla}"
BUILD_DIR="$SCRIPT_DIR/_build_standalone"

# Auto-detect GPU target
GPU_TARGET="${SYCL_GPU_TARGET:-intel_gpu_bmg_g21}"

# Check prerequisites
if ! command -v icpx &>/dev/null; then
    echo "ERROR: icpx not found. Run: source /opt/intel/oneapi/setvars.sh"
    exit 1
fi

if [ ! -d "$SYCL_TLA_DIR/include/cutlass" ]; then
    echo "ERROR: sycl-tla not found at $SYCL_TLA_DIR"
    echo "  Run: git submodule update --init"
    exit 1
fi

mkdir -p "$BUILD_DIR"

# ─── Common compile flags (matching inductor exactly) ───────────────────
INCLUDE_FLAGS="\
    -I$SYCL_TLA_DIR/include \
    -I$SYCL_TLA_DIR/tools/library/include \
    -I$SYCL_TLA_DIR/tools/library/src \
    -I$SYCL_TLA_DIR/tools/util/include"

COMMON_FLAGS="\
    -DCUTLASS_ENABLE_SYCL \
    -DSYCL_INTEL_TARGET \
    -DCUTLASS_VERSIONS_GENERATED \
    -O3 -DNDEBUG \
    -std=c++20 -fPIC \
    -fsycl \
    -fsycl-targets=$GPU_TARGET \
    -Xspirv-translator \
    -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate \
    -fno-sycl-instrument-device-code"

XS_FLAGS='-Xs -options\ \"-igc_opts\ '\''VISAOptions=-perfmodel,VectorAliasBBThreshold=100000000000,ExtraOCLOptions=-cl-intel-256-GRF-per-thread'\''\"\ -options\ -ze-opt-large-register-file'

COMPILER_LIB="$(dirname $(dirname $(which icpx)))/lib"
LINK_FLAGS="-L$COMPILER_LIB -Xlinker -rpath=$COMPILER_LIB -lsycl"

# ─── Helper: compile a CUTLASS kernel .sycl → .so ──────────────────────
compile_kernel() {
    local SRC="$1"
    local SO="$2"
    local BASENAME=$(basename "$SRC" .sycl)
    local CPP="$BUILD_DIR/${BASENAME}.cpp"
    local OBJ="$BUILD_DIR/${BASENAME}.o"

    if [ -f "$SO" ] && [ "$SO" -nt "$SRC" ]; then
        echo "${BASENAME}.so: up to date"
        return
    fi

    cp "$SRC" "$CPP"
    echo "Compiling ${BASENAME}.so..."

    # Step 1: compile to .o
    icpx \
        -I"$SYCL_TLA_DIR/include" \
        -I"$SYCL_TLA_DIR/tools/library/include" \
        -I"$SYCL_TLA_DIR/tools/library/src" \
        -I"$SYCL_TLA_DIR/tools/util/include" \
        -DCUTLASS_ENABLE_SYCL \
        -DSYCL_INTEL_TARGET \
        -DCUTLASS_VERSIONS_GENERATED \
        -O3 -DNDEBUG \
        -std=c++20 -fPIC \
        -fsycl \
        -fsycl-targets=$GPU_TARGET \
        -Xspirv-translator \
        "-spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate" \
        -fno-sycl-instrument-device-code \
        -Xs "-options \"-igc_opts 'VISAOptions=-perfmodel,VectorAliasBBThreshold=100000000000,ExtraOCLOptions=-cl-intel-256-GRF-per-thread'\" -options -ze-opt-large-register-file" \
        -c -o "$OBJ" "$CPP"

    # Step 2: link to .so
    icpx \
        -fsycl \
        -fsycl-targets=$GPU_TARGET \
        -Xspirv-translator \
        "-spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate" \
        -fno-sycl-instrument-device-code \
        -Xs "-options \"-igc_opts 'VISAOptions=-perfmodel,VectorAliasBBThreshold=100000000000,ExtraOCLOptions=-cl-intel-256-GRF-per-thread'\" -options -ze-opt-large-register-file" \
        -L"$COMPILER_LIB" -Xlinker -rpath="$COMPILER_LIB" -lsycl \
        -shared -o "$SO" "$OBJ"
}

# ─── Build kernel 1 (plain GEMM) ───────────────────────────────────────
KERNEL1_SO="$BUILD_DIR/kernel1_plain.so"
compile_kernel "$SCRIPT_DIR/_inductor_kernel1_plain.sycl" "$KERNEL1_SO"

# ─── Build kernel 2 (EVT GEMM) ─────────────────────────────────────────
KERNEL2_SO="$BUILD_DIR/kernel2_evt.so"
compile_kernel "$SCRIPT_DIR/_inductor_kernel2_evt.sycl" "$KERNEL2_SO"

# ─── Build test harness ────────────────────────────────────────────────
HARNESS="$BUILD_DIR/repro_no_pytorch"

if [ ! -f "$HARNESS" ] || [ "$SCRIPT_DIR/repro_no_pytorch.cpp" -nt "$HARNESS" ]; then
    echo "Compiling test harness..."
    icpx -fsycl -fsycl-targets=$GPU_TARGET -std=c++20 -O2 \
        -o "$HARNESS" "$SCRIPT_DIR/repro_no_pytorch.cpp" -ldl
else
    echo "repro_no_pytorch: up to date"
fi

echo "Build complete."

if [ "$1" = "--build-only" ]; then
    exit 0
fi

echo ""
echo "Running test matrix..."
echo ""
HARNESS_BIN="$HARNESS"
SO1="$KERNEL1_SO"
SO2="$KERNEL2_SO"

# Print device info
"$HARNESS_BIN" "$SO1" "$SO2" info
echo "SO1:     $SO1"
echo "SO2:     $SO2"
echo ""
echo "======================================================================"
echo "Test Matrix: .so Load Order vs Kernel Call"
echo "  Each test runs as a separate process for clean GPU state."
echo "======================================================================"

ALL_EXPECTED=true

run_test() {
    local LABEL="$1"
    local TEST="$2"
    local EXPECT_FAIL="${3:-false}"

    # Run in separate process, capture output and exit code
    local EXIT_CODE=0
    OUTPUT=$("$HARNESS_BIN" "$SO1" "$SO2" "$TEST" 2>/dev/null) || EXIT_CODE=$?

    if [ "$EXIT_CODE" -eq 0 ]; then
        RESULT="PASS"
    else
        RESULT="FAIL"
    fi

    if [ "$EXPECT_FAIL" = "true" ]; then
        if [ "$RESULT" = "FAIL" ]; then
            MATCH=""
        else
            MATCH="  *** UNEXPECTED ***"
            ALL_EXPECTED=false
        fi
    else
        if [ "$RESULT" = "PASS" ]; then
            MATCH=""
        else
            MATCH="  *** UNEXPECTED ***"
            ALL_EXPECTED=false
        fi
    fi

    printf "  %-55s [%s]%s\n" "$LABEL" "$RESULT" "$MATCH"
}

run_test "A: plain only"                                        A
run_test "B: EVT only"                                          B
run_test "C: load plain -> load EVT -> call plain  [BUG]"      C true
run_test "D: load EVT -> load plain -> call plain"              D
run_test "E: load+call plain -> load EVT -> call plain"         E
run_test "F: load plain -> load EVT -> call EVT"                F

echo ""
echo "======================================================================"
echo "Analysis"
echo "======================================================================"
if [ "$ALL_EXPECTED" = "true" ]; then
    echo "  All tests match expected behavior."
    echo ""
    echo "  Root cause: dlopen of second SYCL .so invalidates first .so's"
    echo "  un-JIT'd SPIR-V device code in Level Zero runtime."
else
    echo "  Some tests had unexpected results. The bug pattern may differ"
    echo "  on this hardware/driver combination."
fi
