#!/bin/bash

# Independent Beer Test Orchestrator
# Compiles and tests beer model with and without CMSIS-NN independently

set -e

REPO_ROOT="/home/marco/Z-Ant"
BUILD_DIR="$REPO_ROOT/build/beer_independent"
RESULTS_FILE="$BUILD_DIR/results.txt"

echo "=== Independent Beer Test Orchestrator ==="
echo "Repository: $REPO_ROOT"
echo "Build directory: $BUILD_DIR"
echo

# Create build directory
mkdir -p "$BUILD_DIR"
rm -f "$RESULTS_FILE"

# Function to log results
log_result() {
    echo "$1" | tee -a "$RESULTS_FILE"
}

cd "$REPO_ROOT"

echo "=== Step 1: Build Beer Library WITHOUT CMSIS-NN ==="
echo "Building reference beer library (no CMSIS acceleration)..."

# Clean previous builds
rm -rf zig-out/beer/

# Build beer library without CMSIS-NN
ZANT_FBA_SIZE_KB=320 ZANT_FBA_SECTION=.tensor_pool \
zig build lib \
    -Dmodel=beer \
    -Dfuse \
    -Ddo_export \
    -Ddynamic \
    -Dtarget=thumb-freestanding \
    -Dcpu=cortex_m55 \
    -Doptimize=ReleaseSmall

# Copy the reference library
cp zig-out/beer/libzant.a "$BUILD_DIR/libzant_reference.a"
echo "✅ Reference library built and saved"

echo
echo "=== Step 2: Test Reference (No CMSIS-NN) ==="

# Run reference test using our custom build
python3 - << 'EOF'
import subprocess
import sys
from pathlib import Path

# Test configuration
REPO_ROOT = Path("/home/marco/Z-Ant")
BUILD_DIR = REPO_ROOT / "build/beer_independent"
HARNESS_DIR = REPO_ROOT / "tests/stm32n6_qemu"
GENERATED_DIR = REPO_ROOT / "generated/beer"

# Source files (no CMSIS sources)
sources = [
    HARNESS_DIR / "runtime.c",
    HARNESS_DIR / "semihost_arm.c", 
    HARNESS_DIR / "semihost_arm.S",
    HARNESS_DIR / "support.c",
    HARNESS_DIR / "beer_main.c",
    REPO_ROOT / "src/Core/Tensor/Accelerators/stm32n6/conv_f32.c",
    REPO_ROOT / "src/Core/Tensor/Accelerators/stm32n6/ethos_stub.c",
    BUILD_DIR / "libzant_reference.a"
]

# Build reference executable
elf_path = BUILD_DIR / "beer_reference.elf"
cmd = [
    "arm-none-eabi-gcc",
    "-mcpu=cortex-m55", "-mthumb", "-mfloat-abi=soft",
    "-ffreestanding", "-fno-builtin", "-fno-exceptions", "-fno-stack-protector",
    "-Wall", "-Wextra", "-Wno-unused-parameter", "-O2", "-g", "-nostdlib",
    "-Wl,-T", str(HARNESS_DIR / "stm32n6.ld"),
    "-Wl,--gc-sections", "-Wl,--nmagic",
    "-Wl,-Map=" + str(BUILD_DIR / "beer_reference.map"),
    "-o", str(elf_path),
    "-I", str(REPO_ROOT / "src/Core/Tensor/Accelerators/stm32n6"),
    "-I", str(HARNESS_DIR),
    "-I", str(GENERATED_DIR),
] + [str(s) for s in sources if s.exists()] + [
    "-Wl,--start-group", "-lgcc", "-Wl,--end-group"
]

print("Building reference executable...")
subprocess.run(cmd, check=True, cwd=REPO_ROOT)
print("✅ Reference executable built")

# Run reference test
print("Running reference test...")
qemu_cmd = [
    "qemu-system-arm",
    "-M", "mps3-an547",
    "-nographic", "-monitor", "none", "-serial", "stdio",
    "-kernel", str(elf_path)
]

import time
start_time = time.time()
result = subprocess.run(qemu_cmd, capture_output=True, text=True, timeout=30)
end_time = time.time()

if "beer PASS" in result.stdout:
    duration = (end_time - start_time) * 1000
    print(f"✅ Reference test PASSED in {duration:.2f} ms")
    with open(BUILD_DIR / "results.txt", "a") as f:
        f.write(f"beer_reference: {duration:.2f} ms\n")
else:
    print("❌ Reference test FAILED")
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    sys.exit(1)
EOF

echo
echo "=== Step 3: Build Beer Library WITH CMSIS-NN ==="
echo "Building optimized beer library (with CMSIS-NN acceleration)..."

# Clean previous builds
rm -rf zig-out/beer/

# Build beer library with CMSIS-NN
ZANT_FBA_SIZE_KB=320 ZANT_FBA_SECTION=.tensor_pool \
zig build lib \
    -Dmodel=beer \
    -Dfuse \
    -Ddo_export \
    -Ddynamic \
    -Dstm32n6_accel \
    -Dstm32n6_use_cmsis \
    -Dtarget=thumb-freestanding \
    -Dcpu=cortex_m55 \
    -Doptimize=ReleaseSmall

# Copy the CMSIS library
cp zig-out/beer/libzant.a "$BUILD_DIR/libzant_cmsis.a"
echo "✅ CMSIS-NN library built and saved"

echo
echo "=== Step 4: Test CMSIS-NN Optimized ==="

# Run CMSIS-NN test using our custom build
python3 - << 'EOF'
import subprocess
import sys
from pathlib import Path

# Test configuration
REPO_ROOT = Path("/home/marco/Z-Ant")
BUILD_DIR = REPO_ROOT / "build/beer_independent"
HARNESS_DIR = REPO_ROOT / "tests/stm32n6_qemu"
GENERATED_DIR = REPO_ROOT / "generated/beer"

# CMSIS-NN source files
cmsis_nn_sources = [
    REPO_ROOT / "third_party/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_s8.c",
    REPO_ROOT / "third_party/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_get_buffer_sizes_s8.c",
    REPO_ROOT / "third_party/CMSIS-NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s8_s16.c",
    REPO_ROOT / "third_party/CMSIS-NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_row_offset_s8_s16.c",
    REPO_ROOT / "third_party/CMSIS-NN/Source/NNSupportFunctions/arm_s8_to_s16_unordered_with_offset.c",
    REPO_ROOT / "third_party/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s8.c",
    REPO_ROOT / "third_party/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s8.c",
    REPO_ROOT / "third_party/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s8_s32.c",
]

# CMSIS-DSP source files  
cmsis_dsp_sources = [
    REPO_ROOT / "third_party/CMSIS-DSP/Source/BasicMathFunctions/arm_dot_prod_f32.c",
]

# All source files
sources = [
    HARNESS_DIR / "runtime.c",
    HARNESS_DIR / "semihost_arm.c",
    HARNESS_DIR / "semihost_arm.S", 
    HARNESS_DIR / "support.c",
    HARNESS_DIR / "beer_main.c",
    REPO_ROOT / "src/Core/Tensor/Accelerators/stm32n6/conv_f32.c",
    REPO_ROOT / "src/Core/Tensor/Accelerators/stm32n6/ethos_stub.c",
] + cmsis_nn_sources + cmsis_dsp_sources + [
    BUILD_DIR / "libzant_cmsis.a"
]

# Build CMSIS-NN executable
elf_path = BUILD_DIR / "beer_cmsis.elf"
cmd = [
    "arm-none-eabi-gcc",
    "-mcpu=cortex-m55", "-mthumb", "-mfloat-abi=soft",
    "-ffreestanding", "-fno-builtin", "-fno-exceptions", "-fno-stack-protector", 
    "-Wall", "-Wextra", "-Wno-unused-parameter", "-O2", "-g", "-nostdlib",
    "-Wl,-T", str(HARNESS_DIR / "stm32n6.ld"),
    "-Wl,--gc-sections", "-Wl,--nmagic",
    "-Wl,-Map=" + str(BUILD_DIR / "beer_cmsis.map"),
    "-o", str(elf_path),
    "-I", str(REPO_ROOT / "src/Core/Tensor/Accelerators/stm32n6"),
    "-I", str(HARNESS_DIR),
    "-I", str(GENERATED_DIR),
    "-I", str(REPO_ROOT / "third_party/CMSIS-DSP/Include"),
    "-I", str(REPO_ROOT / "third_party/CMSIS-NN/Include"), 
    "-I", str(REPO_ROOT / "third_party/CMSIS_5/CMSIS/Core/Include"),
    "-DZANT_HAS_CMSIS_DSP=1",
    "-DZANT_HAS_CMSIS_NN=1",
] + [str(s) for s in sources if s.exists()] + [
    "-Wl,--start-group", "-lgcc", "-Wl,--end-group"
]

print("Building CMSIS-NN executable...")
subprocess.run(cmd, check=True, cwd=REPO_ROOT)
print("✅ CMSIS-NN executable built")

# Run CMSIS-NN test
print("Running CMSIS-NN optimized test...")
qemu_cmd = [
    "qemu-system-arm",
    "-M", "mps3-an547", 
    "-nographic", "-monitor", "none", "-serial", "stdio",
    "-kernel", str(elf_path)
]

import time
start_time = time.time()
result = subprocess.run(qemu_cmd, capture_output=True, text=True, timeout=30)
end_time = time.time()

if "beer PASS" in result.stdout:
    duration = (end_time - start_time) * 1000
    print(f"✅ CMSIS-NN test PASSED in {duration:.2f} ms")
    with open(BUILD_DIR / "results.txt", "a") as f:
        f.write(f"beer_cmsis_nn: {duration:.2f} ms\n")
else:
    print("❌ CMSIS-NN test FAILED")
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    sys.exit(1)
EOF

echo
echo "=== Results Summary ==="
if [ -f "$RESULTS_FILE" ]; then
    cat "$RESULTS_FILE"
    echo
    
    # Calculate performance improvement
    python3 - << EOF
import re
from pathlib import Path

results_file = Path("$RESULTS_FILE")
if results_file.exists():
    content = results_file.read_text()
    
    ref_match = re.search(r'beer_reference: ([\d.]+) ms', content)
    cmsis_match = re.search(r'beer_cmsis_nn: ([\d.]+) ms', content) 
    
    if ref_match and cmsis_match:
        ref_time = float(ref_match.group(1))
        cmsis_time = float(cmsis_match.group(1))
        
        improvement = ref_time - cmsis_time
        improvement_pct = (improvement / ref_time) * 100
        
        print(f"Performance Analysis:")
        print(f"  Reference (no CMSIS-NN): {ref_time:.2f} ms")
        print(f"  CMSIS-NN optimized:     {cmsis_time:.2f} ms") 
        print(f"  Improvement:             {improvement:.2f} ms ({improvement_pct:.1f}%)")
        
        if improvement > 0:
            print(f"  🚀 CMSIS-NN optimization is FASTER!")
        else:
            print(f"  ⚠️  CMSIS-NN optimization is slower")
EOF
else
    echo "❌ No results file found"
fi

echo
echo "=== Test Complete ==="
echo "Build artifacts saved in: $BUILD_DIR"
