#!/bin/bash

# Beer Performance Comparison Script
# Builds two different beer libraries and tests them independently

set -e

REPO_ROOT="/home/marco/Z-Ant"
BUILD_DIR="$REPO_ROOT/build/beer_comparison"
RESULTS_FILE="$BUILD_DIR/comparison_results.txt"

echo "=== Beer Performance Comparison ==="
echo "Repository: $REPO_ROOT"
echo "Build directory: $BUILD_DIR"
echo

# Create build directory
mkdir -p "$BUILD_DIR"
rm -f "$RESULTS_FILE"

cd "$REPO_ROOT"

echo "=== Step 1: Generate and Build Beer Library WITHOUT CMSIS-NN ==="
echo "Generating beer model with tensor pool support..."

# Generate beer model with tensor pool allocation
zig build lib-gen -Dmodel=beer -Dfuse -Duse_tensor_pool=true -Ddo_export=true

echo "Building reference beer library (no CMSIS acceleration)..."

# Clean previous builds  
rm -rf zig-out/beer/

# Build beer library without CMSIS-NN acceleration
zig build lib \
    -Dmodel=beer \
    -Dfuse \
    -Duse_tensor_pool=true \
    -Ddo_export=true \
    -Dtarget=thumb-freestanding \
    -Dcpu=cortex_m55 \
    -Doptimize=ReleaseSmall

# Backup the reference library
cp zig-out/beer/libzant.a "$BUILD_DIR/libzant_reference.a"
echo "✅ Reference library built and backed up"

echo
echo "=== Step 2: Test Reference (No CMSIS-NN) ==="

# Create a simple test script that uses the existing test infrastructure
cat > "$BUILD_DIR/test_reference.py" << 'EOF'
#!/usr/bin/env python3
import sys
import subprocess
import shutil
from pathlib import Path

# Add the scripts directory to path to import the original test module
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

# Import the original test function components
from test_stm32n6_qemu import run_qemu, detect_qemu, ArmGccToolchain

REPO_ROOT = Path(__file__).parent.parent.parent
BUILD_DIR = Path(__file__).parent
HARNESS_DIR = REPO_ROOT / "tests/stm32n6_qemu"
GENERATED_DIR = REPO_ROOT / "generated/beer"
STM32_DIR = REPO_ROOT / "src/Core/Tensor/Accelerators/stm32n6"

# Copy back the reference library
ref_lib = BUILD_DIR / "libzant_reference.a"
target_lib = REPO_ROOT / "zig-out/beer/libzant.a"
target_lib.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(ref_lib, target_lib)

# Use the exact same build process as the original script
BEER_SOURCES = (
    HARNESS_DIR / "runtime.c",
    HARNESS_DIR / "semihost_arm.c",
    HARNESS_DIR / "semihost_arm.S",
    HARNESS_DIR / "support.c",
    HARNESS_DIR / "beer_main.c",
    STM32_DIR / "conv_f32.c",
    STM32_DIR / "ethos_stub.c",
)

toolchain = ArmGccToolchain("arm-none-eabi")
qemu_path = detect_qemu(None)

elf_path = BUILD_DIR / "beer_reference.elf"

# Build reference executable using original build process
toolchain.build(
    output=elf_path,
    base_sources=BEER_SOURCES,
    macros=(*toolchain.default_macros(),),  # No CMSIS macros
    extra_sources=(target_lib,),  # Only the beer library
    include_dirs=(GENERATED_DIR,),
)

print("✅ Reference executable built using original build process")

# Run test using original QEMU runner
print("Running reference test...")
import time
start_time = time.perf_counter()
result = run_qemu(
    qemu_path,
    elf_path,
    verbose=False,
    success_marker="beer PASS",
    timeout=30.0
)
end_time = time.perf_counter()

if result.returncode == 0:
    duration = (end_time - start_time) * 1000
    print(f"✅ Reference test PASSED in {duration:.2f} ms")
    
    with open(BUILD_DIR / "comparison_results.txt", "w") as f:
        f.write(f"beer_reference: {duration:.2f} ms\n")
else:
    print("❌ Reference test FAILED")
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    sys.exit(1)
EOF

python3 "$BUILD_DIR/test_reference.py"

echo
echo "=== Step 3: Generate and Build Beer Library WITH CMSIS-NN ==="
echo "Regenerating beer model for CMSIS-NN build..."

# Regenerate beer model (same generation works for both)
zig build lib-gen -Dmodel=beer -Dfuse -Duse_tensor_pool=true -Ddo_export=true

echo "Building optimized beer library (with CMSIS-NN acceleration)..."

# Clean previous builds
rm -rf zig-out/beer/

# Build beer library with CMSIS-NN acceleration
zig build lib \
    -Dmodel=beer \
    -Dfuse \
    -Duse_tensor_pool=true \
    -Ddo_export=true \
    -Dstm32n6_accel=true \
    -Dstm32n6_use_cmsis=true \
    -Dstm32n6_use_ethos=true \
    -Dtarget=thumb-freestanding \
    -Dcpu=cortex_m55 \
    -Doptimize=ReleaseSmall

# Backup the CMSIS library
cp zig-out/beer/libzant.a "$BUILD_DIR/libzant_cmsis.a"
echo "✅ CMSIS-NN library built and backed up"

echo
echo "=== Step 4: Test CMSIS-NN Optimized ==="

# Create test script for CMSIS-NN version
cat > "$BUILD_DIR/test_cmsis.py" << 'EOF'
#!/usr/bin/env python3
import sys
import subprocess
import shutil
from pathlib import Path

# Add the scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from test_stm32n6_qemu import run_qemu, detect_qemu, ArmGccToolchain, get_cmsis_sources, get_cmsis_dsp_sources

REPO_ROOT = Path(__file__).parent.parent.parent
BUILD_DIR = Path(__file__).parent
HARNESS_DIR = REPO_ROOT / "tests/stm32n6_qemu"
GENERATED_DIR = REPO_ROOT / "generated/beer"
STM32_DIR = REPO_ROOT / "src/Core/Tensor/Accelerators/stm32n6"

# Copy back the CMSIS library
cmsis_lib = BUILD_DIR / "libzant_cmsis.a"
target_lib = REPO_ROOT / "zig-out/beer/libzant.a"
target_lib.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(cmsis_lib, target_lib)

# Get CMSIS sources
nn_include = REPO_ROOT / "third_party/CMSIS-NN/Include"
dsp_include = REPO_ROOT / "third_party/CMSIS-DSP/Include"
convolve_source = REPO_ROOT / "third_party/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_s8.c"

cmsis_nn_sources = get_cmsis_sources(convolve_source, nn_include)
cmsis_dsp_sources = get_cmsis_dsp_sources(dsp_include)

BEER_SOURCES = (
    HARNESS_DIR / "runtime.c",
    HARNESS_DIR / "semihost_arm.c",
    HARNESS_DIR / "semihost_arm.S",
    HARNESS_DIR / "support.c",
    HARNESS_DIR / "beer_main.c",
    STM32_DIR / "conv_f32.c",
    STM32_DIR / "ethos_stub.c",
)

# Setup includes
include_dirs = [dsp_include]
if nn_include != dsp_include:
    include_dirs.append(nn_include)
cmsis_core = REPO_ROOT / "third_party/CMSIS_5/CMSIS/Core/Include"
if cmsis_core.exists():
    include_dirs.append(cmsis_core)
include_dirs.append(GENERATED_DIR)

toolchain = ArmGccToolchain("arm-none-eabi")
qemu_path = detect_qemu(None)

elf_path = BUILD_DIR / "beer_cmsis.elf"

# Build CMSIS executable using original build process
toolchain.build(
    output=elf_path,
    base_sources=BEER_SOURCES,
    macros=(*toolchain.default_macros(), "ZANT_HAS_CMSIS_DSP=1", "ZANT_HAS_CMSIS_NN=1"),
    extra_sources=(*cmsis_nn_sources, *cmsis_dsp_sources, target_lib),
    include_dirs=tuple(include_dirs),
)

print("✅ CMSIS-NN executable built using original build process")

# Run test using original QEMU runner
print("Running CMSIS-NN optimized test...")
import time
start_time = time.perf_counter()
result = run_qemu(
    qemu_path,
    elf_path,
    verbose=False,
    success_marker="beer PASS",
    timeout=30.0
)
end_time = time.perf_counter()

if result.returncode == 0:
    duration = (end_time - start_time) * 1000
    print(f"✅ CMSIS-NN test PASSED in {duration:.2f} ms")
    
    with open(BUILD_DIR / "comparison_results.txt", "a") as f:
        f.write(f"beer_cmsis_nn: {duration:.2f} ms\n")
else:
    print("❌ CMSIS-NN test FAILED")
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    sys.exit(1)
EOF

python3 "$BUILD_DIR/test_cmsis.py"

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
            print(f"  🚀 CMSIS-NN optimization is {improvement_pct:.1f}% FASTER!")
        else:
            print(f"  ⚠️  CMSIS-NN optimization is {abs(improvement_pct):.1f}% slower")
    else:
        print("Could not parse timing results")
EOF
else
    echo "❌ No results file found"
fi

echo
echo "=== Comparison Complete ==="
echo "Build artifacts saved in: $BUILD_DIR"
