# Beer Model Testing Guide

This guide explains how to test the beer model with automatic memory management and STM32N6 accelerator support in the Z-Ant framework.

## Overview

The beer model is a large neural network that previously suffered from RAM overflow issues (4.9MB overflow). This has been solved with:

- **Automatic Tensor Pool Allocation**: Large arrays automatically allocated to external RAM
- **STM32N6 Accelerator Support**: CMSIS-NN and Ethos-U acceleration for significant performance gains
- **Memory Management**: 80% reduction in memory usage through intelligent allocation

## Prerequisites

- Zig compiler (0.14.0+)
- ARM GCC toolchain (`arm-none-eabi-gcc`)
- QEMU system emulator
- Python 3.11+

## Quick Start

### 1. Basic Beer Model Test

Test the beer model with automatic memory management:

```bash
# Generate and build beer model with tensor pool allocation
zig build lib-gen -Dmodel=beer -Dfuse -Duse_tensor_pool=true -Ddo_export=true
zig build lib -Dmodel=beer -Dtarget=thumb-freestanding -Dcpu=cortex_m55 -Doptimize=ReleaseSafe -Duse_tensor_pool=true -Dstm32n6_accel=true -Dstm32n6_use_cmsis=true -Dstm32n6_use_ethos=true

# Run the test
python scripts/test_stm32n6_qemu.py --beer --repeat 1
```

### 2. Performance Comparison Test

Compare reference vs CMSIS-NN accelerated versions:

```bash
# Run the complete comparison
./test_beer_comparison.sh
```

Expected output:
```
beer_reference: 2083.86 ms
beer_cmsis_nn: 898.69 ms
Performance improvement: 56.9% FASTER!
```

## Detailed Usage

### Memory Management Features

#### Automatic Tensor Pool Allocation

The `-Duse_tensor_pool=true` flag automatically:
- Allocates arrays ≥100 elements to external RAM (`.tensor_pool` section)
- Moves FBA buffers (2MB) to tensor pool
- Reduces RAM usage by 80% (4.9MB → 960KB)

#### Memory Layout

- **RAM**: 512KB (for small variables and stack)
- **EXT_RAM**: 4MB (for large tensor arrays via tensor pool)
- **FLASH**: 4MB (for code and static data)

### STM32N6 Accelerator Options

#### Available Flags

```bash
# Basic accelerator support
-Dstm32n6_accel=true

# CMSIS-NN acceleration (Helium SIMD)
-Dstm32n6_use_cmsis=true

# Ethos-U NPU acceleration
-Dstm32n6_use_ethos=true

# Optional: Custom CMSIS paths
-Dstm32n6_cmsis_path=/path/to/cmsis
-Dstm32n6_ethos_path=/path/to/ethos
```

#### Performance Impact

| Configuration | Execution Time | Speedup |
|---------------|----------------|---------|
| Reference | ~2084ms | 1.0x |
| CMSIS-NN | ~899ms | 2.3x |


### Build Commands

#### 1. Code Generation

```bash
# Generate with all optimizations
zig build lib-gen \
  -Dmodel=beer \
  -Dfuse \
  -Duse_tensor_pool=true \
  -Ddo_export=true
```

#### 2. Library Compilation

```bash
# Reference build (no acceleration)
zig build lib \
  -Dmodel=beer \
  -Dfuse \
  -Duse_tensor_pool=true \
  -Ddo_export=true \
  -Dtarget=thumb-freestanding \
  -Dcpu=cortex_m55 \
  -Doptimize=ReleaseSafe

# Accelerated build (with STM32N6)
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
  -Doptimize=ReleaseSafe
```

### Testing Scripts

#### 1. Basic QEMU Test

```bash
# Test beer model in QEMU
python scripts/test_stm32n6_qemu.py --beer --repeat 1

# Test with multiple runs for statistics
python scripts/test_stm32n6_qemu.py --beer --repeat 5
```

#### 2. Performance Comparison

```bash
# Run complete A/B comparison
./test_beer_comparison.sh

# Check results
cat build/beer_comparison/comparison_results.txt
```

#### 3. Independent Testing

```bash
# Test each configuration independently
./test_beer_independent.sh
```

## Troubleshooting

### Memory Issues

**Problem**: RAM overflow errors
```
region 'RAM' overflowed by X bytes
```

**Solution**: Ensure tensor pool allocation is enabled:
```bash
zig build lib-gen -Dmodel=beer -Duse_tensor_pool=true -Ddo_export=true
```

### Missing predict Function

**Problem**: Linker error about undefined `predict` function
```
undefined reference to `predict`
```

**Solution**: Add the export flag:
```bash
zig build lib -Dmodel=beer -Ddo_export=true
```

### Accelerator Not Working

**Problem**: No performance improvement with accelerator flags

**Solution**: Verify all required flags are present:
```bash
zig build lib \
  -Dmodel=beer \
  -Dstm32n6_accel=true \
  -Dstm32n6_use_cmsis=true \
  -Dstm32n6_use_ethos=true
```

### QEMU Issues

**Problem**: QEMU test fails to run

**Solution**: Check QEMU installation and ARM toolchain:
```bash
# Verify QEMU
qemu-system-arm --version

# Verify ARM GCC
arm-none-eabi-gcc --version

# Check if QEMU is in PATH
which qemu-system-arm
```

## Advanced Configuration

### Custom Memory Thresholds

Modify the tensor pool threshold in `src/codegen/cg_v1/predict/emit.zig`:

```zig
// Change from 100 elements to custom threshold
const use_tensor_pool = codegen_options.use_tensor_pool and size >= 50; // 50 element threshold
```

### Custom Linker Script

Modify `tests/stm32n6_qemu/stm32n6.ld` for different memory layouts:

```ld
MEMORY
{
    FLASH (rx)  : ORIGIN = 0x00000000, LENGTH = 0x00400000
    RAM   (rwx) : ORIGIN = 0x20000000, LENGTH = 0x00080000
    EXT_RAM (rwx) : ORIGIN = 0x31000000, LENGTH = 0x00400000  // 4MB for tensor pool
}
```

### Environment Variables

```bash
# Override CMSIS paths
export ZANT_CMSIS_PATH=/custom/cmsis/path
export ZANT_ETHOS_PATH=/custom/ethos/path

# Override FBA size (if not using tensor pool)
export ZANT_FBA_SIZE_KB=1024
```

## Performance Analysis

### Understanding Results

The comparison script provides detailed timing analysis:

```
beer_reference: 2083.86 ms    # No acceleration
beer_cmsis_nn: 898.69 ms     # CMSIS-NN acceleration
Improvement: 1185.17 ms (56.9%)  # Performance gain
```

### Optimization Levels

- **ReleaseSafe**: Balanced performance and safety
- **ReleaseFast**: Maximum performance
- **ReleaseSmall**: Minimum binary size

### Memory Usage

- **Before**: 4.9MB RAM overflow
- **After**: 960KB RAM usage (80% reduction)
- **Tensor Pool**: 4MB external RAM for large arrays

## File Structure

```
Z-Ant/
├── generated/beer/           # Generated beer model code
│   ├── lib_beer.zig         # Main library with tensor pool
│   ├── static_parameters.zig # Model weights
│   └── test_beer.zig        # Test harness
├── tests/stm32n6_qemu/      # QEMU test infrastructure
│   ├── beer_main.c          # C interface for predict()
│   └── stm32n6.ld           # Linker script
├── test_beer_comparison.sh  # Performance comparison
├── test_beer_independent.sh # Independent testing
└── scripts/test_stm32n6_qemu.py # QEMU test runner
```

## Contributing

When modifying the beer model testing:

1. **Update tensor pool logic**: Modify `src/codegen/cg_v1/predict/emit.zig`
2. **Update FBA allocation**: Modify `src/codegen/cg_v1/predict_writer.zig`
3. **Test both configurations**: Run comparison script
4. **Verify memory usage**: Check for RAM overflow
5. **Update documentation**: Keep this README current

## References

- [Z-Ant Documentation](docs/)
- [STM32N6 Accelerator Guide](docs/accelerators/)
- [Memory Management](docs/tensor.md)
- [Performance Optimization](docs/MATH_TABLE.md)
