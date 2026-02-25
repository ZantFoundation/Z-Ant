# Zant CLI Commands Reference

Zant is a tensor computation framework with ONNX support. This document provides a comprehensive reference for all available CLI commands and options.

## Zant Build System Commands
Run `zig build -h` and look into `Steps:` section to show all the options

### Available Commands
- `test` - Run all unit tests
- `lib-gen` - Generate library code from ONNX models
- `lib-exe` - Build and run generated model executable
- `lib-test` - Run generated library tests
- `lib` - Compile tensor math static library
- `op-codegen-gen` - Codegenerate one-operation test libraries
- `op-codegen-test` - Run generated one-operation tests
- `extractor-gen` - Codegenerate tests for extracted nodes
- `extractor-test` - Start extracted nodes tests
- `benchmark` - Run benchmarks
- `onnx-parser` - Test ONNX parsing functionality
- `build-main` - Build main executable for profiling

### Library Flags Table

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-Dmodel` | string | `"mnist-8"` | Model name |
| `-Dmodel_path` | string | `"datasets/models/{model}/{model}.onnx"` | Model path |
| `-Dgenerated_path` | string | `"generated/{model}/"` | Custom output path for generated code |
| `-Denable_user_tests` | bool | `false` | Enable user test data |
| `-Dlog` | bool | `false` | Run with log output |
| `-Dtype` | string | `"f32"` | Input type |
| `-Doutput_type` | string | `"f32"` | Output type |
| `-Dcomm` | bool | `false` | Codegen with comments |
| `-Ddynamic` | bool | `true` | Dynamic allocation |
| `-Dstatic_planning` | bool | `false` | Static memory planning (ignored when `-Ddynamic=true`) |
| `-Dfuse` | bool | `false` | Enable kernel fusion |
| `-Ddo_export` | bool | `false` | Codegen exportable symbols |
| `-Dv` | string | `"v1"` | Codegen version: v1 or v2 |
| `-Dxip` | bool | `false` | Enable XIP (Execute In Place) for weights |
| `-Duse_tensor_pool` | bool | `false` | Allocate large tensor arrays to tensor_pool section |
| `-Dgen_ino` | bool | `false` | Generate Arduino .INO file |
| `-Dgen_h` | bool | `false` | Generate C .H header file |

See also [BUILD_FLAGS.md](BUILD_FLAGS.md)

### Library Usage Examples
```bash
# Basic library generation
zig build lib-gen

# Generate with custom model
zig build lib-gen -Dmodel="resnet50"

# Generate with specific configuration
zig build lib-gen -Dmodel="custom" -Ddynamic -Dcomm

# Run generated executable
zig build lib-exe -Dmodel="mnist-8" -Dlog
```

## Extractor Commands

### Available Commands
- `extractor-gen` - Generate node extractor tests
- `extractor-test` - Run node extractor tests

### Extractor Flags Table

| Flag | Type | Default | Description | Used By |
|------|------|---------|-------------|---------|
| `-Dmodel` | string | `"mnist-8"` | Model name for node extraction | Both extractor commands |

### Extractor Usage Examples
```bash
# Generate extractor tests for default model
zig build extractor-gen

# Generate extractor tests for specific model
zig build extractor-gen -Dmodel="resnet50"

# Run extractor tests
zig build extractor-test -Dmodel="mnist-8"
```

## OneOp Commands

### Available Commands
- `op-codegen-gen` - Generate one-operation test models
- `op-codegen-test` - Run one-operation tests

### OneOp Flags Table

| Flag | Type | Default | Description | Used By |
|------|------|---------|-------------|---------|
| `-Dop` | string | `"all"` | Specific operation name to test | Both oneop commands |

### OneOp Usage Examples
Remember to first generate the onnx with the onnx_generator.py see [here](#2-test-single-operations)
```bash
# Generate tests for all operations listed in available_operations.txt
zig build op-codegen-gen

# Generate tests for specific operation
zig build op-codegen-gen -Dop="Add"

# Run all operation tests
zig build op-codegen-test

# Run specific operation test
zig build op-codegen-test -Dop="Conv"
```

## Testing Commands

### Available Commands
- `test` - Run all unit tests
- `onnx-parser` - Test ONNX parsing functionality

### Testing Flags Table

| Flag | Type | Default | Description | Used By |
|------|------|---------|-------------|---------|
| `-Dheavy` | bool | `false` | Run heavy/slow tests | `test` |
| `-Dtest_name` | string | `""` | Run specific test by name | `test` |

### Testing Usage Examples
```bash
# Run basic unit tests
zig build test

# Run all tests including heavy ones
zig build test -Dheavy=true

# Run specific test
zig build test -Dtest_name="tensor_operations"

# Test ONNX parser
zig build onnx-parser
```

## Benchmark Commands

### Available Commands
- `benchmark` - Run performance benchmarks

### Benchmark Flags Table

| Flag | Type | Default | Description | Used By |
|------|------|---------|-------------|---------|
| `-Dfull` | bool | `false` | Run complete benchmark suite | `benchmark` |

### Benchmark Usage Examples
```bash
# Run basic benchmarks
zig build benchmark

# Run full benchmark suite
zig build benchmark -Dfull=true
```

## Global Build Flags

These flags can be used with any build command:

### Global Flags Table

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-Dtarget` | string | `"native"` | Target architecture (e.g., "x86_64-linux", "thumb-freestanding") |
| `-Dcpu` | string | `null` | Target CPU model (e.g., "cortex_m33") |
| `-Doptimize` | enum | `Debug` | Optimization mode: Debug, ReleaseFast, ReleaseSafe, ReleaseSmall |
| `-Dtrace_allocator` | bool | `true` | Enable tracing allocator |
| `-Dallocator` | string | `"raw_c_allocator"` | Allocator type to use |

### STM32 N6 Accelerator Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-Dstm32n6_accel` | bool | `false` | Enable STM32 N6 accelerator support |
| `-Dstm32n6_cmsis_path` | string | `null` | Optional CMSIS include path |
| `-Dstm32n6_force_native` | bool | `false` | Force accelerator stubs on non-Thumb targets (for host testing) |
| `-Dstm32n6_use_cmsis` | bool | `false` | Enable CMSIS Helium kernels |
| `-Dstm32n6_use_ethos` | bool | `false` | Enable Ethos-U integration stubs |
| `-Dstm32n6_ethos_path` | string | `""` | Optional Ethos-U driver headers path |

### Global Usage Examples
```bash
# Cross-compile for ARM Cortex-M
zig build lib-gen -Dmodel="my_model" -Dtarget=thumb-freestanding-eabi -Dcpu=cortex_m33

# Build with optimization
zig build lib -Dmodel="my_model" -Doptimize=ReleaseFast

# Build for different platforms
zig build lib -Dtarget=x86_64-windows -Doptimize=ReleaseSmall
zig build lib -Dtarget=aarch64-macos -Doptimize=ReleaseSafe
```

## Common Workflows

### 1. MOST IMPORTANT - Generate and Test a Model

```bash
./zant input_setter --model my_model --shape 1,3,224,224 #use https://netron.app/ to see the input shape
# or, if the model input is already well defined you can run this:
./zant shape_thief --model my_model #shape_thief is in Beta version

# Generate test data
./zant user_tests_gen --model my_model

# --- GENERATING THE LIBRARY and TESTS ---
# Generate code for a specific model
zig build lib-gen -Dmodel="my_model" -Denable_user_tests [ -Ddo_export -Dxip -Dfuse -Dlog -Dcomm ... ]

# Test the generated code
zig build lib-test -Dmodel="my_model" -Denable_user_tests [ -Dfuse -Ddo_export -Dlog -Dcomm ... ]

# Build the static library
zig build lib -Dmodel="my_model" [-Doptimize= [ ReleaseSmall, ReleaseFast ] -Dtarget=... -Dcpu=...]
```

### 2. Test Single Operations
Without specifying `--op Add` and `-Dop="Add"` all the ops are tested
```bash
# Generate ONNX models for testing (using zant wrapper)
./zant onnx_gen --op Add

# Generate code for one-op models
zig build op-codegen-gen -Dop="Add"

# Test the generated operations
zig build op-codegen-test -Dop="Add"
```

### 3. Test Individual Nodes in a Graph
```bash
# --- GENERATING THE Single Node lib and test it ---
# For an N-node model it creates N onnx models, one for each node with respective tests.
./zant onnx_extract --model my_model

# Generate libs for extracted nodes
zig build extractor-gen -Dmodel="my_model"

# Test extracted nodes
zig build extractor-test -Dmodel="my_model"
```

### 4. Prepare ONNX Models
```bash
# Set input shape and infer intermediate shapes
./zant input_setter --model my_model --shape 1,3,224,224

# Generate additional shape information
./zant shape_thief --model my_model

# Generate test data
./zant user_tests_gen --model my_model --iterations 10
```

### 5. Production Workflow
```bash
# Prepare model with proper shapes
./zant input_setter --model my_model --shape 1,3,224,224
./zant shape_thief --model my_model

# Generate optimized library
zig build lib-gen -Dmodel=production-model -Ddo_export

# Build for multiple targets
zig build lib -Dmodel=production-model -Dtarget=x86_64-linux -Doptimize=ReleaseFast
zig build lib -Dmodel=production-model -Dtarget=aarch64-linux -Doptimize=ReleaseFast
zig build lib -Dmodel=production-model -Dtarget=x86_64-windows -Doptimize=ReleaseSmall
```

### 6. Complete Testing Workflow
```bash
# Generate test models for multiple operations
./zant onnx_gen [ --op NameOfTheOperator ( default tests all ops inside available_operations.txt ) ]

# Test specific operations
zig build op-codegen-gen [ -Dop="Conv" ]
zig build op-codegen-test [ -Dop="Conv" ]

# Test node extraction
zig build extractor-gen -Dmodel=test-model
zig build extractor-test -Dmodel=test-model

# Validate ONNX parsing
zig build onnx-parser

# Run comprehensive tests
zig build test -Dheavy=true
```

### 7. Cross-Platform Development
```bash
# Prepare model
./zant input_setter --model my_model --shape 1,1,28,28
./zant shape_thief --model my_model

# Generate for ARM Cortex-M
zig build lib-gen -Dmodel=embedded_model -Ddo_export -Dtarget=thumb-freestanding -Dcpu=cortex_m33 [-Dxip]
zig build lib -Dmodel=embedded_model -Ddo_export -Dtarget=thumb-freestanding -Dcpu=cortex_m33 -Doptimize=ReleaseSmall [-Dxip]

# Test on native platform first
zig build lib-test -Dmodel=embedded_model
```

## Zant Python Tools Integration

The workflows above use the `zant` wrapper script for ONNX model preparation. Here are the available `zant` commands:

### ONNX Model Generation
```bash
# Generate test models for specific operations
./zant onnx_gen --op Add --iterations 5 --seed 42
./zant onnx_gen --op Conv --iterations 3 --output-dir ./conv_models

# Generate models for all operations
./zant onnx_gen --iterations 10 --output-dir ./all_models
```

### Model Preparation
```bash
# Set input shapes
./zant input_setter --model my_model --shape 1,3,224,224
./zant input_setter --model my_model --shape 4,3,256,256

# Infer intermediate shapes
./zant shape_thief --model my_model

# Generate user test data
./zant user_tests_gen --model my_model --iterations 10
```

### Model Profiling
```bash
zig build build-main -Dmodel="my_model"

valgrind --tool=massif --heap=yes --stacks=yes --time-unit=ms ./zig-out/bin/main_profiling_target

ms_print massif.out.* > out_profiling.txt
```

### Zant Script Locations
- **input_setter**: `src/onnx/input_setter.py`
- **shape_thief**: `src/onnx/shape_thief.py`
- **user_tests_gen**: `tests/CodeGen/user_tests_gen.py`
- **onnx_gen**: `tests/CodeGen/Python-ONNX/onnx_gen.py`
- **onnx_extract**: `tests/CodeGen/onnx_node_extractor.py`

## Getting Help

```bash
# Show all available options and steps
zig build --help

# List all build steps
zig build --list-steps

# Show build summary
zig build --summary all
```

## Notes

- Most commands support the `--help` flag for detailed usage information
- The Zig build system uses `-D` prefix for custom options
- The shell wrapper automatically detects and uses `python3` or `python`
- Generated files are typically placed in the `generated/` directory
- Model files should be in ONNX format and placed in `datasets/models/{model_name}/`
