# Zant CLI Commands Reference

Zant is a tensor computation framework with ONNX support. This document provides a comprehensive reference for all available CLI commands and options.


1. [Available Commands](#available-commands): Overview of available build commands
2. [Library Flags](#library-flags-table): Build flags for library generation and testing
3. [Extractor Flags](#extractor-flags-table): Configuration for node extraction tools
4. [OneOp Flags](#oneop-flags-table): Flags for single-operator generation and testing
5. [Benchmark Flags](#benchmark-flags-table): Options for performance benchmarking
6. [Global Build Flags](#global-build-flags): General build options (optimization, target, etc.)
7. [Common Workflows](#common-workflows): Examples of common usage patterns


## Zant Build System Commands
Run `zig build -h` and look into `Steps:` section to show all the options

### Available Commands
- `test` - Run all unit tests
- `lib-gen` - Generate library code from ONNX models
- `lib-exe` - Build and run generated model executable
- `lib-test` - Run generated library tests
- `lib` - Compile zant .a static library
- `op-codegen-gen` - Codegenerate one-operation test libraries
- `op-codegen-test` - Run generated one-operation tests
- `extractor-gen` - Codegenerate tests for extracted nodes
- `extractor-test` - Start extracted nodes tests
- `benchmark` - Run benchmarks
- `onnx-parser` - Test ONNX parsing functionality
- `build-main` - Build main executable for profiling

### Library Flags Table

These are some flags available in Zant, for a complete list check the following examples or `zig build --help`

| Flag | Type | Default | Description | Used By |
|------|------|---------|-------------|---------|
| `-Dmodel` | string | `"mnist-8"` | Model name | All lib commands |
| `-Dmodel_path` | string | `"datasets/models/{model}/{model}.onnx"` | Path to ONNX model file | `lib-gen`, `lib-exe` |
| `-Dgenerated_path` | string | `"generated/{model}/"` | Directory for generated code | All lib commands |
| `-Doutput_path` | string | `""` | Custom output directory for built library | `lib` |
| `-Dshape` | string | `""` | Input tensor shape (e.g., "1,3,224,224") | `lib-gen`, `lib-exe` |
| `-Dtype` | string | `"f32"` | Input tensor data type | `lib-gen`, `lib-exe` |
| `-Doutput_type` | string | `"f32"` | Output tensor data type | `lib-gen`, `lib-exe` |
| `-Dcomm` | bool | `false` | Generate code with comments | `lib-gen`, `lib-exe` |
| `-Ddynamic` | bool | `true` | Enable dynamic allocation | `lib-gen`, `lib-exe` |
| `-Dstatic_planning` | string | `disabled` | Static planning mode used when `-Ddynamic=false` | `lib-gen`, `lib-exe` |
| `-Dforce_bnb` | bool | `false` | Force branch-and-bound static planning | `lib-gen`, `lib-exe` |
| `-Ddo_export` | bool | `false` | Generate exportable functions | `lib-gen`, `lib-exe` |
| `-Dv` | string | `"v1"` | Codegen version | `lib-gen`, `lib-exe` |
| `-Dlog` | bool | `false` | Enable logging during generation | `lib-gen`, `lib-exe` |
| `-Denable_user_tests` | bool | `false` | Generate user test code | `lib-gen`, `lib-exe` |
| `-Dxip` | bool | `false` | XIP (Execute In Place) support for neural network weights | `lib-gen`, `lib-exe` |
| `-Dgen_ino` | bool | `false` | Generate .ino file within generated directory | `lib-gen` | 
| `-Dgen_h` | bool | `false` | Generate .h file within generated directory | `lib-gen` | 

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
Without specifing ` --op Add ` and `-Dop="Add"` all the ops are tested
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
- **input_setter**: `src/codegen/IR_zant/onnx/input_setter.py`
- **shape_thief**: `src/codegen/IR_zant/onnx/shape_thief.py`
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
