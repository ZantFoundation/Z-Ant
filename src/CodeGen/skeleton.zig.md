# Internal Code Documentation: ONNX to Zig Code Generator

## Table of Contents

1. [Overview](#overview)
2. [Module-Level Functions](#module-level-functions)
    * [writeZigFile](#writezigfile)
    * [write_libraries](#write_libraries)
    * [write_logFunction](#write_logfunction)
    * [write_FBA](#write_fba)
    * [write_type_T](#write_type_t)
    * [write_debug](#write_debug)


## <a name="overview"></a>1. Overview

This document details the implementation of a Zig code generator for ONNX (Open Neural Network Exchange) models.  The generator takes an ONNX model as input and produces two Zig source files: a main library file (e.g., `lib_model.zig`) containing the prediction logic and a separate file (e.g., `static_parameters.zig`) containing the initialization of model parameters (tensors).  The generated code utilizes the `zant` library for tensor operations.


## <a name="module-level-functions"></a>2. Module-Level Functions

### <a name="writezigfile"></a>2.1 `writeZigFile`

```zig
pub fn writeZigFile(model_name: []const u8, model_path: []const u8, model: ModelOnnx) !void { ... }
```

This function orchestrates the entire code generation process.  It performs the following steps:

1. **File Creation:** Creates two files:  a main library file and a file for static parameters.  File paths are constructed using `std.fmt.allocPrint`. Error handling is implemented using `try` for file creation and writing operations.


2. **Writer Initialization:** Initializes writers for both files.


3. **Library Imports:** Calls `write_libraries` to write standard library imports to the main library file.


4. **Conditional Logging Function:** Calls `write_logFunction` to add a logging function to the main library file if `codegen_options.log` is true.


5. **Fixed Buffer Allocator:** Calls `write_FBA` to generate code for a fixed buffer allocator in the main library file.


6. **Type Definition:** Calls `write_type_T` to define the tensor data type (currently hardcoded to `f32`) in the main library file.


7. **Parameter Initialization:** Calls `codeGenInitializers.write_parameters` to generate tensor initialization code in the `static_parameters.zig` file.


8. **Prediction Function Generation:** Calls `coddeGenPredict.writePredict` to generate the prediction function in the main library file.



### <a name="write_libraries"></a>2.2 `write_libraries`

```zig
fn write_libraries(writer: std.fs.File.Writer) !void { ... }
```

This function writes the necessary `@import` statements to the generated Zig file.  It includes imports for the standard library (`std`), the `zant` library,  the `codegen` module, and the generated `static_parameters.zig` file.  The imports are hardcoded for now but could be made more dynamic in the future.


### <a name="write_logfunction"></a>2.3 `write_logFunction`

```zig
fn write_logFunction(writer: std.fs.File.Writer) !void { ... }
```

This function generates a `setLogFunction` which allows setting a custom logging function at runtime. This function is only written if the `codegen_options.log` flag is set.  The generated code includes a pointer to a C function taking a null-terminated C string as an argument.


### <a name="write_fba"></a>2.4 `write_FBA`

```zig
fn write_FBA(writer: std.fs.File.Writer) !void { ... }
```

This function generates code for a fixed-size buffer allocator using `std.heap.FixedBufferAllocator`. This allocator is used for memory management within the generated code.  The buffer size is currently hardcoded.


### <a name="write_type_t"></a>2.5 `write_type_T`

```zig
fn write_type_T(writer: std.fs.File.Writer) !void { ... }
```

This function writes the definition for the `T` type, representing the data type of the tensors. Currently, it is hardcoded to `f32` (single-precision floating-point).  Future improvements could dynamically determine the type from the ONNX model.


### <a name="write_debug"></a>2.6 `write_debug`

```zig
fn write_debug(writer: std.fs.File.Writer) !void { ... }
```

This function generates a `debug` function that prints a debug message to the console.  It's currently commented out in `writeZigFile`.  The function prints a simple header and footer message to visually separate debug output.

