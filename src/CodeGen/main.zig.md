# Internal Code Documentation: ONNX Model Code Generator

## Table of Contents

* [1. Introduction](#1-introduction)
* [2. Data Structures](#2-data-structures)
* [3. Modules](#3-modules)
* [4. `main` Function](#4-main-function)
* [5. Code Generation Process](#5-code-generation-process)


<a name="1-introduction"></a>
## 1. Introduction

This document details the implementation of an ONNX model code generator.  The program parses an ONNX model, analyzes its structure, and generates Zig code for inference. The generated code is designed for efficient execution and can be compiled and tested independently.

<a name="2-data-structures"></a>
## 2. Data Structures

The code utilizes several data structures from the `zant` library, primarily the `Tensor` structure for representing tensors and associated mathematical functions from `zant.core.tensor.math_standard`.  The `allocator` from `zant.utils.allocator` manages memory allocation.  The internal representation of the ONNX model is handled by the `onnx` module imported from `zant`.


<a name="3-modules"></a>
## 3. Modules

The code utilizes several custom modules:

| Module Name           | Description                                                                |
|-----------------------|----------------------------------------------------------------------------|
| `std`                 | Zig's standard library.                                                    |
| `zant`                |  External library providing ONNX parsing and tensor manipulation capabilities. |
| `codegen`             | Custom module responsible for code generation.                               |
| `codegen_options`     | Custom module containing configurable options for the code generation process.  (Currently, only `model` option is used).|


The `codegen` module is further broken down into sub-modules:

* `utils`: Utility functions for code generation.
* `parameters`: Functions to handle parameters of operations.
* `math_handler`: Functions that handle the translation of mathematical operations.
* `predict`: Functions for generating prediction code.
* `tests`: Functions for generating test code (currently commented out).
* `skeleton`: Contains the function to write the main Zig file.
* `globals`: Contains the function to set global attributes of the model.


<a name="4-main-function"></a>
## 4. `main` Function

The `main` function orchestrates the entire code generation process:

1. **Initialization:** A general-purpose allocator (`gpa`) is created for memory management.

2. **Model Loading:** The program loads the ONNX model specified by `codegen_options.model` from the path constructed by concatenating "datasets/models/", model name, and ".onnx". The `onnx.parseFromFile` function handles the parsing.  Error handling is implemented using Zig's `try` mechanism.

3. **Directory Creation:** It creates the directory ("generated/" ++ model_name ++ "/") to store the generated code, if it doesn't exist.

4. **Global Attribute Setting:**  `globals.setGlobalAttributes` processes the loaded ONNX model to extract and set global attributes needed for code generation.

5. **Code Generation:** `codeGen.skeleton.writeZigFile` generates the Zig code based on the parsed ONNX model and writes it to the designated directory.

6. **Testing (Commented Out):**  The line `// try codeGen_tests.writeTestFile(model_name, generated_path, model);` indicates provision for generating test code, currently disabled.



<a name="5-code-generation-process"></a>
## 5. Code Generation Process

The code generation process is encapsulated within the `codegen` module, particularly the `skeleton.writeZigFile` function.  The exact algorithm employed within this function is not detailed in this code, but would likely involve traversing the ONNX graph, analyzing each operation node, and generating corresponding Zig code for tensor manipulation and calculations. It leverages the `zant` library's tensor operations for efficient code generation. The generated Zig code will include necessary includes, function definitions, and calls to the appropriate `zant` tensor math functions based on the operations defined in the ONNX model.


