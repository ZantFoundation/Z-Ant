# Internal Code Documentation: Prediction Model

## Table of Contents

* [1. Introduction](#1-introduction)
* [2. Data Structures](#2-data-structures)
    * [2.1. `Tensor` Struct](#21-tensor-struct)
* [3. Functions](#3-functions)
    * [3.1. `setAllocator` Function](#31-setallocator-function)
    * [3.2. `setAllAllocators` Function](#32-setallallocators-function)
    * [3.3. `setLogFunction` Function](#33-setlogfunction-function)
    * [3.4. `predict` Function](#34-predict-function)
* [4. Initialization of Weights and Biases](#4-initialization-of-weights-and-biases)
* [5. Initialization of Output Tensors](#5-initialization-of-output-tensors)


## 1. Introduction

This document details the implementation of a prediction model written in Zig.  The model performs a series of tensor operations, including reshaping, convolution, addition, ReLU activation, max pooling, and matrix multiplication, to produce a prediction based on input data.  The model utilizes the `zant` library for tensor manipulation.  A fixed buffer allocator is used for memory management.


## 2. Data Structures

### 2.1. `Tensor` Struct (from `zant` library)

The `zant.core.tensor.Tensor` struct is used throughout the code to represent tensors.  Details of its internal structure are available in the `zant` library documentation.  Key attributes used here include:

| Attribute     | Description                                          |
|---------------|------------------------------------------------------|
| `allocator`  | Memory allocator used by the tensor.                   |
| `data`        | Pointer to the underlying data array.                 |
| `shape`       | Array of integers representing the tensor's dimensions.|


## 3. Functions

### 3.1. `setAllocator` Function

This function sets the memory allocator for a given tensor.

```zig
fn setAllocator(tensor: *Tensor(T), alloc: *const std.mem.Allocator) void {
    tensor.allocator = alloc;
}
```

### 3.2. `setAllAllocators` Function

This function iterates through a list of tensors and sets their allocators to the global allocator. This ensures that all tensors use the same allocator, simplifying memory management.

```zig
fn setAllAllocators() void {
    setAllocator(&tensor_relu32_output_0, &allocator);
    setAllocator(&tensor_pooling66_output_0, &allocator);
    // ... more tensors ...
}
```

### 3.3. `setLogFunction` Function

This function sets a logging function to be used for debugging purposes. The logging function accepts a C-style string (`[*c]u8`) as input.

```zig
pub export fn setLogFunction(func: ?*const fn ([*c]u8) callconv(.C) void) void {
    log_function = func;
}
```

### 3.4. `predict` Function

This is the main prediction function. It takes an input tensor, its shape, and a pointer to store the result. The function performs the following steps:

1. **Input Validation:** Checks if the input shape is valid (length 4, 28x28 input).
2. **Input Copying:** Copies the input data into a newly allocated tensor (`tensor_input3`).
3. **Reshape Operation:** Reshapes the `tensor_parameter193` weight tensor using `tensMath.reshape_lean`. This function efficiently reshapes a tensor without copying data if possible.  The boolean `false` argument indicates that the reshape should be performed in-place.
4. **Convolution Operation:** Performs a convolution using `tensMath.conv_lean`. This function performs a convolution operation efficiently using optimized algorithms. The "SAME_UPPER" `auto_pad` option ensures that the output tensor has the same dimensions as the input tensor.
5. **Addition Operation:** Adds two tensors element-wise using `tensMath.sum_tensors_lean`. This function is optimized for this specific operation.
6. **ReLU Activation:** Applies the ReLU activation function (element-wise maximum of 0 and the input) using `tensMath.ReLU_lean`.
7. **Max Pooling Operation:** Performs max pooling using `tensMath.onnx_maxpool_lean`.  This function implements the ONNX max pooling algorithm.
8. **Repeated Convolution, Addition, ReLU, and Max Pooling:** Steps 4-7 are repeated with different weights and biases.
9. **Final Reshape and Matrix Multiplication:** The output of the final max pooling is reshaped, and then matrix multiplied with reshaped weights `tensor_parameter193_reshape1` using `tensMath.mat_mul_lean`, which is optimized for matrix multiplication.
10. **Final Addition:** A final addition with `tensor_parameter194` is performed.
11. **Result Return:**  The pointer to the result is assigned to the data of `tensor_plus214_output_0`.

```zig
pub export fn predict(
    input: [*]T,
    input_shape: [*]u32,
    shape_len: u32,
    result: *[*]T,
) void {
    setAllAllocators();
    // ... input validation ...
    // ... input copying ...
    // ... reshape, conv, add, relu, maxpool (repeated) ...
    // ... final reshape and matmul ...
    // ... final addition ...
    result.* = tensor_plus214_output_0.data.ptr;
}
```


## 4. Initialization of Weights and Biases

The code initializes several weight and bias tensors from constant buffers.  These tensors are used in the convolution and fully connected layers of the neural network. The shapes and data for each tensor are defined separately:

| Tensor Name                     | Shape             | Data Source                |
|---------------------------------|--------------------|----------------------------|
| `tensor_parameter193`           | `[16, 4, 4, 10]`   | `array_parameter193`       |
| `tensor_parameter87`           | `[16, 8, 5, 5]`    | `array_parameter87`       |
| `tensor_parameter5`             | `[8, 1, 5, 5]`     | `array_parameter5`         |
| `tensor_parameter6`             | `[8, 1, 1]`        | `array_parameter6`         |
| `tensor_parameter88`           | `[16, 1, 1]`       | `array_parameter88`       |
| `tensor_pooling160_output_0_reshape0_shape` | `[1]`             | `array_pooling160_output_0_reshape0_shape` |
| `tensor_parameter193_reshape1_shape` | `[1]`             | `array_parameter193_reshape1_shape` |
| `tensor_parameter194`           | `[1, 10]`          | `array_parameter194`       |


## 5. Initialization of Output Tensors

Several tensors are initialized to store intermediate and final results of the computation.  Most are initialized with undefined values except for `tensor_pooling66_output_0` and `tensor_convolution110_output_0`, which are initialized to zero:

| Tensor Name                     | Shape                 | Initial Value           |
|---------------------------------|-----------------------|--------------------------|
| `tensor_parameter193_reshape1`  | `[256, 10]`           | `undefined`              |
| `tensor_convolution28_output_0` | `[1, 8, 28, 28]`      | `undefined`              |
| `tensor_plus30_output_0`       | `[1, 8, 28, 28]`      | `undefined`              |
| `tensor_relu32_output_0`       | `[1, 8, 28, 28]`      | `undefined`              |
| `tensor_pooling66_output_0`     | `[1, 8, 14, 14]`      | `[0] * 1568`             |
| `tensor_convolution110_output_0`| `[1, 16, 14, 14]`     | `[0] * 3136`             |
| `tensor_plus112_output_0`      | `[1, 16, 14, 14]`     | `undefined`              |
| `tensor_relu114_output_0`      | `[1, 16, 14, 14]`     | `undefined`              |
| `tensor_pooling160_output_0`   | `[1, 16, 4, 4]`      | `undefined`              |
| `tensor_pooling160_output_0_reshape0` | `[1, 256]`          | `undefined`              |
| `tensor_times212_output_0`     | `[1, 10]`            | `undefined`              |
| `tensor_plus214_output_0`      | `[1, 10]`            | `undefined`              |

