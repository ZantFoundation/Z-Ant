# Internal Code Documentation: Prediction Model

## Table of Contents

* [1. Introduction](#1-introduction)
* [2. Data Structures](#2-data-structures)
    * [2.1. `Tensor` Structure](#21-tensor-structure)
    * [2.2. Fixed Buffer Allocator](#22-fixed-buffer-allocator)
* [3. Global Variables](#3-global-variables)
* [4. Functions](#4-functions)
    * [4.1. `setLogFunction`](#41-setlogfunction)
    * [4.2. `predict`](#42-predict)
* [5. Weight and Bias Initialization](#5-weight-and-bias-initialization)
* [6. Output Tensor Initialization](#6-output-tensor-initialization)


## 1. Introduction

This document details the implementation of a prediction model written in Zig. The model performs a series of tensor operations, including gather, reduce mean, GEMM (General Matrix Multiply), ReLU (Rectified Linear Unit), and sigmoid, to produce a prediction from a given input.  The code utilizes the `zant` library for tensor manipulation and a fixed buffer allocator for memory management.


## 2. Data Structures

### 2.1. `Tensor` Structure

The code leverages the `zant.core.tensor.Tensor` structure.  While the internal implementation of this structure is not shown here, it's understood to represent a multi-dimensional array (tensor) and provide methods for various tensor operations.  These methods are used extensively throughout the `predict` function.

### 2.2. Fixed Buffer Allocator

A fixed-size buffer allocator (`std.heap.FixedBufferAllocator`) is used for memory management. This allocator is initialized with a 40960-byte buffer (`buf`) and provides a more predictable and potentially faster allocation scheme compared to a general-purpose allocator, especially for the relatively small tensors in this model.  It prevents fragmentation and potential memory leaks within the limited scope of this prediction function.


## 3. Global Variables

* `log_function`: A pointer to a C-callable function used for logging.  It's initialized to `null`.  The `setLogFunction` function allows external setting of this for debugging purposes.
* `buf`: A fixed-size buffer of 40960 bytes used by the `FixedBufferAllocator`.
* `fba_state`:  Holds the state of the `FixedBufferAllocator`.
* `fba`: The allocator obtained from `fba_state`.
* `T`: An alias for `f32`, representing the data type of the tensors (single-precision floating-point numbers).


## 4. Functions

### 4.1. `setLogFunction`

This function sets the global logging function.  It simply assigns the provided function pointer to the `log_function` variable. This allows for easy enabling/disabling of logging functionality without modifying the main prediction logic.


```zig
pub export fn setLogFunction(func: ?*const fn ([*c]u8) callconv(.C) void) void {
    log_function = func;
}
```

### 4.2. `predict`

The `predict` function is the core of the prediction model. It takes an input tensor, its shape, and a pointer to store the result as parameters.  The function performs several steps:

1. **Input Parameter Checks:** It validates the input parameters (`shape_len`, `input_shape`) to ensure they are consistent with the model's expectations (1D input tensor of length 15).

2. **Input Tensor Creation:** It allocates memory for the input data using the fixed buffer allocator, copies the input data into this allocated memory, and constructs a `zant.core.tensor.Tensor` from it. Note the conversion from `[*]u32` to `[]usize` for the shape is handled by `utils.u32ToUsize`.

3. **Gather Operation:** It performs a gather operation using `tensMath.gather_lean`. This operation selects specific elements from the `tensor_embedding_weight` based on the indices provided in `tensor_input`, resulting in `tensor__embedding_gather_output_0`.

4. **ReduceMean Operation:** It calculates the mean along a specified axis (axis 1) using `tensMath.reduce_mean_lean`. The result is stored in `tensor__reducemean_output_0`.

5. **GEMM Operations:** It performs two GEMM operations using `tensMath.gemm_lean`.  The first GEMM multiplies `tensor__reducemean_output_0` and `tensor_fc1_weight`, adding `tensor_fc1_bias`, to produce `tensor__fc1_gemm_output_0`. The second GEMM operates on `tensor__relu_relu_output_0` and `tensor_fc2_weight`, adding `tensor_fc2_bias`, to produce `tensor__fc2_gemm_output_0`.


6. **ReLU Operation:** A ReLU activation function is applied to `tensor__fc1_gemm_output_0` using `tensMath.ReLU_lean`, storing the result in `tensor__relu_relu_output_0`.

7. **Sigmoid Operation:** A sigmoid activation function is applied to `tensor__fc2_gemm_output_0` using `tensMath.sigmoid_lean`, producing the final output in `tensor_output`.

8. **Result Return:**  The function returns the pointer to the data of the output tensor.


```zig
pub export fn predict(
    input: [*]T,
    input_shape: [*]u32,
    shape_len: u32,
    result: *[*]T,
) void {
    // ... (function body as described above) ...
}
```

## 5. Weight and Bias Initialization

The model's weights and biases are initialized as constant tensors using `Tensor(f32).fromConstBuffer`.  The weights and biases are stored as arrays of `f32` and their shapes are defined separately.


*   `tensor_embedding_weight`: Shape (243, 16)
*   `tensor_fc1_weight`: Shape (8, 16)
*   `tensor_fc1_bias`: Shape (8)
*   `tensor_fc2_weight`: Shape (1, 8)
*   `tensor_fc2_bias`: Shape (1)

The large `array_embedding_weight` is initialized with a set of predefined floating-point numbers.


## 6. Output Tensor Initialization

Several output tensors are pre-allocated using `Tensor(T).fromConstBuffer` with undefined initial values. These tensors store intermediate results of the computations:

*   `tensor__embedding_gather_output_0`: Shape (1, 15, 16)
*   `tensor__reducemean_output_0`: Shape (1, 16)
*   `tensor__fc1_gemm_output_0`: Shape (1, 8)
*   `tensor__relu_relu_output_0`: Shape (1, 8)
*   `tensor__fc2_gemm_output_0`: Shape (1, 1)
*   `tensor_output`: Shape (1, 1) - This is the final prediction.

The use of `fromConstBuffer` for output tensors pre-allocates space; the actual values will be computed during the prediction process.
