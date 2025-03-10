# Internal Code Documentation: `predict.zig`

## Table of Contents

* [1. Overview](#1-overview)
* [2. Data Structures](#2-data-structures)
* [3. Function: `setLogFunction`](#3-function-setlogfunction)
* [4. Function: `predict`](#4-function-predict)
    * [4.1 Input Validation](#41-input-validation)
    * [4.2 Input Tensor Creation](#42-input-tensor-creation)
    * [4.3 Unsqueeze Operations](#43-unsqueeze-operations)
    * [4.4 Convolutional Operations (Conv)](#44-convolutional-operations-conv)
    * [4.5 ReLU Activation Function](#45-relu-activation-function)
    * [4.6 Max Pooling Operation (MaxPool)](#46-max-pooling-operation-maxpool)
    * [4.7 Shape Operation](#47-shape-operation)
    * [4.8 Gather Operation](#48-gather-operation)
    * [4.9 Concatenation Operation (Concat)](#49-concatenation-operation-concat)
    * [4.10 Reshape Operation](#410-reshape-operation)
    * [4.11 General Matrix Multiplication (GEMM)](#411-general-matrix-multiplication-gemm)
* [5.  Error Handling](#5-error-handling)


## 1. Overview

This document details the implementation of the `predict` function within the `predict.zig` file.  The code performs a forward pass of a convolutional neural network (CNN) using the `zant` tensor library. The network consists of two convolutional layers, each followed by a ReLU activation and max pooling, and a final fully connected layer. The function takes an input tensor, performs the necessary computations, and returns the output tensor.


## 2. Data Structures

The code utilizes the following data structures:

| Data Structure          | Type                                  | Description                                                                  |
|--------------------------|---------------------------------------|------------------------------------------------------------------------------|
| `log_function`           | `?*const fn ([*c]u8) callconv(.C) void` | Pointer to a logging function (optional).                                     |
| `buf`                     | `[4096 * 10]u8`                       | Buffer for the `FixedBufferAllocator`.                                      |
| `fba_state`              | `std.heap.FixedBufferAllocator`       | Fixed buffer allocator state.                                               |
| `fba`                     | `std.mem.Allocator`                   | Allocator derived from `fba_state`.                                          |
| `T`                       | `f32`                                  | Type alias for the floating-point type used in tensors (single-precision). |
| `tensor_conv1_weight`, `tensor_conv1_bias`, `tensor_conv2_weight`, `tensor_conv2_bias`, `tensor_fc_weight`, `tensor_fc_bias` | `Tensor(f32)`                         | Tensors representing the weights and biases of the CNN.                     |
| `tensor_input`, `tensor__unsqueeze_output_0`, `tensor__conv1_conv_output_0`, `tensor__relu1_relu_output_0`, `tensor__pool1_maxpool_output_0`, `tensor__conv2_conv_output_0`, `tensor__relu2_relu_output_0`, `tensor__pool2_maxpool_output_0`, `tensor__shape_output_0`, `tensor__gather_output_0`, `tensor__unsqueeze_1_output_0`, `tensor__unsqueeze_2_output_0`, `tensor__concat_output_0`, `tensor__reshape_output_0`, `tensor_output` | `Tensor(T)` or `Tensor(usize)` or `Tensor(i64)` | Tensors used for intermediate and output results.                          |


## 3. Function: `setLogFunction`

This function sets the optional logging function.  It simply assigns the provided function pointer to the `log_function` variable.


## 4. Function: `predict`

The `predict` function is the core of the code. It takes an input array and its shape, and outputs the prediction result.

### 4.1 Input Validation

The function first validates its input parameters:
* It checks if `shape_len` is zero or not equal to 3 (expected shape is 3D).
* It verifies the dimensions of the input shape are 1x1x8000.

If any of these conditions are false, the function immediately returns.

### 4.2 Input Tensor Creation

* The function calculates the total size of the input data.
* It allocates memory for the input tensor using the `allocator`.
* It copies the input data into the newly allocated memory.
* It converts the input shape from `[*]u32` to `[]usize`, which is the required format for the `zant` library.
* Finally, it creates a `Tensor(T)` from the shape and data.


### 4.3 Unsqueeze Operations

Two unsqueeze operations are performed:

1.  **`tensMath.unsqueeze_lean` on `tensor_input`:** Adds a dimension of size 1 to the input tensor along axis 2. This reshapes the input for compatibility with the convolutional layer.  The algorithm simply creates a new tensor with the added dimension and copies the data.

2.  **`tensMath.unsqueeze_lean` on `tensor__constant_1_output_0`:** Adds a dimension of size 1 to a constant tensor containing -1 along axis 0. This creates a tensor for use in the concatenation operation later. The algorithm is the same as the previous unsqueeze.


### 4.4 Convolutional Operations (Conv)

Two convolutional operations (`tensMath.conv_lean`) are performed:

* **First Convolutional Layer:** This layer uses `tensor__unsqueeze_output_0` as input, `tensor_conv1_weight` as the kernel, `tensor__conv1_conv_output_0` as the output, `tensor_conv1_bias` as bias, a stride of 1x1, padding of 0x1x0x1, and dilation of 1x1.  The algorithm implements a standard discrete convolution. It performs element-wise multiplication between the input and kernel, sums the results, and adds the bias.

* **Second Convolutional Layer:** This layer uses the output of the first max pooling layer (`tensor__pool1_maxpool_output_0`) as input, `tensor_conv2_weight` as the kernel, `tensor__conv2_conv_output_0` as the output, `tensor_conv2_bias` as bias, a stride of 1x1, padding of 0x1x0x1, and dilation of 1x1. The algorithm is identical to the first convolutional layer.


### 4.5 ReLU Activation Function

Two ReLU activation functions (`tensMath.ReLU_lean`) are applied:

*   **First ReLU:** Applied to the output of the first convolutional layer (`tensor__conv1_conv_output_0`), producing `tensor__relu1_relu_output_0`. The algorithm replaces any negative values with zero, leaving positive values unchanged.

*   **Second ReLU:** Applied to the output of the second convolutional layer (`tensor__conv2_conv_output_0`), producing `tensor__relu2_relu_output_0`.  The algorithm is the same as the first ReLU.


### 4.6 Max Pooling Operation (MaxPool)

Two max pooling operations (`tensMath.onnx_maxpool_lean`) are performed:

*   **First MaxPool:**  This takes `tensor__relu1_relu_output_0` as input and produces `tensor__pool1_maxpool_output_0`.  The algorithm uses a kernel shape of 1x3, strides of 1x3, and no padding.  It finds the maximum value within each sliding window defined by the kernel shape and stride.

*   **Second MaxPool:**  This takes `tensor__relu2_relu_output_0` as input and produces `tensor__pool2_maxpool_output_0`. The algorithm is identical to the first max pooling layer.


### 4.7 Shape Operation

The `tensMath.shape_onnx_lean` function determines the shape of  `tensor__pool2_maxpool_output_0` and stores it in `tensor__shape_output_0`.  The algorithm returns a tensor containing the dimensions of the input tensor as integers.


### 4.8 Gather Operation

The `tensMath.gather_lean` function selects specific elements from `tensor__shape_output_0` based on the indices in `usize_tensor__constant_output_0`. The result is stored in `tensor__gather_output_0`. This is essentially index-based element selection. The constant tensor contains index 0, effectively extracting only the first element of the shape.


### 4.9 Concatenation Operation (Concat)

The `tensMath.concatenate` function concatenates `tensor__unsqueeze_1_output_0` and `tensor__unsqueeze_2_output_0` along axis 0, creating `tensor__concat_output_0`. The algorithm simply combines the data from both tensors into a single output tensor.


### 4.10 Reshape Operation

The `tensMath.reshape_lean_f32` function reshapes `tensor__pool2_maxpool_output_0` to the shape specified by `newShape_tensor__concat_output_0`.  The algorithm rearranges the elements of the input tensor into a new shape. `newShape_tensor__concat_output_0` contains the shape information obtained from the concatenation operation.


### 4.11 General Matrix Multiplication (GEMM)

The final fully connected layer uses `tensMath.gemm_lean` to perform general matrix multiplication. This layer multiplies `tensor__reshape_output_0` by the weights (`tensor_fc_weight`) and adds the bias (`tensor_fc_bias`). The result is stored in `tensor_output`. The algorithm is standard matrix multiplication (alpha and beta are set to 1). The function handles the transposition of the matrices efficiently.


## 5. Error Handling

The `predict` function uses `catch` blocks to handle potential errors during memory allocation and tensor operations. If any error occurs, the function returns immediately, preventing further execution and data corruption.  Error information would be logged if a logging function is set using `setLogFunction`.
