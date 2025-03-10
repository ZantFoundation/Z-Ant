# Zant ONNX Code Generation: Math Operations Handler

## Table of Contents

* [1. Introduction](#1-introduction)
* [2. `write_math_op` Function](#2-write_mathop-function)
* [3. `write_op_info` Function](#3-write-opinfo-function)
* [4. `write_add` Function](#4-write-add-function)
* [5. `write_conv` Function](#5-write-conv-function)
* [6. `write_concat` Function](#6-write-concat-function)
* [7. `write_constant` Function](#7-write-constant-function)
* [8. `write_div` Function](#8-write-div-function)
* [9. `write_gather` Function](#9-write-gather-function)
* [10. `write_gemm` Function](#10-write-gemm-function)
* [11. `write_matmul` Function](#11-write-matmul-function)
* [12. `write_maxPool` Function](#12-write-maxpool-function)
* [13. `write_mul` Function](#13-write-mul-function)
* [14. `write_reduceMean` Function](#14-write-reducemean-function)
* [15. `write_ReLU` Function](#15-write-relu-function)
* [16. `write_reshape` Function](#16-write-reshape-function)
* [17. `write_sigmoid` Function](#17-write-sigmoid-function)
* [18. `write_slice` Function](#18-write-slice-function)
* [19. `write_softmax` Function](#19-write-softmax-function)
* [20. `write_sum` Function](#20-write-sum-function)
* [21. `write_shape` Function](#21-write-shape-function)
* [22. `write_unsqueeze` Function](#22-write-unsqueeze-function)
* [23. `write_transpose` Function](#23-write-transpose-function)
* [24. `compute_output_shape` Function](#24-compute-output-shape-function)
* [25. Shape Computation Methods](#25-shape-computation-methods)


<a name="1-introduction"></a>
## 1. Introduction

This document details the implementation of the ONNX operator code generation within the Zant framework.  The code maps ONNX operations to corresponding Zant LeanTensorMath methods.  Detailed ONNX operator specifications can be found at [https://onnx.ai/onnx/operators/?utm_source=chatgpt.com](https://onnx.ai/onnx/operators/?utm_source=chatgpt.com).


<a name="2-write-mathop-function"></a>
## 2. `write_math_op` Function

This function is the main entry point for writing ONNX math operations to a file. It takes a file writer and a `ReadyNode` (containing ONNX node information) as input.  The function dispatches to specific handler functions based on the `op_type` of the ONNX node. If an `op_type` is not handled, it returns an `error.OperationNotSupported`.  Conditional logging and operation information writing are included based on `codegen_options`.

| Parameter | Type                     | Description                                                                   |
| --------- | ------------------------ | ----------------------------------------------------------------------------- |
| `writer`  | `std.fs.File.Writer`     | File writer to write the generated code.                                      |
| `node`    | `*ReadyNode`             | Pointer to the `ReadyNode` struct containing ONNX node information.           |


<a name="3-write-opinfo-function"></a>
## 3. `write_op_info` Function

This helper function writes informative comments to the output file, detailing the ONNX operation being performed, its inputs, and outputs. It's called by `write_math_op` when `codegen_options.comm` is true.


<a name="4-write-add-function"></a>
## 4. `write_add` Function

This function generates code for the ONNX `Add` operation. It uses the `tensMath.sum_tensors_lean` function from Zant's LeanTensorMath library to perform element-wise addition of two tensors.


<a name="5-write-conv-function"></a>
## 5. `write_conv` Function

This function generates code for the ONNX `Conv` operation. It extracts attributes (`auto_pad`, `dilations`, `group`, `kernel_shape`, `pads`, `strides`) from the `NodeProto`.  It handles optional bias tensor B, mandatory strides and optional pads and dilations.  The generated code calls `tensMath.conv_lean` from the Zant LeanTensorMath library. The function meticulously handles attribute parsing and error checking, ensuring that required attributes are present and of the correct type.  Default values are used where appropriate.


<a name="6-write-concat-function"></a>
## 6. `write_concat` Function

This function generates code for the ONNX `Concat` operation. It retrieves the `axis` attribute.  A special handling is implemented for axis 0 when tensors have different ranks. Otherwise, it creates a list of input tensors and calls `tensMath.concatenate`.


<a name="7-write-constant-function"></a>
## 7. `write_constant` Function

Generates code for the ONNX `Constant` operation.  It handles various attribute types (`value`, `value_float`, `value_floats`, `value_int`, `value_ints`, `value_string`, `value_strings`, `sparse_value`) to initialize constant tensors. String constants are currently handled as placeholders.


<a name="8-write-div-function"></a>
## 8. `write_div` Function

Generates code for the ONNX `Div` operation using the `tensMath.div_lean` function.


<a name="9-write-gather-function"></a>
## 9. `write_gather` Function

Generates code for the ONNX `Gather` operation.  It handles the `axis` attribute and utilizes `tensMath.gather_lean`.  The indices are converted to `usize` for use by the LeanTensorMath functions.


<a name="10-write-gemm-function"></a>
## 10. `write_gemm` Function

Generates code for the ONNX `Gemm` operation. This function extracts attributes (`alpha`, `beta`, `transA`, `transB`) and handles optional input tensor C.  It utilizes the Zant LeanTensorMath `gemm_lean` function.


<a name="11-write-matmul-function"></a>
## 11. `write_matmul` Function

Generates code for the ONNX `MatMul` operation using `tensMath.mat_mul_lean`.  Handles cases where input B is a static parameter.


<a name="12-write-maxpool-function"></a>
## 12. `write_maxPool` Function

Generates code for the ONNX `MaxPool` operation.  It extracts numerous attributes (`auto_pad`, `ceil_mode`, `dilations`, `kernel_shape`, `pads`, `strides`, `storage_order`) and calls `tensMath.onnx_maxpool_lean`.  Error handling is implemented for missing required attributes.


<a name="13-write-mul-function"></a>
## 13. `write_mul` Function

Generates code for the ONNX `Mul` operation using `tensMath.mul_lean`. Handles cases where input B is a static parameter.


<a name="14-write-reducemean-function"></a>
## 14. `write_reduceMean` Function

Generates code for the ONNX `ReduceMean` operation using `tensMath.reduce_mean_lean`.  Handles `keepdims` and `noop_with_empty_axes` attributes and optional axes input.


<a name="15-write-relu-function"></a>
## 15. `write_ReLU` Function

Generates code for the ONNX `Relu` operation using `tensMath.ReLU_lean`.


<a name="16-write-reshape-function"></a>
## 16. `write_reshape` Function

Generates code for the ONNX `Reshape` operation using `tensMath.reshape_lean`.  Handles the `allowzero` attribute and retrieves the new shape from the second input tensor.


<a name="17-write-sigmoid-function"></a>
## 17. `write_sigmoid` Function

Generates code for the ONNX `Sigmoid` operation using `tensMath.sigmoid_lean`.


<a name="18-write-slice-function"></a>
## 18. `write_slice` Function

Generates code for the ONNX `Slice` operation using `tensMath.lean_slice_onnx`.  Handles optional `axes` and `steps` inputs.


<a name="19-write-softmax-function"></a>
## 19. `write_softmax` Function

Generates code for the ONNX `Softmax` operation using `tensMath.softmax_lean`.


<a name="20-write-sum-function"></a>
## 20. `write_sum` Function

Generates code for the ONNX `Sum` operation using `tensMath.sum_tensor_list_lean`.  It creates a list of input tensors to sum.


<a name="21-write-shape-function"></a>
## 21. `write_shape` Function

Generates code for the ONNX `Shape` operation using `tensMath.shape_onnx_lean`. Handles `start` and `end` attributes.


<a name="22-write-unsqueeze-function"></a>
## 22. `write_unsqueeze` Function

Generates code for the ONNX `Unsqueeze` operation using `tensMath.unsqueeze_lean`.  It handles axes input either from a tensor input or an attribute.


<a name="23-write-transpose-function"></a>
## 23. `write_transpose` Function

Generates code for the ONNX `Transpose` operation using `tensMath.transpose_onnx_lean`.  It handles the `perm` attribute.


<a name="24-compute-output-shape-function"></a>
## 24. `compute_output_shape` Function

This function computes the output shape for a given ONNX node.  It dispatches to specific shape computation functions based on the node's `op_type`.  It handles cases where the output shape calculation is not implemented, returning an `error.OperationWIP` or `error.OperationNotSupported`.



<a name="25-shape-computation-methods"></a>
## 25. Shape Computation Methods

This section contains helper functions for computing output shapes for specific ONNX operators.  Each function (`compute_constant_output_shape`, `compute_ReLU_output_shape`, `compute_reshape_output_shape`, `compute_softmax_output_shape`, `compute_gemm_output_shape`, `compute_mul_output_shape`, `compute_conv_output_shape`, `compute_maxPool_output_shape`, `compute_reduceMean_output_shape`, `compute_slice_output_shape`, `compute_shape_output_shape`, `compute_gather_output_shape`, `compute_sigmoid_output_shape`, `compute_transpose_output_shape`, `compute_unsqueeze_output_shape`, `compute_concat_output_shape`) implements the shape inference logic for its corresponding ONNX operator.  These functions extensively use debugging print statements to aid in development and troubleshooting.  They also rigorously check input conditions and return appropriate errors if necessary.  The `compute_concat_output_shape` function, for example, has sophisticated logic to handle concatenation along axis 0 with tensors of different ranks, a scenario not directly supported by standard concatenation functions.
